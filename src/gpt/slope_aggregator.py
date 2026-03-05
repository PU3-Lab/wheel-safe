from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np


class StatsDic(TypedDict, total=False):
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    disp_scale: float
    roi_ratio: float
    conf_th: float
    min_d: float
    max_d: float
    conf_ratio: float
    disp_ratio: float
    dist_ratio: float
    external_mask_ratio: float
    final_valid_ratio: float
    final_valid_count: int
    slope_method: str | None
    yz_a: float | None
    yz_b: float | None
    yz_inlier_ratio: float | None
    yz_inlier_thresh_m: float | None
    plane_normal: list[float] | None
    plane_d: float | None
    ransac_inlier_ratio: float | None
    ransac_fit_points: int | None
    plane_avg: float | None
    plane_signed: float | None
    plane_inlier_thresh_m: float


@dataclass
class CameraParams:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    rx_pitch_rad: float | None = None  # optional


class SlopeAggregator:
    """
    Depth_*.conf에서 카메라 파라미터 로드
    disp16 + confidence_save + external_mask로 valid 생성
    1) (권장) Y-Z 선형 RANSAC: Y = a*Z + b  -> slope = atan(a)
    2) (디버그/백업) 3D plane RANSAC: n·p + d = 0 -> slope from normal vs vertical
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        mode: str = 'LEFT_CAM_FHD',
        stereo_section: str = 'STEREO',
        # disp16 scale
        disp_scale: float = 16.0,
        # filters
        conf_th: int = 40,
        min_d: float = 1.0,
        max_d: float = 8.0,
        roi_ratio: float = 0.25,
        # ransac
        ransac_iters: int = 500,
        ransac_inlier_thresh_m: float = 0.05,  # for plane (meters)
        yz_inlier_thresh_m: float = 0.02,  # for YZ line (meters)
        max_points: int = 50000,
        debug_print: bool = True,
    ):
        self.config_path = Path(config_path)
        self.mode = mode
        self.stereo_section = stereo_section

        self.disp_scale = float(disp_scale)

        self.conf_th = int(conf_th)
        self.min_d = float(min_d)
        self.max_d = float(max_d)
        self.roi_ratio = float(roi_ratio)

        self.ransac_iters = int(ransac_iters)
        self.ransac_inlier_thresh_m = float(ransac_inlier_thresh_m)
        self.yz_inlier_thresh_m = float(yz_inlier_thresh_m)
        self.max_points = int(max_points)

        self.debug_print = bool(debug_print)

        self.cam = self._load_camera_params(
            self.config_path, self.mode, self.stereo_section
        )

    # -------------------------
    # Config parsing
    # -------------------------
    def _load_camera_params(
        self, conf_path: Path, mode: str, stereo_section: str
    ) -> CameraParams:
        if not conf_path.exists():
            raise FileNotFoundError(f'config not found: {conf_path}')

        cfg = configparser.ConfigParser()
        cfg.read(conf_path)

        if mode not in cfg:
            raise KeyError(
                f'mode section not found in conf: {mode} (available: {list(cfg.sections())})'
            )
        if stereo_section not in cfg:
            raise KeyError(f'stereo section not found in conf: {stereo_section}')

        fx = float(cfg[mode]['fx'])
        fy = float(cfg[mode].get('fy', cfg[mode]['fx']))
        cx = float(cfg[mode].get('cx', 0.0))
        cy = float(cfg[mode]['cy'])

        baseline_raw = float(cfg[stereo_section]['BaseLine'])
        baseline_m = baseline_raw / 1000.0 if baseline_raw > 10 else baseline_raw

        rx_key = None
        for k in ('RX_FHD', 'RX_2K', 'RX_HD', 'RX_VGA'):
            if k in cfg[stereo_section]:
                rx_key = k
                break
        rx_pitch = float(cfg[stereo_section][rx_key]) if rx_key else None

        print('rx_pitch', rx_pitch)

        return CameraParams(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline_m=baseline_m,
            rx_pitch_rad=rx_pitch,
        )

    # -------------------------
    # IO
    # -------------------------
    def _read_disp(self, disp_path: str | Path) -> np.ndarray:
        disp16 = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        if disp16 is None:
            raise RuntimeError(f'Failed to read disparity: {disp_path}')
        return disp16.astype(np.float32) / self.disp_scale

    def _read_conf_img(self, conf_img_path: str | Path) -> np.ndarray:
        conf = cv2.imread(str(conf_img_path), cv2.IMREAD_UNCHANGED)
        if conf is None:
            raise RuntimeError(f'Failed to read confidence image: {conf_img_path}')
        if conf.ndim == 3:
            conf = cv2.cvtColor(conf.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if conf.dtype != np.uint8:
            conf = np.clip(conf, 0, 255).astype(np.uint8)
        return conf

    # -------------------------
    # Geometry
    # -------------------------
    def _make_3d(self, disp: np.ndarray):
        """
        disp: (H,W) float disparity (pixels)
        returns X,Y,Z in meters
        """
        h, w = disp.shape
        u = np.arange(w, dtype=np.float32)[None, :]
        v = np.arange(h, dtype=np.float32)[:, None]

        Z = (self.cam.fx * self.cam.baseline_m) / np.maximum(disp, 1e-6)
        Z[disp <= 0] = np.inf

        X = (u - self.cam.cx) * Z / self.cam.fx
        Y = (v - self.cam.cy) * Z / self.cam.fy
        return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)

    # ----- Plane RANSAC (디버그/백업) -----
    @staticmethod
    def _plane_from_3pts(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return None
        n = n / nn
        d = -float(np.dot(n, p1))
        return n, d

    @staticmethod
    def _dist_to_plane(points: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
        return points @ n + d

    def _ransac_plane(self, points: np.ndarray):
        N = points.shape[0]
        if N < 300:
            return None, None, None

        rng = np.random.default_rng(42)
        best_model = None
        best_inliers = None
        best_count = 0

        for _ in range(self.ransac_iters):
            idx = rng.choice(N, size=3, replace=False)
            p1, p2, p3 = points[idx]
            model = self._plane_from_3pts(p1, p2, p3)
            if model is None:
                continue
            n, d = model

            dist = np.abs(self._dist_to_plane(points, n, d))
            inliers = dist < self.ransac_inlier_thresh_m
            count = int(inliers.sum())

            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_model = (n, d)

        if best_model is None or best_inliers is None or best_count < 300:
            return None, None, None

        # refine
        P = points[best_inliers]
        centroid = P.mean(axis=0)
        Q = P - centroid
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        n = vh[-1]
        n = n / (np.linalg.norm(n) + 1e-9)
        d = -float(np.dot(n, centroid))

        # final inliers
        dist = np.abs(self._dist_to_plane(points, n, d))
        inliers = dist < self.ransac_inlier_thresh_m

        return n.astype(np.float32), float(d), inliers

    # ----- YZ Line RANSAC (권장) -----
    def _ransac_line_yz(self, yz: np.ndarray):
        """
        Robust fit for Y = a*Z + b using RANSAC.
        yz: (N,2) columns [Y, Z]
        returns: a, b, inliers_mask (N,)
        """
        N = yz.shape[0]
        if N < 300:
            return None, None, None

        Y = yz[:, 0].astype(np.float32)
        Z = yz[:, 1].astype(np.float32)

        rng = np.random.default_rng(42)
        best_inliers = None
        best_count = 0

        thr = float(self.yz_inlier_thresh_m)

        for _ in range(self.ransac_iters):
            i1, i2 = rng.choice(N, size=2, replace=False)
            z1, y1 = float(Z[i1]), float(Y[i1])
            z2, y2 = float(Z[i2]), float(Y[i2])

            dz = z2 - z1
            if abs(dz) < 1e-4:
                continue

            a = (y2 - y1) / dz
            b = y1 - a * z1

            residual = np.abs(Y - (a * Z + b))
            inliers = residual < thr
            count = int(inliers.sum())

            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < 300:
            return None, None, None

        # refine LS on inliers
        Zi = Z[best_inliers]
        Yi = Y[best_inliers]
        A = np.column_stack([Zi, np.ones_like(Zi)])
        (a_ls, b_ls), *_ = np.linalg.lstsq(A, Yi, rcond=None)

        # final inliers against refit
        residual = np.abs(Y - (a_ls * Z + b_ls))
        final_inliers = residual < thr

        return float(a_ls), float(b_ls), final_inliers

    # -------------------------
    # Main
    # -------------------------
    def calculate_slope(
        self,
        disp_path: str | Path,
        conf_img_path: str | Path,
        external_mask: np.ndarray | None = None,
    ) -> dict:
        disp = self._read_disp(disp_path)
        conf = self._read_conf_img(conf_img_path)

        h, w = disp.shape

        # ROI (bottom)
        y0 = int(h * (1.0 - self.roi_ratio))
        roi = np.zeros((h, w), dtype=bool)
        roi[y0:, :] = True

        # dist
        dist = (self.cam.fx * self.cam.baseline_m) / np.maximum(disp, 1e-6)
        dist[disp <= 0] = np.inf

        # valid
        valid = (
            roi
            & (disp > 0)
            & (conf >= self.conf_th)
            & np.isfinite(dist)
            & (dist > self.min_d)
            & (dist < self.max_d)
        )

        # external mask
        em_ratio = 0.0
        if external_mask is not None:
            em = external_mask
            if em.ndim == 3:
                em = cv2.cvtColor(em.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            if em.shape[:2] != (h, w):
                em = cv2.resize(
                    em.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
            em = em > 0
            em_ratio = float(em.mean())
            valid &= em

        stats: StatsDic = {
            'fx': self.cam.fx,
            'fy': self.cam.fy,
            'cx': self.cam.cx,
            'cy': self.cam.cy,
            'baseline_m': self.cam.baseline_m,
            'disp_scale': self.disp_scale,
            'roi_ratio': float(roi.mean()),
            'conf_th': self.conf_th,
            'min_d': self.min_d,
            'max_d': self.max_d,
            'conf_ratio': float((conf >= self.conf_th).mean()),
            'disp_ratio': float((disp > 0).mean()),
            'dist_ratio': float(((dist > self.min_d) & (dist < self.max_d)).mean()),
            'external_mask_ratio': em_ratio,
            'final_valid_ratio': float(valid.mean()),
            'final_valid_count': int(valid.sum()),
            'slope_method': 'none',
            'yz_a': None,
            'yz_b': None,
            'yz_inlier_ratio': None,
            'plane_normal': None,
            'plane_d': None,
            'ransac_inlier_ratio': None,
            'ransac_fit_points': 0,
            'plane_avg': None,
            'plane_signed': None,
        }

        if self.debug_print:
            print('[SLOPE] stats:', stats)

        roi_total_count = roi.sum()
        valid_count = stats['final_valid_count']
        valid_ratio = valid_count / roi_total_count if roi_total_count > 0 else 0
        # 임계값 (예: 0.5%). 해상도에 상관없이 동작함.
        min_ratio_threshold = 5e-3

        if valid_ratio < min_ratio_threshold:
            return {
                'avg_slope': float('nan'),
                'signed_slope': float('nan'),
                'reason': f'too sparse (ratio: {valid_ratio:.4f})',
                'stats': stats,
                'valid': valid,
                'plane_normal': None,
                'plane_d': None,
                'inliers_mask': None,
                'yz_inliers_mask': None,
            }

        # 3D points
        X, Y, Z = self._make_3d(disp)
        pts = np.column_stack([X[valid], Y[valid], Z[valid]]).astype(np.float32)

        # subsample for speed
        if pts.shape[0] > self.max_points:
            rng = np.random.default_rng(0)
            sel = rng.choice(pts.shape[0], size=self.max_points, replace=False)
            pts_fit = pts[sel]
        else:
            pts_fit = pts

        # ------------------------------------------------------
        # (1) Y-Z line RANSAC (권장)
        # ------------------------------------------------------
        yz = np.column_stack([pts_fit[:, 1], pts_fit[:, 2]])  # [Y, Z]
        a_yz, b_yz, yz_inliers = self._ransac_line_yz(yz)

        yz_signed = None
        yz_avg = None
        yz_inlier_ratio = None
        if a_yz is not None:
            # dy/dz = a_yz
            # Y가 아래로 증가한다는 좌표계에서 "오르막=+"로 맞추기 위해 부호 뒤집음
            yz_signed = -float(np.degrees(np.arctan(a_yz)))
            yz_avg = float(abs(yz_signed))
            yz_inlier_ratio = float(yz_inliers.mean())

        stats.update(
            {
                'yz_a': a_yz,
                'yz_b': b_yz,
                'yz_inlier_ratio': yz_inlier_ratio,
                'yz_inlier_thresh_m': float(self.yz_inlier_thresh_m),
            }
        )

        # ------------------------------------------------------
        # (2) Plane RANSAC (디버그/백업 유지)
        # ------------------------------------------------------
        n, d, inliers = self._ransac_plane(pts_fit)

        plane_avg = None
        plane_signed = None
        if n is not None:
            # normal 방향 통일: ny < 0 (vertical = [0,-1,0] 기준)
            if float(n[1]) > 0:
                n = -n
                d = -d

            vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            cos_theta = np.clip(np.abs(float(np.dot(n, vertical))), 0.0, 1.0)
            plane_avg = float(np.degrees(np.arccos(cos_theta)))

            eps = 1e-6
            dy_dz = -float(n[2]) / (float(n[1]) + eps)
            # 오르막=+ 맞추기 위해 부호 뒤집음 (YZ 방식과 동일 기준)
            plane_signed = -float(np.degrees(np.arctan(dy_dz)))

            stats.update(
                {
                    'plane_normal': [float(x) for x in n],
                    'plane_d': float(d),
                    'ransac_inlier_ratio': float(inliers.mean())
                    if inliers is not None
                    else None,
                    'ransac_fit_points': int(pts_fit.shape[0]),
                    'plane_avg': plane_avg,
                    'plane_signed': plane_signed,
                    'plane_inlier_thresh_m': float(self.ransac_inlier_thresh_m),
                }
            )
        else:
            stats.update(
                {
                    'plane_normal': None,
                    'plane_d': None,
                    'ransac_inlier_ratio': None,
                    'ransac_fit_points': int(pts_fit.shape[0]),
                    'plane_avg': None,
                    'plane_signed': None,
                    'plane_inlier_thresh_m': float(self.ransac_inlier_thresh_m),
                }
            )

        # ------------------------------------------------------
        # 최종 slope 선택: YZ가 있으면 YZ 우선, 아니면 plane
        # ------------------------------------------------------
        if yz_avg is not None and np.isfinite(yz_avg):
            avg = yz_avg
            signed = yz_signed
            reason = 'ok'
            stats['slope_method'] = 'yz_line_ransac'
        elif plane_avg is not None and np.isfinite(plane_avg):
            avg = plane_avg
            signed = plane_signed
            reason = 'ok(plane_fallback)'
            stats['slope_method'] = 'plane_ransac'
        else:
            avg = float('nan')
            signed = float('nan')
            reason = 'fit failed'
            stats['slope_method'] = None

        if self.debug_print:
            print(
                '[SLOPE] method:',
                stats.get('slope_method'),
                'yz_inlier:',
                stats.get('yz_inlier_ratio'),
                'plane_inlier:',
                stats.get('ransac_inlier_ratio'),
            )

        # ======================================================
        # [수정 포인트 2] 결과 범위 체크 (리턴 직전)
        # ======================================================
        # 6.26도처럼 장애물 때문에 튀는 값을 잡기 위한 안전장치
        # 1. 설정값에서 읽어온 설치 Pitch 각도 (Radian -> Degree)
        pitch_offset_deg = 0.0
        if self.cam.rx_pitch_rad is not None:
            # RX 값이 양수인지 음수인지에 따라 설치 상태가 다름
            pitch_offset_deg = np.degrees(self.cam.rx_pitch_rad)

        if not np.isnan(avg):
            avg = avg + pitch_offset_deg
            signed = (signed or 0.0) - pitch_offset_deg
            reason = reason

        return {
            'avg_slope': float(avg) if np.isfinite(avg) else float('nan'),
            'signed_slope': float(signed)
            if signed is not None and np.isfinite(signed)
            else float('nan'),
            'reason': reason,
            'stats': stats,
            'valid': valid,
            # plane debug
            'plane_normal': stats.get('plane_normal'),
            'plane_d': stats.get('plane_d'),
            'inliers_mask': inliers,
            # yz debug
            'yz_inliers_mask': yz_inliers,
        }
