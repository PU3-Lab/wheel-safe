import configparser
from pathlib import Path

import cv2
import numpy as np


class SlopeAggregator:
    """
    Disparity16 (scaled by 16) + confidence map => estimate ground slope.

    Key improvements vs. original:
    - Do NOT clamp invalid disparity to tiny values (prevents Z explosion).
    - Build valid mask first, then sample valid points for geometry.
    - Use RANSAC plane fitting on 3D points for a stable ground normal.
    - Provide per-pixel slope map as angle between per-pixel normal and vertical (optional),
      and also a "plane slope map" w.r.t. the estimated ground plane.
    """

    def __init__(
        self,
        conf_path: str | None = None,
        mode: str = 'LEFT_CAM_FHD',
        *,
        # If you use cropped images (e.g., 1920x592 from 1920x1080),
        # and your cx, cy are for the original image, set crop offsets:
        crop_x: int = 0,
        crop_y: int = 0,
    ):
        # Defaults (override via .conf)
        self.fx, self.fy = 1400.15, 1400.15
        self.cx, self.cy = 943.093, 559.187
        self.baseline = 0.12  # meters

        # Analysis params
        self.conf_threshold = 230
        self.min_dist = 0.8
        self.max_dist = 10.0
        self.roi_ratio = 0.35

        # RANSAC params
        self.ransac_iters = 400
        self.ransac_inlier_thresh = 0.03  # meters (3cm)
        self.max_points = 60_000  # sample cap for speed

        # Crop offsets (principal point shift)
        self.crop_x = int(crop_x)
        self.crop_y = int(crop_y)

        if conf_path and Path(conf_path).exists():
            self._load_config(conf_path, mode)

    def _load_config(self, conf_path, mode):
        config = configparser.ConfigParser()
        try:
            config.read(conf_path)

            if mode in config:
                self.fx = float(config[mode]['fx'])
                self.fy = float(config[mode]['fy'])
                self.cx = float(config[mode]['cx'])
                self.cy = float(config[mode]['cy'])

            if 'STEREO' in config and 'BaseLine' in config['STEREO']:
                # mm -> m
                self.baseline = float(config['STEREO']['BaseLine']) / 1000.0

        except Exception as e:
            print(f'설정 로드 실패: {e}')

    @staticmethod
    def _plane_from_3pts(p1, p2, p3):
        """Return (n, d) for plane n·x + d = 0, with |n|=1. If degenerate, return None."""
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
    def _point_plane_dist(points, n, d):
        """Signed distance."""
        return points @ n + d

    def _ransac_plane(self, points):
        """
        Fit a plane using RANSAC.
        points: (N,3)
        return: best_n, best_d, inlier_mask
        """
        N = points.shape[0]
        if N < 200:
            return None, None, None

        # Random sampling indices
        rng = np.random.default_rng(42)
        best_inliers = None
        best_count = 0
        best_model = None

        for _ in range(self.ransac_iters):
            idx = rng.choice(N, size=3, replace=False)
            p1, p2, p3 = points[idx]
            model = self._plane_from_3pts(p1, p2, p3)
            if model is None:
                continue

            n, d = model

            # Prefer "ground-like" planes: normal should be reasonably close to vertical
            # (vertical axis defined as (0,-1,0) in your coordinate convention)
            vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(n, vertical))) < np.cos(np.radians(55)):  # too wall-ish
                continue

            dist = np.abs(self._point_plane_dist(points, n, d))
            inliers = dist < self.ransac_inlier_thresh
            count = int(inliers.sum())

            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_model = (n, d)

        if best_model is None or best_inliers is None or best_count < 300:
            return None, None, None

        # Refit using least squares on inliers (SVD)
        inlier_pts = points[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        Q = inlier_pts - centroid
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        n = vh[-1, :]
        n = n / (np.linalg.norm(n) + 1e-9)
        d = -float(np.dot(n, centroid))

        # Make normal point "upwards-ish" (match vertical sign convention)
        vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        if float(np.dot(n, vertical)) < 0:
            n = -n
            d = -d

        return n.astype(np.float32), float(d), best_inliers

    def calculate_slope(self, disp16_path, conf_path, external_mask=None):
        disp16_raw = cv2.imread(str(disp16_path), cv2.IMREAD_UNCHANGED)
        conf_raw = cv2.imread(str(conf_path), cv2.IMREAD_GRAYSCALE)

        if disp16_raw is None or conf_raw is None:
            return {
                'avg_slope': 0.0,
                'slope_map_roi': None,
                'valid_mask': None,
                'roi_y_range': None,
            }

        # Confidence binary mask
        _, conf_mask = cv2.threshold(
            conf_raw, self.conf_threshold, 255, cv2.THRESH_BINARY
        )

        # Smooth disparity a bit (median can be okay; keep small to preserve edges)
        disp16_filtered = cv2.medianBlur(disp16_raw, 5)

        h, w = disp16_filtered.shape[:2]
        roi_v_start = h - int(h * self.roi_ratio)
        roi_v_end = h

        disp_roi = disp16_filtered[roi_v_start:roi_v_end, :].astype(np.float32)
        conf_roi = conf_mask[roi_v_start:roi_v_end, :]

        # Disparity (scaled by 16)
        d = disp_roi / 16.0

        # Build validity BEFORE computing Z/geometry
        valid = (conf_roi > 0) & (d > 0.5)  # disparity > 0.5px as a sane minimum

        # Build validity BEFORE computing Z/geometry
        valid_conf = conf_roi > 0
        valid_disp = d > 0.5
        valid = valid_conf & valid_disp

        print('[DEBUG] ROI shape:', disp_roi.shape)
        print('[DEBUG] conf_raw min/max:', int(conf_raw.min()), int(conf_raw.max()))
        print('[DEBUG] conf_roi>0 ratio:', float(valid_conf.mean()))
        print('[DEBUG] d min/max:', float(np.nanmin(d)), float(np.nanmax(d)))
        print('[DEBUG] d>0.5 ratio:', float(valid_disp.mean()))
        print('[DEBUG] valid (before ext mask) ratio:', float(valid.mean()))

        if external_mask is not None:
            if external_mask.shape[:2] != (h, w):
                ext = cv2.resize(external_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                ext = external_mask

            model_mask_roi = ext[roi_v_start:roi_v_end, :] > 0
            print('[DEBUG] ext mask roi>0 ratio:', float(model_mask_roi.mean()))

            valid &= model_mask_roi
            print('[DEBUG] valid (after ext mask) ratio:', float(valid.mean()))
        else:
            print('[DEBUG] external_mask: None')

        if external_mask is not None:
            # 1) 외부 마스크가 disp/conf와 같은 해상도인지 확인
            if external_mask.shape[:2] != (h, w):
                # segmentation이 다른 해상도에서 왔을 가능성 대비
                ext = cv2.resize(
                    external_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                ext = external_mask

        # 2) ROI에 맞춰 자르고, 0/1 이든 0/255든 상관없이 처리
        model_mask_roi = ext[roi_v_start:roi_v_end, :] > 0
        valid &= model_mask_roi

        if not np.any(valid):
            return {
                'avg_slope': 0.0,
                'slope_map_roi': np.zeros_like(d, dtype=np.float32),
                'valid_mask': valid,
                'roi_y_range': (roi_v_start, roi_v_end),
            }

        # Depth (meters) only for valid pixels
        Z = np.full_like(d, np.nan, dtype=np.float32)
        Z[valid] = (self.fx * self.baseline) / d[valid]

        # Distance filtering
        valid &= np.isfinite(Z) & (self.min_dist < Z) & (self.max_dist > Z)
        if not np.any(valid):
            return {
                'avg_slope': 0.0,
                'slope_map_roi': np.zeros_like(d, dtype=np.float32),
                'valid_mask': valid,
                'roi_y_range': (roi_v_start, roi_v_end),
            }

        # Pixel grid (note: apply crop offsets to principal point if needed)
        # If cx,cy come from original image but you're using a crop, set crop_y accordingly.
        cx = self.cx - self.crop_x
        cy = self.cy - self.crop_y

        u, v = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(roi_v_start, roi_v_end, dtype=np.float32),
        )

        # 3D reconstruction for valid pixels only
        X = np.full_like(Z, np.nan, dtype=np.float32)
        Y = np.full_like(Z, np.nan, dtype=np.float32)
        X[valid] = (u[valid] - cx) * Z[valid] / self.fx
        Y[valid] = (v[valid] - cy) * Z[valid] / self.fy

        # Collect points for RANSAC
        pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)

        # Subsample for speed
        if pts.shape[0] > self.max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(pts.shape[0], size=self.max_points, replace=False)
            pts_fit = pts[idx]
        else:
            pts_fit = pts

        n, d_plane, inliers = self._ransac_plane(pts_fit)

        if n is None:
            # Fallback: simple median normal from local gradients (rough)
            # (kept minimal; better to rely on RANSAC)
            avg_slope = 0.0
            slope_map = np.zeros_like(d, dtype=np.float32)
            return {
                'avg_slope': avg_slope,
                'slope_map_roi': slope_map,
                'valid_mask': valid,
                'roi_y_range': (roi_v_start, roi_v_end),
                'plane_normal': None,
            }

        # Ground slope angle: angle between plane normal and vertical
        vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        cos_nv = np.clip(abs(float(np.dot(n, vertical))), 0.0, 1.0)
        avg_slope = float(np.degrees(np.arccos(cos_nv)))  # 0° flat, bigger = steeper

        # --- slope map (ROI) ---
        # Option 1 (recommended): per-pixel "plane deviation" angle:
        # angle between (estimated ground normal) and per-pixel normal is expensive.
        # Instead we make a simple "height residual" map OR a plane-based pseudo slope.
        #
        # Here: make a "plane residual" (meters) -> convert to pseudo angle by local scale is non-trivial,
        # so we provide a simpler and more interpretable map:
        # per-pixel "vertical slope" using depth gradients on a filled Z.

        # Fill NaNs in Z for gradient (inpaint requires 8-bit; use nearest fill)
        Z_fill = Z.copy()
        if np.any(~np.isfinite(Z_fill)):
            # nearest neighbor fill using distance transform
            mask_nan = (~np.isfinite(Z_fill)).astype(np.uint8)
            # Need finite pixels for distance transform; if too sparse, keep zeros
            if np.any(np.isfinite(Z_fill)):
                Z0 = Z_fill.copy()
                Z0[~np.isfinite(Z0)] = 0.0
                dist, labels = cv2.distanceTransformWithLabels(
                    mask_nan,
                    distanceType=cv2.DIST_L2,
                    maskSize=5,
                    labelType=cv2.DIST_LABEL_PIXEL,
                )
                labels = labels - 1
                # map labels -> nearest finite pixel coords
                coords = np.column_stack(np.where(np.isfinite(Z_fill)))
                # guard
                if coords.shape[0] > 0:
                    nearest = coords[np.clip(labels, 0, coords.shape[0] - 1)]
                    Z_fill[mask_nan.astype(bool)] = Z_fill[nearest[:, 0], nearest[:, 1]]

        dz_dv, dz_du = np.gradient(Z_fill)

        # Approx local surface slope angle from depth gradients (small-angle approx in camera space is imperfect,
        # but works as a stable "steepness" heatmap when masked to valid ground region)
        # Convert gradient magnitude to angle: tan(theta) ~ |grad(Z)|
        grad_mag = np.sqrt(dz_du**2 + dz_dv**2)
        slope_map = np.degrees(np.arctan(grad_mag)).astype(np.float32)
        slope_map[~valid] = 0.0

        return {
            'avg_slope': avg_slope,
            'slope_map_roi': slope_map,
            'valid_mask': valid,
            'roi_y_range': (roi_v_start, roi_v_end),
            'plane_normal': n,
            'plane_d': d_plane,
        }
