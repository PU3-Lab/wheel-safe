import configparser
from pathlib import Path

import cv2
import numpy as np


class GlobalSlopeAggregator:
    def __init__(self, conf_path=None, mode='LEFT_CAM_FHD'):
        # 1. 카메라 물리 파라미터 초기값 (INI 설정 대응)
        self.fx, self.fy = 1400.15, 1400.15
        self.cx, self.cy = 943.093, 559.187
        self.baseline = 119.975 / 1000.0  # mm -> m

        # 2. 분석 하이퍼파라미터 세팅
        self.conf_threshold = 230
        self.min_dist = 1.0  # 오르막 감지 시작 거리
        self.max_dist = 15.0  # 배경 노이즈 차단 거리
        self.roi_ratio = 0.2  # 지면 분석 영역 비율

        # 사다리꼴 가로 폭 설정 (원근 왜곡 보정용)
        self.top_width_ratio = 0.1
        self.bottom_width_ratio = 0.3

        if conf_path and Path(conf_path).exists():
            self._load_config(conf_path, mode)

    def _load_config(self, conf_path, mode):
        """INI 포맷의 카메라 명세를 읽어 파라미터를 업데이트합니다."""
        config = configparser.ConfigParser()
        try:
            config.read(conf_path)
            if mode in config:
                self.fx = float(config[mode]['fx'])
                self.fy = float(config[mode]['fy'])
                self.cx = float(config[mode]['cx'])
                self.cy = float(config[mode]['cy'])
            if 'STEREO' in config:
                self.baseline = float(config['STEREO']['BaseLine']) / 1000.0
        except Exception as e:
            print(f'설정 로드 실패: {e}')

    def calculate_slope(self, disp16_path, conf_path, external_mask=None):
        """
        경사도를 계산합니다.
        external_mask: ONNX 모델 등에서 생성한 '지면' 마스크 (Optional)
        """
        # 데이터 로드 (PNG 무손실 대응)
        disp16_img = cv2.imread(str(disp16_path), cv2.IMREAD_UNCHANGED)
        conf_img = cv2.imread(str(conf_path), cv2.IMREAD_GRAYSCALE)

        if disp16_img is None or conf_img is None:
            raise ValueError('이미지를 불러올 수 없습니다.')

        disp16_img = cv2.medianBlur(disp16_img, 5)  # 노이즈 제거

        # ROI 설정 및 3D 좌표 복원
        total_h, crop_h = 1080, 592
        _, end_y = (total_h - crop_h) // 2, (total_h - crop_h) // 2 + crop_h
        roi_v_start = end_y - int(crop_h * self.roi_ratio)
        roi_v_end = end_y

        d = (
            disp16_img[roi_v_start:roi_v_end, :].astype(np.float32) / 16.0
        )  # Sub-pixel 복원
        d[d < 0.1] = 0.1
        Z = (self.fx * self.baseline) / d  # 거리(Z) 계산

        u, v = np.meshgrid(np.arange(Z.shape[1]), np.arange(roi_v_start, roi_v_end))
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # 법선 벡터 계산 및 방향 정규화
        dz_dv, dz_du = np.gradient(Z)
        dx_dv, dx_du = np.gradient(X)
        dy_dv, dy_du = np.gradient(Y)
        normal_vectors = np.cross(
            np.stack([dx_du, dy_du, dz_du], axis=-1),
            np.stack([dx_dv, dy_dv, dz_dv], axis=-1),
        )
        normal_vectors /= np.linalg.norm(normal_vectors, axis=-1, keepdims=True) + 1e-6
        normal_vectors[normal_vectors[..., 1] > 0] *= -1  # 하늘 방향 고정

        # 물리 기반 마스크 생성 (사다리꼴 + 거리 + 벽면 필터)
        z_mask = (self.min_dist < Z) & (self.max_dist > Z)

        # 사다리꼴 가로 마스크
        width_mask = np.zeros_like(d, dtype=bool)
        for row in range(Z.shape[0]):
            curr_ratio = self.top_width_ratio + (row / Z.shape[0]) * (
                self.bottom_width_ratio - self.top_width_ratio
            )
            w_start, w_end = (
                int(Z.shape[1] * (0.5 - curr_ratio)),
                int(Z.shape[1] * (0.5 + curr_ratio)),
            )
            width_mask[row, w_start:w_end] = True

        # 수직 벽면 필터 (카메라 시선 방향 내적 이용)
        forward_vec = np.array([0, 0, 1])
        cos_with_forward = np.abs(np.sum(normal_vectors * forward_vec, axis=-1))
        wall_mask = cos_with_forward > np.cos(np.radians(40))

        vertical_vec = np.array([0, -1, 0])
        cos_with_vertical = np.abs(np.sum(normal_vectors * vertical_vec, axis=-1))
        ground_mask = cos_with_vertical > np.cos(np.radians(45))

        # 최종 유효 마스크 통합
        valid_mask = (
            (conf_img[roi_v_start:roi_v_end, :] > self.conf_threshold)
            & (d > 0.5)
            & z_mask
            & width_mask
            & (~wall_mask)
        )

        # 최종 유효 마스크 업데이트
        valid_mask &= (~wall_mask) & ground_mask

        # 모델 마스크(external_mask)가 제공된 경우 결합
        if external_mask is not None:
            model_mask_roi = external_mask[roi_v_start:roi_v_end, :].astype(bool)
            valid_mask &= model_mask_roi

        slope_angle_roi = np.zeros_like(d)
        if np.any(valid_mask):
            # 지면 절대 기울기(Pitch) 및 픽셀 편차 계산
            mean_n = np.median(normal_vectors[valid_mask], axis=0)
            mean_n /= np.linalg.norm(mean_n)

            vertical_vec = np.array([0, -1, 0])
            ground_pitch = np.arccos(np.abs(np.dot(mean_n, vertical_vec))) * (
                180.0 / np.pi
            )

            cos_theta = np.clip(
                np.abs(np.sum(normal_vectors * mean_n, axis=-1)), 0.0, 1.0
            )
            slope_angle_roi = ground_pitch + (np.arccos(cos_theta) * (180.0 / np.pi))

            avg_slope = np.mean(slope_angle_roi[valid_mask])
            max_slope = np.percentile(slope_angle_roi[valid_mask], 95)
        else:
            avg_slope, max_slope = 0.0, 0.0

        return {
            'avg_slope': avg_slope,
            'max_slope': max_slope,
            'slope_map_roi': slope_angle_roi,
            'valid_mask': valid_mask,
            'roi_y_range': (roi_v_start, roi_v_end),
        }
