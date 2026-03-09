import configparser

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

CONF_TH = 0.5
DISP_TH = 0


class SlopeEstimator:
    def __init__(self, road_index=0, debug=False):
        self.road_index = road_index
        self.debug = debug

    def set_config_params(self, config_path):
        self.cam_params = self.__load_cam_params(config_path)

    def run(self, pred_index, conf_map, disp_map):
        # 1. 고정밀 마스크 생성 (도로 영역 + 확신도 높음 + 노이즈 제외)
        # Disparity가 0이면 거리가 무한대이므로 제외해야 합니다.
        if pred_index.shape != disp_map.shape:
            pred_index_resized = cv2.resize(
                pred_index,
                (disp_map.shape[1], disp_map.shape[0]),  # (width, height) 순서 주의
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            pred_index_resized = pred_index

        valid_mask = (
            (pred_index_resized == self.road_index)
            & (conf_map > CONF_TH)
            & (disp_map > DISP_TH)
        )

        y_coords, x_coords = np.where(valid_mask)
        if len(x_coords) < 10:  # 최소 10개 이상의 포인트가 있을 때만 실행
            print('Warning: 유효한 도로 영역 데이터가 부족합니다. (Skipping...)')
            return 0.0, None  # 기본값 또는 에러를 알릴 수 있는 값 반환

        if self.debug:
            print('pred shape:', pred_index.shape)
            print(
                'conf shape:',
                conf_map.shape,
                'min/max:',
                conf_map.min(),
                conf_map.max(),
            )
            print(
                'disp shape:',
                disp_map.shape,
                'min/max:',
                disp_map.min(),
                disp_map.max(),
            )
            print('pred unique:', np.unique(pred_index))

            # 예시: road class = 0 이 아니라 8일 수도 있으니 확인 필요
            debug_road_mask = pred_index == self.road_index
            print(
                'road pixels:', debug_road_mask.sum(), 'ratio:', debug_road_mask.mean()
            )

            debug_conf_valid = conf_map > CONF_TH
            print(
                'conf valid:', debug_conf_valid.sum(), 'ratio:', debug_conf_valid.mean()
            )

            debug_disp_valid = disp_map > 0
            print(
                'disp valid:', debug_disp_valid.sum(), 'ratio:', debug_disp_valid.mean()
            )

            debug_valid_mask = debug_road_mask & debug_conf_valid & debug_disp_valid
            print(
                'final valid:',
                debug_valid_mask.sum(),
                'ratio:',
                debug_valid_mask.mean(),
            )

        # 2. Disparity를 실제 거리(Depth, Z)로 변환
        # 공식: Z = (fx * baseline) / disparity
        # 주의: disp_map의 값이 픽셀 단위인지 확인이 필요합니다.
        actual_disp = disp_map[y_coords, x_coords]
        z_depth = (self.cam_params['fx'] * self.cam_params['baseline']) / actual_disp

        # 3. 3D 공간 좌표 복원 (카메라 좌표계)
        # X = (x - cx) * Z / fx
        # Y = (y - cy) * Z / fy
        real_x = (x_coords - self.cam_params['cx']) * z_depth / self.cam_params['fx']
        real_y = (y_coords - self.cam_params['cy']) * z_depth / self.cam_params['fy']

        # 4. RANSAC 회귀 분석
        # 입력(X): 실제 바닥면의 X, Z 좌표 / 타겟(y): 실제 높이인 Y 좌표
        X_input = np.column_stack((real_x, z_depth))
        y_target = real_y

        model = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=1000,
            residual_threshold=30.0,
            max_trials=300,
            max_skips=500,
            stop_probability=0.99,
            loss='absolute_error',
            random_state=42,
        )
        model.fit(X_input, y_target)

        # 5. 경사도 계산
        # z_depth(종방향)에 대한 real_y(높이)의 변화율이 실제 경사입니다.
        slope_rate = model.estimator_.coef_[1]
        actual_slope_z = -slope_rate

        # angle_rad = np.arctan(actual_slope_z)
        # angle_deg = np.degrees(angle_rad)  # 각도(degree)
        # grade_pct = actual_slope_z * 100  # 경사율(%)

        # 카메라 피치(Pitch) 보정: 카메라가 아래를 향하고 있다면 해당 각도만큼 보정 필요
        rad = np.arctan(actual_slope_z)
        deg = np.degrees(rad)
        real_slope_deg = deg - self.cam_params['pitch_deg']

        if self.debug:
            print('valid count:', len(x_coords))
            print('z min/max:', z_depth.min(), z_depth.max())
            print('real_y min/max:', real_y.min(), real_y.max())
            print('coef:', model.estimator_.coef_)
            print('intercept:', model.estimator_.intercept_)

            inlier_mask = model.inlier_mask_
            print('inlier count:', inlier_mask.sum())
            print('inlier ratio:', inlier_mask.mean())

            print('pred shape:', pred_index.shape)
            print('disp shape:', disp_map.shape)
            print('conf shape:', conf_map.shape)

            print('disp min/max:', disp_map.min(), disp_map.max())
            print('conf min/max:', conf_map.min(), conf_map.max())

            print(
                f'real_slope_deg : {real_slope_deg} pitch{self.cam_params["pitch_deg"]}'
            )

        return real_slope_deg, valid_mask

    def __load_cam_params(self, config_path, resolution='2K'):
        config = configparser.ConfigParser()
        config.read(config_path)

        # 1. 내재 파라미터 (Intrinsic)
        section_name = f'LEFT_CAM_{resolution}'
        params = {
            'fx': config.getfloat(section_name, 'fx'),
            'fy': config.getfloat(section_name, 'fy'),
            'cx': config.getfloat(section_name, 'cx'),
            'cy': config.getfloat(section_name, 'cy'),
            'baseline': config.getfloat('STEREO', 'BaseLine'),
        }

        # 2. 외재 파라미터 (Extrinsic) - Pitch 값 추출
        # RX 값이 Pitch를 의미함
        rx_key = f'RX_{resolution}'
        pitch_rad = config.getfloat('STEREO', rx_key)

        # 만약 RX가 라디안 단위라면 도(degree)로 변환
        params['pitch_deg'] = np.degrees(pitch_rad)

        return params
