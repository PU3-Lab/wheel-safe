import configparser

import numpy as np
from sklearn.linear_model import RANSACRegressor


class SlopeEstimator:
    def __init__(self, config_path, road_index=0):
        self.model = RANSACRegressor(residual_threshold=0.05, random_state=42)
        self.cam_params = self.__load_cam_params(config_path)
        self.road_index = road_index

    def run(self, pred_index, conf_map, disp_map):
        # 1. 고정밀 마스크 생성 (도로 영역 + 확신도 높음 + 노이즈 제외)
        # Disparity가 0이면 거리가 무한대이므로 제외해야 합니다.
        valid_mask = (pred_index == self.road_index) & (conf_map > 0.8) & (disp_map > 0)

        y_coords, x_coords = np.where(valid_mask)
        if len(x_coords) < 100:
            return None

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
        Y_target = real_y

        ransac = RANSACRegressor(residual_threshold=0.05)  # 5cm 오차 허용
        ransac.fit(X_input, Y_target)

        # 5. 경사도 계산
        # z_depth(종방향)에 대한 real_y(높이)의 변화율이 실제 경사입니다.
        slope_rate = ransac.estimator_.coef_[1]
        actual_slope_z = -slope_rate

        # angle_rad = np.arctan(actual_slope_z)
        # angle_deg = np.degrees(angle_rad)  # 각도(degree)
        # grade_pct = actual_slope_z * 100  # 경사율(%)

        # 카메라 피치(Pitch) 보정: 카메라가 아래를 향하고 있다면 해당 각도만큼 보정 필요
        rad = np.arctan(actual_slope_z)
        deg = np.degrees(np.arctan(rad))
        real_slope_deg = deg - self.cam_params['pitch_deg']

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
