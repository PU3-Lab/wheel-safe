import configparser

import numpy as np
from sklearn.linear_model import RANSACRegressor

CONF_TH = 0.5
DISP_TH = 0.2
MIN_POINTS = 30


class SlopeEstimator:
    def __init__(self, road_index=0, debug=False):
        self.road_index = road_index
        self.debug = debug

    def set_config_params(self, config_path):
        self.cam_params = self.__load_cam_params(config_path)

    def run(self, pred_index, conf_map, disp_map):
        valid_mask = (
            (pred_index == self.road_index)
            & (conf_map > CONF_TH)
            & (disp_map > DISP_TH)
        )

        y_coords, x_coords = np.where(valid_mask)
        n_points = len(x_coords)
        if n_points < MIN_POINTS:  # 최소 10개 이상의 포인트가 있을 때만 실행
            print(
                f'Warning: 유효 데이터 부족 ({n_points}/{MIN_POINTS}). 계산을 건너뜁니다.'
            )
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

        actual_disp = disp_map[y_coords, x_coords]
        z_depth = (self.cam_params['fx'] * self.cam_params['baseline']) / actual_disp

        real_x = (x_coords - self.cam_params['cx']) * z_depth / self.cam_params['fx']
        real_y = (y_coords - self.cam_params['cy']) * z_depth / self.cam_params['fy']

        X_input = np.column_stack((real_x, z_depth))
        y_target = real_y

        model = RANSACRegressor(
            min_samples=max(10, int(n_points * 0.2)),
            residual_threshold=0.5,  # 단위가 미터라면 0.1~0.2(10~20cm) 추천
            max_trials=1000,
            random_state=42,
        )

        try:
            model.fit(X_input, y_target)
        except ValueError:
            return 0.0, None

        slope_rate = model.estimator_.coef_[1]

        rad = np.arctan(slope_rate)
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
