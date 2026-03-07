import configparser

import numpy as np
from sklearn.linear_model import RANSACRegressor


# 1. Config 파일 로드 함수
def load_cam_params(config_path, resolution='2K'):
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


# --- 보정된 경사도 계산 ---
def get_calibrated_slope(disp, conf, mask, config_path, res='2K'):
    params = load_cam_params(config_path, res)

    print(f'params: {params}')

    # Raw 경사도 계산 (카메라 기준)
    raw_angle, _ = estimate_road_slope(disp, conf, mask, params)

    if raw_angle is not None:
        # 실제 경사 = 계산된 각도 - 카메라 설치 각도(Pitch)
        # (주의: 좌표계 방향에 따라 더할지 뺄지는 테스트가 필요함)
        actual_road_slope = raw_angle - params['pitch_deg']
        return actual_road_slope

    return None


def estimate_road_slope(disp_map, conf_map, road_mask, params):
    # valid_indices = (road_mask == 255) & (conf_map > 0.9) & (disp_map > 0.5)

    v_coords, u_coords = np.where(road_mask)
    disparities = disp_map[road_mask]

    if len(disparities) < 100:  # 샘플 수가 너무 적으면 계산 불가
        return None, None

    # 2. 3차원 좌표(X, Y, Z) 복원 (mm 단위)
    # Z = (f * Baseline) / disparity
    Z = (params['fx'] * params['baseline']) / disparities
    X = (u_coords - params['cx']) * Z / params['fx']
    Y = (v_coords - params['cy']) * Z / params['fy']

    # 3. RANSAC을 이용한 지면 평면 피팅 (Y = aX + bZ + c)
    # 지면의 높이 Y를 X(가로)와 Z(깊이)를 통해 예측
    X_input = np.column_stack((X, Z))
    Y_target = Y

    # residual_threshold: 50mm(5cm) 이상의 오차를 가진 점은 노이즈(아웃라이어)로 취급
    ransac = RANSACRegressor(residual_threshold=50, random_state=42)
    ransac.fit(X_input, Y_target)

    # slope_x: 좌우 기울기, slope_z: 진행 방향(거리 대비 높이) 기울기
    _, slope_z = ransac.estimator_.coef_

    # 4. 경사도(Slope) 계산
    # 수직 변화량(dy) / 수평 변화량(dz) = tan(theta)
    # 주의: 카메라 좌표계상 Y는 아래가 +이므로 오르막일 때 slope_z는 음수가 나옵니다.
    # 따라서 직관적인 이해를 위해 -1을 곱해줍니다.
    actual_slope_z = -slope_z

    angle_rad = np.arctan(actual_slope_z)
    angle_deg = np.degrees(angle_rad)  # 각도(degree)
    grade_pct = actual_slope_z * 100  # 경사율(%)

    return angle_deg, grade_pct


# --- 가상의 실행 예시 ---
"""
# 데이터 로드 (실제 환경에 맞게 수정)
disp = np.fromfile("disparity.bin", dtype=np.float32).reshape(1080, 1920)
conf = np.fromfile("confidence.bin", dtype=np.float32).reshape(1080, 1920)
mask = cv2.imread("pidnet_result.png", cv2.IMREAD_GRAYSCALE)

deg, pct = estimate_road_slope(disp, conf, mask, cam_params)

if deg is not None:
    print(f"Road Slope: {deg:.2f}° ({pct:.2f}%)")
    if deg > 0:
        print("상태: 오르막 (Uphill)")
    else:
        print("상태: 내리막 (Downhill)")
"""
