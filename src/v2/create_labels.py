import os
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
import pandas as pd

from v2.estimate_road_slope import get_calibrated_slope
from v2.pidnet_onnx_predictor import create_model


def calculate_slope(disparity_map):
    # 16비트 시차 지도의 경사도 계산
    grad_x = cv2.Sobel(disparity_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disparity_map, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(magnitude)


def process_all_depth_data(root_path):

    model = create_model()
    # labels = [7, 8]
    labels = range(0, 19)
    labels = [8]

    all_results = []

    # 1. Depth_ 로 시작하는 모든 폴더 찾기
    depth_folders = sorted(glob(os.path.join(root_path, 'Depth_001')))

    for folder in depth_folders:
        folder_name = os.path.basename(folder)
        # 폴더 내 모든 파일 리스트업
        files = os.listdir(folder)

        config_path = f'{folder}/{folder_name}.conf'
        # 2. 파일명에서 공통 접두어(Prefix) 추출하여 그룹화
        # 예: 'frame1_left.png' -> key: 'frame1'
        file_groups = defaultdict(dict)

        for f in files:
            if '_L' in f:
                key = f.split('_L')[0]
                file_groups[key]['left'] = f
            elif '_confidence' in f:
                key = f.split('_confidence')[0]
                file_groups[key]['confidence'] = f
            elif '_disp16' in f:
                key = f.split('_disp16')[0]
                file_groups[key]['disp16'] = f

        # 3. 그룹화된 세트별로 처리
        for prefix, components in file_groups.items():
            # 세 가지 파일이 모두 존재하는 경우에만 처리
            if all(k in components for k in ['left', 'confidence', 'disp16']):
                img_path = os.path.join(folder, components['left'])
                disp_path = os.path.join(folder, components['disp16'])
                conf_path = os.path.join(folder, components['confidence'])

                # 이미지 로드 및 경사도 계산
                disp_map = (
                    cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
                    / 16
                )
                conf_map = (
                    cv2.imread(str(conf_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    / 255
                )

                print(f'disp {disp_map.shape} conf {conf_map.shape}')

                road_mask = model.get_road_mask(img_path, conf_map, disp_map, labels)

                print(f'road_mask {np.unique(road_mask)}')

                slope = get_calibrated_slope(disp_map, conf_map, road_mask, config_path)

                all_results.append(
                    {
                        'folder': folder_name,
                        'prefix': prefix,
                        'left_file': components['left'],
                        'conf_file': components['confidence'],
                        'disp_file': components['disp16'],
                        'slope_avg': round(slope, 4),
                    }
                )
                print(f'Matched & Processed: {folder_name} / {prefix}')

    # 4. CSV 저장
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv('depth_analysis_multi.csv', index=False, encoding='utf-8-sig')
        print(f'\n✅ 완료! 총 {len(all_results)}개의 데이터 셋이 CSV로 저장되었습니다.')
    else:
        print('❌ 매칭되는 파일 세트를 찾지 못했습니다.')


# 실행
# process_all_depth_data('data/raw')
