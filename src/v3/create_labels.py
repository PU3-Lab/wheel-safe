import os
from collections import defaultdict
from datetime import datetime
from glob import glob

import cv2
import numpy as np
import pandas as pd

from lib.utils.path import output_path, raw_path
from v3.slope_pipeline import SlopePipeline


def save_to_csv(root_path, folder_name='Depth_*'):
    print(f'DEBUG: root_path is {root_path}')  # 이 부분이 None인지 확인
    if root_path is None:
        raise ValueError('root_path가 None입니다. lib/utils/path.py를 확인하세요.')

    all_results = []

    # 1. Depth_ 로 시작하는 모든 폴더 찾기
    depth_folders = sorted(glob(os.path.join(root_path, folder_name)))

    pipeline = SlopePipeline()

    for folder in depth_folders:
        folder_name = os.path.basename(folder)
        # 폴더 내 모든 파일 리스트업
        files = os.listdir(folder)

        config_path = f'{folder}/{folder_name}.conf'

        pipeline.set_config_params(config_path)

        # 2. 파일명에서 공통 접두어(Prefix) 추출하여 그룹화
        # 예: 'frame1_left.png' -> key: 'frame1'
        file_groups = defaultdict(dict)

        for f in files:
            if '_left' in f:
                key = f.split('_left')[0]
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

                disp_map = (
                    cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
                    / 16
                )
                conf_map = (
                    cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
                    / 255.0
                )

                pipeline.run(img_path)
                slope, mask = pipeline.estimate(conf_map, disp_map)

                if mask is None:
                    continue

                all_results.append(
                    {
                        'folder': folder_name,
                        'prefix': prefix,
                        'path': f'{folder}/{components["left"]}',
                        # 'left_file': components['left'],
                        # 'conf_file': components['confidence'],
                        # 'disp_file': components['disp16'],
                        'slope_origin': slope,
                        'slope_avg': abs(slope),
                    }
                )
                print(f'Matched & Processed: {folder_name} / {prefix}')

    # 4. CSV 저장
    if all_results:
        df = pd.DataFrame(all_results)

        now = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 파일명에 날짜/시간 추가 (예: slope_labels_20231027_143005.csv)
        file_name = f'slope_labels_{now}.csv'
        save_path = os.path.join(output_path(), file_name)

        df.to_csv(
            save_path,
            index=False,
            encoding='utf-8-sig',
            float_format='%.2f',
        )
        print(f'\n✅ 완료! 총 {len(all_results)}개의 데이터 셋이 CSV로 저장되었습니다.')
    else:
        print('❌ 매칭되는 파일 세트를 찾지 못했습니다.')


if __name__ == '__main__':
    save_to_csv(raw_path())
