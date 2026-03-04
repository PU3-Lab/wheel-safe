# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: wheel-safe (3.11.14)
#     language: python
#     name: python3
# ---

# %%
import cv2
import numpy as np

from lib.utils.path import raw_data_path
from v2.estimate_road_slope import get_calibrated_slope
from v2.pidnet_onnx_predictor import create_model

model = create_model()

image_names = ['ZED1_KSC_001032', 'ZED4_KSC_010545']
foder_nums = ['001', '007']

for _, (name, num) in enumerate(list(zip(image_names, foder_nums, strict=False))):
    folder_path = raw_data_path(num)
    conf_file = folder_path / f'Depth_{num}.conf'
    mask_img_file = folder_path / f'{name}_left.png'
    conf_img_file = folder_path / f'{name}_confidence_save.png'
    disp_img_file = folder_path / f'{name}_disp16.png'

    mask = model.get_road_mask(mask_img_file)

    disp_img = cv2.imread(str(disp_img_file), cv2.IMREAD_UNCHANGED).astype(np.float32)
    conf_img = (
        cv2.imread(str(conf_img_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    )

    # 5. 최종 경사도 계산 (RANSAC + Pitch 보정)
    # 이전에 정의한 함수 호출
    final_slope = get_calibrated_slope(disp_img, conf_img, mask, conf_file)

    print(final_slope)

    if final_slope is not None:
        print(f'[{name}] 실제 도로 경사도: {final_slope:.2f} 도')

