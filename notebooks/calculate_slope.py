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

labels = [7, 8]

for _, (name, num) in enumerate(list(zip(image_names, foder_nums, strict=False))):
    folder_path = raw_data_path(num)
    conf_file = folder_path / f'Depth_{num}.conf'
    left_img = folder_path / f'{name}_left.png'
    conf_img = folder_path / f'{name}_confidence.png'
    disp_img = folder_path / f'{name}_disp16.png'

    disp_map = cv2.imread(str(disp_img), cv2.IMREAD_UNCHANGED).astype(np.float32)
    conf_map = (
        cv2.imread(str(conf_img), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    )

    road_mask = model.get_road_mask(left_img, conf_map, disp_map, labels)

    slope = get_calibrated_slope(disp_map, conf_map, road_mask, conf_file)

    print(f'[{name}] 실제 도로 경사율: {slope:.2f} 도')
    print('=' * 100)


# %%
from lib.utils.path import raw_path
from v2.create_labels import process_all_depth_data

process_all_depth_data(raw_path())
