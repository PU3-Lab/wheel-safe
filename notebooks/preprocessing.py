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

from gpt.debug_heatmap import draw_slope_points
from gpt.onnx_slope_pipeline import ONNXSlopePipeline
from lib.utils.path import model_path, output_path, raw_data_path

depth_nums = [('001', 'ZED1_KSC_001032'), ('007', 'ZED4_KSC_010545')]

for _, num in enumerate(depth_nums):
    depth_path = raw_data_path(num[0])

    pipe = ONNXSlopePipeline(
        onnx_path=model_path() / 'pidnet.onnx',
        config_path=depth_path / f'Depth_{num[0]}.conf',
    )

    prefix = num[1]

    res = pipe.run_pipeline(
        left_path=depth_path / f'{prefix}_left.png',
        disp_path=depth_path / f'{prefix}_disp16.png',
        conf_path=depth_path / f'{prefix}_confidence_save.png',
    )
    print(res)

    img = cv2.imread(depth_path / f'{prefix}_left.png')

    vis = draw_slope_points(
        img,
        res['valid'],
        res['inliers_mask'],
        None,
        save_path=output_path() / f'{prefix}_debug_points.png',
    )

