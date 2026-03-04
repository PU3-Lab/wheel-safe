from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_debug_figure(
    img_bgr: np.ndarray,
    pred_full: np.ndarray,
    result: dict,
    out_path: str | Path = 'debug_out/debug_viz.png',
    class_cmap: str = 'tab20',  # 19-class 보기 좋음
    slope_cmap: str = 'turbo',  # heatmap
):
    """
    img_bgr: (H,W,3) 입력 이미지 (Crop_Left 또는 Raw_Left 중 '세그/disp/conf와 동일한' 것)
    pred_full: (H,W) 클래스 인덱스 맵
    result: GlobalSlopeAggregator.calculate_slope(...) 반환 dict
        - avg_slope
        - slope_map_roi: (roi_h, W)
        - valid_mask: (roi_h, W) bool
        - roi_y_range: (roi_v_start, roi_v_end)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = pred_full.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    avg_slope = float(result.get('avg_slope', 0.0))
    slope_roi = result.get('slope_map_roi')
    valid_roi = result.get('valid_mask')
    roi_range = result.get('roi_y_range')

    # slope를 full 해상도로 올리기 (ROI만 채움)
    slope_full = np.full((H, W), np.nan, dtype=np.float32)
    if slope_roi is not None and roi_range is not None:
        y0, y1 = roi_range
        slope_full[y0:y1, :] = slope_roi.astype(np.float32)

        # valid_mask가 있으면 invalid는 nan 처리(보기 깔끔)
        if valid_roi is not None:
            m = valid_roi.astype(bool)
            tmp = slope_full[y0:y1, :]
            tmp[~m] = np.nan
            slope_full[y0:y1, :] = tmp

    # ---------------------------
    # Figure
    # ---------------------------
    fig = plt.figure(figsize=(12, 8), dpi=150)

    # 1) Input
    ax1 = plt.subplot(3, 1, 1)
    ax1.imshow(img_rgb)
    ax1.set_title('Input')
    ax1.axis('off')

    # 2) Raw class map
    ax2 = plt.subplot(3, 1, 2)
    im2 = ax2.imshow(
        pred_full, cmap=class_cmap, vmin=0, vmax=18, interpolation='nearest'
    )
    ax2.set_title('Raw Model Prediction (Class Map)')
    ax2.axis('off')
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.025, pad=0.02)
    cb2.set_label('Class ID')

    # 3) Slope map overlay (이미지 위에 slope를 반투명으로)
    ax3 = plt.subplot(3, 1, 3)
    ax3.imshow(img_rgb, alpha=0.35)  # 배경을 살짝 보이게
    # slope 값 범위는 데이터에 따라 조절 가능(예시처럼 0~20도)
    im3 = ax3.imshow(
        slope_full, cmap=slope_cmap, vmin=0, vmax=20, interpolation='nearest'
    )
    ax3.set_title(f'Refined Avg Slope: {avg_slope:.2f}° (All-in-One Filtered)')
    ax3.axis('off')
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.025, pad=0.02)
    cb3.set_label('Slope (deg)')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    return str(out_path)
