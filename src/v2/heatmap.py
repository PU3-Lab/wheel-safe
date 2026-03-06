import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap_comparison(img_left, pred):
    h_orig, w_orig = img_left.shape[:2]
    img_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))  # 가로로 길게 설정

    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Original Image ({w_orig}x{h_orig})')
    axes[0].axis('off')

    im = axes[1].imshow(pred, cmap='tab20')
    axes[1].set_title(f'Full-Size Heatmap ({w_orig}x{h_orig})')

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def plot_refined_heatmap_with_slope(img_left, pred, conf_map, angle=0.0):
    h_orig, w_orig = img_left.shape[:2]

    # 1. pred를 원본 크기로 리사이즈
    pred_full = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # 2. 레터박스 제거 (Confidence 필터링)
    # 신뢰도가 0.5 이하인 곳은 클래스 ID를 0(배경)으로 초기화
    # 이렇게 하면 히트맵에서 레터박스 영역이 '파란색(0번)'으로 깨끗하게 지워집니다.
    pred_refined = np.where(conf_map > 0.5, pred_full, 0)

    # 3. 시각화
    _, axes = plt.subplots(1, 2, figsize=(22, 10))
    axes[0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')

    # 정제된 pred_refined로 히트맵 그리기
    axes[1].imshow(pred_refined, cmap='tab20')
    axes[1].set_title(f'Refined Heatmap (Slope: {angle:.2f} deg)')

    # 경사도 수치 표시
    axes[1].text(
        w_orig * 0.5,
        h_orig * 0.9,
        f'Angle: {angle:.2f}°',
        color='yellow',
        weight='bold',
        fontsize=25,
        ha='center',
        bbox={'facecolor': 'black', 'alpha': 0.7},
    )

    plt.show()


# --- 루프 적용 시 ---
# plot_heatmap_full_size(img_left, pred)
