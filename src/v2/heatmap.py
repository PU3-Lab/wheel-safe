import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap_comparison(img_left, pred, mask):
    h_orig, w_orig = img_left.shape[:2]
    img_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))  # 가로로 길게 설정

    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Original Image ({w_orig}x{h_orig})')
    axes[0].axis('off')

    im = axes[1].imshow(pred, cmap='tab20')
    axes[1].set_title(f'Full-Size Heatmap ({w_orig}x{h_orig})')

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(mask, cmap='tab20')
    axes[2].set_title(f'Mask Full-Size Heatmap ({w_orig}x{h_orig})')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

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


# Cityscapes 클래스 이름 (인덱스 확인용)
class_names = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
]


def show_heatmap(pred_index, title='Segmentation Index Heatmap'):
    plt.figure(figsize=(15, 8))

    # 1. 히트맵 그리기
    # cmap='nipy_spectral' 또는 'jet'은 인덱스 간 색상 차이가 뚜렷해서 보기 좋습니다.
    im = plt.imshow(pred_index, cmap='nipy_spectral')

    # 2. 컬러바(인덱스 설명) 설정
    # vmin, vmax를 설정하여 0~18 범위가 고정되게 합니다.
    plt.clim(0, 18)
    cbar = plt.colorbar(im, fraction=0.02, pad=0.04)

    # 컬러바에 0~18 인덱스 표시
    cbar.set_ticks(range(19))
    cbar.set_ticklabels(
        class_names
    )  # 클래스 이름으로 라벨링 (숫자만 보고 싶으면 이 줄 삭제)

    plt.title(title, fontsize=15)
    plt.axis('off')  # 격자 숨기기
    plt.tight_layout()
    plt.show()


def compare_results(img_path, pred_index):
    # 1. 이미지 로드 (문자열 경로가 들어올 경우 대비)
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 2. 왼쪽: 원본 이미지
    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image', fontsize=15)
    axes[0].axis('off')

    # 3. 오른쪽: 히트맵 (반드시 im 변수에 할당)
    # vmin, vmax를 직접 설정하면 plt.clim을 따로 부를 필요가 없습니다.
    im = axes[1].imshow(pred_index, cmap='nipy_spectral', vmin=0, vmax=18)
    axes[1].set_title('Index Heatmap', fontsize=15)
    axes[1].axis('off')

    # 4. 컬러바 추가 (im 객체를 명시적으로 지정)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_ticks(range(19))
    cbar.set_ticklabels(class_names)

    plt.tight_layout()
    plt.show()


# --- 기존 코드 루프 내 적용 예시 ---
# pred_index = postprocess_output(output, org_size)
# show_heatmap(pred_index)


# --- 루프 적용 시 ---
# plot_heatmap_full_size(img_left, pred)
