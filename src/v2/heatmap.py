import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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

mask_index = list(map(str, range(19)))


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


def compare_results(img_path, pred_index, color_map):
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
    cbar.set_ticklabels(mask_index)

    plt.tight_layout()
    plt.show()


def visualize_mask_1(img_path, pred_index, labels):
    # 1. 1번 인덱스만 추출 (Boolean Mask)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_1 = np.isin(pred_index, labels)

    # 2. 시각화를 위해 원본 이미지 복사
    overlay = image.copy()

    # 3. 1번 영역에만 특정 색상(예: 밝은 하늘색) 채우기
    # [R, G, B] 순서 (0~255)
    overlay[mask_1] = [0, 255, 255]

    # 4. 원본과 마스크를 적절히 합성 (Alpha Blending)
    alpha = 0.4
    combined = ((1 - alpha) * image + alpha * overlay).astype(np.uint8)

    # 5. 결과 출력
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Index 1 Mask Highlight')
    plt.imshow(combined)
    plt.axis('off')

    plt.show()


def visualize_road_report(img_path, output, palette):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 0. 전처리 및 예측값 추출 ---
    # output: [1, C, H, W], image: [H, W, 3]
    probs = F.softmax(output, dim=1)
    conf, pred = torch.max(probs, dim=1)

    pred = pred[0].cpu().numpy()  # [H, W] 인덱스 맵
    conf = conf[0].cpu().numpy()  # [H, W] 확신도 맵

    # --- 1. 컬러 매핑 (Color Mapping) ---
    color_map = palette[pred]

    # --- 2. 1번 인덱스 마스킹 (Masking) ---
    mask_1 = pred == 0
    mask_overlay = image.copy()
    mask_overlay[mask_1] = [0, 255, 255]  # 하늘색 마스크
    mask_final = cv2.addWeighted(image, 0.6, mask_overlay, 0.4, 0)  # 투명도 조절

    # --- 3. 2x2 Subplot 생성 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (1,1) Original Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('1. Original Image', fontsize=15)
    axes[0, 0].axis('off')

    # (1,2) Color Map (Cityscapes Palette)
    axes[0, 1].imshow(color_map)
    axes[0, 1].set_title('2. Semantic Color Map', fontsize=15)
    axes[0, 1].axis('off')

    # (2,1) Index 1 Mask Highlight
    axes[1, 0].imshow(mask_final)
    axes[1, 0].set_title('3. Index 1 (Road/Sidewalk) Mask', fontsize=15)
    axes[1, 0].axis('off')

    # (2,2) Confidence Map (확신도 시각화)
    im4 = axes[1, 1].imshow(conf, cmap='jet')
    axes[1, 1].set_title('4. Prediction Confidence', fontsize=15)
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_refined_road(img_path, pred_index, conf_map, disp_map, palette):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. 0번(Road) 기본 마스크
    road_mask = (pred_index == 0).astype(np.uint8)

    # 2. Conf와 Disp를 결합한 정밀 마스킹 (Refinement)
    # 확신도가 0.5 이상인 곳만 도로로 인정하고, 경계선(Disp)은 제외하거나 강조
    refined_mask = (road_mask == 1) & (conf_map > 0.5)

    # 3. 마스킹 이미지 생성 (원본 + 연두색 마스크)
    mask_overlay = image.copy()
    mask_overlay[refined_mask] = [0, 255, 100]  # 정밀 마스크 영역

    # 경계선(Disp)을 빨간색 외곽선으로 표시 (시각적 확인용)
    mask_overlay[disp_map > 0.8] = [255, 0, 0]

    road_result = cv2.addWeighted(image, 0.6, mask_overlay, 0.4, 0)

    # 4. 2x2 Subplot 구성
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (0,0) 원본 이미지
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('1. Original Image', fontsize=14)
    axes[0, 0].axis('off')

    # (0,1) 전체 컬러 매핑 (Palette 적용)
    axes[0, 1].imshow(palette[pred_index])
    axes[0, 1].set_title('2. Full Semantic Map', fontsize=14)
    axes[0, 1].axis('off')

    # (1,0) Conf/Disp 기반 정밀 마스킹 (결과물)
    axes[1, 0].imshow(road_result)
    axes[1, 0].set_title('3. Refined Road Mask (Conf+Disp)', fontsize=14)
    axes[1, 0].axis('off')

    # (1,1) 시각적 분석 (Conf와 Disp 중첩)
    axes[1, 1].imshow(conf_map, cmap='jet', alpha=0.7)
    axes[1, 1].imshow(disp_map, cmap='gray', alpha=0.3)
    axes[1, 1].set_title('4. Confidence & Boundary Analysis', fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


# --- 실행 ---
# palette = np.array(color_map, dtype=np.uint8)
# visualize_refined_road(img, pred, conf, disp, palette)

# 실행 예시
# palette = np.array(color_map, dtype=np.uint8) # 이전 단계에서 정의한 리스트
# visualize_all_in_one(img_np, model_output, palette)

# 사용 예시 (변수명은 환경에 맞게 수정하세요)
# visualize_mask_1(original_img_numpy, pred_index)

# --- 기존 코드 루프 내 적용 예시 ---
# pred_index = postprocess_output(output, org_size)
# show_heatmap(pred_index)


# --- 루프 적용 시 ---
# plot_heatmap_full_size(img_left, pred)
