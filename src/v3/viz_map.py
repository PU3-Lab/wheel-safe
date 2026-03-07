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


def report(img_path, output, palette):
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


def report_2(img_path, pred_index, conf_map, disp_map, palette):
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
    # axes[1, 1].imshow(conf_map, cmap='jet', alpha=0.7)
    # axes[1, 1].imshow(disp_map, cmap='gray', alpha=0.5)
    # axes[1, 1].imshow(conf_map, cmap='gray', alpha=0.5)
    # 경계선(Disp)을 더 강렬하게 보고 싶을 때
    axes[1, 1].imshow(disp_map, cmap='gray', alpha=0.5)

    # 혹은 확신도(Conf)를 직관적으로 볼 때
    axes[1, 1].imshow(conf_map, cmap='gray', alpha=0.8)
    axes[1, 1].set_title('4. Confidence & Boundary Analysis', fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def report_road(img_path, mask, slope_deg):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    # 마스킹된 영역 표시
    overlay = image.copy()
    overlay[mask] = [0, 255, 0]  # 유효 픽셀 녹색 표시

    plt.imshow(overlay)
    plt.title(f'Estimated Road Slope: {slope_deg:.2f}°', fontsize=15)
    plt.axis('off')
    plt.show()
