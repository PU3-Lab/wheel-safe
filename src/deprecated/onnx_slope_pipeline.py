from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from pyprojroot import here

from slope_aggregator_old import GlobalSlopeAggregator

# 위에서 정의한 GlobalSlopeAggregator 클래스가 같은 파일에 있거나 import 되었다고 가정합니다.


class ONNXSlopePipeline:
    def __init__(self, onnx_path, config_path, device='cuda'):
        # 1. ONNX 세션 초기화 (GPU 에러 시 CPU로 자동 전환)
        try:
            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if device == 'cuda'
                else ['CPUExecutionProvider']
            )
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(
                onnx_path, providers=['CPUExecutionProvider']
            )

        self.input_name = self.session.get_inputs()[0].name
        self.slope_engine = GlobalSlopeAggregator(conf_path=config_path)

        # 정규화 파라미터
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def get_road_mask(self, img_bgr):
        """다양한 환경의 바닥 인덱스를 통합하고 장애물을 제외합니다."""
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 모델 규격에 맞춘 리사이즈 (1024x2048)
        # input_h, input_w = 1024, 2048
        # img_resized = cv2.resize(img_rgb, (input_w, input_h))

        input_data = cv2.resize(img_rgb, (2048, 1024))
        input_data = (input_data / 255.0 - self.mean) / self.std
        input_data = input_data.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: input_data})
        pred = np.argmax(outputs[0], axis=1).squeeze()
        pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        road_indices = [7, 13]
        road_mask = np.isin(pred_full, road_indices).astype(np.uint8)

        # 2. [전처리] 기하학적 담장 차단 (Geometric Wall Clipping)
        # (A) 화면 상단 50% 제거: 휠체어 주행 경사도는 하단에서 결정됩니다.
        road_mask[: int(h * 0.5), :] = 0

        # (B) 화면 우측 20% 제거: 이미지상 담장이 위치하는 구간을 계산에서 제외합니다.
        road_mask[:, int(w * 0.8) :] = 0

        # 3. 노이즈 정리 (Morphology Open)
        kernel = np.ones((7, 7), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        return road_mask, pred_full

    def run_pipeline(self, left_path, disp_path, conf_path, output_dir='./output'):
        left_img = cv2.imread(str(left_path))
        if left_img is None:
            return

        # 1. 모델 마스크 및 원본 예측 맵 생성
        road_mask, raw_pred = self.get_road_mask(left_img)

        # 2. 물리 엔진 연산 (모델 마스크 적용)
        # 이제 담장 영역은 road_mask에 의해 1차 차단, 물리 엔진의 벽면 필터로 2차 차단됩니다.
        result = self.slope_engine.calculate_slope(
            disp_path, conf_path, external_mask=road_mask
        )

        # 3. 결과 시각화 및 저장
        self.visualize_result(left_img, raw_pred, result, left_path, output_dir)
        return result

    def visualize_result(self, left_img, raw_pred, result, left_path, output_dir):
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        roi_start, roi_end = result['roi_y_range']
        display_slope = result['slope_map_roi'].copy()
        display_slope[~result['valid_mask']] = np.nan

        plt.figure(figsize=(15, 12))

        # (1) 입력 이미지
        plt.subplot(3, 1, 1)
        plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Input: {Path(left_path).name}')
        plt.axis('off')

        # (2) 모델 예측 맵 (디버깅용)
        plt.subplot(3, 1, 2)
        plt.imshow(raw_pred, cmap='tab20')
        plt.colorbar(label='Class ID', orientation='vertical')
        plt.title('Raw Model Prediction (Class Map)')
        plt.axis('off')

        # (3) 최종 경사도 결과 (오버레이)
        plt.subplot(3, 1, 3)
        plt.imshow(
            cv2.cvtColor(left_img[roi_start:roi_end, :], cv2.COLOR_BGR2RGB), alpha=0.5
        )
        im = plt.imshow(display_slope, cmap='jet', vmin=0, vmax=20, alpha=0.8)
        plt.colorbar(im, label='Slope (Deg)')
        plt.title(
            f'Refined Avg Slope: {result["avg_slope"]:.2f}° (All-in-One Filtered)'
        )
        plt.axis('off')

        save_path = out_p / f'integrated_result_{Path(left_path).stem}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f'분석 완료: {save_path}')


# --- 실행 예시 ---
if __name__ == '__main__':
    # 파일 경로 설정
    path = here() / 'data'
    depth_001 = path / 'raw/Depth_001'
    onnx_model = path / 'models' / 'pidnet.onnx'
    camera_conf = depth_001 / 'Depth_001.conf'

    pipeline = ONNXSlopePipeline(onnx_model, camera_conf)

    left = depth_001 / 'ZED1_KSC_001032_left.png'
    disp16 = depth_001 / 'ZED1_KSC_001032_disp16.png'
    conf = depth_001 / 'ZED1_KSC_001032_confidence.png'

    # 실제 데이터 경로로 수정하여 실행
    pipeline.run_pipeline(left, disp16, conf)
# if __name__ == '__main__':
#     # 파일 경로 설정
#     path = here() / 'data'
#     depth_007 = path / 'raw/Depth_007'
#     onnx_model = path / 'models' / 'pidnet.onnx'
#     camera_conf = depth_007 / 'Depth_007.conf'

#     pipeline = ONNXSlopePipeline(onnx_model, camera_conf)

#     num = '010542'

#     left = depth_007 / f'ZED4_KSC_{num}_left.png'
#     disp16 = depth_007 / f'ZED4_KSC_{num}_disp16.png'
#     conf = depth_007 / f'ZED4_KSC_{num}_confidence.png'

#     # 실제 데이터 경로로 수정하여 실행
#     pipeline.run_pipeline(left, disp16, conf)
