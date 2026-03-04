import cv2
import numpy as np
import onnxruntime as ort

from lib.utils.path import model_path
from v2.heatmap import plot_heatmap_comparison


class PIDNetOnnxPredictor:
    def __init__(self, model_path, device='cuda'):
        # 1. ONNX 세션 초기화 (GPU 우선 사용)
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if device == 'cuda'
            else ['CPUExecutionProvider']
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img_bgr, target_size=(2048, 1024)):
        # PIDNet 입력 사이즈에 맞춰 리사이즈 (모델에 따라 다를 수 있음)
        img = cv2.resize(img_bgr, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 정규화 (ImageNet 기준)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # dtype 추가
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # dtype 추가
        img = (img - mean) / std

        # HWC -> CHW 및 배치 차원 추가
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def get_road_mask(self, img_path, conf_map):
        img_bgr = cv2.imread(str(img_path))
        h_orig, w_orig = img_bgr.shape[:2]

        # 2. 전처리 및 추론
        blob = self.preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: blob})

        # PIDNet ONNX는 보통 최종 세그멘테이션 결과 하나만 출력하거나
        # [p, b, d] 순서로 출력합니다. 첫 번째 결과(p)를 사용합니다.
        main_out = outputs[0]

        # 3. Argmax로 클래스 결정
        pred = np.argmax(main_out[0], axis=0).astype(np.uint8)

        pred_full = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        ROAD_LABEL = 8
        road_mask = np.where(pred == ROAD_LABEL, 255, 0).astype(np.uint8)

        print(f'road_mask: {np.sum(road_mask)}')

        # [디버깅 추가] 만약 7번도 0개라면, 현재 예측된 값 중 가장 많이 나온 번호를 찾아보세요.
        if np.sum(road_mask) == 0:
            unique, counts = np.unique(pred, return_counts=True)
            # 가장 많이 나타난 클래스 ID 확인 (보통 도로가 면적이 제일 넓음)
            most_frequent_id = unique[np.argmax(counts)]
            print(f'가장 넓은 면적의 클래스 ID: {most_frequent_id}')
            # 임시로 가장 넓은 면적을 도로로 설정해서 경사도가 계산되는지 확인
            road_mask = np.where(pred == most_frequent_id, 255, 0).astype(np.uint8)
            # 4. 도로 클래스(0번)만 마스크로 생성
            # Cityscapes 기준 0번이 도로입니다.

        # road_mask = np.where(pred == 0, 255, 0).astype(np.uint8)

        # 원래 이미지 크기로 복원
        road_mask = cv2.resize(
            road_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
        )

        h, _ = road_mask.shape
        road_mask[: int(h * 0.25), :] = 0  # 상단 검은 박스 제거
        road_mask[int(h * 0.85) :, :] = 0  # 하단 검은 박스 제거

        if conf_map is not None:
            valid_road_mask = (pred_full == 8) & (conf_map > 0.5)
            pred_full_refined = np.where(valid_road_mask, 8, 0).astype(np.uint8)

            plot_heatmap_comparison(img_bgr, pred_full_refined, conf_map)
        else:
            plot_heatmap_comparison(img_bgr, pred_full, conf_map)

        return road_mask

    # pidnet_onnx_predictor.py 내부
    def predict_raw(self, img_path):
        # 1. 전처리 (2048x1024 등 모델 규격에 맞게)
        img_bgr = cv2.imread(str(img_path))
        blob = self.preprocess(img_bgr)

        # 2. ONNX 추론
        outputs = self.session.run(None, {self.input_name: blob})

        # 3. Argmax로 클래스 ID 추출 (Batch, Channel, H, W -> H, W)
        # PIDNet의 출력은 보통 [1, 19, H, W] 형태입니다.
        pred = np.argmax(outputs[0][0], axis=0)

        return pred  # [128, 256] 또는 [512, 1024] 형태의 raw 데이터 반환


# --- 사용 예시 ---
# predictor = PIDNetOnnxPredictor("pidnet_model.onnx")
# frame = cv2.imread("road_scene.jpg")
# mask = predictor.get_road_mask(frame)
# cv2.imshow("Road Mask", mask)
# cv2.waitKey(0)
def create_model():
    return PIDNetOnnxPredictor(model_path=model_path() / 'pidnet.onnx')


if __name__ == '__main__':
    create_model()
