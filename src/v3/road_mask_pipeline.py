import cv2
import numpy as np
import torch
import torch.nn.functional as F

import models.pidnet as pidnet
from lib.utils.device import get_device
from lib.utils.path import model_path


class RoadMaskPipeline:
    def __init__(self, debug=False):
        self.device = get_device()
        self.model = self.__create_model()
        self.debug = debug

    def run(self, img_path):
        with torch.no_grad():
            input_tensor, org_size = self.__preprocess_image(img_path, self.device)

            output = self.model(input_tensor)

            preds = self.__postprocess_output(output, org_size)

            if self.debug:
                print(f'preds : {np.unique(preds)}')

            return preds, org_size

    def __create_model(self):
        model = pidnet.get_pred_model('pidnet_s', num_classes=19)

        checkpoint_path = str(model_path() / 'PIDNet_S_Cityscapes_val.pt')

        self.model = self.__load_pretrained(model, checkpoint_path)

        model.to(self.device)
        model.eval()
        print('모델 세팅 완료')

        return model

    def __load_pretrained(self, model, pretrained):
        pretrained_dict = torch.load(pretrained, map_location=self.device)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {
            k[6:]: v
            for k, v in pretrained_dict.items()
            if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)
        }
        msg = f'Loaded {len(pretrained_dict)} parameters!'
        print('Attention!!!')
        print(msg)
        print('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        return model

    def __preprocess_image(self, img_path, device):
        # 1. 이미지 로드 (BGR -> RGB 변환 필수)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f'이미지를 찾을 수 없습니다: {img_path}')

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. 정규화 (Cityscapes/ImageNet 표준 값 사용)
        # PIDNet 학습 시 사용된 표준적인 mean/std입니다.
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = img_rgb.astype(np.float32) / 255.0
        img = (img - mean) / std

        # 3. HWC (Height, Width, Channel) -> CHW (Channel, Height, Width)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        return img_tensor, img_bgr.shape[:2]  # 원본 크기도 함께 반환 (Interpolate용)

    def __postprocess_output(self, pred, org_size):
        # 모델 출력을 원본 이미지 크기로 복원 및 인덱스 추출
        # PIDNet은 [1, classes, h, w] 형태의 텐서를 출력함
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        pred = F.interpolate(pred, size=org_size, mode='bilinear', align_corners=True)
        pred_index = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

        if self.debug:
            unique, counts = torch.unique(
                torch.from_numpy(pred_index), return_counts=True
            )
            index_counts = dict(zip(unique.tolist(), counts.tolist(), strict=False))

            # 3. 빈도수 순으로 정렬 (보통 길이나 하늘이 가장 넓음)
            sorted_indices = sorted(
                index_counts.items(), key=lambda x: x[1], reverse=True
            )

            print('면적이 넓은 순서대로 인덱스 확인:')
            for idx, count in sorted_indices:
                print(f'인덱스 {idx}: {count} 픽셀')

        return pred_index
