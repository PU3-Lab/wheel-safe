import cv2
import numpy as np
import torch
import torch.nn.functional as F

import models.pidnet as pidnet
from lib.utils.path import model_path


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
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


def create_model():
    model = pidnet.get_pred_model('pidnet_s', num_classes=19)

    checkpoint_path = str(model_path() / 'PIDNet_S_Cityscapes_val.pt')

    load_pretrained(model, checkpoint_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # checkpoint = torch.load(checkpoint_path, map_location=device)

    # if 'state_dict' in checkpoint:
    #     model.load_state_dict(checkpoint['state_dict'])
    # else:
    #     model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print('모델 세팅 완료')

    return model


def preprocess_image(img_path, device='cuda'):
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
    img = img.transpose((2, 0, 1))

    # 4. Tensor 변환 및 배치 차원 추가 (1, C, H, W)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device).float()

    return img_tensor, img_bgr.shape[:2]  # 원본 크기도 함께 반환 (Interpolate용)


def postprocess_output(pred, org_size):
    # 모델 출력을 원본 이미지 크기로 복원 및 인덱스 추출
    # PIDNet은 [1, classes, h, w] 형태의 텐서를 출력함
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred = F.interpolate(pred, size=org_size, mode='bilinear', align_corners=True)
    pred_index = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    return pred_index
