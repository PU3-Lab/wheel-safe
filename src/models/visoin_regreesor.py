import os

import timm
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from lib.utils.device import get_device


class VisionRegressor:
    def __init__(self, model_name='efficientnet_b0', lr=1e-3):
        self.device = get_device()

        # 모델 생성 (회귀용이므로 출력 노드 1개)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=1).to(
            self.device
        )
        self.criterion = nn.MSELoss()

        # 초기 상태: Backbone 고정 (Head만 학습 준비)
        self._freeze_backbone()

        # 학습 가능한 파라미터만 옵티마이저에 등록
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )
        self.best_loss = float('inf')

    def _freeze_backbone(self):
        """기본적으로 모든 층을 얼리고 마지막 층(Head)만 풉니다."""
        for param in self.model.parameters():
            param.requires_grad = False

        # timm 모델별 Head 이름 대응 (classifier 또는 head)
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'head'):
            for param in self.model.head.parameters():
                param.requires_grad = True

    def unfreeze_all(self, lr=1e-5):
        """Fine-tuning을 위해 전체 레이어 해제"""
        for param in self.model.parameters():
            param.requires_grad = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        print('>>> [알림] 모든 레이어 활성화 (Fine-tuning 모드)')

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0

        # tqdm 진행바 설정
        pbar = tqdm(dataloader, desc=f'Epoch {epoch_idx}')

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze()  # [B, 1] -> [B]
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(mse=f'{total_loss / (pbar.n + 1):.4f}')

        return total_loss / len(dataloader)

    def save_checkpoint(self, current_loss, path='best_model.pth'):
        """최저 손실값 갱신 시 모델 저장"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save(self.model.state_dict(), path)
            print(f'--- 모델 저장됨 (Best Loss: {current_loss:.4f}) ---')

    @torch.no_grad()  # 경사도 계산을 꺼서 메모리 절약 및 속도 향상
    def evaluate(self, dataloader):
        """검증 데이터셋으로 모델 성능 평가"""
        self.model.eval()  # 추론 모드 전환 (Dropout, BatchNorm 비활성화)
        total_loss = 0

        # 검증 단계에도 tqdm 적용
        pbar = tqdm(dataloader, desc='Evaluating')

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            pbar.set_postfix(val_mse=f'{total_loss / (pbar.n + 1):.4f}')

        return total_loss / len(dataloader)

    @torch.no_grad()
    def predict(self, image_path, transform):
        """이미지 한 장에 대해 slope_avg 예측"""
        self.model.eval()

        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # 추론
        prediction = self.model(image).squeeze().item()
        return prediction

    def load_best_model(self, path='best_model.pth'):
        """저장된 최적의 가중치 불러오기"""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f'>>> [완료] {path}로부터 최적 가중치 로드됨')


# ==========================================
# 3. Execution: 실제 실행 루프
# ==========================================
# (데이터프레임 'df'가 준비되었다고 가정)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# trainer = VisionRegressor(model_name='efficientnet_b0', lr=1e-3)
# train_loader = DataLoader(SlopeDataset(df, transform), batch_size=16, shuffle=True)

# # Stage 1: Head만 학습
# for e in range(1, 6):
#     loss = trainer.train_epoch(train_loader, e)
#     trainer.save_checkpoint(loss)

# # Stage 2: 전체 미세 조정
# trainer.unfreeze_all(lr=1e-5)
# for e in range(6, 11):
#     loss = trainer.train_epoch(train_loader, e)
#     trainer.save_checkpoint(loss)
# 1. 학습 및 검증 루프
# for e in range(1, 11):
#     train_loss = trainer.train_epoch(train_loader, e)
#     val_loss = trainer.evaluate(val_loader) # 검증 수행

#     # 검증 손실 기준으로 체크포인트 저장
#     trainer.save_checkpoint(val_loss, path='best_slope_model.pth')

# # 2. 학습 종료 후 최적 모델 불러오기
# trainer.load_best_model('best_slope_model.pth')

# # 3. 새로운 이미지 테스트
# test_img_path = "data/test_sample.png"
# result = trainer.predict(test_img_path, transform)
# print(f"예측된 경사도(Slope): {result:.4f}")
