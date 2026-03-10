import os
from datetime import datetime

import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    mean_absolute_error,  # 추가
    r2_score,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.utils.device import get_device


class VisionRegressor:
    def __init__(
        self, model_name='efficientnet_b0', lr=1e-3, log_dir='runs/vision_regressor'
    ):
        self.device = get_device()

        # 모델 생성 (회귀용이므로 출력 노드 1개)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=1).to(
            self.device
        )
        self.criterion = nn.HuberLoss(delta=1.0)

        # 초기 상태: Backbone 고정 (Head만 학습 준비)
        self._freeze_backbone()

        # 학습 가능한 파라미터만 옵티마이저에 등록
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )
        self.best_loss = float('inf')

        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=f'{log_dir}_{timestamp}')
        self.global_step = 0

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

    def train_epoch(
        self,
        dataloader,
        epoch_idx,
        val_dataloader=None,
        eval_interval=None,
        checkpoint_path='best_model.pth',
    ):

        self.model.train()
        total_loss = 0

        # tqdm 진행바 설정
        pbar = tqdm(dataloader, desc=f'Epoch {epoch_idx}')

        for step, (images, labels) in enumerate(pbar, start=1):
            images = images.to(self.device)
            labels = labels.to(self.device).float().view(-1)

            self.optimizer.zero_grad()

            outputs = self.model(images).view(-1)  # [B, 1] -> [B]
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            train_loss = total_loss / step
            self.global_step += 1

            pbar.set_postfix(train_mse=f'{train_loss:.4f}')

            # TensorBoard step 로그
            self.writer.add_scalar('train/step_mse', loss.item(), self.global_step)
            self.writer.add_scalar('train/running_mse', train_loss, self.global_step)
            self.writer.add_scalar(
                'train/lr',
                self.optimizer.param_groups[0]['lr'],
                self.global_step,
            )

            # 중간 평가
            if (
                val_dataloader is not None
                and eval_interval is not None
                and step % eval_interval == 0
            ):
                val_loss, r2 = self.__evaluate(val_dataloader)
                print(
                    f'\n[Epoch {epoch_idx} | Step {step}/{len(dataloader)}] '
                    f'train_mse={train_loss:.4f}, val_mse={val_loss:.4f}, r2={r2}'
                )
                self.save_checkpoint(val_loss, checkpoint_path)
                self.model.train()  # evaluate 후 다시 train 모드로 복귀

        epoch_loss = total_loss / len(dataloader)
        self.writer.add_scalar('train/epoch_mse', epoch_loss, epoch_idx)
        return epoch_loss

    def save_checkpoint(self, current_loss, path='best_model.pth'):
        """최저 손실값 갱신 시 모델 저장"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save(self.model.state_dict(), path)
            print(f'--- 모델 저장됨 (Best Loss: {current_loss:.4f}) ---')

    @torch.no_grad()
    def __evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc='Evaluating')

        for step, (images, labels) in enumerate(pbar, start=1):
            images = images.to(self.device)
            labels = labels.to(self.device).float().view(-1)

            outputs = self.model(images).view(-1)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix(val_mse=f'{total_loss / step:.4f}')

        avg_loss = total_loss / len(dataloader)
        r2 = r2_score(all_labels, all_preds)

        return avg_loss, r2

    @torch.no_grad()
    def test(self, dataloader):
        self.model.eval()
        total_loss = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc='testing...')

        for step, (images, labels) in enumerate(pbar, start=1):
            images = images.to(self.device)
            labels = labels.to(self.device).float().view(-1)

            outputs = self.model(images).view(-1)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix(val_mse=f'{total_loss / step:.4f}')

        avg_loss = total_loss / len(dataloader)
        r2 = r2_score(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)  # 추가

        return avg_loss, r2, mae

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
