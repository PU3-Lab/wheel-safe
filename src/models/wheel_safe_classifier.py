import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class WheelSafeClassifier(pl.LightningModule):
    def __init__(self, model_name='efficientnet_b0', lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 회귀 모델이므로 num_classes=1로 설정
        self.model = timm.create_model(model_name, pretrained=True, num_classes=1)

        # 회귀용 손실 함수 (Mean Squared Error)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x).squeeze()  # [Batch, 1] -> [Batch]

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log('train_mse', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log('val_mse', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


# class WheelSafeClassifier(pl.LightningModule):
#     def __init__(self, model_name='convnext_tiny', num_classes=10, lr=1e-4):
#         super().__init__()
#         self.save_hyperparameters()
#         # timm을 통해 어떤 모델이든 동적으로 생성
#         self.model = timm.create_model(
#             model_name, pretrained=True, num_classes=num_classes
#         )
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y)
#         self.log('train_loss', loss, prog_bar=True, on_epoch=True)
#         return loss

#     def configure_optimizers(self):
#         # AdamW는 최신 논문들(ConvNeXt, EfficientNetV2 등)에서 공통적으로 권장하는 옵티마이저입니다.
#         optimizer = torch.optim.AdamW(
#             self.parameters(), lr=self.hparams.lr, weight_decay=0.05
#         )
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
#         return [optimizer], [scheduler]


class WheelSafeGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook 등록: 어떤 모델의 레이어든 경사도와 활성화 값을 가로챕니다.
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        logit = self.model(input_tensor)

        if class_idx is None:
            class_idx = logit.argmax(1).item()

        self.model.zero_grad()
        logit[0, class_idx].backward()

        # Global Average Pooling on Gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
