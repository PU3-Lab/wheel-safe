import torch
from PIL import Image
from torch.utils.data import Dataset


class SlopeDataset(Dataset):
    def __init__(self, csv_data, transform=None):
        """
        Args:
            csv_data: 이미지 경로와 slope_avg가 포함된 Pandas DataFrame
            transform: 이미지 전처리 (Compose)
        """
        self.data = csv_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 이미지 로드
        img_path = self.data.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')

        # 2. 경사도(Slope) 값 로드
        slope = torch.tensor(self.data.iloc[idx]['slope_avg'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, slope
