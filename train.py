import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from dataset.slope_dataset import SlopeDataset
from lib.utils.path import data_path, model_path
from models.visoin_regreesor import VisionRegressor


def main():
    transform = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    trainer = VisionRegressor(model_name='efficientnet_b0', lr=1e-3)

    path = str(data_path() / 'train' / 'slope_lables_005.csv')
    df = pd.read_csv(path)

    train_loader = DataLoader(SlopeDataset(df, transform), batch_size=16, shuffle=True)
    output_model_path = model_path() / 'best_model.pth'

    for e in range(1, 6):
        loss = trainer.train_epoch(train_loader, e)
        trainer.save_checkpoint(loss, output_model_path)

    trainer.unfreeze_all(lr=1e-5)
    for e in range(6, 11):
        loss = trainer.train_epoch(train_loader, e)
        trainer.save_checkpoint(loss, output_model_path)


if __name__ == '__main__':
    main()
