import argparse
import glob
import os

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import lib.utils.const as const
from dataset.slope_dataset import SlopeDataset
from lib.utils.path import data_path, model_path, runs_path
from models.visoin_regressor import VisionRegressor
from run.predict import get_eval_transform


def get_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((const.RESIZE_SIZE, const.RESIZE_SIZE)),
            transforms.RandomCrop((const.CROP_SIZE, const.CROP_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1)],
                p=0.5,
            ),
            transforms.RandomRotation(degrees=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=const.IMAGENET_MEAN, std=const.IMAGENET_STD),
        ]
    )


def execute(model_name, train_path, val_path):
    train_files = glob.glob(os.path.join(train_path, '*.csv'))
    val_files = glob.glob(os.path.join(val_path, '*.csv'))

    # 리스트 컴프리헨션으로 읽어온 뒤 하나로 합치기
    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    val_df = pd.concat([pd.read_csv(f) for f in val_files], ignore_index=True)

    train_df = train_df[['path', 'slope_avg']]
    val_df = val_df[['path', 'slope_avg']]

    train_transform = get_train_transform()
    val_transform = get_eval_transform()

    train_loader = DataLoader(
        SlopeDataset(train_df, train_transform), batch_size=16, shuffle=True
    )

    val_loader = DataLoader(SlopeDataset(val_df, val_transform), batch_size=16)

    trainer = VisionRegressor(model_name=model_name, lr=1e-3, log_dir=runs_path())
    path = str(model_path() / 'best_model.pth')
    trainer.load_best_model(path)

    for e in range(1, 3):
        trainer.train_epoch(
            train_loader, e, val_loader, checkpoint_path=path, eval_interval=50
        )

    trainer.unfreeze_all(lr=1e-5)
    for e in range(3, 5):
        trainer.train_epoch(
            train_loader, e, val_loader, checkpoint_path=path, eval_interval=50
        )


def main():
    print('=' * 100)
    print('uv run train.py efficientnet_b0 ./data/train/')
    print('=' * 100)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_name', nargs='?', help='모델 이름', default='convnext_tiny'
    )
    parser.add_argument(
        'train',
        nargs='?',
        help='label 경로(.csv)',
        default=str(data_path() / 'train'),
    )
    parser.add_argument(
        'val',
        nargs='?',
        help='label 경로(.csv)',
        default=str(data_path() / 'val'),
    )

    args = parser.parse_args()

    model_name = args.model_name
    train_path = args.train
    val_path = args.val

    execute(model_name, train_path, val_path)


if __name__ == '__main__':
    main()
