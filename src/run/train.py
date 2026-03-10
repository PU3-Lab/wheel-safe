import argparse
import glob
import os

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import lib.utils.const as const
from dataset.slope_dataset import SlopeDataset
from lib.utils.path import data_path, model_path
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
    train = glob.glob(os.path.join(train_path, '*.csv'))
    val = glob.glob(os.path.join(val_path, '*.csv'))

    train_df = pd.read_csv(train[0])
    val_df = pd.read_csv(val[0])

    train_df = train_df[['path', 'slope_avg']]
    val_df = val_df[['path', 'slope_avg']]

    train_transform = get_train_transform()
    val_transform = get_eval_transform()

    train_loader = DataLoader(
        SlopeDataset(train_df, train_transform), batch_size=16, shuffle=True
    )

    val_loader = DataLoader(SlopeDataset(val_df, val_transform), batch_size=16)

    trainer = VisionRegressor(model_name=model_name, lr=1e-3)
    path = str(model_path() / 'best_model.pth')
    trainer.load_best_model(path)

    for e in range(1, 3):
        trainer.train_epoch(
            train_loader, e, val_loader, checkpoint_path=path, eval_interval=50
        )

    trainer.unfreeze_all(lr=1e-5)
    for e in range(3, 6):
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
