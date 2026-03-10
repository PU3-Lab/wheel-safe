import argparse
import glob
import os

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from dataset.slope_dataset import SlopeDataset
from lib.utils.path import data_path, model_path
from models.visoin_regreesor import VisionRegressor


def execute(model_name, path):
    all_files = glob.glob(os.path.join(path, '*.csv'))
    df_list = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=0)

        df_list.append(df)
        print(f'{filename} 읽기 완료! (행 개수: {len(df)})')

    combine_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f'all df{combine_df}')
    preprocess_df = combine_df[['path', 'slope_avg']]
    print(f'preprossed df : {preprocess_df}')

    transform = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_loader = DataLoader(
        SlopeDataset(preprocess_df, transform), batch_size=16, shuffle=True
    )

    trainer = VisionRegressor(model_name=model_name, lr=1e-3)
    path = str(model_path() / 'best_model.pth')
    trainer.load_best_model(path)

    for e in range(1, 6):
        loss = trainer.train_epoch(train_loader, e)
        trainer.save_checkpoint(loss, path)

    trainer.unfreeze_all(lr=1e-5)
    for e in range(6, 11):
        loss = trainer.train_epoch(train_loader, e)
        trainer.save_checkpoint(loss, path)


def main():
    print('=' * 100)
    print('uv run train.py efficientnet_b0 ./data/train/')
    print('=' * 100)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_name', nargs='?', help='모델 이름', default='efficientnet_b0'
    )
    parser.add_argument(
        'root',
        nargs='?',
        help='label 경로(.csv)',
        default=str(data_path() / 'train'),
    )

    args = parser.parse_args()

    model_name = args.model_name
    path = args.root

    execute(model_name, path)


if __name__ == '__main__':
    main()
