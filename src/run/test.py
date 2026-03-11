import glob
import os

import pandas as pd
from torch.utils.data.dataloader import DataLoader

from dataset.slope_dataset import SlopeDataset
from lib.utils.path import data_path, model_path
from models.visoin_regressor import VisionRegressor
from run.predict import get_eval_transform


def execute_test(model_name, test_path):
    test_files = glob.glob(os.path.join(test_path, '*.csv'))

    for file in test_files:
        test_df = pd.read_csv(file)

    test_df = test_df[['path', 'slope_avg']]

    transform = get_eval_transform()
    test_loader = DataLoader(SlopeDataset(test_df, transform), batch_size=16)

    trainer = VisionRegressor(model_name=model_name, lr=1e-3)
    path = str(model_path() / 'best_model.pth')
    trainer.load_best_model(path)

    loss, r2, mae, mse, rmse = trainer.run_test(test_loader)

    print('✅ 최종 테스트 결과')
    print(f'LOSS: {loss:.4f}')
    print(f'MAE (평균 각도 오차): {mae:.2f}°')
    print(f'R2 Score: {r2:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')


if __name__ == '__main__':
    execute_test('convnext_tiny', data_path() / 'test')
