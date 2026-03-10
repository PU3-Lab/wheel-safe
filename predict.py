import argparse
import os
from glob import glob

from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from lib.utils.path import data_path, model_path
from models.visoin_regreesor import VisionRegressor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_name', nargs='?', help='모델 이름', default='efficientnet_b0'
    )
    parser.add_argument(
        'input_path', nargs='?', help='이미지 경로', default=str(data_path() / 'test')
    )

    args = parser.parse_args()

    model_name = args.model_name
    input_path = args.input_path

    transform = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    trainer = VisionRegressor(model_name=model_name, lr=1e-3)
    trainer.load_best_model(model_path() / 'best_model.pth')

    jpg_files = glob(os.path.join(input_path, '*.jpg'))
    png_files = glob(os.path.join(input_path, '*.png'))
    test_folder = sorted(jpg_files + png_files)

    for _, image in enumerate(test_folder):
        result = trainer.predict(image, transform)
        print(f'예측된 경사도(Slope): {result:.4f}')


if __name__ == '__main__':
    main()
