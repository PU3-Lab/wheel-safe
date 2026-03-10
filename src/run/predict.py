import argparse
import os
from glob import glob

from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from lib.utils.path import data_path, model_path
import lib.utils.const as const
from models.visoin_regreesor import VisionRegressor

from torchvision import transforms

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((const.RESIZE_SIZE, const.RESIZE_SIZE)),
        transforms.CenterCrop((const.CROP_SIZE, const.CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=const.IMAGENET_MEAN, std=const.IMAGENET_STD),
    ])

def execute(model_name, input_path, show_image=None):
    transform = get_eval_transform()

    trainer = VisionRegressor(model_name=model_name, lr=1e-3)
    trainer.load_best_model(model_path() / 'best_model.pth')

    jpg_files = glob(os.path.join(input_path, '*.jpg'))
    png_files = glob(os.path.join(input_path, '*.png'))
    test_folder = sorted(jpg_files + png_files)

    for _, image in enumerate(test_folder):
        slope = trainer.predict(image, transform)
        print(f'예측된 경사도(Slope): {slope:.4f}')
        if show_image is not None:
            show_image(image, slope)


def main():
    print('=' * 100)
    print('uv run ./src/run/predict.py')
    print('=' * 100)

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

    execute(model_name, input_path)


if __name__ == '__main__':
    main()
