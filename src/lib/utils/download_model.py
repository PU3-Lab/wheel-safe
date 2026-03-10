import gdown

from lib.utils.path import model_path


def download_model():
    url = 'https://drive.google.com/drive/folders/1KseIwH6qszZPig-11tFOaAnsTGqo3WY2?usp=sharing'

    gdown.download_folder(url, output=str(model_path()))


if __name__ == '__main__':
    download_model()
