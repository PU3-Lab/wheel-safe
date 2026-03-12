from pathlib import Path

from pyprojroot import here

root = here()


def data_path():
    path = root / 'data'
    Path.mkdir(path, parents=True, exist_ok=True)
    return path


def model_path():
    path = root / 'data' / 'models'
    Path.mkdir(path, parents=True, exist_ok=True)
    return path


def raw_data_path(depth_num):
    path = root / 'data' / 'raw' / f'depth_{depth_num}'
    Path.mkdir(path, parents=True, exist_ok=True)
    return path


def raw_path():
    path = root / 'data' / 'raw'
    Path.mkdir(path, parents=True, exist_ok=True)
    return path


def output_path():
    path = root / 'output'
    Path.mkdir(path, parents=True, exist_ok=True)
    return path


def runs_path():
    path = root / 'runs'
    Path.mkdir(path, parents=True, exist_ok=True)
    return path
