import sys

import torch


def get_device():
    device = 'cpu'
    if sys.platform == 'win32' or sys.platform == 'linux':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'mps' if torch.backends.mps else 'cpu'
    print(f'Using {device}')
    return device
