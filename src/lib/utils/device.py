import sys

import torch


def get_device():
    if sys.platform == 'win32' or sys.platform == 'linux':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return 'mps' if torch.backends.mps else 'cpu'
