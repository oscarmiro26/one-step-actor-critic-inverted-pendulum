# util/device.py

import torch

def fetch_device():
    """
    Fetches the available device ('cuda' if available, else 'cpu').

    :return: torch.device object
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
