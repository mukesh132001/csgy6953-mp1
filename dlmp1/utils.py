"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""

import base64
import datetime
from pathlib import Path

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
from torch import Tensor


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
def count_parameters(model: nn.Module, include_non_trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (include_non_trainable or p.requires_grad))


def serialize_rng_state_str(state: Tensor) -> str:
    np_state: np.ndarray = state.clone().detach().cpu().numpy()
    print(np_state.shape)
    print(np_state.dtype)
    np_state_bytes = np_state.tobytes()
    encoded = base64.encodebytes(np_state_bytes).decode('us-ascii')
    return encoded


def get_repo_root() -> Path:
    return Path(__file__).absolute().parent.parent


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")