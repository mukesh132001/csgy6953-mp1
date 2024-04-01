"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import os
import sys
import time
import base64

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
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

def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


class ProgressBar:

    def __init__(self):
        try:
            with os.popen('stty size', 'r') as p:
                term_width = p.read().split()[-1]
        except Exception as e:
            print("terminal width could not be determined due to", type(e).__name__, e, file=sys.stderr)
            term_width = 80
        self.term_width = int(term_width)

        self.TOTAL_BAR_LENGTH = 65.
        self.last_time = time.time()
        self.begin_time = self.last_time
    def update(self, current, total, msg=None):
        if current == 0:
            self.begin_time = time.time()  # Reset for new bar.

        cur_len = int(self.TOTAL_BAR_LENGTH*current/total)
        rest_len = int(self.TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        L = []
        L.append('  Step: %s' % format_time(step_time))
        L.append(' | Tot: %s' % format_time(tot_time))
        if msg:
            L.append(' | ' + msg)

        msg = ''.join(L)
        sys.stdout.write(msg)
        for i in range(self.term_width-int(self.TOTAL_BAR_LENGTH)-len(msg)-3):
            sys.stdout.write(' ')

        # Go back to the center of the bar.
        for i in range(self.term_width-int(self.TOTAL_BAR_LENGTH/2)+2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current+1, total))

        if current < total-1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
def count_parameters(model: nn.Module, include_non_trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (include_non_trainable or p.requires_grad))


def serialize_rng_state_str(state: Tensor) -> str:
    np_state: np.ndarray = state.clone().detach().cpu().numpy()
    print(np_state.shape)
    print(np_state.dtype)
    np_state_bytes = np_state.tobytes()
    encoded = base64.b64encode(np_state_bytes).decode('us-ascii')
    return encoded
