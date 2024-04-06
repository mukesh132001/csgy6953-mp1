import os
import json
import tempfile
from unittest import TestCase

import torch.random
import torch.optim
import torch.optim.lr_scheduler
import torchsummary

import dlmp1.train
from dlmp1.train import Partitioning
from dlmp1.models.resnet import ResNet18
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from dlmp1.train import TrainConfig
from dlmp1.models.resnet import CustomResNet
from dlmp1.models.resnet import CustomResNetWithDropout
from dlmp1.models.resnet import Hyperparametry
from dlmp1.models.resnet import BlockSpec

import dlmp1.select

class ModuleMethodsTest(TestCase):

    def test_iterate_model_factories(self):
        factories = dlmp1.select.iterate_model_factories([
            [2, 1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [3, 4, 3],
            [2, 5, 3],
            [3, 5, 3],
            [2, 4, 3],
            [2, 5, 2],
            [2, 4, 2],
            # [2, 6, 3],  # too big
            # [2, 5, 4],  # too big
        ])
        for factory in factories:
            with self.subTest():
                model = factory()
                stats = torchsummary.summary(model, verbose=0)
                self.assertLessEqual(stats.trainable_params, 5_000_000, msg=model.summary_text)
                model.eval()
                with torch.random.fork_rng():
                    torch.random.manual_seed(498682)
                    x = torch.randn(1, 3, 32, 32)
                    model(x)
