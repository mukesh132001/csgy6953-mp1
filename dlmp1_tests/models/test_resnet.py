#!/usr/bin/env python3

from unittest import TestCase

import torch
import torch.random
from torch import nn
from torch import Tensor

from dlmp1.models.resnet import ResNet18
from dlmp1.models.resnet import ResNet34
from dlmp1.models.resnet import ResNet50
from dlmp1.models.resnet import CustomResNet
from dlmp1.models.resnet import BlockSpec

class ResNetTest(TestCase):

    def test_resnet_builtins(self):
        for model_factory in [
            ResNet18,
            ResNet34,
            ResNet50,
        ]:
            with self.subTest():
                print(model_factory)
                model = model_factory()
                self._try_forward(model)

    def _try_forward(self, model: nn.Module):
        y: Tensor = model(torch.randn(1, 3, 32, 32))
        self.assertIsInstance(y, Tensor)
        print(y.shape)


class CustomResNetTest(TestCase):

    def test_same(self):
        seed = 1255
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            model1 = ResNet18()
            model1.eval()
            torch.random.manual_seed(seed)
            model2 = CustomResNet([
                BlockSpec(2, 64, stride=1),
                BlockSpec(2),
                BlockSpec(2),
                BlockSpec(2),
            ])
            model2.eval()
            with torch.no_grad():
                x: Tensor = torch.randn(1, 3, 32, 32)
                y1 = model1(x)
                # y1_verify = model1(x)
                # self.assertTrue(torch.equal(y1, y1_verify))
                y2 = model2(x)
                print("y1", y1)
                print("y2", y2)
                self.assertTrue(torch.equal(y1, y2))

    def test__make_block_layers(self):
        container = CustomResNet._make_block_layers([
            BlockSpec(2, 64, stride=1),
            BlockSpec(2, 128, stride=2),
            BlockSpec(2, 256, stride=2),
            BlockSpec(2, 512, stride=2),
        ])
        self.assertEqual(4, container.pool_kernel_size)
        self.assertEqual(512, container.out_size)
        self.assertEqual(64, container.in_size)
