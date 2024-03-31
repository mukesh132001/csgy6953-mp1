#!/usr/bin/env python3

from unittest import TestCase

import torch
import torch.random
from torch import nn
from torch import Tensor

from dlmp1.models.resnet import ResNet
from dlmp1.models.resnet import BasicBlock
from dlmp1.models.resnet import ResNet18
from dlmp1.models.resnet import ResNet34
from dlmp1.models.resnet import ResNet50
from dlmp1.models.resnet import CustomResNet
from dlmp1.models.resnet import BlockSpec
from dlmp1.models.resnet import ThreeLayerResNet

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

    def test_three_layer(self):
        custom3 = ThreeLayerResNet(BasicBlock, [2, 4, 3])
        self._try_forward(custom3)


class CustomResNetTest(TestCase):

    def test_conv_decompose(self):
        import torch.nn.functional as F
        # out = F.relu(self.bn1(self.conv1(x)))
        def _compute(conv, bn, x_):
            y = conv(x_)
            # y = bn(y)
            # y = F.relu(y)
            return y
        with torch.random.fork_rng():
            torch.random.manual_seed(9255)
            conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            torch.random.manual_seed(9255)
            conv2 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            bn1 = nn.BatchNorm2d(64)
            bn2 = nn.BatchNorm2d(64)
            with torch.no_grad():
                x1: Tensor = torch.randn(1, 3, 32, 32)
                x2: Tensor = x1.clone().detach()
                y2 = _compute(conv2, bn2, x2)
                y1 = _compute(conv1, bn1, x1)
                n = 4
                print(torch.flatten(y1)[:n])
                print(torch.flatten(y2)[:n])
                self.assertTrue(torch.equal(y1, y2))
    def test_conv(self):
        import torch.nn.functional as F
        # out = F.relu(self.bn1(self.conv1(x)))
        def _compute(model, x_):
            y = model.conv1(x_)
            y = model.bn1(y)
            return F.relu(y)
        seed = 9255
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            # torch.use_deterministic_algorithms(True)
            model1 = ResNet18()
            model1.eval()
            torch.random.manual_seed(seed)
            model2 = CustomResNet([
                BlockSpec(2, 64, stride=1),
                BlockSpec(2, 128, stride=2),
                BlockSpec(2, 256, stride=2),
                BlockSpec(2, 512, stride=2),
            ])
            model2.eval()
            with torch.no_grad():
                x: Tensor = torch.randn(1, 3, 32, 32)
                y1 = _compute(model1, x)
                y2 = _compute(model2, x)
                n = 4
                print(torch.flatten(y1)[:n])
                print(torch.flatten(y2)[:n])
                self.assertTrue(torch.equal(y1, y2))

    def test_same(self):
        seed = 1255
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            model1 = ResNet18()
            model1.eval()
            torch.random.manual_seed(seed)
            model2 = CustomResNet([
                BlockSpec(2, 64, stride=1),
                BlockSpec(2, 128, stride=2),
                BlockSpec(2, 256, stride=2),
                BlockSpec(2, 512, stride=2),
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
