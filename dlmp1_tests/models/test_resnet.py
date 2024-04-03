#!/usr/bin/env python3

from unittest import TestCase

import torch
import torch.random
from torch import nn
from torch import Tensor
# noinspection PyUnresolvedReferences
from torchsummary import summary

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


def creator(block_sizes: list[int], conv_kernel_sizes: list[int] = None, start_planes: int = 64, pool_kernel_size: int = None):
    block_specs = []
    conv_kernel_sizes = conv_kernel_sizes or ([None] * len(block_sizes))
    for i, (block_size, conv_kernel_size) in enumerate(zip(block_sizes, conv_kernel_sizes)):
        stride = 2
        planes = None
        if i == 0:
            stride = 1
            planes = start_planes
        conv_kernel_size = conv_kernel_size or 3
        block_spec = BlockSpec(block_size, planes, stride, conv_kernel_size=conv_kernel_size, pool_kernel_size=pool_kernel_size)
        block_specs.append(block_spec)
    def create():
        model_ = CustomResNet(block_specs)
        return model_
    create._num_blocks_list = list(block_sizes)
    return create


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

    def test_creator(self):
        created = creator([2, 2, 2, 2])
        self.assertIsInstance(created(), CustomResNet)

    def test_constructable(self):
        create_fns = [
            creator([2, 2, 2, 2]),  # resnet18
            creator([2, 5, 3]),
            creator([2, 6, 3]),
            creator([2, 5, 4]),
            creator([3, 5, 3]),
            creator([2, 4, 4]),
            creator([3, 5, 4]),
            creator([2, 2, 1, 2]),
            creator([2, 1, 1, 2]),
            creator([1, 1, 1]),
            creator([2, 3, 1, 2]),
            creator([2, 4, 3], conv_kernel_sizes=[5, 3, 3]),
            creator([2, 5, 3], conv_kernel_sizes=[5, 3, 3]),
            creator([2, 2, 2, 2], start_planes=32),
        ]
        self._test_constructable(create_fns)

    def test_crazy_resnets(self):
        self._test_constructable([
            creator([2, 2], pool_kernel_size=16),
        ])

    def _test_constructable(self, create_fns):
        for create_fn in create_fns:
            # noinspection PyUnresolvedReferences
            num_blocks_list = create_fn._num_blocks_list
            with self.subTest(num_blocks_list):
                with torch.random.fork_rng():
                    model = create_fn()
                    x = torch.randn(1, 3, 32, 32)
                    model_stats = summary(model, input_data=x, verbose=0)
                    print(f"{str(num_blocks_list):<20}", f"{model_stats.trainable_params:9d} parameters")
                    # self.assertLessEqual(model_stats.trainable_params, 5_000_000, f"trainable param count: {num_blocks_list}")
                    y: Tensor = model(x)
                    self.assertEqual(torch.Size((1, 10)), y.shape)





