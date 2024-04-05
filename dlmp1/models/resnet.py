"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    expansion = 1  # class field

    def __init__(self, in_planes: int, planes: int, stride: int = 1, conv_kernel_size: int = 3):
        super(BasicBlock, self).__init__()
        padding = conv_kernel_size // 2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=conv_kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, conv_kernel_size: int = 3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


class BlockSpec(NamedTuple):

    num_blocks: int
    planes: Optional[int] = None
    stride: Optional[int] = 2
    block_type: Union[Type[BasicBlock], Type[Bottleneck]] = BasicBlock
    conv_kernel_size: int = 3
    pool_kernel_size: Optional[int] = None


class BlockLayerListContainer(NamedTuple):

    layers: nn.ModuleList
    in_size: int
    out_size: int
    pool_kernel_size: int

    @staticmethod
    def make_block_layers(block_specs: Sequence[BlockSpec]) -> 'BlockLayerListContainer':
        in_planes = block_specs[0].planes
        assert in_planes is not None, "planes must be specified for first block"
        in_size = in_planes
        def _make_layer(block_spec: BlockSpec):
            nonlocal in_planes
            num_blocks = block_spec.num_blocks
            strides = [block_spec.stride] + [1]*(num_blocks-1)
            block = block_spec.block_type
            layers = []
            planes = block_spec.planes or (in_planes * 2)
            for stride in strides:
                layers.append(block(in_planes, planes, stride, block_spec.conv_kernel_size))
                in_planes = planes * block.expansion
            return nn.Sequential(*layers)
        block_layers = nn.ModuleList()
        pool_kernel_size = None
        for block_spec_ in block_specs:
            layer = _make_layer(block_spec_)
            block_layers.append(layer)
            pool_kernel_size = block_spec_.pool_kernel_size
        out_size = in_planes
        if pool_kernel_size is None:
            # it's really best to specify this, but certain situations are known
            if out_size == 512:
                pool_kernel_size = 4
            elif out_size == 256:
                if in_size == 32:
                    pool_kernel_size = 4
                else:
                    pool_kernel_size = 8
            else:
                raise ValueError("could not determine pool kernel size; pool_kernel_size must be specified on last block spec")
        return BlockLayerListContainer(block_layers, in_size, out_size, pool_kernel_size)


class Hyperparametry(NamedTuple):

    first_conv_kernel_size: int = 3
    pre_blocks_dropout_rate: float = 0.0
    post_blocks_dropout_rate: float = 0.0

    def first_conv_kernel_padding(self) -> int:
        return self.first_conv_kernel_size // 2


class CustomResNet(nn.Module):
    def __init__(self, block_specs: list[BlockSpec], hyperparametry: Hyperparametry = None, num_classes=10):
        super(CustomResNet, self).__init__()
        hyperparametry = hyperparametry or Hyperparametry()
        in_planes = block_specs[0].planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=hyperparametry.first_conv_kernel_size, stride=1, padding=hyperparametry.first_conv_kernel_padding(), bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        container = BlockLayerListContainer.make_block_layers(block_specs)

        self.block_layers = container.layers
        self.linear = nn.Linear(container.out_size, num_classes)
        self.pool_kernel_size = container.pool_kernel_size
        self.summary_text = "resnet:" + "-".join(map(str, [b.num_blocks for b in block_specs])) + ";" + str(hyperparametry)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.block_layers:
            out = layer(out)
        out = F.avg_pool2d(out, self.pool_kernel_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class CustomResNetWithDropout(CustomResNet):

    def __init__(self,
                 block_specs: list[BlockSpec],
                 hyperparametry: Hyperparametry = None,
                 num_classes=10):
        super().__init__(block_specs, hyperparametry, num_classes=num_classes)
        hyperparametry = hyperparametry or Hyperparametry()
        self.pre_blocks_dropout = nn.Dropout(hyperparametry.pre_blocks_dropout_rate)
        self.post_blocks_dropout = nn.Dropout(hyperparametry.post_blocks_dropout_rate)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pre_blocks_dropout(out)
        for layer in self.block_layers:
            out = layer(out)
        out = self.post_blocks_dropout(out)
        out = F.avg_pool2d(out, self.pool_kernel_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
