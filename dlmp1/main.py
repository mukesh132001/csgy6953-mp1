"""Train CIFAR10 with PyTorch."""

import os
import sys
from typing import Sequence
from argparse import ArgumentParser

from torch import nn

from dlmp1 import train
from dlmp1.train import Dataset
from dlmp1.train import TrainConfig
from dlmp1.models.resnet import ResNet18

MODEL_FACTORIES = {
    ResNet18.__name__: ResNet18,
}

def model_from_name(model_name: str) -> nn.Module:
    net_factory = MODEL_FACTORIES[model_name]
    return net_factory()


def main(argv1: Sequence[str] = None) -> int:
    parser = ArgumentParser(
        description='PyTorch CIFAR10 Training',
        epilog=f"""\
Models available:
  {(os.linesep + "  ").join(sorted(MODEL_FACTORIES.keys()))}
""")
    parser.add_argument("--model", metavar="NAME", choices=set(MODEL_FACTORIES.keys()))
    parser.add_argument("--epoch-count", type=int, default=200)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument("--batch-size", type=int, help="train batch size")
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args(argv1)
    model_name = args.model or ResNet18.__name__
    config = TrainConfig(
        epoch_count=args.epoch_count,
        learning_rate=args.lr,
    )
    batch_size_train: int = args.batch_size or 128
    batch_size_test: int = 100
    dataset = Dataset.acquire(batch_size_train, batch_size_test)
    train.perform(
        model_provider=lambda: model_from_name(model_name),
        dataset=dataset,
        config=config,
        resume=args.resume,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
