#!/usr/bin/env python3

from typing import Any
from typing import Iterable
from typing import Iterator
from typing import NamedTuple
from typing import TypeVar
from typing import Sequence
from typing import Callable

from dlmp1.train import TrainConfig
from dlmp1.train import ModelFactory
from dlmp1.models.resnet import Hyperparametry
from dlmp1.models.resnet import BlockSpec
from dlmp1.models.resnet import CustomResNet

T = TypeVar("T")


def construct(type_: Callable[..., T], kwargs: dict[str, Any]) -> T:
    not_none_kwargs = {}
    for k, v in kwargs.items():
        if v is not None:
            not_none_kwargs[k] = v
    return type_(**not_none_kwargs)


def iterate_hyperparametries(first_conv_kernel_sizes: Iterable[int] = None,
                             dropout_rates: Iterable[float] = None) -> Iterator[Hyperparametry]:
    for first_conv_kernel_size in (first_conv_kernel_sizes or [None]):
        for dropout_rate in (dropout_rates or [None]):
            # pre_blocks_dropout_rate: float = 0.0
            #     between_blocks_dropout_rate: float = 0.0
            #     post_blocks_dropout_rate: float = 0.0
            kwargs = {
                "first_conv_kernel_size": first_conv_kernel_size,
                "pre_blocks_dropout_rate": dropout_rate,
                "between_blocks_dropout_rate": dropout_rate,
                "post_blocks_dropout_rate": dropout_rate,
            }
            yield construct(Hyperparametry, kwargs)


class TaggedModelFactory:

    def __init__(self, creator: ModelFactory, description: str):
        self.creator = creator
        self.description = description

    def __call__(self):
        return self.creator()


def _model_factory(numblocks_seq: Sequence[int],
                   hyperparametry: Hyperparametry,
                   conv_kernel_size: int = 3) -> ModelFactory:
    def _create_model():
        block_specs = []
        for index, numblocks in enumerate(numblocks_seq):
            stride = 1 if index == 0 else 2
            planes = 64 if index == 0 else None
            block_specs.append(BlockSpec(num_blocks=numblocks, planes=planes, stride=stride, conv_kernel_size=conv_kernel_size))
        return CustomResNet(block_specs, hyperparametry)
    return TaggedModelFactory(_create_model, f"{'-'.join(map(str, numblocks_seq))};h={hyperparametry.describe()};k={conv_kernel_size}")


def iterate_model_factories(numblocks_specs: Iterable[Sequence[int]],
                            hyperparametries: Iterable[Hyperparametry] = None,
                            block_conv_kernel_sizes: Iterable[int] = None) -> Iterator[ModelFactory]:
    block_conv_kernel_sizes = block_conv_kernel_sizes or [3]
    for numblocks_seq in numblocks_specs:
        for hyperparametry in (hyperparametries or [Hyperparametry()]):
            for block_conv_kernel_size in block_conv_kernel_sizes:
                yield _model_factory(numblocks_seq, hyperparametry, conv_kernel_size=block_conv_kernel_size)


class Selectable(NamedTuple):

    model_factory: ModelFactory
    train_config: TrainConfig
    description: str = ""


def iterate_selectables(model_factories: Iterable[ModelFactory],
                        train_configs: Iterable[TrainConfig] = None) -> Iterator[Selectable]:
    for model_factory in model_factories:
        for train_config in (train_configs or [TrainConfig()]):
            model_desc = getattr(model_factory, "description", "") or "model"
            config_desc = f"lr={train_config.learning_rate};opt={train_config.optimizer_type};sch={train_config.lr_scheduler_spec}"
            yield Selectable(model_factory, train_config, description=f"{model_desc};{config_desc}")
