#!/usr/bin/env python3

"""Evaluate a model on the CIFAR test dataset.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch
from torch import nn

import dlmp1.select
import dlmp1.train
import dlmp1.utils
from dlmp1.models.resnet import Hyperparametry
from dlmp1.train import EpochInference
from dlmp1.train import ModelFactory
from dlmp1.train import Partitioning


def load_model_parameters(model: nn.Module, checkpoint_file: Path, device: Optional[str] = None):
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_file, map_location=device)
    state_dict = ckpt['net']
    state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)


def predict(model_factory: ModelFactory,
            checkpoint_file: Path,
            hide_progress: bool = False) -> EpochInference:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model_factory()
    load_model_parameters(model, checkpoint_file, device=device)
    dataloader = Partitioning.prepare_test_loader(num_workers=0)
    inference = dlmp1.train.inference_all(model, device, dataloader, show_progress=not hide_progress)
    return inference


def parse_structure(structure_token: str) -> list[int]:
    return [int(t) for t in structure_token.split('-')]


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True, metavar="FILE", action='append')
    parser.add_argument("-s", "--structure", required=True, metavar="N-N-N", action='append')
    args = parser.parse_args()
    if len(args.structure) == 1 and len(args.checkpoint) > 1:
        args.structure = args.structure * len(args.checkpoint)
    if len(args.structure) != len(args.checkpoint):
        parser.error("invalid number of structures and checkpoints; must be equal or exactly 1 structure that is assumed for all checkpoints")
    for structure, checkpoint_file in zip(args.structure, args.checkpoint):
        checkpoint_file = Path(checkpoint_file)
        structure = parse_structure(structure)
        model_factory = dlmp1.select.create_model_factory(structure, Hyperparametry())
        inference = predict(model_factory, checkpoint_file)
        description = getattr(model_factory, "description", checkpoint_file.name)
        print(f"{inference.accuracy()*100:.2f}%", f"({inference.correct}/{inference.total})", description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
