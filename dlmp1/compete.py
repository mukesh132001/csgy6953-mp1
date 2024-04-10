#!/usr/bin/env python3

"""Evaluate a model on the Kaggle no-labels test dataset.
"""

import csv
import os
import sys
import glob
import pickle
import contextlib
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional
from typing import TextIO

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import dlmp1.train
import dlmp1.utils
import dlmp1.select
from dlmp1.models.resnet import Hyperparametry
from dlmp1.train import ModelFactory
from dlmp1.evaluate import parse_structure
from dlmp1.evaluate import load_model_parameters


def write_image(array: np.ndarray, image_id: int, output_dir: Path):
    hwc_image = np.moveaxis(array.reshape(3, 32, 32), 0, -1)
    image = Image.fromarray(hwc_image)
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(str(output_dir / f"{image_id}.png"))


def save_images(input_file, output):
    output_dir = Path(output or (dlmp1.utils.get_repo_root() / "data" / "cifar_test_nolabels"))
    custom_test_dataset_pickle_file = input_file or (dlmp1.utils.get_repo_root() / "data" / "cifar_test_nolabels.pkl")
    with open(custom_test_dataset_pickle_file, "rb") as ifile:
        custom_test_dataset = pickle.load(ifile)
    count = 0
    for array, image_id in zip(custom_test_dataset[b'data'], custom_test_dataset[b'ids']):
        write_image(array, image_id, output_dir)
        count += 1
    print(count, "files saved")


@contextlib.contextmanager
def open_csv_write(output_file: Optional[str] = None) -> TextIO:
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", newline="") as ofile:
            yield ofile
    else:
        yield sys.stdout


def infer(images_dir: Path,
          model_factory: ModelFactory,
          checkpoint_file: Path,
          output_file: Optional[str] = None,
          device: Optional[str] = None):
    model = model_factory()
    load_model_parameters(model, checkpoint_file, device=device)
    model.eval()
    transform = dlmp1.train.get_test_set_transform()
    with torch.no_grad():
        with open_csv_write(output_file) as ofile:
            csv_writer = csv.writer(ofile)
            csv_writer.writerow(["ID", "Labels"])
            progress_disable = not output_file
            image_pathnames = glob.glob(os.path.join(images_dir, "*.png"))
            image_pathnames.sort()
            for image_pathname in tqdm(image_pathnames, disable=progress_disable):
                with Image.open(image_pathname) as im:
                    image_id = Path(image_pathname).stem
                    image_tensor: torch.Tensor = transform(im)
                    image_tensor.to(device)
                    outputs = model(image_tensor.unsqueeze(dim=0))
                    _, predicted = outputs.max(1)
                    csv_writer.writerow([image_id, predicted.item()])


def perform(images_dir: Path,
            model_structure: list[int],
            checkpoint_file: Path,
            output_file: Optional[str] = None):
    model_factory = dlmp1.select.create_model_factory(model_structure, Hyperparametry())
    infer(images_dir, model_factory, checkpoint_file, output_file=output_file)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", metavar="DIR", required=True)
    parser.add_argument("-c", "--checkpoint", metavar="FILE", required=True)
    parser.add_argument("-s", "--structure", metavar="N-N-N", required=True)
    parser.add_argument("-o", "--output", metavar="FILE")
    args = parser.parse_args()
    images_dir = Path(args.input)
    model_structure = parse_structure(args.structure)
    checkpoint_file = Path(args.checkpoint)
    perform(images_dir, model_structure, checkpoint_file, output_file=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
