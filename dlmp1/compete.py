#!/usr/bin/env python3

import sys
import pickle
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from PIL import Image

import dlmp1.utils


def write(array: np.ndarray, image_id: int, output_dir: Path):
    hwc_image = np.moveaxis(array.reshape(3, 32, 32), 0, -1)
    image = Image.fromarray(hwc_image)
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(str(output_dir / f"{image_id}.png"))


def save_images(input_file, output_dir: Path):
    custom_test_dataset_pickle_file = input_file or (dlmp1.utils.get_repo_root() / "data" / "cifar_test_nolabels.pkl")
    with open(custom_test_dataset_pickle_file, "rb") as ifile:
        custom_test_dataset = pickle.load(ifile)
    count = 0
    for array, image_id in zip(custom_test_dataset[b'data'], custom_test_dataset[b'ids']):
        write(array, image_id, output_dir)
        count += 1
    print(count, "files saved")


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-m", "--model")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    output_dir = Path(args.output or (dlmp1.utils.get_repo_root() / "data" / "cifar_test_nolabels"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
