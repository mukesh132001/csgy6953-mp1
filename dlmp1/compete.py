#!/usr/bin/env python3
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
import torchvision.transforms.v2 as transforms

import dlmp1.main
import dlmp1.utils
from dlmp1.models.resnet import CustomResNet
from dlmp1.models.resnet import BlockSpec

def write(array: np.ndarray, image_id: int, output_dir: Path):
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
        write(array, image_id, output_dir)
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


def infer(images_dir, model_file, output_file):
    images_dir = Path(images_dir or dlmp1.utils.get_repo_root() / "data" / "cifar_test_nolabels")
    model_file = Path(model_file or dlmp1.utils.get_repo_root() / "checkpoints" / "ckpt-20240403-001830.pth")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_file, map_location=device)
    net = CustomResNet([
        BlockSpec(2, 64, stride=1),
        BlockSpec(5, 128, stride=2),
        BlockSpec(3, 256, stride=2),
    ])
    state_dict = ckpt['net']
    state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    transform = dlmp1.main.get_test_set_transform()
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
                    outputs = net(image_tensor.unsqueeze(dim=0))
                    _, predicted = outputs.max(1)
                    csv_writer.writerow([image_id, predicted.item()])


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-m", "--model")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    infer(args.input, args.model, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
