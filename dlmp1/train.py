"""Train CIFAR10 with PyTorch."""

import sys
import datetime
from pathlib import Path
from typing import Any
from typing import Protocol
from typing import Iterator
from typing import Literal
from typing import NamedTuple
from typing import Optional

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch import Tensor
from torch.nn import Parameter
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.transforms.v2 as transforms

import os

from tqdm import tqdm

import dlmp1.utils
from dlmp1.utils import serialize_rng_state_str


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
# these are pre-calculated with utils.get_mean_and_std()
TRAIN_SET_MEAN = (0.4914, 0.4822, 0.4465)
TRAIN_SET_STDEV = (0.2023, 0.1994, 0.2010)


def get_test_set_transform() -> transforms.Transform:
    return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(TRAIN_SET_MEAN, TRAIN_SET_STDEV),
        ])


class TrainConfig(NamedTuple):

    learning_rate: float = 0.1
    epoch_count: int = 200
    checkpoint_file: Optional[str] = None
    seed: Optional[int] = None
    lr_scheduler_spec: Optional[str] = None
    optimizer_type: Literal["sgd", "adam"] = "sgd"
    sgd_momentum: float = 0.9
    weight_decay: float = 5e-4
    quiet: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self._asdict()

    def create_optimizer(self, parameters: Iterator[Parameter]) -> Optimizer:
        if self.optimizer_type == "sgd":
            return optim.SGD(parameters, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.sgd_momentum)
        if self.optimizer_type == "adam":
            return optim.Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        raise NotImplementedError("unsupported optimizer type")


    def create_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        lr_scheduler_spec = self.lr_scheduler_spec
        if not lr_scheduler_spec:
            return StepLR(optimizer, gamma=0.1, step_size=40)
        parts = lr_scheduler_spec.split(":", maxsplit=1)
        scheduler_type = parts[0]
        if lr_scheduler_spec == "upstream":
            return CosineAnnealingLR(optimizer, T_max=200)
        scheduler_class = {
            "cosine_anneal": CosineAnnealingLR,
            "step": StepLR,
            "exponential": ExponentialLR,
            "constant": ConstantLR,
            "multistep": MultiStepLR,
        }[scheduler_type]
        kwargs = {
            "cosine_anneal": {"T_max": self.epoch_count},
        }.get(scheduler_type, {})
        kwarg_types = {
            "cosine_anneal": {"T_max": int},
            "step": {"step_size": int},
            "multistep": {"milestones": lambda milestones_str: [int(m) for m in milestones_str.split(",")]}
        }
        params = dict(p.split('=', maxsplit=1) for p in parts[1].split(";"))
        for k, v in params.items():
            if k == "last_epoch":
                value_type = int
            else:
                value_type = kwarg_types.get(scheduler_type, {}).get(k, float)
            params[k] = value_type(v)
        kwargs.update(params)
        return scheduler_class(optimizer, **kwargs)


def _truncate(dataset: torch.utils.data.Dataset, limit: int):
    dataset.data = dataset.data[:limit]
    dataset.targets = dataset.targets[:limit]


class Dataset(NamedTuple):

    trainloader: DataLoader
    valloader: DataLoader

    @staticmethod
    def acquire(batch_size_train: int,
                batch_size_test: int = 100,
                truncate_train: Optional[int] = None,
                truncate_test: Optional[int] = None,
                quiet: bool = False) -> 'Dataset':
        if not quiet:
            print(f"==> Preparing data; batch size: {batch_size_train} train, {batch_size_test} test")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(TRAIN_SET_MEAN, TRAIN_SET_STDEV),
        ])

        transform_test = get_test_set_transform()

        data_dir = str(dlmp1.utils.get_repo_root() / "data")

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        if truncate_train:
            _truncate(trainset, truncate_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        if truncate_test:
            _truncate(testset, truncate_test)
        valloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size_test, shuffle=False, num_workers=2)
        return Dataset(trainloader, valloader)


class History:

    def __init__(self, losses: list[float] = None, accs: list[float] = None):
        self.losses = [] if losses is None else losses
        self.accs = [] if accs is None else accs

    def __str__(self) -> str:
        return f"History(losses={self.losses}, accs={self.accs})"


class TrainResult(NamedTuple):

    checkpoint_file: Path
    timestamp: str
    train_history: History
    test_history: History


class ModelFactory(Protocol):

    def __call__(self) -> nn.Module: ...


def perform(model_provider: ModelFactory, dataset: Dataset, *, config: TrainConfig = None, resume: bool = False) -> TrainResult:
    config = config or TrainConfig()
    checkpoint_file = config.checkpoint_file or './checkpoint/ckpt.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    was_seeded = False
    if config.seed is not None:
        torch.random.manual_seed(config.seed)
        was_seeded = True
    if not resume:
        print("random seed:", torch.random.initial_seed())
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    train_hist, test_hist = History(), History()

    # Model
    net = model_provider()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if resume:
        # Load checkpoint.
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        train_hist = History(checkpoint.get('train_losses', []), checkpoint.get('train_accs', []))
        test_hist = History(checkpoint.get('test_losses', []), checkpoint.get('test_accs', []))
        rng_state = checkpoint.get('rng_state', None)
        if rng_state is not None:
            was_seeded = True
            torch.random.set_rng_state(rng_state)
        rng_state = torch.random.get_rng_state()
        print("rng state")
        print()
        print(serialize_rng_state_str(rng_state))
        print()
        print("==> Resuming from checkpoint", checkpoint_file, "at epoch", start_epoch, "with best acc", best_acc)

    criterion = nn.CrossEntropyLoss()
    optimizer = config.create_optimizer(net.parameters())
    scheduler = config.create_lr_scheduler(optimizer)

    def _report_progress(message: str):
        if not config.quiet:
            print(message)

    # Training
    def train():
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        mean_train_loss = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataset.trainloader), total=len(dataset.trainloader), file=sys.stdout, disable=config.quiet):
            inputs: Tensor
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            mean_train_loss = train_loss/(batch_idx+1)
        epoch_train_acc = correct / total
        train_hist.losses.append(mean_train_loss)
        train_hist.accs.append(epoch_train_acc)
        _report_progress(f"\nTrain Loss: {mean_train_loss:.3f} | Acc: {100 * epoch_train_acc:.2f}% ({correct}/{total})")


    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        mean_test_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataset.valloader):
                inputs: Tensor
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                mean_test_loss = test_loss/(batch_idx+1)
        epoch_test_acc = correct / total
        test_hist.accs.append(epoch_test_acc)
        test_hist.losses.append(mean_test_loss)
        _report_progress(f" Test Loss: {mean_test_loss:.3f} | Acc: {100 * epoch_test_acc:.2f}% ({correct}/{total})")

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'train_losses': train_hist.losses,
                'test_losses': test_hist.losses,
                'train_accs': train_hist.accs,
                'test_accs': test_hist.accs,
                'model_description': str(net),
                'summary_text': getattr(net, "summary_text", ""),
                'train_config': config.to_dict()
            }
            if was_seeded:
                state['rng_state'] = torch.random.get_rng_state()
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_file)
            print('saved checkpoint to', checkpoint_file)
            best_acc = acc


    for epoch_ in range(start_epoch, start_epoch + config.epoch_count):
        _report_progress(f'\nEpoch: {epoch_+1}/{start_epoch + config.epoch_count}')
        train()
        test(epoch_)
        _report_progress(f"{scheduler.get_last_lr()} was learning rate for epoch {epoch_+1}")
        scheduler.step()

    return TrainResult(
        checkpoint_file=Path(checkpoint_file),
        timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        train_history=train_hist,
        test_history=test_hist,
    )
