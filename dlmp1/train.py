"""Train CIFAR10 with PyTorch."""

import sys
from pathlib import Path
from typing import Any
from typing import Protocol
from typing import Iterator
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Union
from typing import Callable

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
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms.v2 import Transform

import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.datasets import VisionDataset

import os

from tqdm import tqdm

import dlmp1.utils
from dlmp1.utils import serialize_rng_state_str


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
# these are pre-calculated with utils.get_mean_and_std()
TRAIN_SET_MEAN = (0.4914, 0.4822, 0.4465)
TRAIN_SET_STDEV = (0.2023, 0.1994, 0.2010)
Device = Union[str, torch.device]
Criterion = Callable[[Tensor, Tensor], Tensor]

def get_test_set_transform() -> transforms.Transform:
    return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(TRAIN_SET_MEAN, TRAIN_SET_STDEV),
        ])


class TrainConfig(NamedTuple):

    """Value class that represents training configuration.

    Some examples of lr_scheduler_spec values:

    * step:gamma=0.1;step_size=20
    * multistep:gamma=0.1;milestones=[50,75,100]
    * cosine_anneal:eta_min=0.0001

    Use lr_scheduler_spec="step:gamma=1;step_size=1" for a constant learning rate.
    """

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
    if hasattr(dataset, "data"):
        dataset.data = dataset.data[:limit]
        dataset.targets = dataset.targets[:limit]
    elif isinstance(dataset, torch.utils.data.Subset):
        indices = dataset.indices[:limit]
        dataset.indices = indices
    elif isinstance(dataset, TransformedDataset):
        _truncate(dataset.untransformed, limit)
    else:
        raise NotImplementedError(f"unsupported dataset type: {type(dataset)}")



class TransformedDataset(Dataset):

    def __init__(self, untransformed: VisionDataset, transform: Transform):
        self.untransformed = untransformed
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.untransformed[index]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.untransformed)



class Partitioning(NamedTuple):

    trainloader: DataLoader
    valloader: DataLoader

    @staticmethod
    def prepare_test_loader(batch_size: int = 100) -> DataLoader:
        transform_test = get_test_set_transform()
        data_dir = str(dlmp1.utils.get_repo_root() / "data")
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        return testloader

    @staticmethod
    def prepare(batch_size_train: int,
                batch_size_val: int = 100,
                val_proportion: float = 0.1,
                truncate_train: Optional[int] = None,
                truncate_val: Optional[int] = None,
                random_seed: Optional[int] = None,
                quiet: bool = False) -> 'Partitioning':
        if not quiet:
            print(f"==> Preparing data; batch size: {batch_size_train} train, {batch_size_val} validation")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(TRAIN_SET_MEAN, TRAIN_SET_STDEV),
        ])

        transform_val = get_test_set_transform()

        data_dir = str(dlmp1.utils.get_repo_root() / "data")

        full_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
        rng = None if random_seed is None else torch.Generator().manual_seed(random_seed)
        trainset, valset = torch.utils.data.random_split(full_trainset, [1 - val_proportion, val_proportion], generator=rng)
        trainset = TransformedDataset(trainset, transform_train)
        valset = TransformedDataset(valset, transform_val)
        if truncate_train:
            _truncate(trainset, truncate_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

        if truncate_val:
            _truncate(valset, truncate_val)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size_val, shuffle=False, num_workers=2)
        return Partitioning(trainloader, valloader)


class History:

    def __init__(self, losses: list[float] = None, accs: list[float] = None):
        self.losses = [] if losses is None else losses
        self.accs = [] if accs is None else accs

    def __str__(self) -> str:
        return f"History(losses={self.losses}, accs={self.accs})"


class TrainResult(NamedTuple):

    checkpoint_file: Path
    model: nn.Module
    device: Device
    timestamp: str
    train_history: History
    val_history: History


class Restored(NamedTuple):

    train_history: History
    val_history: History
    was_seeded: bool


def restore(checkpoint_file: str, net: nn.Module, quiet: bool = False) -> Restored:
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    train_hist = History(checkpoint.get('train_losses', []), checkpoint.get('train_accs', []))
    val_hist = History(checkpoint.get('val_losses', []), checkpoint.get('val_accs', []))
    rng_state = checkpoint.get('rng_state', None)
    was_seeded = False
    if rng_state is not None:
        was_seeded = True
        torch.random.set_rng_state(rng_state)
    rng_state = torch.random.get_rng_state()
    if not quiet:
        print("rng state")
        print()
        print(serialize_rng_state_str(rng_state))
        print()
        print("==> Resuming from checkpoint", checkpoint_file, "at epoch", start_epoch, "with best acc", best_acc)
    return Restored(train_hist, val_hist, was_seeded)

class EpochInference(NamedTuple):

    correct: int
    total: int
    mean_loss: float

    def accuracy(self) -> float:
        return self.correct / self.total


def inference_all(net: nn.Module,
                  device: Device,
                  dataloader: DataLoader,
                  criterion: Optional[Criterion] = None,
                  show_progress: bool = False) -> EpochInference:
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    mean_loss = float("nan") if criterion is None else 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), disable=not show_progress):
            inputs: Tensor
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                mean_loss = val_loss / (batch_idx+1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return EpochInference(correct, total, mean_loss)


class ModelFactory(Protocol):

    def __call__(self) -> nn.Module: ...


def perform(model_provider: ModelFactory,
            dataset: Partitioning,
            *,
            config: TrainConfig = None,
            resume: bool = False) -> TrainResult:
    config = config or TrainConfig()
    checkpoint_file = config.checkpoint_file or './checkpoint/ckpt.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    was_seeded = False
    if config.seed is not None:
        torch.random.manual_seed(config.seed)
        was_seeded = True
    if not resume:
        print("random seed:", torch.random.initial_seed())
    best_acc = 0  # best validation accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    train_hist, val_hist = History(), History()

    # Model
    net = model_provider()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if resume:
        restored = restore(checkpoint_file, net)
        train_hist = restored.train_history
        val_hist = restored.val_history
        was_seeded = was_seeded or restored.was_seeded
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
        inf_result = inference_all(net, device, dataset.valloader, criterion=criterion)
        epoch_val_acc = inf_result.accuracy()
        val_hist.accs.append(epoch_val_acc)
        val_hist.losses.append(inf_result.mean_loss)
        _report_progress(f"  Val Loss: {inf_result.mean_loss:.3f} | Acc: {100 * epoch_val_acc:.2f}% ({inf_result.correct}/{inf_result.total})")

        # Save checkpoint.
        acc = 100. * inf_result.accuracy()
        if acc > best_acc:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'train_losses': train_hist.losses,
                'val_losses': val_hist.losses,
                'train_accs': train_hist.accs,
                'val_accs': val_hist.accs,
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
        model=net,
        device=device,
        timestamp=dlmp1.utils.timestamp(),
        train_history=train_hist,
        val_history=val_hist,
    )
