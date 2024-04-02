"""Train CIFAR10 with PyTorch."""

import sys
import datetime
from pathlib import Path
from typing import NamedTuple
from typing import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch import Tensor
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm import tqdm
from dlmp1.models.resnet import ResNet18
from dlmp1.utils import serialize_rng_state_str


MODEL_FACTORIES = {
    ResNet18.__name__: ResNet18,
}
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
# these are pre-calculated with utils.get_mean_and_std()
TRAIN_SET_MEAN = (0.4914, 0.4822, 0.4465)
TRAIN_SET_STDEV = (0.2023, 0.1994, 0.2010)



class TrainConfig(NamedTuple):

    epoch_count: int = 200
    learning_rate: float = 0.1
    verbose_scheduler: bool = False
    checkpoint_file: Optional[str] = None
    seed: Optional[int] = None
    lr_scheduler_spec: Optional[str] = None

    def create_lr_scheduler(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        lr_scheduler_spec = self.lr_scheduler_spec or "multistep:0.1,40,80,120,160"
        if lr_scheduler_spec == "upstream":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        parts = lr_scheduler_spec.split(":")
        scheduler_type, params = parts[0], parts[1:]
        if scheduler_type == "multistep":
            gamma = float(params[0])
            steps = [int(p) for p in params[1:]]
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=gamma)
        raise NotImplementedError(f"unrecognized lr scheduler: {repr(lr_scheduler_spec)}")


class Dataset(NamedTuple):

    trainloader: DataLoader
    testloader: DataLoader

    @staticmethod
    def acquire(batch_size_train: int, batch_size_test: int = 100, quiet: bool = False) -> 'Dataset':
        if not quiet:
            print(f"==> Preparing data; batch size: {batch_size_train} train, {batch_size_test} test")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_SET_MEAN, TRAIN_SET_STDEV),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_SET_MEAN, TRAIN_SET_STDEV),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size_test, shuffle=False, num_workers=2)
        return Dataset(trainloader, testloader)


class TrainResult(NamedTuple):

    checkpoint_file: Path
    timestamp: str


def model_from_name(model_name: str) -> nn.Module:
    net_factory = MODEL_FACTORIES[model_name]
    return net_factory()


def perform(model: nn.Module, dataset: Dataset, *, config: TrainConfig = None, resume: bool = False) -> TrainResult:
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
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    # Model
    net = model
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
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        test_accs = checkpoint.get('test_accs', [])
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
    optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = config.create_lr_scheduler(optimizer)

    def _report_progress(message: str):
        print(message)

    # Training
    def train():
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        mean_train_loss = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataset.trainloader), total=len(dataset.trainloader), file=sys.stdout):
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
        train_losses.append(mean_train_loss)
        train_accs.append(epoch_train_acc)
        _report_progress(f"\nTrain Loss: {mean_train_loss:.3f} | Acc: {100 * epoch_train_acc:.2f}% ({correct}/{total})")


    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        mean_test_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataset.testloader):
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
        test_accs.append(epoch_test_acc)
        test_losses.append(mean_test_loss)
        _report_progress(f" Test Loss: {mean_test_loss:.3f} | Acc: {100 * epoch_test_acc:.2f}% ({correct}/{total})")

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accs': train_accs,
                'test_accs': test_accs,
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
        timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )


def main(argv1: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(
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
    model = model_from_name(model_name)
    batch_size_train: int = args.batch_size or 128
    batch_size_test: int = 100
    dataset = Dataset.acquire(batch_size_train, batch_size_test)
    perform(
        model=model,
        dataset=dataset,
        config=config,
        resume=args.resume,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
