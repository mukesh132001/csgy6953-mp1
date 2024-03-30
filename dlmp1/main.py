"""Train CIFAR10 with PyTorch."""

import sys
from typing import NamedTuple
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch import Tensor

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm import tqdm
from dlmp1.models.resnet import ResNet18


MODEL_FACTORIES = {
    ResNet18.__name__: ResNet18,
}



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
        batch_size_train=args.batch_size or 128
    )
    perform(
        model_name=model_name,
        config=config,
        resume=args.resume,
    )
    return 0


class TrainConfig(NamedTuple):

    epoch_count: int = 200
    learning_rate: float = 0.1
    batch_size_train: int = 128
    batch_size_test: int = 100
    verbose_scheduler: bool = False


def perform(*, model: nn.Module = None, model_name: str = None, config: TrainConfig = None, resume: bool = False):
    config = config or TrainConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print(f"==> Preparing data; batch size: {config.batch_size_train} train, {config.batch_size_test} test")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size_train, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size_test, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    if model_name:
        assert model is None, f"must provide exactly one of {model, model_name}"
        print('==> Building model', model_name, "on", device)
        net_factory = MODEL_FACTORIES[model_name]
        net = net_factory()
    else:
        net = model
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    def _report_progress(message: str):
        print(message)

    # Training
    def train():
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        mean_train_loss = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader), file=sys.stdout):
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
        _report_progress(f"\nTrain Loss: {mean_train_loss:.3f} | Acc: {100 * correct / total:.2f}% ({correct}/{total})")


    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        mean_test_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs: Tensor
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                mean_test_loss = test_loss/(batch_idx+1)
        _report_progress(f" Test Loss: {mean_test_loss:.3f} | Acc: {100 * correct / total:.2f}% ({correct}/{total})")

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            checkpoint_file = './checkpoint/ckpt.pth'
            torch.save(state, checkpoint_file)
            print('saved checkpoint to', checkpoint_file)
            best_acc = acc


    for epoch_ in range(start_epoch, start_epoch + config.epoch_count):
        print(f'\nEpoch: {epoch_+1}/{start_epoch + config.epoch_count}')
        train()
        test(epoch_)
        scheduler.step()
        if config.verbose_scheduler:
            print(scheduler.get_last_lr(), "is learning rate")


if __name__ == '__main__':
    sys.exit(main())
