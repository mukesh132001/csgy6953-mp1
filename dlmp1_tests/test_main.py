
from unittest import TestCase
from dlmp1.models.resnet import ResNet18
import torch.random
import torch.optim
import torch.optim.lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from dlmp1.main import TrainConfig


class RandomSeedTest(TestCase):

    def test_initial_seed(self):
        with torch.random.fork_rng():
            a = torch.random.initial_seed()
            print(a)

    def test_manual_seed_report(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(1234)
            a = torch.random.initial_seed()
            print(a)
            torch.rand(size=(10, 10))
            b = torch.random.initial_seed()
            print(b)
            self.assertEqual(a, b)


def _test_optimizer(initial_lr: float = 0.1) -> torch.optim.Optimizer:
    net = ResNet18()
    optimizer = torch.optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    return optimizer


class SchedulerTest(TestCase):

    def test_annealing(self):
        with torch.random.fork_rng():
            optimizer = _test_optimizer()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
            for epoch in range(205):
                lr = scheduler.get_last_lr()
                print(f"{epoch:3d} {lr}")
                optimizer.step()
                scheduler.step()


def _run_schedule(optimizer: Optimizer, scheduler: LRScheduler, epochs: int = 500) -> list[float]:
    lrs = []
    for _ in range(epochs):
        lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        lrs.append(lr)
    return lrs


class TrainConfigTest(TestCase):

    def test_create_lr_scheduler_default(self):
        with torch.random.fork_rng():
            optimizer = _test_optimizer()
            scheduler = TrainConfig().create_lr_scheduler(optimizer)
            self.assertIsInstance(scheduler, MultiStepLR)
            _run_schedule(optimizer, scheduler)
            self.assertGreater(len(list(scheduler.milestones)), 0)
            self.assertEqual(0.1, scheduler.gamma)

    def test_create_lr_scheduler_multistep(self):
        initial_lr = 0.1
        gamma = 0.1
        with torch.random.fork_rng():
            optimizer = _test_optimizer(initial_lr)
            scheduler = TrainConfig(lr_scheduler_spec=f"multistep:{gamma},40,80,120,160").create_lr_scheduler(optimizer)
            self.assertIsInstance(scheduler, MultiStepLR)
            learning_rates = _run_schedule(optimizer, scheduler)
        for epoch, expected_lr in [
            (0, initial_lr),
            (39, initial_lr),
            (40, initial_lr * gamma),
            (41, initial_lr * gamma),
            (79, initial_lr * gamma),
            (80, initial_lr * gamma ** 2),
            (81, initial_lr * gamma ** 2),
            (119, initial_lr * gamma ** 2),
            (120, initial_lr * gamma ** 3),
            (121, initial_lr * gamma ** 3),
        ]:
            with self.subTest(epoch=epoch):
                self.assertAlmostEqual(expected_lr, learning_rates[epoch], msg=f"learning rate at {epoch}", delta=1e-6)

    def test_constant(self):
        with torch.random.fork_rng():
            lr = 0.01
            optimizer = _test_optimizer(lr)
            scheduler = TrainConfig(lr_scheduler_spec=f"multistep:1.0").create_lr_scheduler(optimizer)
            rates = _run_schedule(optimizer, scheduler)
            self.assertSetEqual({lr}, set(rates))


    def test_scheduler_upstream(self):
        with torch.random.fork_rng():
            scheduler = TrainConfig(lr_scheduler_spec="upstream").create_lr_scheduler(_test_optimizer())
            self.assertIsInstance(scheduler, CosineAnnealingLR)
            self.assertEqual(200, scheduler.T_max)
