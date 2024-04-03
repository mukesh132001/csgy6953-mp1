import os
import json
import tempfile
from unittest import TestCase

from dlmp1.main import Dataset
from dlmp1.models.resnet import ResNet18
import torch.random
import torch.optim
import torch.optim.lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from dlmp1.main import TrainConfig
from dlmp1.models.resnet import CustomResNet
from dlmp1.models.resnet import BlockSpec
from dlmp1.main import perform


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


def _run_schedule(optimizer: Optimizer, scheduler: LRScheduler, epochs: int = 500, verbose: bool = False) -> list[float]:
    lrs = []
    width = len(str(epochs))
    for epoch in range(epochs):
        lr = scheduler.get_last_lr()[0]
        for _ in range(2):  # simulate multiple batches
            optimizer.step()
        scheduler.step()
        lrs.append(lr)
        if verbose:
            print(f"{epoch:{width}d} {lr:.6f}")
    return lrs


class TrainConfigTest(TestCase):

    def test_create_lr_scheduler_default(self):
        with torch.random.fork_rng():
            optimizer = _test_optimizer()
            config = TrainConfig()
            scheduler = config.create_lr_scheduler(optimizer)
            self.assertIsInstance(scheduler, StepLR)
            _run_schedule(optimizer, scheduler)
            self.assertEqual(0.1, scheduler.gamma)

    def test_create_lr_scheduler_multistep(self):
        initial_lr = 0.1
        gamma = 0.1
        with torch.random.fork_rng():
            optimizer = _test_optimizer(initial_lr)
            scheduler = TrainConfig(lr_scheduler_spec=f"multistep:gamma={gamma};milestones=40,70,90,100").create_lr_scheduler(optimizer)
            self.assertIsInstance(scheduler, MultiStepLR)
            learning_rates = _run_schedule(optimizer, scheduler)
        for epoch, expected_lr in [
            (0, initial_lr),
            (39, initial_lr),
            (40, initial_lr * gamma),
            (41, initial_lr * gamma),
            (69, initial_lr * gamma),
            (70, initial_lr * gamma ** 2),
            (71, initial_lr * gamma ** 2),
            (89, initial_lr * gamma ** 2),
            (90, initial_lr * gamma ** 3),
            (91, initial_lr * gamma ** 3),
            (99, initial_lr * gamma ** 3),
            (100, initial_lr * gamma ** 4),
            (101, initial_lr * gamma ** 4),
        ]:
            with self.subTest(epoch=epoch):
                self.assertAlmostEqual(expected_lr, learning_rates[epoch], msg=f"learning rate at {epoch}", delta=1e-6)

    def test_constant(self):
        with torch.random.fork_rng():
            lr = 0.025
            optimizer = _test_optimizer(lr)
            scheduler = TrainConfig(lr_scheduler_spec=f"step:gamma=1.0;step_size=1").create_lr_scheduler(optimizer)
            rates = _run_schedule(optimizer, scheduler)
            self.assertSetEqual({lr}, set(rates))


    def test_scheduler_upstream(self):
        with torch.random.fork_rng():
            scheduler = TrainConfig(lr_scheduler_spec="upstream").create_lr_scheduler(_test_optimizer())
            self.assertIsInstance(scheduler, CosineAnnealingLR)
            self.assertEqual(200, scheduler.T_max)

    def test_cosine_anneal_default(self):
        eta_min = 0.0001
        config = TrainConfig(lr_scheduler_spec=f"cosine_anneal:eta_min={eta_min}")
        optimizer = _test_optimizer(config.learning_rate)
        scheduler = config.create_lr_scheduler(optimizer)
        self.assertIsInstance(scheduler, CosineAnnealingLR)
        rates = _run_schedule(optimizer, scheduler)
        self.assertEqual(config.learning_rate, rates[0])
        self.assertAlmostEqual(rates[config.epoch_count - 1], eta_min, delta=1e-5)

    def test_cosine_anneal_optimal(self):
        eta_min = 0.00001
        min_epoch = 165
        config = TrainConfig(epoch_count=160, lr_scheduler_spec=f"cosine_anneal:eta_min={eta_min};T_max={min_epoch}")
        optimizer = _test_optimizer(config.learning_rate)
        scheduler = config.create_lr_scheduler(optimizer)
        self.assertIsInstance(scheduler, CosineAnnealingLR)
        rates = _run_schedule(optimizer, scheduler)
        self.assertEqual(config.learning_rate, rates[0])
        self.assertAlmostEqual(rates[min_epoch - 1], eta_min, delta=1e-5)
        for epoch in [0, 40, 80, 120, 159, 160]:
            print(f"{epoch:3d} {rates[epoch]:.6f}")

    def test_demo_step(self):
        config = TrainConfig()
        optimizer = _test_optimizer(config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        self._test_demo(scheduler, config)

    def test_demo_exponential(self):
        config = TrainConfig(learning_rate=0.1)
        optimizer = _test_optimizer(config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        self._test_demo(scheduler, config)

    def test_demo_cosine_anneal_ex2(self):
        eta_min = 0.00001
        # min_epoch = 100
        # config = TrainConfig(epoch_count=100, lr_scheduler_spec=f"cosine_anneal:eta_min={eta_min},T_max={min_epoch}")
        config = TrainConfig(epoch_count=100, lr_scheduler_spec=f"cosine_anneal:eta_min={eta_min}")
        optimizer = _test_optimizer(config.learning_rate)
        scheduler = config.create_lr_scheduler(optimizer)
        self._test_demo(scheduler, config)

    def _test_demo(self, scheduler: LRScheduler, config: TrainConfig):
        rates = _run_schedule(scheduler.optimizer, scheduler, epochs=config.epoch_count, verbose=True)
        self.assertEqual(config.learning_rate, rates[0])

    def test_to_dict(self):
        s = json.dumps(TrainConfig().to_dict())
        self.assertIsInstance(s, str)


class DatasetTest(TestCase):

    def test_truncate(self):
        dataset = Dataset.acquire(batch_size_train=25, batch_size_test=10, truncate_train=200, truncate_test=100, quiet=True)
        self.assertEqual(8, len(dataset.trainloader))
        self.assertEqual(10, len(dataset.testloader))


class ModuleMethodsTest(TestCase):

    def test_perform(self):
        self._test_perform(seed=12345)

    def test_perform_anneal(self):
        self._test_perform(seed=23456, lr_scheduler_spec="cosine_anneal:eta_min=0.0001;T_max=100")

    def _test_perform(self, seed: int, **config_kwargs):
        with torch.random.fork_rng():
            dataset = Dataset.acquire(batch_size_train=10, batch_size_test=10, truncate_train=100, truncate_test=100, quiet=True)
            modeler = lambda: CustomResNet([
                BlockSpec(2, 64, stride=1),
                BlockSpec(5, 128, stride=2),
                BlockSpec(3, 256, stride=2),
            ])
            with tempfile.TemporaryDirectory() as tempdir:
                checkpoint_file = os.path.join(tempdir, "ckpt.pth")
                all_config_kwargs = {
                    "epoch_count": 2,
                    "checkpoint_file": checkpoint_file,
                    "seed": seed,
                    "quiet": True,
                }
                all_config_kwargs.update(config_kwargs)
                config = TrainConfig(**all_config_kwargs)
                result = perform(modeler, dataset, config=config)
                print("train", result.train_history)
                print(" test", result.test_history)
                self.assertTrue(result.checkpoint_file.is_file(), f"expect checkpoint file exists {result.checkpoint_file}")
