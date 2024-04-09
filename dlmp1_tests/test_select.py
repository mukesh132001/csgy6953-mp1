from unittest import TestCase

import torch.optim
import torch.optim.lr_scheduler
import torch.random
import torchsummary

import dlmp1.select
import dlmp1.train
from dlmp1.train import TrainConfig


class ModuleMethodsTest(TestCase):

    def test_iterate_model_factories(self):
        factories = dlmp1.select.iterate_model_factories([
            [2, 1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [3, 4, 3],
            [2, 5, 3],
            [3, 5, 3],
            [2, 4, 3],
            [2, 5, 2],
            [2, 4, 2],
            # [2, 6, 3],  # too big
            # [2, 5, 4],  # too big
        ])
        for factory in factories:
            with self.subTest():
                model = factory()
                stats = torchsummary.summary(model, verbose=0)
                self.assertLessEqual(stats.trainable_params, 5_000_000, msg=model.summary_text)
                model.eval()
                with torch.random.fork_rng():
                    torch.random.manual_seed(498682)
                    x = torch.randn(1, 3, 32, 32)
                    model(x)

    # noinspection PyMethodMayBeStatic
    def test_confirm_train_config_valid_ok(self):
        dlmp1.select.confirm_train_config_valid(TrainConfig())
        dlmp1.select.confirm_train_config_valid(TrainConfig(lr_scheduler_spec="cosine_anneal:eta_min=0.001;T_max=200"))

    def test_confirm_train_config_valid_bad(self):
        for train_config in [
            TrainConfig(epoch_count=0),
            TrainConfig(lr_scheduler_spec="cosine_anneal:eta_min=0.001,T_max=200"),  # typo ',' instead of ';'
        ]:
            with self.subTest():
                with self.assertRaises(Exception) as cm:
                    dlmp1.select.confirm_train_config_valid(train_config)
