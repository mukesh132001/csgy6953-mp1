import unittest
from unittest import TestCase
import base64
import numpy as np
import torch.random
import dlmp1.utils

class UtilsTest(TestCase):

    def test_print_random_state(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(1234)
            state = torch.random.get_rng_state()
            def _get_random_tensor():
                return torch.randint(0, 1_000_000, (10,))
            first_tensor_ref = _get_random_tensor()
            print(state.shape)
            print(state)
            np_state: np.ndarray = state.clone().detach().cpu().numpy()
            print(np_state.shape)
            print(np_state.dtype)
            np_state_bytes = np_state.tobytes()
            encoded = base64.b64encode(np_state_bytes).decode('us-ascii')
            self.assertIsInstance(encoded, str)
            decoded = base64.b64decode(encoded.encode('us-ascii'))
            decoded_np = np.frombuffer(decoded, dtype=np.uint8)
            deserialized = torch.from_numpy(np.copy(decoded_np))
            self.assertTrue(torch.equal(state, deserialized))
            torch.random.set_rng_state(deserialized)
            first_tensor_que = _get_random_tensor()
            self.assertTrue(torch.equal(first_tensor_ref, first_tensor_que))

    @unittest.skip("just a demo")
    def test_get_mean_and_std(self):
        from torchvision import transforms
        import torchvision.datasets
        directory = str(dlmp1.utils.get_repo_root() / "data")
        testset = torchvision.datasets.CIFAR10(root=directory, train=True, download=True, transform=transforms.ToTensor())
        mean, std = dlmp1.utils.get_mean_and_std(testset)
        print("mean", mean)
        print("std", std)