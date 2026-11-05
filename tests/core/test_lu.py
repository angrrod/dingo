"""CPU parity checks; CUDA checks run when invoked in a GPU allocation.

Run from the repository root:
    python External/dingo/tests/core/test_lu.py
"""

import unittest

import torch
from glasflow.nflows import transforms
from glasflow.nflows.transforms.lu import LULinear

from dingo.core.nn.lu import DeviceLULinear
from dingo.core.nn.nsf import create_linear_transform


class TestDeviceLULinear(unittest.TestCase):
    def check_parity(self, device):
        for features in (1, 15):
            for dtype in (torch.float32, torch.float64):
                for inverse in (False, True):
                    with self.subTest(features=features, dtype=dtype, inverse=inverse):
                        # Nonidentity factors exercise lower/upper entries and diagonal.
                        torch.manual_seed(17)
                        original = LULinear(features, identity_init=False).to(device, dtype)
                        optimized = DeviceLULinear(features).to(device, dtype)
                        optimized.load_state_dict(original.state_dict(), strict=True)
                        x = torch.randn(7, features, device=device, dtype=dtype,
                                        requires_grad=True)
                        y = x.detach().clone().requires_grad_()
                        fn = "inverse" if inverse else "forward"
                        expected = getattr(original, fn)(x)
                        actual = getattr(optimized, fn)(y)
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b)
                        sum(t.square().sum() for t in expected).backward()
                        sum(t.square().sum() for t in actual).backward()
                        torch.testing.assert_close(y.grad, x.grad)
                        for (name, a), (_, b) in zip(
                            optimized.named_parameters(), original.named_parameters()
                        ):
                            if a.grad is None or b.grad is None:
                                self.assertIs(a.grad, b.grad, name)
                            else:
                                torch.testing.assert_close(a.grad, b.grad, msg=name)
                        torch.testing.assert_close(optimized.weight(), original.weight())
                        torch.testing.assert_close(
                            optimized.weight_inverse(), original.weight_inverse())
                        for name, indices in optimized.named_buffers():
                            self.assertEqual(indices.dtype, torch.long, name)
                            self.assertEqual(indices.device, optimized.bias.device, name)

    def test_cpu_outputs_and_gradients(self):
        self.check_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_outputs_and_gradients(self):
        self.check_parity("cuda")

    def test_factory_and_checkpoint_compatibility(self):
        original = transforms.CompositeTransform([
            transforms.RandomPermutation(features=15), LULinear(15)])
        optimized = create_linear_transform(15)
        self.assertIsInstance(optimized._transforms[1], DeviceLULinear)
        # Strict loads both ways: derived buffers must not add checkpoint keys.
        self.assertEqual(set(original.state_dict()), set(optimized.state_dict()))
        optimized.load_state_dict(original.state_dict(), strict=True)
        original.load_state_dict(optimized.state_dict(), strict=True)
        x = torch.randn(7, 15)
        for a, b in zip(optimized(x), original(x)):
            torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
