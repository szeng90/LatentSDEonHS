import unittest

import torch

from core.models import Chebyshev


class TestChebyshevBatchedTime(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.time_fn = Chebyshev(
            input_dim=3,
            degree=4,
            out_dim=2,
        )
        self.h = torch.randn(2, 3)

    def test_shared_time_grid_shape(self):
        t = torch.tensor([0.0, 0.5, 1.0])

        result = self.time_fn(self.h, t)

        self.assertEqual(result.shape, (2, 3, 2))

    def test_batched_time_grids_match_individual_evaluation(self):
        t = torch.tensor([
            [0.0, 0.25, 0.5],
            [0.2, 0.6, 1.0],
        ])

        expected = torch.cat([
            self.time_fn(self.h[i:i + 1], t[i])
            for i in range(self.h.shape[0])
        ])

        result = self.time_fn(self.h, t)

        self.assertEqual(result.shape, (2, 3, 2))
        torch.testing.assert_close(result, expected)

    def test_batched_time_grids_preserve_gradients(self):
        h = self.h.clone().requires_grad_()
        t = torch.tensor([
            [0.0, 0.25, 0.5],
            [0.2, 0.6, 1.0],
        ])

        result = self.time_fn(h, t)
        result.square().sum().backward()

        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(h.grad).all())

        self.assertIsNotNone(self.time_fn.map.weight.grad)
        self.assertTrue(torch.isfinite(self.time_fn.map.weight.grad).all())


if __name__ == "__main__":
    unittest.main()