import unittest

import torch

from core.models import (
    Chebyshev,
    default_SOnPathDistributionEncoder,
)
from core.sde_solvers import geometric_euler
from core.pathdistribution import SOnPathDistribution
from core.power_spherical import PowerSpherical
from core.pathdistribution import (
    BrownianMotionOnSphere,
    SOnPathDistribution,
)
from torch.distributions import kl_divergence


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

class TestGeometricEulerBatchedTime(unittest.TestCase):
    def test_batched_time_steps_match_individual_integrations(self):
        z0 = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        drift = torch.tensor([
            [[0.4], [0.3], [0.2]],
            [[-0.2], [0.1], [0.5]],
        ])

        dt = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.1, 0.2],
        ])

        basis = torch.tensor([
            [[0.0, -1.0],
             [1.0,  0.0]],
        ])

        zero_diffusion = torch.tensor(0.0)

        expected = torch.cat([
            geometric_euler(
                z0[i:i + 1],
                drift[i:i + 1],
                zero_diffusion,
                dt[i],
                basis,
            )
            for i in range(z0.shape[0])
        ])

        result = geometric_euler(
            z0,
            drift,
            zero_diffusion,
            dt,
            basis,
        )

        self.assertEqual(result.shape, (2, 4, 2))
        torch.testing.assert_close(result, expected)

    def test_batched_time_steps_support_sample_dimensions(self):
        z0 = torch.tensor([
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            [
                [0.0, -1.0],
                [-1.0, 0.0],
            ],
            [
                [2.0**-0.5, 2.0**-0.5],
                [-2.0**-0.5, 2.0**-0.5],
            ],
        ])

        drift = torch.tensor([
            [[0.4], [0.3], [0.2]],
            [[-0.2], [0.1], [0.5]],
        ])

        dt = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.1, 0.2],
        ])

        basis = torch.tensor([
            [[0.0, -1.0],
            [1.0,  0.0]],
        ])

        zero_diffusion = torch.tensor(0.0)

        expected = torch.stack([
            geometric_euler(
                sample_z0,
                drift,
                zero_diffusion,
                dt,
                basis,
            )
            for sample_z0 in z0
        ])

        result = geometric_euler(
            z0,
            drift,
            zero_diffusion,
            dt,
            basis,
        )

        self.assertEqual(result.shape, (3, 2, 4, 2))
        torch.testing.assert_close(result, expected)

class TestSOnPathDistributionBatchedTime(unittest.TestCase):
    def setUp(self):
        location = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        scale = torch.tensor([10.0, 10.0])
        p0 = PowerSpherical(location, scale)

        self.t = torch.tensor([
            [0.0, 0.1, 0.2, 0.4],
            [0.2, 0.4, 0.7, 1.0],
        ])

        group_dim = 3

        def K(time_steps):
            return torch.zeros(
                *time_steps.shape,
                group_dim,
                device=time_steps.device,
                dtype=time_steps.dtype,
            )

        self.dist = SOnPathDistribution(
            p0,
            K,
            torch.tensor([0.1]),
            self.t,
        )

    def test_batched_time_metadata(self):
        self.assertEqual(self.dist.batch_shape, torch.Size([2]))
        self.assertEqual(self.dist.event_shape, torch.Size([4, 3]))
        self.assertEqual(self.dist.steps, 4)
        self.assertEqual(self.dist.dt.shape, (2, 3))
        self.assertEqual(self.dist.Kt.shape, (2, 3, 3))

    def test_batched_time_sampling(self):
        torch.manual_seed(0)

        sample = self.dist.rsample((3,))

        self.assertEqual(sample.shape, (3, 2, 4, 3))

        norms = sample.norm(dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_batched_brownian_prior_drift_shape(self):
        prior = BrownianMotionOnSphere(
            dim=3,
            sigma=torch.tensor([0.1]),
            t=self.t,
        )

        self.assertEqual(prior.dt.shape, (2, 3))
        self.assertEqual(prior.Kt.shape, (2, 3, 3))

    def test_batched_time_kl_is_finite_per_batch_element(self):
        prior = BrownianMotionOnSphere(
            dim=3,
            sigma=torch.tensor([0.1]),
            t=self.t,
        )

        torch.manual_seed(0)
        self.dist.rsample((3,))

        kl_path, kl_initial = kl_divergence(self.dist, prior)

        self.assertEqual(kl_path.shape, (2,))
        self.assertEqual(kl_initial.shape, (2,))
        self.assertTrue(torch.isfinite(kl_path).all())
        self.assertTrue(torch.isfinite(kl_initial).all())

    def test_shared_time_grid_remains_supported(self):
        location = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        scale = torch.tensor([10.0, 10.0])
        p0 = PowerSpherical(location, scale)

        t = torch.tensor([0.0, 0.2, 0.6, 1.0])

        def K(time_steps):
            return torch.zeros(
                2,
                time_steps.shape[-1],
                3,
                device=time_steps.device,
                dtype=time_steps.dtype,
            )

        dist = SOnPathDistribution(
            p0,
            K,
            torch.tensor([0.1]),
            t,
        )

        torch.manual_seed(0)
        sample = dist.rsample()

        self.assertEqual(dist.batch_shape, torch.Size([2]))
        self.assertEqual(dist.event_shape, torch.Size([4, 3]))
        self.assertEqual(dist.steps, 4)
        self.assertEqual(dist.dt.shape, (3,))
        self.assertEqual(dist.Kt.shape, (2, 3, 3))
        self.assertEqual(sample.shape, (2, 4, 3))

class TestSOnPathDistributionEncoderBatchedTime(unittest.TestCase):
    def test_encoder_supports_different_time_ranges(self):
        torch.manual_seed(0)

        encoder = default_SOnPathDistributionEncoder(
            h_dim=4,
            z_dim=3,
            n_deg=4,
            learnable_prior=False,
        )

        h = torch.randn(2, 4, requires_grad=True)
        t = torch.tensor([
            [0.0, 0.1, 0.2, 0.4, 0.6],
            [0.2, 0.4, 0.6, 0.8, 1.0],
        ])

        posterior, prior = encoder(h, t)
        sample = posterior.rsample((3,))
        kl_path, kl_initial = kl_divergence(posterior, prior)

        self.assertEqual(posterior.batch_shape, torch.Size([2]))
        self.assertEqual(posterior.event_shape, torch.Size([5, 3]))
        self.assertEqual(posterior.Kt.shape, (2, 4, 3))
        self.assertEqual(prior.Kt.shape, (2, 4, 3))
        self.assertEqual(sample.shape, (3, 2, 5, 3))
        self.assertEqual(kl_path.shape, (2,))
        self.assertEqual(kl_initial.shape, (2,))

        self.assertTrue(torch.isfinite(sample).all())
        self.assertTrue(torch.isfinite(kl_path).all())
        self.assertTrue(torch.isfinite(kl_initial).all())

        loss = (
            sample[..., 0].sum()
            + kl_path.sum()
            + kl_initial.sum()
        )
        loss.backward()

        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(h.grad).all())

        self.assertIsNotNone(encoder._time_fn.map.weight.grad)
        self.assertTrue(
            torch.isfinite(encoder._time_fn.map.weight.grad).all()
        )

if __name__ == "__main__":
    unittest.main()