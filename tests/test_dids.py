"""Minimal shape and gate tests for hierarchical Avg/Max DIDS."""

import unittest

import torch

from model.operators import DIDS


class DIDSTest(unittest.TestCase):
    def _check_shape(self, input_shape, out_channels, expected_shape):
        module = DIDS(input_shape[1], out_channels, reduction=4)
        inputs = torch.randn(*input_shape)
        outputs, gates = module(inputs, return_gate_weights=True)

        self.assertEqual(tuple(outputs.shape), expected_shape)
        self.assertEqual(set(gates), {"alpha", "beta", "gamma"})
        for name in ("alpha", "beta"):
            self.assertEqual(tuple(gates[name].shape), (input_shape[0], 3))
            self.assertTrue(torch.allclose(gates[name].sum(dim=1), torch.ones(input_shape[0]), atol=1e-6))
        self.assertEqual(tuple(gates["gamma"].shape), (input_shape[0], 2))
        self.assertTrue(torch.allclose(gates["gamma"].sum(dim=1), torch.ones(input_shape[0]), atol=1e-6))
        self.assertEqual(tuple(module.last_alpha_mean.shape), (3,))
        self.assertEqual(tuple(module.last_beta_mean.shape), (3,))
        self.assertEqual(tuple(module.last_gamma_mean.shape), (2,))
        self.assertEqual(set(module.last_gate_means), {"alpha", "beta", "gamma"})
        self.assertEqual(module.avg_scale_gate[0].in_features, 6 * input_shape[1])
        self.assertEqual(module.max_scale_gate[0].in_features, 6 * input_shape[1])
        self.assertEqual(module.type_gate[0].in_features, 2 * input_shape[1])

        features = [
            module.avg_pool_2(inputs), module.avg_pool_3(inputs), module.avg_pool_4(inputs),
            module.max_pool_2(inputs), module.max_pool_3(inputs), module.max_pool_4(inputs),
        ]
        self.assertTrue(all(tuple(feature.shape) == tuple(features[0].shape) for feature in features[1:]))

    def test_even_stage_shapes(self):
        self._check_shape((2, 32, 28, 28), 64, (2, 64, 14, 14))
        self._check_shape((2, 32, 56, 56), 64, (2, 64, 28, 28))
        self._check_shape((2, 64, 14, 14), 128, (2, 128, 7, 7))

    def test_final_odd_stage_shape_is_explicitly_aligned(self):
        self._check_shape((2, 128, 7, 7), 256, (2, 256, 4, 4))

    def test_crc5000_native_resolution_through_three_downsamplers(self):
        """150x150 CRC-5000 tiles retain aligned branches at every stage."""
        x = torch.randn(2, 32, 150, 150)
        stage_1 = DIDS(32, 64)(x)
        stage_2 = DIDS(64, 128)(stage_1)
        stage_3 = DIDS(128, 256)(stage_2)
        self.assertEqual(tuple(stage_1.shape), (2, 64, 75, 75))
        self.assertEqual(tuple(stage_2.shape), (2, 128, 38, 38))
        self.assertEqual(tuple(stage_3.shape), (2, 256, 19, 19))


if __name__ == "__main__":
    unittest.main()
