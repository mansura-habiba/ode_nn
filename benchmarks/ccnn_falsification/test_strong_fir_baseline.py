"""Tests for an expressive query-conditioned FIR baseline (not publication evidence)."""
import unittest
import torch
from torch import nn

from strong_fir_baseline import (
    QueryMLPFIR, AttentionPlusMLPFIR, causal_history,
    freeze_attention, parameter_summary,
)


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x, q):
        return self.linear(torch.stack((x, q), -1)).mean(1).squeeze(-1)


class StrongFIRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(35)
        self.x = torch.randn(5, 32)
        self.q = torch.linspace(-0.8, 0.8, 32).expand(5, -1)

    def test_kernel_is_causal_and_position_conditioned(self):
        module = QueryMLPFIR(lags=14, hidden=16)
        k = module.weights(self.q)
        self.assertEqual(k.shape, (5, 32, 14))
        self.assertTrue(torch.isfinite(k).all().item())
        self.assertGreater((k[:, -1] - k[:, 0]).abs().max().item(), 1e-4)
        history = causal_history(self.x, 14)
        self.assertEqual(history.shape, (5, 32, 14))
        self.assertTrue(torch.equal(history[:, 0, 1:], torch.zeros_like(history[:, 0, 1:])))

    def test_query_conditioning_is_nonconstant(self):
        net = QueryMLPFIR(hidden=16)
        k1 = net.weights(self.q)
        k2 = net.weights(-self.q)
        self.assertGreater((k1-k2).abs().mean().item(), 1e-6)

    def test_no_future_signal_leakage(self):
        model = QueryMLPFIR(lags=14, hidden=12)
        h1 = causal_history(self.x, 14)
        corrupted = self.x.clone()
        corrupted[:, 22:] += 100
        h2 = causal_history(corrupted, 14)
        self.assertTrue(torch.equal(h1[:, :22], h2[:, :22]))

    def test_frozen_attention_and_learnable_residual(self):
        network = AttentionPlusMLPFIR(ToyAttention(), hidden=12)
        freeze_attention(network)
        self.assertEqual(sum(p.requires_grad for p in network.att.parameters()), 0)
        loss = network(self.x, self.q).square().mean()
        loss.backward()
        self.assertIsNotNone(network.branch.gain.grad)
        self.assertTrue(torch.isfinite(network.branch.gain.grad).item())
        self.assertGreater(parameter_summary(network)["total_parameters"], 0)

    def test_no_timestamp_hidden_advantage(self):
        network = AttentionPlusMLPFIR(ToyAttention(), hidden=12)
        t = torch.arange(32).float().view(1, -1).expand(5, -1) / 32
        a = network(self.x, self.q)
        b = network(self.x, self.q, t)
        self.assertTrue(torch.allclose(a, b, atol=1e-7, rtol=1e-7))


if __name__ == "__main__":
    unittest.main()
