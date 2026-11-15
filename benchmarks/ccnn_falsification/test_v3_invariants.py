"""Fast mathematical invariants for the CCNN V3 falsification protocol.

Run: python -m unittest discover -s benchmarks/ccnn_falsification -p 'test_*.py'
These are invariant checks, NOT empirical evidence of predictive superiority.
"""
import unittest
import torch

from ccnn_generalization_benchmark import (
    Model, UngatedAdapter, N, L, FAMILIES,
    kernel, teacher_k, ungated_predict,
)

ARMS = (
    ("structured", 1), ("tangent", 1), ("spectral", 1),
    ("structured", 5), ("tangent", 5), ("spectral", 5),
    ("structured", 18), ("spectral", 18), ("rank1", 18),
    ("structured", 36), ("spectral", 36), ("rank2", 36),
    ("structured", 56), ("spectral", 56), ("full", 56),
)


class V3ProtocolInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        cls.base = Model("lag").eval()
        for p in cls.base.parameters():
            p.requires_grad_(False)
        cls.x = torch.randn(3, N)
        cls.q = torch.linspace(-0.8, 0.8, N)[None, :].expand(3, -1)
        cls.y0 = ungated_predict(cls.base, cls.x, cls.q).detach()

    def test_every_adapter_has_exact_trainable_budget_and_identity(self):
        for method, budget in ARMS:
            with self.subTest(method=method, budget=budget):
                adapter = UngatedAdapter(self.base, method, budget)
                free = sum(p.numel() for p in adapter.parameters()
                           if p.requires_grad)
                self.assertEqual(free, budget)
                with torch.no_grad():
                    y = ungated_predict(self.base, self.x, self.q, adapter)
                self.assertTrue(torch.isfinite(y).all().item())
                self.assertTrue(torch.allclose(y, self.y0, rtol=5e-5, atol=5e-6))

    def test_tangent_is_first_order_equivalent_to_structured_shift(self):
        center = UngatedAdapter(self.base, "structured", 1)
        tangent = UngatedAdapter(self.base, "tangent", 1)
        with torch.no_grad():
            center.edit[0] = 1e-3
            tangent.edit[0] = 1e-3
            k1 = center.corrected_kernel(self.q)
            k2 = tangent.corrected_kernel(self.q)
        self.assertLess((k1 - k2).abs().max().item(), 5e-4)

    def test_trainable_tangent_gradient(self):
        adapter = UngatedAdapter(self.base, "tangent", 1)
        y = ungated_predict(self.base, self.x, self.q, adapter).square().mean()
        y.backward()
        self.assertIsNotNone(adapter.edit.grad)
        self.assertTrue(torch.isfinite(adapter.edit.grad).all().item())

    def test_unmatched_teachers_are_not_chirp_family(self):
        with torch.no_grad():
            matched = teacher_k(self.q, "matched_chirp")
            for family in FAMILIES[1:]:
                with self.subTest(family=family):
                    other = teacher_k(self.q, family)
                    self.assertEqual(tuple(other.shape), (3, N, L))
                    self.assertGreater((matched - other).abs().mean().item(), 0.02)

    def test_prediction_has_no_domain_identity(self):
        with torch.no_grad():
            adapter = UngatedAdapter(self.base, "structured", 1)
            adapter.edit[0] = 0.04
            y1 = ungated_predict(self.base, self.x, self.q, adapter)
            y2 = ungated_predict(self.base, self.x, self.q, adapter)
        self.assertTrue(torch.equal(y1, y2))


if __name__ == "__main__":
    unittest.main()
