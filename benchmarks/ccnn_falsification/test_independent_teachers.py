"""Isolated RQ1 teachers: numerical, family and split invariants."""
import unittest
import torch

from independent_teachers import (
    FAMILIES, make_split, teacher_target, sample_inputs,
    split_digest, assert_disjoint_splits,
)


class IndependentTeacherTests(unittest.TestCase):
    def test_shapes_reproducibility_and_finite(self):
        for family in FAMILIES:
            a = make_split(family, "train", 11, count=10)
            b = make_split(family, "train", 11, count=10)
            self.assertEqual(split_digest(a), split_digest(b))
            self.assertEqual(a.y.shape, (10,))
            self.assertTrue(torch.isfinite(a.y).all().item())

    def test_splits_are_independent(self):
        a = make_split("bimodal", "train", 11, 12)
        b = make_split("bimodal", "validation", 11, 12)
        c = make_split("bimodal", "locked_test", 11, 12)
        assert_disjoint_splits(a, b, c)
        self.assertFalse(torch.equal(a.x, c.x))
        with self.assertRaises(AssertionError):
            assert_disjoint_splits(a, b, a)

    def test_distinct_independent_mechanisms(self):
        x, q = sample_inputs(6, 42)
        outputs = [teacher_target(x, q, family) for family in FAMILIES]
        for a in range(len(outputs)):
            for b in range(a + 1, len(outputs)):
                self.assertGreater((outputs[a] - outputs[b]).abs().max().item(), 1e-4)

    def test_no_ccnn_generator_dependency(self):
        import inspect
        import independent_teachers as teachers
        source = inspect.getsource(teachers)
        self.assertNotIn("from ccnn_generalization_benchmark import", source)
        self.assertNotIn("import ccnn_generalization_benchmark", source)
        self.assertNotIn("integrate_drive(", source)
        self.assertNotIn("kernel(", source)

    def test_invalid_inputs_rejected(self):
        x, q = sample_inputs(3, 8)
        with self.assertRaises(ValueError):
            teacher_target(x, q[:, 1:], "bimodal")
        with self.assertRaises(ValueError):
            make_split("matched_chirp", "train", 4)


if __name__ == "__main__":
    unittest.main()
