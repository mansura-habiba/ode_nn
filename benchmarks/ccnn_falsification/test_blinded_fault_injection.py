"""Blinded model fault tests: fixed targets and private oracle separation."""
import tempfile
import unittest
from pathlib import Path

import torch

from blinded_fault_injection import (
    FAULT_SURFACES, inject_model_fault, state_digest,
    save_separate_fault_artifacts, load_repair_input, classify_fault_headroom,
)


def checkpoint():
    return {name: torch.tensor([float(i + 1)]) for i, name in enumerate(FAULT_SURFACES.values())} | {
        "att.embed.weight": torch.randn(2, 2)
    }


class FaultInjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(45)
        self.state = checkpoint()

    def test_exactly_one_weight_is_modified(self):
        original = {k: v.clone() for k, v in self.state.items()}
        for family, parameter in FAULT_SURFACES.items():
            public, oracle = inject_model_fault(original, family=family,
                                                magnitude=0.12, sample_id="seed11")
            self.assertEqual(oracle.modified_parameter, parameter)
            self.assertNotEqual(public.model_sha256, oracle.clean_sha256)
            for key, value in original.items():
                if key == parameter:
                    self.assertFalse(torch.equal(value, public.faulted_state[key]))
                else:
                    self.assertTrue(torch.equal(value, public.faulted_state[key]))
                self.assertTrue(torch.equal(value, original[key]))

    def test_zero_fault_is_perfect_placebo(self):
        public, oracle = inject_model_fault(self.state, family="delay_center",
                                            magnitude=0.0, sample_id="placebo")
        self.assertEqual(oracle.clean_sha256, public.model_sha256)

    def test_public_package_never_discloses_fault_identity_or_clean_weights(self):
        public, oracle = inject_model_fault(self.state, family="frequency",
                                            magnitude=0.25, sample_id="trial15")
        with tempfile.TemporaryDirectory() as d:
            repair, private = save_separate_fault_artifacts(
                public, oracle,
                repair_dir=Path(d)/"repair_candidate",
                evaluator_dir=Path(d)/"private_evaluator")
            restored = load_repair_input(repair)
            self.assertEqual(restored.model_sha256, public.model_sha256)
            self.assertEqual(restored.sample_id, "trial15")
            self.assertNotIn("frequency", str(restored))
            self.assertNotIn("modified_parameter", str(restored))
            self.assertIn("frequency", private.read_text())
            self.assertNotIn("frequency", repair.name)

    def test_replay_and_validation_rejects_invalid_state(self):
        a, _ = inject_model_fault(self.state, family="chirp", magnitude=0.07, sample_id="a")
        b, _ = inject_model_fault(self.state, family="chirp", magnitude=0.07, sample_id="b")
        self.assertEqual(state_digest(a.faulted_state), state_digest(b.faulted_state))
        with self.assertRaises(ValueError):
            inject_model_fault(self.state, family="unknown", magnitude=0.5, sample_id="x")

    def test_headroom_is_calibration_only(self):
        self.assertEqual(classify_fault_headroom(0.02, 0.021), "fault_not_detectable")
        self.assertEqual(classify_fault_headroom(0.02, 0.035), "fault_detectable")
        with self.assertRaises(ValueError):
            classify_fault_headroom(float("nan"), 0.02)


if __name__ == "__main__":
    unittest.main()
