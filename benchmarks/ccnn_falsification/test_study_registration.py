"""Checks that protect the preregistered study from silent protocol drift."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class ProtocolIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = json.loads((ROOT / "protocol.yaml").read_text())

    def test_no_confirmatory_label_while_draft(self):
        self.assertIn(self.p["registration_status"], ("DRAFT_NOT_FROZEN", "FROZEN_PRE_RUN"))
        self.assertEqual(self.p["evidence_levels"][0], "proposed")

    def test_unmatched_primary_excludes_matched_teacher(self):
        a = self.p["tasks"]["RQ1-M"]
        self.assertFalse(set(a["primary_families"]) & set(a["positive_calibration_only"]))
        self.assertIn("mlp_dynamic_fir", a["required_controls"])

    def test_repair_cannot_be_teacher_label_shift(self):
        b = self.p["tasks"]["RQ6-M"]
        self.assertTrue(b["target_labels_fixed"])
        self.assertEqual(b["corrupted_object"], "model_parameter")
        self.assertTrue(b["no_domain_gate"])
        self.assertIn("no_fault", b["negative_controls"])
        self.assertEqual(b["fault_oracle_visibility"].split(";")[0], "evaluator_only")

    def test_blind_test_and_independent_seeds(self):
        p = self.p
        ids = p["confirmatory"]["paired_seed_ids"]
        self.assertGreaterEqual(len(ids), 10)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(p["sample_splits"]["locked_test_usage"], "evaluation_only_no_checkpoint_fault_amplitude_hyperparameter_or_edit_selection")

    def test_repair_and_preservation_budgets_predeclared(self):
        b = self.p["tasks"]["RQ6-M"]
        self.assertGreaterEqual(len(b["required_controls"]), 8)
        self.assertLessEqual(b["maximum_false_positive_no_fault_rate"], 0.05)
        self.assertEqual(b["allowable_updated_parameter_budgets"], [1,5,18,36,56])


if __name__ == "__main__":
    unittest.main()
