"""RQ6 test evaluation: rejects test leakage and false mechanism attribution."""
import unittest
import torch
from repair_evaluation import TestCohort, select_on_validation, score_locked_test


def cohort(prefix, labels, clean, faulted, repaired, placebo):
    return TestCohort(tuple(f"{prefix}-{i}" for i in range(len(labels))),
        torch.tensor(labels,dtype=torch.float32),torch.tensor(clean,dtype=torch.float32),
        torch.tensor(faulted,dtype=torch.float32),torch.tensor(repaired,dtype=torch.float32),
        torch.tensor(placebo,dtype=torch.float32))


class RepairEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.source=cohort("old",[0.,0.],[0.,0.],[0.05,0.],[0.01,0.],[0.,0.])
        self.new=cohort("affected",[0.,0.],[0.,0.],[0.5,0.5],[0.1,0.1],[0.,0.])
        self.val=select_on_validation(baseline_shift_mse=0.25,candidate_shift_mse=0.01,
            baseline_source_mse=0.001,candidate_source_mse=0.0002)

    def test_detectable_recovery_and_heldout_preservation(self):
        r=score_locked_test(source=self.source,affected=self.new,validation=self.val)
        self.assertTrue(r["accepted_on_validation"])
        self.assertTrue(r["source_preserved_on_test"])
        self.assertEqual(r["status"],"fault_specific_candidate")
        self.assertAlmostEqual(r["recovered_fraction"],0.96,places=5)

    def test_no_fault_does_not_fabricate_recovery_fraction(self):
        nofault=cohort("affected",[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.])
        r=score_locked_test(source=self.source,affected=nofault,validation=self.val)
        self.assertEqual(r["status"],"fault_not_detectable")
        self.assertIsNone(r["recovered_fraction"])

    def test_invalid_selection_reverts_identity(self):
        rejected=select_on_validation(baseline_shift_mse=.25,candidate_shift_mse=.01,
            baseline_source_mse=.001,candidate_source_mse=.10)
        r=score_locked_test(source=self.source,affected=self.new,validation=rejected)
        self.assertFalse(r["accepted_on_validation"])
        self.assertAlmostEqual(r["affected_after"],r["affected_before"])
        self.assertEqual(r["status"],"rejected_by_validation")

    def test_test_ids_must_be_locked_and_distinct(self):
        with self.assertRaises(ValueError):
            score_locked_test(source=self.source,affected=self.new,validation=self.val,
                source_ids_train_validation=["old-0"])
        with self.assertRaises(ValueError):
            score_locked_test(source=self.source,affected=self.source,validation=self.val)

    def test_false_positive_placebo_weakens_specificity(self):
        biased=cohort("affected",[0.,0.],[0.5,0.5],[0.5,0.5],[0.4,0.4],[0.,0.])
        r=score_locked_test(source=self.source,affected=biased,validation=self.val)
        self.assertEqual(r["status"],"fault_not_detectable")
        self.assertIsNone(r["recovered_fraction"])

    def test_validation_is_separate_from_test_criterion(self):
        val=select_on_validation(baseline_shift_mse=.25,candidate_shift_mse=.01,
            baseline_source_mse=.001,candidate_source_mse=.001)
        bad_source=cohort("old",[0.,0.],[0.,0.],[0.05,0.],[1.,1.],[0.,0.])
        result=score_locked_test(source=bad_source,affected=self.new,validation=val)
        self.assertFalse(result["source_preserved_on_test"])
        self.assertEqual(result["status"],"source_regression_on_test")


if __name__ == "__main__":
    unittest.main()
