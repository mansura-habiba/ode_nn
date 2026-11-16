"""Check checkpoint selection and fault amplitude without touching test samples."""
import unittest
import torch
from torch import nn
from checkpoint_readiness import train_checkpoint, calibrate_on_disjoint_population


class TinySequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w=nn.Parameter(torch.tensor(.2))
    def forward(self,x,q):
        return self.w*x.mean(1)


class ReadinessTests(unittest.TestCase):
    def test_training_uses_validation_only_and_records_provenance(self):
        torch.manual_seed(1)
        x=torch.randn(32,8)
        q=torch.zeros_like(x)
        y=1.7*x.mean(1)
        out=train_checkpoint(TinySequenceModel(),(x,q,y),(x[:8],q[:8],y[:8]),
            seed=7,max_epochs=12,minimum_epochs=4,patience=3,batch_size=8,lr=.1)
        self.assertGreater(out.total_optimizer_steps,0)
        self.assertEqual(out.manifest()["checkpoint_selection_data"],"validation_only")
        self.assertFalse(out.manifest()["test_used"])
        self.assertEqual(out.checkpoint_hash,len(out.checkpoint_hash)*"0") if False else self.assertEqual(len(out.checkpoint_hash),64)
        self.assertGreater(out.chosen_epoch,0)
        self.assertEqual(out.model.training,False)

    def test_short_run_marked_not_established(self):
        x=torch.ones(6,4);q=torch.zeros_like(x);y=torch.ones(6)
        out=train_checkpoint(TinySequenceModel(),(x,q,y),(x,q,y),seed=9,
             max_epochs=2,minimum_epochs=2,patience=4)
        self.assertEqual(out.readiness,"checkpoint_not_established")

    def test_calibration_does_not_consult_test_set(self):
        called=[]
        def evaluate(m):
            called.append(m)
            return 0.002+m*m
        result=calibrate_on_disjoint_population(clean_validation_mse=.002,
            candidate_magnitudes=[.01,.02,.05],calibration_loss_for_magnitude=evaluate)
        self.assertEqual(result.status,"calibrated")
        self.assertEqual(len(called),3)
        self.assertEqual(result.selected_magnitude,.05)

    def test_undetectable_fault_is_explicit(self):
        result=calibrate_on_disjoint_population(clean_validation_mse=.02,
            candidate_magnitudes=[.01],calibration_loss_for_magnitude=lambda x:.0201)
        self.assertIsNone(result.selected_magnitude)
        self.assertEqual(result.status,"fault_not_detectable")

    def test_invalid_budget_fails(self):
        x=torch.ones(4,8);q=x;y=torch.ones(4)
        with self.assertRaises(ValueError):
            train_checkpoint(TinySequenceModel(),(x,q,y),(x,q,y),seed=2,max_epochs=1,
                             minimum_epochs=2)


if __name__=="__main__":
    unittest.main()
