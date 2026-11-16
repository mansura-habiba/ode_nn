# Blinded model-weight fault injector — RQ6-M Issue #10

Use this only for **fixed-target, model-weight corruption**. The clean
checkpoint produces the same true target labels as faulted and repaired
checkpoints. Do NOT change teacher labels as the primary fault mechanism.

API:

```python
from blinded_fault_injection import inject_model_fault, save_separate_fault_artifacts
public, oracle = inject_model_fault(clean.state_dict(), family="query_delay_gain",
                                    magnitude=0.12, sample_id="seed101_fault1")
repair_file, evaluator_file = save_separate_fault_artifacts(
    public, oracle, repair_dir="repair_input",
    evaluator_dir="private_evaluator")
# Give ONLY repair_file to the repair algorithm; the oracle never crosses that interface.
```

**Privacy is procedural, not cryptographic**: the evaluator metadata must
actually be excluded from repair-worker mounts, training samples and prompts.
Separate directories in one process are *not* a security boundary.

This module supplies deterministic corruption and evaluator-only metadata,
with a calibration helper to label insufficient fault headroom. **Not yet
implemented:** hard sandbox isolation, automatic fault amplitude calibration,
source/affected cohorts, no-fault continued Adam, true RQ6 recovery selection,
multi-coordinate/matched-norm random faults, or validation/test evaluator.
Those are tracked under #9, #11, #12 and #13; do not claim the whole issue
completed until those acceptance gates pass.
