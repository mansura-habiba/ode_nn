# Independent teacher mechanisms — RQ1-M task #6

These five families are implemented in `independent_teachers.py` using explicit,
non-CCNN functions: nonlinear discrete delays, sharp boxcar history windows, a
driven two-state oscillator, hard switching, and nonlinear feedback. They do
**not** use `ccnn_generalization_benchmark.py`, its lag kernel or its shared
leaky ODE when creating targets. Matched chirp remains an *independent positive
calibration control* in the historic V3 experiment and must not be included in
the primary unmatched-family aggregate.

Usage (from this directory):

```python
from independent_teachers import make_split, assert_disjoint_splits
train = make_split("bimodal", "train", seed=101)
valid = make_split("bimodal", "validation", seed=101)
test = make_split("bimodal", "locked_test", seed=101)
assert_disjoint_splits(train, valid, test)
```

**Current limitations:** still synthetic, fixed observation intervals; not
yet wired to equal-capacity architectures, training curves or external
dataset. These are *independent mechanisms*, not proof of out-of-domain
generalization. Random process inputs use distinct seed offsets by split; the
sample IDs and content hashes allow independent replay.

**Next integration:** replace the V3 target-generation path only in RQ1
experimental runs; do not repurpose it as the primary RQ6 fault generator.
Record teacher family, seed, split hashes and data code SHA in the run manifest.
