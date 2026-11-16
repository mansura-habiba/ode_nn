"""Held-out fault vs placebo evaluation for RQ6-M (no test-based edit selection).

This module only SCORES outputs after candidates were selected using validation.
It never chooses or optimizes an edit. A zero/undetectable fault produces
undefined recovery_fraction, not a fabricated zero or divide-by-epsilon score.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Sequence
import math
import torch


@dataclass(frozen=True)
class TestCohort:
    ids: tuple[str, ...]
    labels: torch.Tensor
    clean: torch.Tensor
    faulted: torch.Tensor
    repaired: torch.Tensor
    placebo: torch.Tensor

    def losses(self) -> dict[str, float]:
        if len(set(self.ids)) != len(self.ids) or len(self.ids) == 0:
            raise ValueError("duplicate/empty sample IDs")
        n = len(self.ids)
        for x in (self.labels, self.clean, self.faulted, self.repaired, self.placebo):
            if x.ndim != 1 or x.numel() != n or not torch.isfinite(x).all().item():
                raise ValueError("all predictions/labels must be 1D, finite, and aligned to IDs")
        with torch.no_grad():
            return {
                "clean": float((self.clean-self.labels).square().mean()),
                "faulted": float((self.faulted-self.labels).square().mean()),
                "repaired": float((self.repaired-self.labels).square().mean()),
                "placebo": float((self.placebo-self.labels).square().mean()),
            }


@dataclass(frozen=True)
class ValidationDecision:
    """Immutable decision produced from validation only; pass to test scorer."""
    accepted: bool
    baseline_shift_mse: float
    candidate_shift_mse: float
    baseline_source_mse: float
    candidate_source_mse: float
    preservation_ceiling: float
    reason: str


def select_on_validation(*, baseline_shift_mse: float, candidate_shift_mse: float,
                         baseline_source_mse: float, candidate_source_mse: float,
                         relative_source_limit: float = 0.10,
                         absolute_source_limit: float = 0.0001,
                         minimum_relative_improvement: float = 0.005) -> ValidationDecision:
    values=(baseline_shift_mse,candidate_shift_mse,baseline_source_mse,
            candidate_source_mse,relative_source_limit,absolute_source_limit,
            minimum_relative_improvement)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("nonnegative finite losses/thresholds required")
    ceiling=baseline_source_mse*(1+relative_source_limit)+absolute_source_limit
    preserving=candidate_source_mse<=ceiling
    improving=candidate_shift_mse<baseline_shift_mse*(1-minimum_relative_improvement)
    accepted=preserving and improving
    reason=("accepted" if accepted else
            "source_preservation_failed" if not preserving else
            "insufficient_shift_improvement")
    return ValidationDecision(accepted, baseline_shift_mse,candidate_shift_mse,
                              baseline_source_mse,candidate_source_mse,ceiling,reason)


def score_locked_test(*, source: TestCohort, affected: TestCohort,
                      validation: ValidationDecision,
                      minimum_abs_headroom: float = 0.001,
                      minimum_relative_headroom: float = 0.10,
                      source_relative_limit: float = 0.10,
                      source_absolute_limit: float = 0.0001,
                      source_ids_train_validation: Sequence[str] = ()) -> dict:
    """Evaluation only: NEVER call from a checkpoint/adapter selection loop."""
    if any(v < 0 or not math.isfinite(v) for v in (
       minimum_abs_headroom,minimum_relative_headroom,source_relative_limit,source_absolute_limit)):
        raise ValueError("invalid prespecified thresholds")
    used=set(source_ids_train_validation)
    if used.intersection(source.ids) or used.intersection(affected.ids):
        raise ValueError("locked test overlaps train/validation")
    if set(source.ids).intersection(affected.ids):
        raise ValueError("source and affected test IDs overlap")
    s=source.losses()
    a=affected.losses()
    headroom=a["faulted"]-a["clean"]
    detectable=(headroom>=minimum_abs_headroom and
                headroom>=minimum_relative_headroom*max(a["clean"],1e-12))
    # Rejected edits must be compared with no-change, not the rejected raw proposal.
    if not validation.accepted:
        after=a["faulted"]
        source_after=s["faulted"]
    else:
        after=a["repaired"]
        source_after=s["repaired"]
    improvement=a["faulted"]-after
    placebo_improvement=a["clean"]-a["placebo"]
    specificity_excess=improvement-placebo_improvement
    source_ceiling=s["clean"]*(1+source_relative_limit)+source_absolute_limit
    preserved=source_after<=source_ceiling
    status=("fault_not_detectable" if not detectable else
            "rejected_by_validation" if not validation.accepted else
            "source_regression_on_test" if not preserved else
            "no_fault_specific_gain" if specificity_excess<=0 else
            "fault_specific_candidate") 
    return {
        "status":status, "accepted_on_validation":validation.accepted,
        "validation_reason":validation.reason,
        "fault_headroom":headroom,
        "recovered_fraction":(improvement/headroom if detectable else None),
        "fault_improvement":improvement,
        "placebo_improvement":placebo_improvement,
        "specificity_excess":specificity_excess,
        "source_drift_vs_clean":source_after-s["clean"],
        "source_drift_vs_faulted":source_after-s["faulted"],
        "source_preservation_limit":source_ceiling,
        "source_preserved_on_test":preserved,
        "affected_before":a["faulted"], "affected_after":after,
        "source_clean":s["clean"], "source_faulted":s["faulted"],
        "source_after":source_after,
        "source_test_n":len(source.ids), "affected_test_n":len(affected.ids),
    }
