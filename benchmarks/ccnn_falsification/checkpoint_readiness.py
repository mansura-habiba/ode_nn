"""Training-checkpoint adequacy and calibration-only fault selection (Issue #9).

Training and checkpoint selection receive ONLY train and validation tensors.
No test interface is accepted. Plateau is diagnostic, not a proof of correctness.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Sequence
import copy
import json
import math
import time

import torch

from blinded_fault_injection import state_digest, classify_fault_headroom


@dataclass(frozen=True)
class TrainingObservation:
    epoch: int
    training_mse: float
    validation_mse: float
    batches_seen: int
    elapsed_seconds: float


@dataclass
class TrainingOutcome:
    model: torch.nn.Module
    chosen_epoch: int
    best_validation_mse: float
    stop_reason: str
    readiness: str
    train_metrics: list[TrainingObservation]
    checkpoint_hash: str
    optimizer_state: dict
    torch_rng_state: torch.Tensor
    total_optimizer_steps: int

    def manifest(self) -> dict:
        return {
            "chosen_epoch":self.chosen_epoch, "best_validation_mse":self.best_validation_mse,
            "stop_reason":self.stop_reason, "readiness":self.readiness,
            "checkpoint_hash":self.checkpoint_hash, "optimizer_steps":self.total_optimizer_steps,
            "training_curves":[asdict(item) for item in self.train_metrics],
            "checkpoint_selection_data":"validation_only", "test_used":False,
            "notes":"A plateau or epoch cap is a diagnostic, not proof of adequate mechanism identification.",
        }


def train_checkpoint(model: torch.nn.Module,
                     train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     *, seed: int, max_epochs: int = 80,
                     batch_size: int = 64, patience: int = 10,
                     minimum_epochs: int = 20, lr: float = 0.003,
                     minimum_delta: float = 1e-6) -> TrainingOutcome:
    if max_epochs < 1 or batch_size < 1 or patience < 1 or minimum_epochs < 1 or minimum_epochs > max_epochs:
        raise ValueError("invalid train/checkpoint controls")
    if lr <= 0 or minimum_delta < 0 or not all(math.isfinite(x) for x in (lr, minimum_delta)):
        raise ValueError("invalid optimizer choices")
    xt, qt, yt = train
    xv, qv, yv = validation
    for x, q, y in (train, validation):
        if x.ndim != 2 or q.shape != x.shape or y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("invalid train/validation tensor shapes")
        if not all(torch.isfinite(t).all().item() for t in (x,q,y)):
            raise ValueError("nonfinite train/validation data")
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    best=float("inf")
    best_weights=None
    best_optim=None
    best_rng=None
    chosen=-1
    stalled=0
    steps=0
    trace=[]
    start=time.perf_counter()
    for ep in range(max_epochs):
        model.train()
        ep_losses=[]
        for indices in torch.randperm(len(xt)).split(batch_size):
            optimizer.zero_grad()
            loss=(model(xt[indices],qt[indices])-yt[indices]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("diverged source model during training")
            loss.backward()
            optimizer.step()
            ep_losses.append(float(loss.detach()))
            steps+=1
        model.eval()
        with torch.no_grad():
            validation_loss=float((model(xv,qv)-yv).square().mean())
        if not math.isfinite(validation_loss):
            raise ValueError("nonfinite validation loss")
        trace.append(TrainingObservation(ep+1, sum(ep_losses)/len(ep_losses),
                    validation_loss,steps,time.perf_counter()-start))
        if validation_loss < best-minimum_delta:
            best=validation_loss
            best_weights=copy.deepcopy(model.state_dict())
            best_optim=copy.deepcopy(optimizer.state_dict())
            best_rng=torch.get_rng_state().clone()
            chosen=ep+1
            stalled=0
        else:
            stalled+=1
        if ep+1>=minimum_epochs and stalled>=patience:
            reason="validation_plateau"
            break
    else:
        reason="epoch_budget_exhausted"
    if best_weights is None:
        raise RuntimeError("no valid validation-selected checkpoint")
    model.load_state_dict(best_weights)
    model.eval()
    return TrainingOutcome(model,chosen,best,reason,
        "plateau_observed" if reason=="validation_plateau" else "checkpoint_not_established",
        trace,state_digest(best_weights),best_optim,best_rng,steps)


@dataclass(frozen=True)
class FaultCalibration:
    selected_magnitude: float | None
    calibration_clean_mse: float
    trials: tuple[dict,...]
    status: str


def calibrate_on_disjoint_population(
    *, clean_validation_mse: float,
    candidate_magnitudes: Sequence[float],
    calibration_loss_for_magnitude: Callable[[float],float],
    minimum_abs_headroom: float=0.001,
    minimum_relative_headroom: float=0.10,
) -> FaultCalibration:
    """Evaluator-only: caller MUST supply an isolated calibration cohort."""
    if not math.isfinite(clean_validation_mse) or clean_validation_mse < 0:
        raise ValueError("invalid clean calibration loss")
    if len(candidate_magnitudes)==0:
        raise ValueError("no fault magnitude candidates")
    rows=[]
    chosen=None
    for value in sorted(set(float(x) for x in candidate_magnitudes),key=abs):
        if value == 0 or not math.isfinite(value):
            raise ValueError("fault candidates must be finite and nonzero")
        measured=float(calibration_loss_for_magnitude(value))
        label=classify_fault_headroom(clean_validation_mse,measured,
                minimum_abs=minimum_abs_headroom,minimum_relative=minimum_relative_headroom)
        rows.append({"magnitude":value,"calibration_faulted_mse":measured,"headroom":measured-clean_validation_mse,
                     "status":label})
        if chosen is None and label=="fault_detectable":
            chosen=value
    return FaultCalibration(chosen,clean_validation_mse,tuple(rows),
        "calibrated" if chosen is not None else "fault_not_detectable")
