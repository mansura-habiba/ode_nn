"""Blinded model-weight fault injection for CCNN RQ6-M.

Corrupt only a saved model state, NOT a teacher or target. Public repair inputs
contain faulted weights and a neutral identifier. Fault identity and the
clean-checkpoint digest are private evaluator metadata, never repair inputs.

This module does not train repair algorithms or make scientific success claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile
from typing import Mapping

import torch

FAULT_SURFACES = {
    "delay_center": "branch.mu0",
    "query_delay_gain": "branch.muq",
    "bandwidth": "branch.logsigma",
    "frequency": "branch.freq",
    "chirp": "branch.chirp",
    "response_gain": "branch.gain",
}


def clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone().cpu() for name, tensor in state.items()}


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Content-stable checkpoint fingerprint, independent of serialization ZIP metadata."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        raw = tensor.detach().contiguous().cpu()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(raw.shape)).encode("ascii"))
        h.update(str(raw.dtype).encode("ascii"))
        h.update(raw.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class PublicFaultPackage:
    """Only these fields can be supplied to a repair algorithm."""
    faulted_state: dict[str, torch.Tensor]
    model_sha256: str
    sample_id: str


@dataclass(frozen=True)
class EvaluatorOnlyFaultOracle:
    """Must NOT be supplied to editor or included in repair diagnostics."""
    fault_family: str
    modified_parameter: str
    signed_magnitude: float
    clean_sha256: str
    faulted_sha256: str
    sample_id: str


def inject_model_fault(
    clean_state: Mapping[str, torch.Tensor],
    *,
    family: str,
    magnitude: float,
    sample_id: str,
) -> tuple[PublicFaultPackage, EvaluatorOnlyFaultOracle]:
    if family not in FAULT_SURFACES:
        raise ValueError(f"unsupported fault family {family}")
    if not math.isfinite(magnitude):
        raise ValueError("fault magnitude must be finite")
    if not sample_id or "/" in sample_id or "\\\\" in sample_id:
        raise ValueError("sample_id must be a plain nonempty token")

    name = FAULT_SURFACES[family]
    if name not in clean_state:
        raise ValueError(f"checkpoint lacks candidate surface {name}")
    if not clean_state[name].is_floating_point() or clean_state[name].numel() != 1:
        raise ValueError("requested CCNN parameter must be a floating scalar")

    state = clone_state(clean_state)
    state[name] = state[name] + torch.as_tensor(magnitude, dtype=state[name].dtype)
    clean_sha = state_digest(clean_state)
    faulted_sha = state_digest(state)
    public = PublicFaultPackage(state, faulted_sha, sample_id)
    oracle = EvaluatorOnlyFaultOracle(
        family, name, float(magnitude), clean_sha, faulted_sha, sample_id)
    return public, oracle


def save_separate_fault_artifacts(
    public: PublicFaultPackage,
    oracle: EvaluatorOnlyFaultOracle,
    *,
    repair_dir: str | Path,
    evaluator_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist strict repair/evaluator files in different root directories.

    Caller must enforce OS/CI access restrictions: separate folders alone are
    NOT a security boundary. Never give repair code the evaluator_dir path.
    """
    repair = Path(repair_dir).resolve()
    private = Path(evaluator_dir).resolve()
    if repair == private or repair in private.parents or private in repair.parents:
        raise ValueError("evaluator and repair directories must not be nested")
    if public.sample_id != oracle.sample_id or public.model_sha256 != oracle.faulted_sha256:
        raise ValueError("package and oracle mismatch")
    if state_digest(public.faulted_state) != public.model_sha256:
        raise ValueError("public checkpoint digest mismatch")
    repair.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    repair_file = repair / (public.sample_id + ".pt")
    oracle_file = private / (public.sample_id + ".json")
    if repair_file.exists() or oracle_file.exists():
        raise FileExistsError("refusing to overwrite fault evidence")
    torch.save({"state_dict": public.faulted_state,
                "model_sha256": public.model_sha256,
                "sample_id": public.sample_id}, repair_file)
    oracle_file.write_text(json.dumps(vars(oracle), indent=2))
    return repair_file, oracle_file


def load_repair_input(path: str | Path) -> PublicFaultPackage:
    """Load only the public artifact; never read the evaluator oracle."""
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    digest = state_digest(state)
    if digest != payload["model_sha256"]:
        raise ValueError("invalid corrupted checkpoint digest")
    return PublicFaultPackage(state, digest, str(payload["sample_id"]))


def classify_fault_headroom(clean_mse: float, faulted_mse: float,
                            minimum_abs: float = 0.001,
                            minimum_relative: float = 0.10) -> str:
    """Evaluator-only calibration decision: not a test-set acceptance rule."""
    if not all(math.isfinite(x) for x in [clean_mse, faulted_mse, minimum_abs, minimum_relative]):
        raise ValueError("metrics must be finite")
    if min(clean_mse, faulted_mse, minimum_abs, minimum_relative) < 0:
        raise ValueError("negative loss/threshold")
    improvement = faulted_mse - clean_mse
    return ("fault_detectable" if improvement >= minimum_abs and
            improvement >= minimum_relative * max(clean_mse, 1e-12)
            else "fault_not_detectable")
