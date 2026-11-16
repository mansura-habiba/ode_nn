"""Independent temporal-memory teachers for CCNN RQ1-M.

These five targets DO NOT import the CCNN benchmark, its kernel, or integrate_drive.
Inputs and target functions are fully observable; matched chirp remains in V3
strictly as a positive calibration example, NOT the primary generalization score.

The time steps here are discrete and equally spaced. Timestamp/irregular stress
tests must not be claimed from this code.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

FAMILIES = ("bimodal", "boxcar", "damped_oscillator", "switching", "feedback_ode")
SPLIT_OFFSETS = {"train": 100, "validation": 200, "calibration": 250, "locked_test": 300}


class SequenceSet(NamedTuple):
    x: torch.Tensor
    q: torch.Tensor
    y: torch.Tensor
    sample_ids: tuple[str, ...]
    family: str
    split: str


def _previous(x: torch.Tensor, delay: int) -> torch.Tensor:
    """Strictly causal zero-prehistory samples at integer lag."""
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    if delay == 0:
        return x
    if delay >= x.shape[1]:
        return torch.zeros_like(x)
    return torch.cat([torch.zeros_like(x[:, :delay]), x[:, :-delay]], dim=1)


def _mean_lags(x: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    return torch.stack([_previous(x, i) for i in range(start, stop + 1)], -1).mean(-1)


def teacher_target(x: torch.Tensor, q: torch.Tensor, family: str) -> torch.Tensor:
    """Map observed sequence and condition to a scalar, without CCNN machinery."""
    if family not in FAMILIES:
        raise ValueError(f"Unsupported independent teacher family: {family}")
    if x.shape != q.shape or x.ndim != 2 or x.shape[1] < 16:
        raise ValueError("Expected [batch,time>=16] x and q with the same shape")
    if not torch.isfinite(x).all() or not torch.isfinite(q).all():
        raise ValueError("non-finite teacher input")

    if family == "bimodal":
        # Two distinct delays interact nonlinearly; no Gaussian/chirp formula.
        early, late = _previous(x, 3), _previous(x, 10)
        excitation = torch.tanh(1.1 * early - 0.9 * late +
                                0.45 * (early * late) * (1 + 0.2 * q))
        return excitation[:, -12:].mean(1)

    if family == "boxcar":
        # Sharp interval supports, rather than a smooth learned envelope.
        recent = _mean_lags(x, 2, 5)
        ancient = _mean_lags(x, 8, 12)
        response = torch.tanh((1 + 0.3 * q) * recent -
                              (0.7 + 0.2 * q.square()) * ancient)
        return response[:, -10:].mean(1)

    if family == "damped_oscillator":
        # A driven 2-state damped oscillator, unrelated to a time-lag integral.
        position = torch.zeros_like(x[:, 0])
        velocity = torch.zeros_like(position)
        trace = []
        dt = 0.08
        for t in range(x.shape[1]):
            omega = 1.5 + 0.3 * q[:, t]
            accel = -0.45 * velocity - omega.square() * position + x[:, t]
            velocity = velocity + dt * accel
            position = position + dt * velocity
            trace.append(position)
        return torch.stack(trace, 1)[:, -10:].mean(1)

    if family == "switching":
        # Abrupt regime switching; a scalar nonlinear recurrent state.
        state = torch.zeros_like(x[:, 0])
        trace = []
        for t in range(x.shape[1]):
            choice = torch.where(q[:, t] > 0.0, x[:, t], -_previous(x, 7)[:, t])
            state = 0.68 * state + 0.32 * torch.tanh(2 * choice)
            trace.append(state)
        return torch.stack(trace, 1)[:, -10:].mean(1)

    # feedback_ode: nonlinear, state-dependent dynamics with observable forcing.
    state = torch.zeros_like(x[:, 0])
    derivative = torch.zeros_like(state)
    for t in range(x.shape[1]):
        derivative = 0.81 * derivative + 0.19 * torch.tanh(
            x[:, t] + q[:, t] * state - 0.3 * state.square())
        state = 0.85 * state + 0.15 * derivative
    return state


def sample_inputs(count: int, seed: int, steps: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent randomized inputs with predictable per-seed replay."""
    if count <= 0 or steps < 16:
        raise ValueError("count must be positive and sequence must have >=16 steps")
    rng = np.random.default_rng(seed)
    tt = np.linspace(0, 1, steps, dtype=np.float32)[None, :]
    phase = rng.uniform(-1, 1, size=(count, 1))
    q0 = rng.uniform(-0.85, 0.85, size=(count, 1))
    slope = rng.uniform(-0.7, 0.7, size=(count, 1))
    q = np.clip(q0 + slope * (tt - 0.5) + 0.25 * np.sin(2 * np.pi * (tt + phase)), -1.25, 1.25)
    x = rng.normal(size=(count, steps)).astype("float32") * 0.35
    for lag in (0.13, 0.41, 0.76):
        centers = rng.uniform(max(0, lag - 0.12), min(1, lag + 0.12), size=(count, 1))
        x += (rng.uniform(-1.0, 1.0, size=(count, 1)) *
              np.exp(-0.5 * ((tt - centers) / 0.055) ** 2)).astype("float32")
    return torch.from_numpy(x), torch.from_numpy(q.astype("float32"))


def make_split(family: str, split: str, seed: int, count: int = 128) -> SequenceSet:
    if family not in FAMILIES or split not in SPLIT_OFFSETS:
        raise ValueError("unknown family or split")
    x, q = sample_inputs(count, seed * 1009 + SPLIT_OFFSETS[split])
    with torch.no_grad():
        y = teacher_target(x, q, family)
    ids = tuple(f"{family}:{seed}:{split}:{i}" for i in range(count))
    return SequenceSet(x, q, y, ids, family, split)


def split_digest(ds: SequenceSet) -> str:
    h = hashlib.sha256()
    for tensor in (ds.x, ds.q, ds.y):
        h.update(tensor.contiguous().numpy().tobytes())
    for sample_id in ds.sample_ids:
        h.update(sample_id.encode("utf-8"))
    return h.hexdigest()


def assert_disjoint_splits(*splits: SequenceSet) -> None:
    seen: set[str] = set()
    for ds in splits:
        ids = set(ds.sample_ids)
        if seen.intersection(ids):
            raise AssertionError("sample IDs overlap: invalid data split")
        seen.update(ids)
