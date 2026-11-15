"""Small, auditable causal-lag research demonstration.

All data are synthetic and deliberately simple. The attention probe is not a
Transformer; the structured kernel is a prior-matched template, not a full
reproduction of the 2021 CCNN training algorithm. Results must not be used as
claims of benchmark superiority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

L = 64
LAGS = np.linspace(0.0, 1.0, L)
WIDTHS = (0.075, 0.10, 0.13)
FREQUENCIES = (6.7, 7.5, 8.3)
BIASES = (-0.055, 0.0, 0.055)


def norm_rows(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.sqrt(np.sum(a * a, axis=-1, keepdims=True)), 1e-9)


def lag_kernel(q: np.ndarray, *, width: float = 0.10,
               frequency: float = 7.5, bias: float = 0.0) -> np.ndarray:
    """Compact, signed, chirped time-lag kernel, evaluated at query q."""
    q = np.asarray(q, dtype=float).reshape(-1, 1)
    d = LAGS[None, :] - q - bias
    raw = np.exp(-0.5 * (d / width) ** 2) * np.cos(
        2 * np.pi * (frequency * d + 0.8 * d * d)
    )
    return norm_rows(raw)


def true_kernel(q: np.ndarray, scenario: str, shift: float = 0.0) -> np.ndarray:
    """Hidden ground truth; accessed only by dataset creation and visual reference."""
    q = np.asarray(q, dtype=float).reshape(-1, 1)
    if scenario == "Matched delayed chirps":
        center = q + shift
        d = LAGS[None, :] - center
        width = 0.085 + 0.025 * q
        freq = 7.2 + 0.65 * q
        raw = np.exp(-0.5 * (d / width) ** 2) * np.cos(
            2 * np.pi * (freq * d + 0.8 * d * d)
        )
    elif scenario == "Nonlocal negative control":
        qq = q + shift
        raw = (np.sin(2 * np.pi * ((3.5 + 3 * qq) * LAGS[None, :] + 0.8 * LAGS[None, :] ** 2))
               + 0.8 * np.cos(2 * np.pi * (13 * LAGS[None, :] - 2.6 * qq)))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return norm_rows(raw)


@dataclass
class Dataset:
    x: np.ndarray
    q: np.ndarray
    y: np.ndarray


def make_dataset(n: int, seed: int, scenario: str, noise: float = 0.025,
                 shift: float = 0.0) -> Dataset:
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.21, 0.77, n)
    phases = rng.uniform(-np.pi, np.pi, (n, 1))
    freq = rng.uniform(2.0, 13.0, (n, 1))
    x = (rng.normal(size=(n, L)) * 0.65
         + 0.38 * np.sin(2 * np.pi * freq * LAGS[None, :] + phases)
         + 0.22 * np.cos(2 * np.pi * (freq / 2) * LAGS[None, :] + phases))
    weights = true_kernel(q, scenario, shift)
    y = np.sum(x * weights, axis=-1) + rng.normal(0, noise, n)
    return Dataset(x=x, q=q, y=y)


def query_basis(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.stack([np.ones_like(q), 2 * (q - 0.5),
                     (2 * (q - 0.5)) ** 2,
                     np.sin(2 * np.pi * q), np.cos(2 * np.pi * q)], axis=-1)


def features_query(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (query_basis(q)[:, :, None] * x[:, None, :]).reshape(len(x), -1)


def attention_weights(q: np.ndarray, width: float = 0.11) -> np.ndarray:
    """Query-centred normalized lag weights: an attention-LIKE probe only."""
    q = np.asarray(q)[:, None]
    logits = -0.5 * ((LAGS[None, :] - q) / width) ** 2
    a = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return a / a.sum(axis=-1, keepdims=True)


def features_attention(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    return x * attention_weights(q)


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.ones(len(x))])


def ridge_fit(x: np.ndarray, y: np.ndarray, regularization: float = .15) -> np.ndarray:
    a = add_bias(x)
    gram = a.T @ a
    penalty = np.eye(gram.shape[0]) * regularization
    penalty[-1, -1] = 1e-9
    return cho_solve(cho_factor(gram + penalty, check_finite=False),
                     a.T @ y, check_finite=False)


def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y_hat - y) ** 2))


@dataclass
class StructuredModel:
    width: float
    frequency: float
    bias: float
    gain: float
    intercept: float
    
    def predict(self, d: Dataset, delta: float = 0.0) -> np.ndarray:
        m = np.sum(d.x * lag_kernel(d.q, width=self.width,
                                    frequency=self.frequency,
                                    bias=self.bias + delta), axis=-1)
        return self.gain * m + self.intercept


def select_structured(train: Dataset, val: Dataset) -> StructuredModel:
    winner = None
    best = float("inf")
    for width in WIDTHS:
        for freq in FREQUENCIES:
            for bias in BIASES:
                z_train = np.sum(train.x * lag_kernel(train.q, width=width,
                                      frequency=freq, bias=bias), axis=-1)
                a, b = ridge_fit(z_train[:, None], train.y, regularization=.001)
                candidate = StructuredModel(width, freq, bias, float(a), float(b))
                val_loss = mse(candidate.predict(val), val.y)
                if val_loss < best:
                    best, winner = val_loss, candidate
    assert winner is not None
    return winner


def fit_methods(train: Dataset, val: Dataset) -> dict:
    t0 = time.perf_counter()
    structured = select_structured(train, val)
    static = ridge_fit(train.x, train.y)
    attention = ridge_fit(features_attention(train.x, train.q), train.y, .01)
    unstructured = ridge_fit(features_query(train.x, train.q), train.y, .45)
    # A two-branch hybrid: the learned lag-attention output and structured lag integral.
    att_train = add_bias(features_attention(train.x, train.q)) @ attention
    struct_train = structured.predict(train)
    hybrid = ridge_fit(np.column_stack([att_train, struct_train]), train.y, .01)
    return dict(structured=structured, static=static, attention=attention,
                unstructured=unstructured, hybrid=hybrid,
                fit_seconds=time.perf_counter() - t0)


def predict_methods(d: Dataset, models: dict) -> dict[str, np.ndarray]:
    att = add_bias(features_attention(d.x, d.q)) @ models["attention"]
    struct = models["structured"].predict(d)
    return {
        "Static convolution (64 weights)": add_bias(d.x) @ models["static"],
        "Lag attention probe (64 weights)": att,
        "Query-conditioned conv (320 weights)": add_bias(features_query(d.x, d.q)) @ models["unstructured"],
        "Structured lag kernel (5 parameters)": struct,
        "Attention + structured lag kernel (hybrid)": (
            add_bias(np.column_stack([att, struct])) @ models["hybrid"]),
    }


def attention_experiment(*, scenario: str, noise: float = .025,
                         seed: int = 17, ntrain: int = 450) -> dict:
    train = make_dataset(ntrain, seed, scenario, noise)
    val = make_dataset(160, seed + 1000, scenario, noise)
    test = make_dataset(220, seed + 2000, scenario, noise)
    models = fit_methods(train, val)
    preds = predict_methods(test, models)
    result = [dict(model=name, test_mse=mse(pred, test.y),
                   test_mae=float(np.mean(np.abs(pred-test.y))),
                   approximate_parameters=int({
                       "Static convolution (64 weights)": 65,
                       "Lag attention probe (64 weights)": 65,
                       "Query-conditioned conv (320 weights)": 321,
                       "Structured lag kernel (5 parameters)": 5,
                       "Attention + structured lag kernel (hybrid)": 8,
                   }[name])) for name, pred in preds.items()]
    scores = {r["model"]: r["test_mse"] for r in result}
    return {
        "config": dict(scenario=scenario, noise=noise, seed=seed, train_size=ntrain,
                       val_size=len(val.y), test_size=len(test.y)),
        "metrics": result,
        "models": models,
        "sample": dict(lags=LAGS.copy(), x=test.x[0].copy(), query=float(test.q[0]),
                       reference_kernel=true_kernel(test.q[:1], scenario)[0],
                       learned_kernel=lag_kernel(test.q[:1], width=models["structured"].width,
                                                 frequency=models["structured"].frequency,
                                                 bias=models["structured"].bias)[0]),
        "preview": dict(reference=test.y[:50],
                        attention=preds["Lag attention probe (64 weights)"][:50],
                        hybrid=preds["Attention + structured lag kernel (hybrid)"][:50]),
        "hypotheses": {
            "H-A1 hybrid improves attention probe on this synthetic test": (
                "supported_in_this_run" if scores["Attention + structured lag kernel (hybrid)"] <
                scores["Lag attention probe (64 weights)"] else "not_supported_in_this_run"),
            "H-A2 hybrid improves query-conditioned unstructured baseline": (
                "supported_in_this_run" if scores["Attention + structured lag kernel (hybrid)"] <
                scores["Query-conditioned conv (320 weights)"] else "not_supported_in_this_run"),
        },
        "fit_seconds":models["fit_seconds"],
        "warning":"Analytic lag-template, query-centred attention probe; not Transformer/2021-CCNN reproduction.",
    }


def best_single_edit(r: np.ndarray, features: np.ndarray,
                     bound: float = 2.5) -> tuple[int, float]:
    # Exact training-set coordinate search; no validation or test labels used.
    dot = features.T @ r
    norm = np.sum(features ** 2, axis=0) + .2
    gain = np.clip(dot / norm, -bound, bound)
    reduction = 2 * gain * dot - gain * gain * norm
    idx = int(np.argmax(reduction))
    return idx, float(gain[idx])


def repair_experiment(*, scenario: str, shift: float = .11,
                      noise: float = .025, scope: str = "Condition-gated (observable shift)",
                      seed: int = 17) -> dict:
    # The original domain and shifted domain have disjoint train/val/test RNG streams.
    old_train = make_dataset(400, seed, scenario, noise)
    old_val = make_dataset(140, seed+1000, scenario, noise)
    old_test = make_dataset(180, seed+2000, scenario, noise)
    new_train = make_dataset(300, seed+3100, scenario, noise, shift)
    new_val = make_dataset(140, seed+4100, scenario, noise, shift)
    new_test = make_dataset(180, seed+5100, scenario, noise, shift)
    models = fit_methods(old_train, old_val)
    s = models["structured"]
    ungated = scope == "Global edit (shared across domains)"
    base_old_val = mse(s.predict(old_val), old_val.y)
    base_new_val = mse(s.predict(new_val), new_val.y)
    tolerance = max(.002, .10 * base_old_val)
    # Choose the delay edit from the REPAIR-TRAIN set only. Apply validation gate separately.
    offsets = np.linspace(-.22, .22, 89)
    train_curve = [mse(s.predict(new_train, float(d)), new_train.y) for d in offsets]
    idx = int(np.argmin(train_curve))
    delta = float(offsets[idx])
    edited_new_val = mse(s.predict(new_val, delta), new_val.y)
    edited_old_val = mse(s.predict(old_val, delta if ungated else 0.), old_val.y)
    accepted_struct = bool(edited_new_val < .99 * base_new_val and
                           edited_old_val <= base_old_val + tolerance)
    chosen_delta = delta if accepted_struct else 0.

    # Less structured: query-conditioned convolution, with only one learnable
    # coefficient update (selected from 320 coordinates on repair training data).
    base_w = models["unstructured"]
    ftr = features_query(new_train.x, new_train.q)
    base_new_train_un = add_bias(ftr) @ base_w
    edit_idx, edit_gain = best_single_edit(new_train.y-base_new_train_un, ftr)
    def unstructured_pred(d: Dataset, modify: bool, gated: bool = True):
        feat = features_query(d.x, d.q)
        return add_bias(feat) @ base_w + ((edit_gain * feat[:, edit_idx]) if modify else 0.)
    ub_val_new = mse(unstructured_pred(new_val, False),new_val.y)
    uc_val_new = mse(unstructured_pred(new_val, True),new_val.y)
    ub_val_old = mse(unstructured_pred(old_val, False),old_val.y)
    uc_val_old = mse(unstructured_pred(old_val, ungated),old_val.y)
    accept_un = bool(uc_val_new < .99 * ub_val_new and uc_val_old <= ub_val_old + max(.002,.1*ub_val_old))

    # Stronger control: 64-weight linear correction, same observable domain gate.
    # Fit using shifted TRAIN residual; validation chooses accept/reject, not parameters.
    f_wide = ridge_fit(new_train.x, new_train.y-unstructured_pred(new_train,False),2.0)
    def wide_pred(d: Dataset, modify: bool):
        return unstructured_pred(d,False)+(add_bias(d.x)@f_wide if modify else 0.)
    wide_new_val=mse(wide_pred(new_val,True),new_val.y)
    wide_old_val=mse(wide_pred(old_val,ungated),old_val.y)
    wide_accept=bool(wide_new_val<.99*ub_val_new and wide_old_val<=ub_val_old+max(.002,.1*ub_val_old))

    structured_mse_old = mse(s.predict(old_test,chosen_delta if ungated else 0.),old_test.y)
    structured_mse_new = mse(s.predict(new_test,chosen_delta),new_test.y)
    rows = [
        dict(method="Structured frozen", new_test_mse=mse(s.predict(new_test),new_test.y),
             original_test_mse=mse(s.predict(old_test),old_test.y), edited_weights=0, accepted=False),
        dict(method="Structured delay edit (1 scalar)",new_test_mse=structured_mse_new,
             original_test_mse=structured_mse_old,edited_weights=int(accepted_struct),accepted=accepted_struct),
        dict(method="Unstructured frozen",new_test_mse=mse(unstructured_pred(new_test,False),new_test.y),
             original_test_mse=mse(unstructured_pred(old_test,False),old_test.y),edited_weights=0,accepted=False),
        dict(method="Unstructured best 1 coefficient",new_test_mse=mse(unstructured_pred(new_test,accept_un),new_test.y),
             original_test_mse=mse(unstructured_pred(old_test,accept_un and ungated),old_test.y),
             edited_weights=int(accept_un),accepted=accept_un),
        dict(method="Unstructured 64-weight adapter",new_test_mse=mse(wide_pred(new_test,wide_accept),new_test.y),
             original_test_mse=mse(wide_pred(old_test,wide_accept and ungated),old_test.y),
             edited_weights=64 if wide_accept else 0,accepted=wide_accept),
    ]
    return {
        "config":dict(scenario=scenario,shift=float(shift),noise=float(noise),scope=scope,seed=seed,
                      n_train=300,n_val=140,n_test=180),
        "metrics":rows,
        "repair_sweep":dict(offsets=offsets,train_mse=np.array(train_curve),
                             proposed_delta=delta,applied_delta=chosen_delta),
        "validation_gate":dict(shifted_frozen=base_new_val,shifted_candidate=edited_new_val,
                                original_frozen=base_old_val,original_candidate=edited_old_val,
                                allowed_original_increase=tolerance),
        "hypotheses":{
            "H-R1 structured single-parameter repair accepted": (
                "supported_in_this_run" if accepted_struct else "not_supported_in_this_run"),
            "H-R2 structured repair has lower shifted-domain MSE than single unstructured edit":(
                "supported_in_this_run" if structured_mse_new < rows[3]["new_test_mse"] else "not_supported_in_this_run"),
        },
        "notes":"Structured family matches the positive generator; validation gating is independent; "
                "condition-gated mode preserves original by construction; the attention proxy is not a Transformer; "
                "this is not LLM Weight-Space Programming.",
    }


def serializable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, StructuredModel):
        return dict(width=v.width,frequency=v.frequency,bias=v.bias,gain=v.gain,intercept=v.intercept)
    if isinstance(v, dict):
        return {k:serializable(x) for k,x in v.items() if k != "models"}
    if isinstance(v, (list,tuple)):
        return [serializable(x) for x in v]
    return v


def as_json(d: dict) -> str:
    return json.dumps(serializable(d),indent=2,allow_nan=False)