"""Strong unrestricted query-conditioned FIR control for RQ1-M.

This is deliberately *not* an oracle chirp: an MLP predicts L unconstrained
causal lag coefficients from observed q and position. It can plug into the
same attention backbone as the proposed CCNN and has no access to fault family.
The attention module is provided externally, allowing frozen or joint training.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def causal_history(x: torch.Tensor, lags: int) -> torch.Tensor:
    if x.ndim != 2 or lags < 1:
        raise ValueError("expected [batch,time] input and >=1 lag")
    values = []
    for j in range(lags):
        if j >= x.shape[1]:
            values.append(torch.zeros_like(x))
        elif j == 0:
            values.append(x)
        else:
            values.append(F.pad(x[:, :-j], (j, 0)))
    return torch.stack(values, -1)


class QueryMLPFIR(nn.Module):
    """A flexible non-Gaussian lag basis conditioned on observable query and time."""

    def __init__(self, lags: int = 14, hidden: int = 32,
                 layers: int = 2, state_decay: float = 8.0):
        super().__init__()
        if lags < 1 or hidden < 1 or layers < 1 or state_decay <= 0:
            raise ValueError("positive dimensions, layers and decay required")
        self.lags = lags
        self.state_decay = state_decay
        blocks = [nn.Linear(5, hidden), nn.GELU()]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden, hidden), nn.GELU()])
        blocks.append(nn.Linear(hidden, lags))
        self.coefficient_net = nn.Sequential(*blocks)
        self.gain = nn.Parameter(torch.tensor(0.9))

    def weights(self, q: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if q.ndim != 2:
            raise ValueError("expected query tensor [batch,time]")
        n = q.shape[1]
        if t is None:
            elapsed = torch.arange(n, device=q.device, dtype=q.dtype).view(1, n).expand_as(q) / n
        else:
            if t.shape != q.shape:
                raise ValueError("timestamp shape must match query")
            elapsed = t
        features = torch.stack(
            [q, q.square(), torch.sin(torch.pi*q), elapsed, elapsed.square()], -1)
        coeff = self.coefficient_net(features)
        return F.normalize(coeff, p=2, dim=-1, eps=1e-7)

    def forward(self, x: torch.Tensor, q: torch.Tensor,
                t: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape != q.shape or x.ndim != 2:
            raise ValueError("x, q must have identical [batch,time] shape")
        if not torch.isfinite(x).all().item() or not torch.isfinite(q).all().item():
            raise ValueError("invalid input")
        k = self.weights(q, t)
        m = (causal_history(x, self.lags) * k).sum(-1)
        # Same downstream exponential leaky-state operation as the V3 lag-CCNN
        # comparison; the hypothesis difference is lag-kernel parameterization.
        alpha = 1 - math.exp(-self.state_decay / x.shape[1])
        z = torch.zeros_like(x[:, 0])
        for step in range(x.shape[1]):
            z = (1 - alpha)*z + alpha*torch.tanh(2*m[:, step])
        return self.gain*z


class AttentionPlusMLPFIR(nn.Module):
    """Shared attention module is injected, not secretly changed between arms."""

    def __init__(self, attention: nn.Module, lags: int = 14,
                 hidden: int = 32, layers: int = 2):
        super().__init__()
        self.att = attention
        self.branch = QueryMLPFIR(lags=lags, hidden=hidden, layers=layers)
        self.gate = nn.Parameter(torch.tensor(0.95))

    def forward(self, x: torch.Tensor, q: torch.Tensor,
                t: torch.Tensor | None = None) -> torch.Tensor:
        return self.att(x, q) + self.gate*self.branch(x, q, t=t)


def freeze_attention(augmented: AttentionPlusMLPFIR) -> None:
    """Enable frozen-backbone residual studies: branch parameters remain live."""
    for p in augmented.att.parameters():
        p.requires_grad_(False)


def parameter_summary(module: nn.Module) -> dict:
    return {"total_parameters":sum(p.numel() for p in module.parameters()),
            "trainable_parameters":sum(p.numel() for p in module.parameters() if p.requires_grad)}
