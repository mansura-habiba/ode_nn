"""Invariant checks for the falsification benchmark; no trained outcome asserted."""
import importlib.util
from pathlib import Path

import torch

source = Path(__file__).with_name("ccnn_generalization_benchmark.py")
spec = importlib.util.spec_from_file_location("ccnn_v3", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_kernel_families_are_finite():
    x,q = module.make_data(3,1729)
    for family in module.FAMILIES:
        weights=module.teacher_k(q,family)
        assert weights.shape == (3,module.N,module.L)
        assert torch.isfinite(weights).all()
        pred=None
        module.CURRENT_FAMILY=family
        y=module.target(x,q)
        assert y.shape==(3,)
        assert torch.isfinite(y).all()

def test_exact_edit_budgets_and_identity_mapping():
    torch.manual_seed(4)
    frozen=module.Model('lag')
    frozen.eval()
    for p in frozen.parameters():p.requires_grad_(False)
    x,q=module.make_data(3,1701)
    with torch.no_grad():
        expected=module.ungated_predict(frozen,x,q)
    for kind,budget in (
        ('structured',1),('structured',5),('structured',18),
        ('structured',36),('structured',56),
        ('spectral',1),('spectral',5),('spectral',18),
        ('spectral',36),('spectral',56),
        ('rank1',18),('rank2',36),('full',56)):
        adapter=module.UngatedAdapter(frozen,kind,budget)
        assert sum(p.numel() for p in adapter.parameters() if p.requires_grad)==budget
        with torch.no_grad():
            actual=module.ungated_predict(frozen,x,q,adapter)
        assert torch.allclose(actual,expected,atol=1e-5,rtol=1e-5),(kind,budget)

def test_shift_changes_teacher_not_model_domain_gate():
    x,q=module.make_data(4,1907)
    for family in module.FAMILIES:
        module.CURRENT_FAMILY=family
        source_target=module.target(x,q,shift=0)
        shifted_target=module.target(x,q,shift=.12)
        assert not torch.allclose(source_target,shifted_target),family
    # The same prediction function accepts no source/shifted-domain switch.
    import inspect
    assert 'domain' not in inspect.signature(module.ungated_predict).parameters
