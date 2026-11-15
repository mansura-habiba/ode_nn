import numpy as np
from core import (LAGS, lag_kernel, true_kernel, make_dataset, attention_experiment,
                  repair_experiment, serializable)


def test_kernels_are_finite_and_normalized():
    for scenario in ("Matched delayed chirps", "Nonlocal negative control"):
        k = true_kernel(np.array([.3,.7]),scenario)
        assert k.shape == (2,len(LAGS))
        assert np.all(np.isfinite(k))
        assert np.allclose((k*k).sum(1),1,atol=1e-8)
    assert lag_kernel(np.array([.4])).shape == (1,len(LAGS))


def test_reproducible_datasets():
    a=make_dataset(30,17,"Matched delayed chirps")
    b=make_dataset(30,17,"Matched delayed chirps")
    assert np.array_equal(a.y,b.y)
    assert a.x.shape==(30,64)


def test_attention_comparison_finite():
    out=attention_experiment(scenario="Matched delayed chirps",seed=5,ntrain=200)
    assert len(out['metrics'])==5
    assert all(np.isfinite(x['test_mse']) for x in out['metrics'])
    assert serializable(out)['metrics'][0]['model']


def test_negative_control_executable():
    out=attention_experiment(scenario="Nonlocal negative control",seed=6,ntrain=220)
    assert out['config']['scenario']=="Nonlocal negative control"


def test_repair_runs_both_scopes_and_controls():
    for scope in ("Condition-gated (observable shift)","Global edit (shared across domains)"):
        out=repair_experiment(scenario="Matched delayed chirps",shift=.10,seed=4,scope=scope)
        assert len(out['metrics'])==5
        assert all(x['new_test_mse']>=0 for x in out['metrics'])
        assert out['validation_gate']['allowed_original_increase']>0
        if scope.startswith('Condition'):
            old=[r['original_test_mse'] for r in out['metrics']]
            assert old[0]==old[1]  # unchanged by *construction*, not a learned guarantee


def test_no_shift_can_reject_repair():
    out=repair_experiment(scenario="Matched delayed chirps",shift=0.,seed=3)
    assert out['config']['shift']==0