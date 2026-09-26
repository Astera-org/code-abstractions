"""The exact observer's function-test pass rate when it knows nothing about the function.

Every designed probe opens with a return, which resets every function machine, so the observer's prediction
after "description, blanks, probe" depends on the description only through its function posterior. A
carrier that holds no function information therefore scores what the observer scores from its prior on the
probe alone. `floor()` enumerates every probe draw (9 for the sum test, 3 each for the others), applies the
paper's pass rule, cubes for the three-probes-all-pass requirement, and averages over the three target
functions. `check_reset()` confirms the reset on real descriptions. Used by paper_numbers.py.

    uv run python programming_minimal_model/toy/analysis/observer_floor.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import program_process as pp  # noqa: E402

CONFIG = HERE.parents[2] / "training" / "configs" / "generative_process" / "nonergodic_24x_program_call.yaml"


def setup(config=CONFIG):
    """The process and its parameters, built from the config file (only the process is used, no trained run)."""
    import hydra
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"device": "cpu", "generative_process": OmegaConf.load(config)})
    params = pp.ProgramParams.from_run_config(cfg)
    process = hydra.utils.instantiate(cfg.generative_process.instance)
    return params, pp.NumpyObserver(process)


def draws(test):
    if test == "sum":
        return [([pp.RET, a, b], (a + b) % 3) for a in range(3) for b in range(3)]
    if test == "no_repeat":
        return [([pp.RET, s], s) for s in range(3)]
    return [([pp.RET, s, s, s], s) for s in range(3)]


def floor(params, obs, n_probes=3, rule="designed"):
    """Per-test single-probe pass rate, all-probes rate, and the mean over target functions, for a predictor with a
    flat prior over the functions that updates on the probe tokens. rule = "designed" (the per-test thresholds) or
    "nearest" (the probe passes when the predictor's symbol distribution is closer to the target function's exact
    predictive than to either other function's)."""
    lay = params.layout
    single, allp = {}, {}
    for test in pp.UNIT_TESTS:
        ok = []
        for suffix, target in draws(test):
            comp, local, glob = obs(np.array([pp.render(suffix, 0, "shared", lay=lay)]))
            q = pp.symbol_distribution(glob[0, -1], lay)
            fr = pp.function_readout(glob[0, -1], comp[0, -1], local[0, -1], lay)
            if rule == "nearest":
                ok.append(int(fr["argmin"] == pp.UNIT_TEST_FUNCTION[test]))
            else:
                ok.append(pp.unit_exhibits(test, float(q[target]), fr["argmin"]))
        single[test] = float(np.mean(ok))
        allp[test] = single[test] ** n_probes
    return single, allp, float(np.mean(list(allp.values())))


def language_reference(params, obs, n_langs=None):
    """Language readout for a predictor with a uniform posterior over the languages: total private-token mass
    and off-target mass at a blank (does the 0.05 other-mass clause reject it?), and the expected pass rate
    with uniform tie breaking, 1/n_langs. Computed on the observer after one shared return followed by a blank,
    which leaves its language posterior uniform."""
    lay = params.layout
    n_langs = n_langs or params.n_langs
    comp, local, glob = obs(np.array([[lay.shared(pp.RET), lay.blank]]))
    lr = pp.language_readout(glob[0, -1], comp[0, -1], lay)
    private = float(lr["lang_mass"].sum())
    off_target = private * (n_langs - 1) / n_langs
    return {"private_mass": private, "off_target_mass": off_target, "threshold_rejects": off_target >= 0.05,
            "pass": 1.0 / n_langs if off_target < 0.05 else 0.0}


def references(params, obs, n_probes=3, rule="designed"):
    """Expected joint success per figure row for an exact Bayesian predictor with uniform priors over the
    edited factors and the correct values of the remaining factors: function rows = function floor x 1,
    language rows = 1 x language pass, composition = product."""
    single, allp, f_floor = floor(params, obs, n_probes, rule)
    lang = language_reference(params, obs)
    return {"transport_F": f_floor, "install_F": f_floor, "transport_L": lang["pass"], "install_L": lang["pass"],
            "compose_FL": f_floor * lang["pass"]}, lang


def belief_references(params):
    """Exact references for the belief score (mean belief in the target factor): an uninformed predictor holds
    1/3 on each function and 1/8 on each language; composition is the product. A certain predictor scores 1."""
    f, l = 1.0 / pp.N_FUNC, 1.0 / params.n_langs
    return {"transport_F": f, "install_F": f, "transport_L": l, "install_L": l, "compose_FL": f * l}


def check_reset(params, obs, n_paths=6, T=36, blanks=4):
    """Largest difference between per-function predictives after description+blanks+probe and after the
    probe alone, over real descriptions in every function."""
    from program_transport import make_paths, render_path
    lay = params.layout
    worst = 0.0
    for p in make_paths(params, np.random.default_rng(3), n_paths, T, obs, blanks):
        for f in range(3):
            toks = render_path(p, f, 1, lay)
            for test in pp.UNIT_TESTS:
                for suffix, _ in draws(test):
                    pr = pp.render(suffix, 1, "shared", lay=lay)
                    c1, l1, _ = obs(np.array([toks + pr]))
                    c0, l0, _ = obs(np.array([pr]))
                    q1, _ = pp.function_predictives(c1[0, -1], l1[0, -1], 3)
                    q0, _ = pp.function_predictives(c0[0, -1], l0[0, -1], 3)
                    worst = max(worst, float(np.abs(q1 - q0).max()))
    return worst


if __name__ == "__main__":
    params, obs = setup()
    single, allp, mean = floor(params, obs)
    print(f"single-probe pass rate {single}\nall three probes {allp}\nmean over target functions {mean:.3f}")
    refs, lang = references(params, obs)
    print(f"language reference {lang}\nper-row references (designed rule) {refs}")
    single_n, allp_n, mean_n = floor(params, obs, rule="nearest")
    refs_n, _ = references(params, obs, rule="nearest")
    print(f"nearest rule: single-probe pass {single_n}; all three probes {allp_n}; mean {mean_n:.3f}; per-row {refs_n}")
    print(f"belief references {belief_references(params)}")
    print(f"reset check: max difference {check_reset(params, obs):.2e}")
