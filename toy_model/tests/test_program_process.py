"""Behavioral helpers for the program toy's second process (programming_minimal_model/toy/program_process.py).

Self-consistency against the exact observer built from the 4-language config: the codec round-trips,
the watcher agrees with the observer's fire probability, the function / language readouts applied to
the observer's *own* predictions recover the true component, and the continuation log-likelihood
identifies the function and language of sampled sequences.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "programming_minimal_model" / "toy"))

import program_process as pp

CONFIG = REPO / "training" / "configs" / "generative_process" / "nonergodic_12x_program_call.yaml"


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.create({"device": "cpu", "generative_process": OmegaConf.load(CONFIG)})


@pytest.fixture(scope="module")
def params(cfg):
    return pp.ProgramParams.from_run_config(cfg)


@pytest.fixture(scope="module")
def process(cfg):
    import hydra
    return hydra.utils.instantiate(cfg.generative_process.instance)


@pytest.fixture(scope="module")
def observer(process):
    return pp.make_observer(process)


def test_params_from_config(params):
    assert params.n_langs == 4 and params.pattern == (0, 1)
    assert (params.p_fire, params.p_floor, params.q_shared) == (0.9, 0.01, 0.8)
    assert [g["generator"] for g in params.generators] == list(pp.FUNCTIONS)
    assert params.generators[1]["rho"] == 0.05 and params.generators[2]["delta"] == 0.1
    assert params.layout.blank == 20 and params.layout.bos == 21


def test_codec_round_trip():
    lay = pp.Layout(4)
    for lang in range(4):
        for k in range(4):
            t = lay.render(k, lang, "lang")
            assert lay.token_symbol(t) == k and lay.token_language(t) == lang
            s = lay.render(k, lang, "shared")
            assert lay.token_symbol(s) == k and lay.token_language(s) is None
    assert lay.token_symbol(lay.blank) is None and not lay.is_content(lay.blank)
    assert lay.is_return(3) and lay.is_return(lay.lang(2, 3)) and not lay.is_return(lay.lang(2, 1))
    assert lay.token_name(3) == "S:F" and lay.token_name(lay.lang(1, 2)) == "L1:2" and lay.token_name(20) == "B"
    assert lay.local_to_global(1) == [0, 1, 2, 3, 8, 9, 10, 11, 20]


def test_watcher_ignores_blanks_and_resets_on_return(params):
    lay = params.layout
    toks = [lay.shared(0), lay.blank, lay.blank, lay.lang(2, 1), lay.shared(3), lay.shared(0), lay.shared(0), lay.shared(1)]
    m = pp.watcher_trajectory(toks, params)
    assert m.tolist() == [1, 1, 1, 2, 0, 1, 1, 2]
    assert pp.expected_fire_positions(toks, params).tolist() == [3, 7]


def test_observer_fire_probability_tracks_watcher(params, observer):
    lay = params.layout
    toks = np.array([[lay.shared(2), lay.shared(0), lay.blank, lay.blank, lay.blank, lay.lang(1, 1), lay.shared(2)]])
    _, _, glob = observer(toks)
    # after 'S:0 B B B L1:1' the pattern is complete across the blank run: 0.9 * P(content next)
    p_content_after_content = 1 - params.p_content_to_filler
    assert pp.return_mass(glob[0, 6], lay) == pytest.approx(0.9 * p_content_after_content, abs=1e-5)
    assert pp.return_mass(glob[0, 7], lay) == pytest.approx(0.01 * p_content_after_content, abs=1e-5)
    # after a blank the next token is a blank with prob 1 - p_filler_to_content
    assert pp.blank_mass(glob[0, 3], lay) == pytest.approx(1 - params.p_filler_to_content, abs=1e-5)
    # a language token was seen, so the language is known: all language mass on language 1
    masses = pp.language_masses(glob[0, 7], lay)
    assert masses[1] == pytest.approx(0.2 * p_content_after_content, abs=1e-5)
    assert masses.sum() - masses[1] == pytest.approx(0.0, abs=1e-7)


def test_readouts_on_the_observer_recover_the_truth(params, observer):
    """Feed the observer's own next-token distribution as if it were a model."""
    rng = np.random.default_rng(3)
    lay = params.layout
    n = 40
    hits_f, hits_l, n_disc, n_lang = 0, 0, 0, 0
    for trial in range(24):
        lang, f = trial % 4, (trial // 4) % 3
        toks = np.array([pp.sample_component_sequence(lang, f, n, rng, params)])
        comp, local, glob = observer(toks)
        c, lc, g = comp[0, -1], local[0, -1], glob[0, -1]
        fr = pp.function_readout(g, c, lc, lay)
        assert fr["kl_bayes"] == pytest.approx(0.0, abs=1e-6)
        if fr["disagreement"] > 0.05 and fr["post_f"][f] > 0.9:
            n_disc += 1
            hits_f += int(fr["argmin"] == f)
        lr = pp.language_readout(g, c, lay)
        if lr["post_l"][lang] > 0.99:
            n_lang += 1
            hits_l += int(lr["argmax"] == lang and lr["other_mass"] < 1e-6)
    assert n_disc >= 8 and hits_f == n_disc, (n_disc, hits_f)
    assert n_lang >= 20 and hits_l == n_lang, (n_lang, hits_l)


def test_continuation_loglik_identifies_function_and_language(params, observer):
    rng = np.random.default_rng(5)
    lay = params.layout
    ok_f = ok_l = 0
    trials = 24
    for trial in range(trials):
        lang, f = trial % 4, (trial // 4) % 3
        toks = np.array([pp.sample_component_sequence(lang, f, 48, rng, params)])
        comp, local, _ = observer(toks)
        res = pp.continuation_loglik(comp[0], local[0], toks[0], start=16, lay=lay)
        ok_f += int(res["argmax_f"] == f)
        ok_l += int(res["argmax_l"] == lang)
    assert ok_f >= 20, ok_f
    assert ok_l >= 22, ok_l


def test_constructed_probe_helpers(params):
    rng = np.random.default_rng(0)
    lay = params.layout
    ev = pp.simulate_function_machine(1, 30, rng, params, fire="never")
    assert pp.RET not in ev and all(0 <= e < 3 for e in ev)
    ev = pp.simulate_function_machine(2, 60, rng, params, fire="always")
    # every completed pattern is followed by a return, and no return appears elsewhere
    m = pp.watcher_trajectory(pp.render(ev, 0, "shared", lay=lay), params)
    for i, e in enumerate(ev):
        if e == pp.RET:
            assert m[i - 1] == len(params.pattern)
        elif i > 0 and m[i - 1] == len(params.pattern):
            raise AssertionError("pattern completed without a return under fire='always'")
    toks = pp.render([0, 2, 1], 3, "lang", lay=lay)
    assert toks == [lay.lang(3, 0), lay.lang(3, 2), lay.lang(3, 1)]
    assert pp.insert_blank_run(toks, 1, 2, lay) == [toks[0], lay.blank, lay.blank, toks[1], toks[2]]


def test_numpy_observer_matches_jax_observer(params, process, observer):
    rng = np.random.default_rng(11)
    toks = np.array([pp.sample_component_sequence(t % 4, t % 3, 30, rng, params) for t in range(6)])
    comp, local, glob = observer(toks)
    npo = pp.NumpyObserver(process)
    comp2, local2, glob2 = npo(toks)
    np.testing.assert_allclose(comp2, comp, atol=1e-5)
    np.testing.assert_allclose(local2, local, atol=1e-5)
    np.testing.assert_allclose(glob2, glob, atol=1e-5)
    w, b = npo.filter_state(toks)
    np.testing.assert_allclose(w, comp[:, -1], atol=1e-5)
    cont = npo.sample_continuations(w, b, 12, rng)
    assert cont.shape == (6, 12) and cont.min() >= 0 and cont.max() < params.layout.n_tokens


def test_designed_unit_tests_pass_on_the_observer(params, observer):
    """The observer itself passes every unit test on a faithful context of the right function and, for
    sum / no-repeat, fails it on contexts it has identified as another function (controls). Mess3 is
    identified by exclusion, so an atypical Mess3 context can legitimately look like no-repeat: controls
    are only counted where the observer's posterior on the context's function exceeds 0.9."""
    rng = np.random.default_rng(21)
    lay = params.layout
    for test in pp.UNIT_TESTS:
        g = pp.UNIT_TEST_FUNCTION[test]
        for ctx_f in range(3):
            passes = counted = 0
            n = 8
            for _ in range(n):
                events = pp.simulate_function_machine(ctx_f, 32, rng, params, fire="sample")
                suffix, target = pp.designed_suffix(test, rng)
                toks = np.array([pp.render(events + suffix, 1, "coin", rng, params.q_shared, lay)])
                comp, local, glob = observer(toks)
                fr = pp.function_readout(glob[0, -1], comp[0, -1], local[0, -1], lay)
                r = pp.unit_readout(glob[0, -1], glob[0, -1], test, target, fr["argmin"], fr["argmin"], lay)
                if ctx_f == g or fr["post_f"][ctx_f] > 0.9:
                    counted += 1
                    passes += r["exhibits"]
            if ctx_f == g:
                assert passes >= n - 1, (test, ctx_f, passes)
            elif test != "mess3_run":
                assert counted >= 3 and passes == 0, (test, ctx_f, counted, passes)


def test_continuation_stats_signatures(params):
    rng = np.random.default_rng(4)
    lay = params.layout
    stats = {}
    for f in range(3):
        toks = pp.sample_component_sequence(2, f, 400, rng, params)
        stats[f] = pp.continuation_stats(toks[:20], toks[20:], 2, params)
    assert stats[1]["repeat_rate"] < 0.12 and stats[0]["repeat_rate"] > 0.3
    assert stats[2]["sum_correct"] > 0.75 and stats[0]["sum_correct"] < 0.55 and stats[1]["sum_correct"] < 0.55
    for f in range(3):
        s = stats[f]
        assert s["ret_precision"] > 0.6 and 0.7 < s["ret_recall"] <= 1.0, s
        assert 0.15 < s["blank_frac"] < 0.35 and 0.1 < s["lang_share"] < 0.3 and s["other_lang"] == 0
    toks = [lay.lang(0, 0), lay.blank, lay.blank, lay.shared(1), lay.shared(3), lay.lang(0, 2), lay.lang(0, 2)]
    s = pp.continuation_stats([], toks, 0, params)
    assert s["n_returns"] == 1 and s["ret_precision"] == 1.0 and s["n_completions"] == 1 and s["ret_recall"] == 1.0
    assert s["n_pairs"] == 2 and s["repeat_rate"] == 0.5 and s["mean_blank_run"] == 2.0


def test_controlled_sum_contexts(params):
    rng = np.random.default_rng(8)
    for cond in ("unseen", "seen", "seen_wrong"):
        for _ in range(10):
            pair = (int(rng.integers(3)), int(rng.integers(3)))
            if pair == params.pattern:
                continue
            ev = pp.simulate_sum_context(rng, params, 30, pair, cond)
            seen = pp.sum_pairs_seen(ev)
            if cond == "unseen":
                assert pair not in seen, (pair, ev)
            else:
                assert pair in seen
                answers = seen[pair]
                correct = (pair[0] + pair[1]) % 3
                if cond == "seen":
                    assert all(a == correct for a in answers)
                else:
                    assert all(a != correct for a in answers)


def test_observer_computes_unseen_pairs_and_ignores_wrong_answers(params, observer):
    """The exact observer predicts a + b for an unseen pair as confidently as for a seen one; a shown wrong
    answer lowers its posterior on the sum function (it hedges) but it does not copy the wrong answer."""
    rng = np.random.default_rng(9)
    lay = params.layout
    res = {c: [] for c in ("unseen", "seen", "seen_wrong")}
    copy = []
    pairs = [(a, b) for a in range(3) for b in range(3) if (a, b) != params.pattern]   # the pattern pair's answer is pre-empted by the return
    for cond in res:
        for _ in range(8):
            pair = pairs[int(rng.integers(len(pairs)))]
            ev = pp.simulate_sum_context(rng, params, 27, pair, cond)
            suffix = [pp.RET, pair[0], pair[1]]
            toks = np.array([pp.render(ev + suffix, 2, "coin", rng, params.q_shared, lay)])
            _, _, glob = observer(toks)
            q = pp.symbol_distribution(glob[0, -1], lay)
            res[cond].append(q[(pair[0] + pair[1]) % 3])
            if cond == "seen_wrong":
                wrong = pp.sum_pairs_seen(ev)[pair][0]
                copy.append(q[wrong])
    # unseen pairs are answered as confidently as seen ones (computation, not lookup); a shown wrong answer is
    # evidence against the sum function, so the observer hedges there but still does not copy the wrong answer
    assert np.mean(res["unseen"]) > 0.75 and np.mean(res["seen"]) > 0.75, res
    assert abs(np.mean(res["unseen"]) - np.mean(res["seen"])) < 0.1, res
    assert 0.35 < np.mean(res["seen_wrong"]) < 0.8, res["seen_wrong"]
    assert np.mean(copy) < 0.3, copy


def test_multi_rendering_suffixes(params):
    rng = np.random.default_rng(2)
    lay = params.layout
    variants = pp.render_with_suffix_renderings([0, 1, 2], [pp.RET, 1, 2], 3, rng, params)
    assert len(variants) == 4 and all(len(v) == 6 for v in variants)
    assert all(v[:4] == variants[0][:4] for v in variants)
    assert [lay.token_symbol(t) for t in variants[1][-2:]] == [1, 2]
    assert lay.token_language(variants[1][-2]) is None and lay.token_language(variants[1][-1]) == 3
    assert lay.token_language(variants[3][-2]) == 3 and lay.token_language(variants[3][-1]) == 3
