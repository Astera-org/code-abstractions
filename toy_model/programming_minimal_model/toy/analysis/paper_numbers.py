"""Emit every toy number quoted in the paper as a LaTeX macro, from the result files (never hand-copied).

Writes paper_tables/toy_numbers.tex, the tables (toy_table_belief.tex is Table 3) and the reference ticks of the figures. Macro scheme: \\tb<Panel><Model><Row><Stat>[Zero]
  Panel: Out (16 blanks, description outside the attention window) | Acc (4 blanks, partially accessible)
  Model: W (window 8, rotary, seeds 1-3) | D (dense 4 layers, rotary, seeds 1-3) | E (dense 8 layers, rotary, seeds 1-3)
         X (W at width 512) | Y (D at width 512): the capacity test, masked and zeroed ablations only
  Row:   TF transport function | TL transport language | IF install function | IL install language | C compose
  Stat:  Loo (seed mean) | Lo / Hi (seed range) | Prompt | Sev (prompted, description ablated) | Uned | Ctrl (largest control) | Rec (recovery %)
         SevF / SevL (function half / language half of Sev) | LooL (language half of Loo) | Mean (the mean-over-values control)
The paper's ablation masks the description out of attention (results/..._MASK); the same macros with the
suffix Zero come from the zeroed-cache ablation of the language-model experiments.
\\toyObserverFloor: the exact observer's all-three-probes pass rate from its prior (analysis/observer_floor.py),
with \\toyObserverFloor<Sum|NoRepeat|Mess> per test; \\toyRef<F|L|FL>: the Bayesian reference per figure row
(also written to toy_references.json for paper_figure.py --reference).
Training: \\toyExcess<W|D|E|X|Y><Min|Max> (evaluation excess over the batch optimum, nats, three seeds), \\toyOptimum.

    uv run python programming_minimal_model/toy/analysis/paper_numbers.py        # from toy_model/, after run_toy.sh suites
"""
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper_tables"                # generated macros, tables and reference ticks
PAPER.mkdir(exist_ok=True)
P = "nonergodic_24x_program_call_transformer_d64_ctx64_"
PW = "nonergodic_24x_program_call_transformer_d512_ctx64_"   # width-scaled (d_model 512) models of the capacity test
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "1 2 3").split())   # training seeds (run_toy.sh passes SEEDS; the paper: 1 2 3)
MODELS = {"W": [P + f"8L_ln_w8_rope_s{s}" for s in SEEDS], "D": [P + f"4L_ln_rope_s{s}" for s in SEEDS],
          "E": [P + f"8L_ln_rope_s{s}" for s in SEEDS],   # E: dense control matched in depth (8 layers) to W
          # X, Y: the width-512 capacity test (JOURNAL 2026-09-18). Peak learning rate 2.5e-4 (1e-3 diverges at this
          # width); the dense model was continued from its 1e-3 run for a second 50K-step schedule. Optional: skipped
          # with a warning while seeds are still training, so the main tables always build.
          "X": [PW + f"8L_ln_w8_rope_s{s}_lr2.5e-4" for s in (1, 2, 3)],
          "Y": [PW + "4L_ln_rope_s1_cont_lr2.5e-4"]}
MODEL_LABELS = {"W": "windowed attention", "D": "dense attention (control)", "E": "dense attention, 8 layers",
                "X": "window 8, width 512", "Y": "dense, 4 layers, width 512"}
# the wide models were run through the masked and zeroed suites only (not the language-model protocol suites)
MODEL_ABLATIONS = {"X": ("", "Zero"), "Y": ("", "Zero")}
OPTIONAL = {"E", "X", "Y"}   # appendix controls: skipped with a warning unless their runs exist (the paper's figures use W and D)
PANELS = {"Out": "programming_toy_v2_n24_b16_p777", "Acc": "programming_toy_v2_n24_b4_p777"}   # zeroed ablation
# macro suffix -> (results-directory suffix, joint metric, function-half metric, language-half metric)
#   ""         the designed-test rule (per-test thresholds), description masked: the paper's current headline
#   Zero       the same rule under the zeroed-cache ablation (no longer in the text)
#   Paper/PaperMask  the language-model protocol (--sever-scope paper), zeroed / masked
#   Near       the nearest-exact-predictive rule on every probe (results of run_belief_suites.sh, JOURNAL 2026-09-21)
#   Bel        the belief readout: mean belief in the target function (mixture fit over continuations), share of
#              private-token mass on the target language, and their product for the joint
RULES = {"": ("_MASK", "fl_exhibits_all", "f_exhibits_all", "l_exhibits"),
         "Zero": ("", "fl_exhibits_all", "f_exhibits_all", "l_exhibits"),
         "Paper": ("_PAPER", "fl_exhibits_all", "f_exhibits_all", "l_exhibits"),
         "PaperMask": ("_PAPERMASK", "fl_exhibits_all", "f_exhibits_all", "l_exhibits"),
         "Near": ("_MASK_belief", "fl_nearest_all", "f_nearest_all", "l_exhibits"),
         "Bel": ("_MASK_belief", "fl_belief", "f_belief_target", "l_belief_target")}
ABLATIONS = {k: v[0] for k, v in RULES.items()}
OPTIONAL_RULES = {"Zero", "Paper", "PaperMask", "Near", "Bel"}    # skipped with a warning when their result directories are absent
# The function-unspecified carriers of the figures hold function-ambiguous symbols (tiers install_neutral and
# compose_neutral); the blank carriers (tiers install and compose) are kept as the rows IFBlank and CBlank so the
# appendix can state that the two agree.
ROWS = {"TF": ("transport_F", "F"), "TL": ("transport_L", "L"), "IF": ("install_neutral_F", "F"),
        "IL": ("install_L", "L"), "C": ("compose_neutral_FL", "FL"),
        "IFBlank": ("install_F", "F"), "CBlank": ("compose_FL", "FL")}
TABLE_ROWS = ("TF", "TL", "IF", "IL", "C")
JOINT = "fl_exhibits_all"


def metrics(rule):
    """(joint, function half, language half) metric names of a rule; half_of(kind) picks the half for a row kind."""
    _, joint, fm, lm = RULES[rule]
    return joint, {"F": fm, "L": lm, "FL": joint}
CONTROLS = ("random", "wrong")      # the controls of the figures; the mean-over-values control is not reported (JOURNAL 2026-09-16)
SHUFFLED = "shuffled"               # reported in the appendix tables only: on installation rows it is a partial correct edit


def summ(d, run):
    fs = sorted((ROOT / "results" / d / run).glob("step_*/summary.json"))
    if not fs and d.endswith("_MASK_belief"):
        # the belief readout (--belief-probes 1, the runner's default) is part of every masked suite in this package,
        # so the Bel / Near rules read the _MASK results unless a separate _MASK_belief run exists
        fs = sorted((ROOT / "results" / d[: -len("_belief")] / run).glob("step_*/summary.json"))
    return json.load(open(fs[-1]))["tables"] if fs else None


def val(tab, tier, key, metric):
    try:
        return tab[tier][key][metric]
    except (KeyError, TypeError):
        return None


def f2(v):
    return f"{v:.2f}"


def main():
    out = ["% generated by programming_minimal_model/toy/analysis/paper_numbers.py -- do not edit by hand"]
    add = lambda name, v: out.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    for suffix, (dsuf, joint, fm, lm) in RULES.items():
        halves = {"F": fm, "L": lm, "FL": joint}
        for pn, pdir in PANELS.items():
            for mn, runs in MODELS.items():
                if suffix not in MODEL_ABLATIONS.get(mn, RULES) and suffix not in OPTIONAL_RULES:
                    continue
                tabs = [t for t in (summ(pdir + dsuf, r) for r in runs) if t]
                if len(tabs) != len(runs):
                    if mn in OPTIONAL or suffix in OPTIONAL_RULES:
                        print(f"warning: {mn} {pdir + dsuf}: {len(tabs)} of {len(runs)} seeds present; using those", file=sys.stderr)
                        if not tabs:
                            continue
                    else:
                        raise AssertionError(f"{pdir + dsuf}: {len(tabs)} of {len(runs)} seeds present")
                if tabs and val(tabs[0], ROWS["TF"][0], "position|loo|1.0", joint) is None:
                    print(f"warning: {mn} {pdir + dsuf}: metric {joint} absent; skipping rule {suffix or 'designed'}", file=sys.stderr)
                    continue
                for rn, (tier, kind) in ROWS.items():
                    half = halves[kind]
                    def g(key, metric=joint, tier=tier, tabs=tabs):
                        return [v for v in (val(t, tier, key, metric) for t in tabs) if v is not None]
                    loo = g("position|loo|1.0")
                    add(f"tb{pn}{mn}{rn}Loo{suffix}", f2(st.mean(loo)))
                    add(f"tb{pn}{mn}{rn}Lo{suffix}", f2(min(loo)))
                    add(f"tb{pn}{mn}{rn}Hi{suffix}", f2(max(loo)))
                    add(f"tb{pn}{mn}{rn}Prompt{suffix}", f2(st.mean(g("none|ceiling|0.0"))))
                    add(f"tb{pn}{mn}{rn}Sev{suffix}", f2(st.mean(g("none|ceiling_severed|0.0"))))
                    add(f"tb{pn}{mn}{rn}Uned{suffix}", f2(st.mean(g("none|none|0.0"))))
                    add(f"tb{pn}{mn}{rn}SevF{suffix}", f2(st.mean(g("none|ceiling_severed|0.0", fm))))
                    add(f"tb{pn}{mn}{rn}SevL{suffix}", f2(st.mean(g("none|ceiling_severed|0.0", lm))))
                    add(f"tb{pn}{mn}{rn}LooL{suffix}", f2(st.mean(g("position|loo|1.0", lm))))
                    # function half of every figure condition (for the belief rule these are function beliefs alone)
                    for key, cname in (("position|loo|1.0", "LooF"), ("position|exact|1.0", "ExactF"), ("none|none|0.0", "UnedF"),
                                       ("position|random|1.0", "RandF"), ("position|wrong|1.0", "WrongF"), ("none|ceiling|0.0", "PromptF")):
                        if g(key, fm):
                            add(f"tb{pn}{mn}{rn}{cname}{suffix}", f2(st.mean(g(key, fm))))
                    if g("position|mean|1.0"):
                        add(f"tb{pn}{mn}{rn}Mean{suffix}", f2(st.mean(g("position|mean|1.0"))))
                    ctrl = [st.mean(g(f"position|{c}|1.0")) for c in CONTROLS if g(f"position|{c}|1.0")]
                    add(f"tb{pn}{mn}{rn}Ctrl{suffix}", f2(max(ctrl)))
                    for c, cname in (("random", "Rand"), ("wrong", "Wrong")):
                        if g(f"position|{c}|1.0"):
                            add(f"tb{pn}{mn}{rn}{cname}{suffix}", f2(st.mean(g(f"position|{c}|1.0"))))
                    if g(f"position|{SHUFFLED}|1.0"):
                        add(f"tb{pn}{mn}{rn}Shuf{suffix}", f2(st.mean(g(f"position|{SHUFFLED}|1.0"))))
                    hl, hs, hn = (st.mean(g(k, half)) for k in ("position|loo|1.0", "none|ceiling_severed|0.0", "none|none|0.0"))
                    add(f"tb{pn}{mn}{rn}Rec{suffix}", f"{100 * (hl - hn) / (hs - hn):.0f}" if hs - hn >= 0.05 else "--")
    # belief readout diagnostics over the width-64 headline models (masked, belief results): the largest per-condition
    # mean divergence between a model's continuation distribution and its best mixture of the three functions, and the
    # mean largest belief in any single function on the unedited neutral carrier (uniform on average, not per trial)
    fits, others = [], []
    for pdir in PANELS.values():
        for mn in ("W", "D"):
            for t in (summ(pdir + RULES["Bel"][0], r) for r in MODELS[mn]):
                if not t:
                    continue
                for tier in ("transport_F", "install_neutral_F", "compose_neutral_FL"):
                    for key in t.get(tier, {}):
                        v = val(t, tier, key, "f_belief_fit_kl")
                        if v is not None and "@" not in key:
                            fits.append(v)
                o = val(t, "install_neutral_F", "none|none|0.0", "f_belief_max_other")
                if o is not None:
                    others.append(o)
    if fits:
        add("toyBeliefFitMax", f"{max(fits):.3f}")
        add("toyBeliefUnedOther", f2(st.mean(others)))
    # the exact observer's floor for a carrier with no function information
    import observer_floor
    params, obs = observer_floor.setup()
    single, allp, mean = observer_floor.floor(params, obs)
    add("toyObserverFloor", f2(mean))
    for test, name in (("sum", "Sum"), ("no_repeat", "NoRepeat"), ("mess3_run", "Mess")):
        add(f"toyObserverFloor{name}", f2(allp[test]))
        add(f"toyObserverFloor{name}Single", f2(single[test]))
    add("toyObserverReset", f"{observer_floor.check_reset(params, obs):.0e}")
    refs, lang = observer_floor.references(params, obs)
    add("toyRefF", f2(refs["transport_F"]))
    add("toyRefL", f"{refs['transport_L']:.3f}")
    add("toyRefFL", f"{refs['compose_FL']:.3f}")
    add("toyRefPrivateMass", f"{lang['private_mass']:.3f}")
    add("toyRefOffTargetMass", f"{lang['off_target_mass']:.3f}")
    json.dump(refs, open(PAPER / "toy_references.json", "w"), indent=1)
    single_n, allp_n, mean_n = observer_floor.floor(params, obs, rule="nearest")
    refs_n, _ = observer_floor.references(params, obs, rule="nearest")
    add("toyObserverFloorNear", f2(mean_n))
    for test, name in (("sum", "Sum"), ("no_repeat", "NoRepeat"), ("mess3_run", "Mess")):
        add(f"toyObserverFloor{name}Near", f2(allp_n[test]))
        add(f"toyObserverFloor{name}SingleNear", f2(single_n[test]))
    add("toyRefFNear", f2(refs_n["transport_F"]))
    add("toyRefFLNear", f"{refs_n['compose_FL']:.3f}")
    json.dump(refs_n, open(PAPER / "toy_references_near.json", "w"), indent=1)
    refs_b = observer_floor.belief_references(params)
    add("toyRefFBel", f"{refs_b['transport_F']:.3f}")
    add("toyRefLBel", f"{refs_b['transport_L']:.3f}")
    add("toyRefFLBel", f"{refs_b['compose_FL']:.3f}")
    json.dump(refs_b, open(PAPER / "toy_references_belief.json", "w"), indent=1)
    # training: evaluation excess over the batch optimum from the lean-trainer logs (logs/lean_<run name>.log, written by run_toy.sh)
    for mn, runs in MODELS.items():
        ex = []
        for r in runs:
            log = ROOT / "logs" / f"lean_{r}.log"
            m = re.search(r"excess \+([0-9.]+)\)", log.read_text()) if log.exists() else None
            if m:
                ex.append(float(m.group(1)))
        if not ex:
            print(f"warning: no training logs for {mn}; toyExcess{mn}* macros not written", file=sys.stderr)
            continue
        add(f"toyExcess{mn}Min", f"{min(ex):.4f}")
        add(f"toyExcess{mn}Max", f"{max(ex):.4f}")
        add(f"toyExcess{mn}Mean", f"{st.mean(ex):.4f}")
        add(f"toyExcess{mn}MeanE", f"{st.mean(ex) * 1e4:.1f}\\times 10^{{-4}}")   # for use in math mode
    opt = sorted((ROOT / "logs").glob("lean_*.log")) if (ROOT / "logs").exists() else []
    m = next((mm for mm in (re.search(r"eval-set-specific optimal loss ([0-9.]+) nats", p.read_text()) for p in opt) if mm), None)
    if m:
        add("toyOptimum", f"{float(m.group(1)):.4f}")
    (PAPER / "toy_numbers.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {PAPER / 'toy_numbers.tex'} with {len(out) - 1} macros")
    write_table("_MASK", "toy_table.tex", ("W", "D"))                        # the designed-test rule, description masked (the figures' numbers)
    # appendix tables of the manuscript: written only when their result sets exist
    for dsuf, fname, models, rule in (("", "toy_table_zero.tex", ("W", "D"), ""), ("_MASK", "toy_table_dense8.tex", ("W", "D", "E"), ""),
                                      ("", "toy_table_dense8_zero.tex", ("W", "D", "E"), ""), ("_PAPER", "toy_table_paper.tex", ("W", "D", "E"), ""),
                                      ("_PAPERMASK", "toy_table_paper_mask.tex", ("W", "D", "E"), ""),
                                      ("_MASK", "toy_table_wide.tex", ("W", "D", "X", "Y"), ""), ("", "toy_table_wide_zero.tex", ("W", "D", "X", "Y"), ""),
                                      (RULES["Near"][0], "toy_table_near.tex", ("W", "D"), "Near")):
        try:
            write_table(dsuf, fname, models, rule=rule)
        except (AssertionError, IndexError, TypeError, st.StatisticsError, ValueError) as e:
            print(f"warning: {fname} not written ({type(e).__name__}: {e}); its result sets are absent or incomplete", file=sys.stderr)
    write_belief_table_compact("toy_table_belief.tex")      # Table 3


ROW_NAMES = {"TF": "transport, function", "TL": "transport, language", "IF": "installation, function",
             "IL": "installation, language", "C": "composition"}


def write_table(dsuf, fname, models=("W", "D"), rule=""):
    """One tabular whose columns are exactly the marks of the figure: leave-one-context-out edit (mean over seeds,
    seed range), same-context-only edit, ordinary prompting, prompting with the description masked, the unedited
    prompt, and the random-direction and wrong-value controls. dsuf selects the results directory suffix and rule the
    scoring rule (see RULES); for the belief rule the entries are mean beliefs rather than pass rates."""
    JOINT, halves = metrics(rule)
    score = "mean belief" if rule == "Bel" else "fraction passing"
    lines = [r"% generated by programming_minimal_model/toy/analysis/paper_numbers.py -- do not edit by hand",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             rf"blanks & model & row & leave-one-context-out (seed range) & same-context-only & prompted & prompted, masked & unedited & random direction & wrong value \\",
             r"\midrule"]
    lines[1] = r"\begin{tabular}{llrrrrrrrr}"
    for pn, pdir, blanks in (("Acc", PANELS["Acc"], "4"), ("Out", PANELS["Out"], "16")):
        for mn, runs, mlabel in ((m, MODELS[m], MODEL_LABELS[m]) for m in models):
            tabs = [t for t in (summ(pdir + dsuf, r) for r in runs) if t]
            first = True
            for rn in TABLE_ROWS:
                tier, kind = ROWS[rn]
                def g(key, metric=JOINT, tier=tier, tabs=tabs):
                    return [v for v in (val(t, tier, key, metric) for t in tabs) if v is not None]
                def cell(key):
                    v = g(key)
                    return f2(st.mean(v)) if v else "--"
                loo = g("position|loo|1.0")
                lines.append(" & ".join([blanks if first else "", mlabel if first else "", ROW_NAMES[rn],
                                         f"{st.mean(loo):.2f} ({min(loo):.2f}--{max(loo):.2f})",
                                         cell("position|exact|1.0"), cell("none|ceiling|0.0"), cell("none|ceiling_severed|0.0"),
                                         cell("none|none|0.0"), cell("position|random|1.0"), cell("position|wrong|1.0")]) + r" \\")
                first = False
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (PAPER / fname).write_text("\n".join(lines) + "\n")
    print(f"wrote {PAPER / fname} ({score})")


def write_belief_table_compact(fname, models=("W", "D")):
    """The compact belief check (JOURNAL 2026-09-25): four blanks, the three function rows, FUNCTION belief alone
    (not the joint product), for the conditions whose beliefs no figure shows: leave-one-context-out edit (seed mean
    and range), same-context-only edit, unedited prompt, random direction, wrong value. Built from the same summaries
    as the belief macros (results/..._MASK_belief)."""
    dsuf = RULES["Bel"][0]
    metric = "f_belief_target"
    rows = (("TF", "transport"), ("IF", "installation"), ("C", "composition"))
    lines = [r"% generated by programming_minimal_model/toy/analysis/paper_numbers.py -- do not edit by hand",
             r"{\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llccccc}", r"\toprule",
             r"model & row & \shortstack{leave-one-\\context-out} & \shortstack{same-\\context-only} & unedited & \shortstack{random\\direction} & \shortstack{wrong\\value} \\",
             r"\midrule"]
    for mn in models:
        tabs = [t for t in (summ(PANELS["Acc"] + dsuf, r) for r in MODELS[mn]) if t]
        assert len(tabs) == len(MODELS[mn]), f"{mn}: {len(tabs)} of {len(MODELS[mn])} seeds"
        first = True
        for rn, rname in rows:
            tier = ROWS[rn][0]
            def g(key, tier=tier, tabs=tabs):
                return [v for v in (val(t, tier, key, metric) for t in tabs) if v is not None]
            loo = g("position|loo|1.0")
            cell = lambda key: f2(st.mean(g(key))) if g(key) else "--"
            short = {"W": "windowed", "D": "dense (control)"}.get(mn, MODEL_LABELS[mn])
            lines.append(" & ".join([short if first else "", rname,
                                     f"{st.mean(loo):.2f} ({min(loo):.2f}--{max(loo):.2f})",
                                     cell("position|exact|1.0"), cell("none|none|0.0"), cell("position|random|1.0"), cell("position|wrong|1.0")]) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}}")
    (PAPER / fname).write_text("\n".join(lines) + "\n")
    print(f"wrote {PAPER / fname} (compact belief check)")


if __name__ == "__main__":
    main()
