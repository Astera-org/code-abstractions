"""The draft's "Factors are Mostly Independent" figure (cfs.png), drawn for the programming toy from the
result files of the paper-faithful intervention suite (program_transport.py --suffix-blanks B
--edit-positions suffix --sever --n-probes k).

Layout as in the paper: rows transport / install / compose, one line per factor (f = function, lambda =
language); one column per model, grouped into PANELS (one results directory each, e.g. the 16-blank and the
4-blank suites). A model column may pool several training seeds: filled blue = mean leave-one-context-out
score at dose 1 across seeds, with a thin bar spanning the seeds' minimum to maximum; open blue = exact
same-context-only factor; gray = controls (circle random direction, x wrong value, square mean over values,
diamond random labelling); dashed orange = prompted, no intervention; dotted gray (--severed-ceiling) =
prompted, then content keys and values zeroed. Control and ceiling markers are seed means. The score is the
fraction of trials where every designed test passes for the target function AND the language readout is the
target language (fl_exhibits_all).

    uv run python programming_minimal_model/toy/paper_figure.py \
        --panel "description outside the attention window (16 blanks)=results/programming_toy_v2_n24_b16_p777" \
        --panel "description partially accessible (4 blanks)=results/programming_toy_v2_n24_b4_p777" \
        --model "window 8, rotary=<run seed 1>,<run seed 2>,<run seed 3>" \
        --model "dense, rotary (control)=<run seed 1>,<run seed 2>,<run seed 3>" \
        --severed-ceiling --out programming_minimal_model/toy/figures/toy_cfs_rotary.png

`--dir D` is shorthand for a single untitled panel. Prints the plotted numbers as a markdown table (seed
means, the leave-one-out seed range, and the recovery fraction of the factor half: (leave-one-out minus
unedited) / (prompted-severed minus unedited) on f_exhibits_all for function rows and l_exhibits for
language rows), so write-ups can quote them without hand-copying.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import textwrap

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

ROWS = [("transport", [("f", "transport_F"), (r"$\lambda$", "transport_L")]),
        ("install", [("f", "install_F"), (r"$\lambda$", "install_L")]),
        ("compose", [(r"$f,\lambda$", "compose_FL")])]
# --neutral-carriers: the function-unspecified carriers hold function-ambiguous symbols (tiers install_neutral /
# compose_neutral) instead of blanks (install / compose); the language-unspecified carrier is the same in both
NEUTRAL_TIERS = {"install_F": "install_neutral_F", "compose_FL": "compose_neutral_FL"}
HALF = {"transport_F": "f_exhibits_all", "install_F": "f_exhibits_all", "transport_L": "l_exhibits",
        "install_L": "l_exhibits", "compose_FL": "fl_exhibits_all",
        "install_neutral_F": "f_exhibits_all", "compose_neutral_FL": "fl_exhibits_all"}
BLUE, DBLUE, GRAY, ORANGE = "#2f7ed8", "#1f4e79", "#8c8c8c", "#c8641e"
CONTROLS = [("random", "o", "random control"), ("wrong", "x", "wrong value"), ("mean", "s", "mean over values"),
            ("shuffled", "D", "random labelling")]


def load(dir_, run):
    fs = sorted(glob.glob(f"{dir_}/{run}/step_*/summary.json"), key=lambda p: int(p.split("step_")[1].split("/")[0]))
    if not fs:
        raise FileNotFoundError(f"no summary.json under {dir_}/{run}")
    return json.load(open(fs[-1]))


def values(summs, tier, key, metric):
    """One value per seed (seeds lacking the cell are skipped); empty list if none has it."""
    out = []
    for s in summs:
        try:
            out.append(s["tables"][tier][key][metric])
        except KeyError:
            pass
    return out


def mean(vs):
    return st.mean(vs) if vs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="single untitled panel (shorthand for one --panel)")
    ap.add_argument("--panel", action="append", default=[], help='"title=results directory" (repeatable)')
    ap.add_argument("--model", action="append", required=True, help='"label=run name[,run name of another seed,...]"')
    ap.add_argument("--metric", default="fl_exhibits_all")
    ap.add_argument("--alpha", default="1.0", help="dose shown for the leave-one-out point")
    ap.add_argument("--observer", action="store_true", help="also mark the exact observer's own score on the prompted contexts")
    ap.add_argument("--severed-ceiling", action="store_true", help="also mark the prompted target context with its content severed")
    ap.add_argument("--severed-label", default="prompted, content severed", help="legend text for the severed-prompt line")
    ap.add_argument("--severed-style", default="line", choices=["line", "band", "title", "row"],
                    help="how to show the severed prompt: dotted line per row (line), shaded band between the severed and "
                         "the intact prompt per row (band), once per column in the title with no per-row mark (title), or "
                         "an extra line at the top of each column showing the intact and the severed prompt with the drop (row)")
    ap.add_argument("--prompted-label", default="prompted (no intervention)", help="legend text for the intact-prompt line")
    ap.add_argument("--top-row-label", default="target\nprompt", help="row label of the extra top line (--severed-style row)")
    ap.add_argument("--out", default="programming_minimal_model/toy/figures/toy_cfs.png")
    ap.add_argument("--stack", action="store_true",
                    help="stack the model columns vertically (one narrow column, for a half-width figure); single panel only")
    ap.add_argument("--no-legend", action="store_true", help="omit the legend (describe the markers in the caption instead)")
    ap.add_argument("--neutral-carriers", action="store_true",
                    help="use the function-ambiguous-symbol carriers (install_neutral / compose_neutral tiers) for the "
                         "function-installation and composition rows instead of the blank carriers")
    ap.add_argument("--drop-control", action="append", default=[], help="control condition to leave out (repeatable), e.g. mean")
    ap.add_argument("--reference-label", default="Bayesian reference: edited factors unknown", help="legend text for the reference tick")
    ap.add_argument("--xlabel", default="", help="x-axis text (default: derived from --metric)")
    ap.add_argument("--font", default="default", choices=["default", "stix"],
                    help="stix: Times-like serif text and math (matches the code-model figure)")
    ap.add_argument("--legend", default="bottom", choices=["bottom", "right"], help="legend below the panels or beside them")
    ap.add_argument("--no-seed-count", action="store_true", help="do not append '(n seeds)' to the column titles")
    ap.add_argument("--row-height", type=float, default=0.62, help="inches per plotted line (default 0.62)")
    ap.add_argument("--hspace", type=float, default=0.35, help="vertical gap between row groups, as a fraction of a row (default 0.35)")
    ap.add_argument("--reference", default="",
                    help='Bayesian reference per tier, "transport_F=0.26,transport_L=0.125,..." drawn as a black tick on each row')
    args = ap.parse_args()
    controls = [c for c in CONTROLS if c[0] not in args.drop_control]
    reference = {k: float(v) for k, v in (kv.split("=") for kv in args.reference.split(",") if kv)}
    panels = [p.split("=", 1) for p in args.panel] or [("", args.dir)]
    if panels[0][1] is None:
        raise SystemExit("give --dir or at least one --panel")
    specs = [(m.split("=", 1)[0], m.split("=", 1)[1].split(",")) for m in args.model]
    columns = [(ptitle, label, [load(pdir, r) for r in runs]) for ptitle, pdir in panels for label, runs in specs]
    est = columns[0][2][0]["estimators"][0]
    n_lines = sum(len(r[1]) for r in ROWS)
    n_cols = len(columns)

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    if args.font == "stix":
        plt.rcParams.update({"font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                             "mathtext.fontset": "stix", "font.size": 11})
    if args.stack:
        if len(panels) != 1:
            raise SystemExit("--stack needs exactly one panel")
        if args.severed_style == "row":
            raise SystemExit("--severed-style row is not available with --stack")
        top_axes = None
        # one narrow column: every model gets its own block of rows, blocks separated by a wider gap
        ratios = [len(r[1]) for r in ROWS] * n_cols
        fig, grid = plt.subplots(len(ROWS) * n_cols, 1, figsize=(3.6, 0.4 * n_lines * n_cols + 1.0 * n_cols),
                                 gridspec_kw={"height_ratios": ratios, "hspace": 0.55}, squeeze=False)
        axes = [[grid[j * len(ROWS) + i, 0] for j in range(n_cols)] for i in range(len(ROWS))]
    else:
        top = 1 if (args.severed_ceiling and args.severed_style == "row") else 0
        legend_w = 2.9 if args.legend == "right" else 0.0
        fig, grid = plt.subplots(len(ROWS) + top, n_cols, figsize=(3.3 * n_cols + 0.6 + legend_w, args.row_height * (n_lines + top) + (1.4 if args.legend == "right" else 2.6)),
                                 gridspec_kw={"height_ratios": [1] * top + [len(r[1]) for r in ROWS], "hspace": args.hspace, "wspace": 0.12},
                                 squeeze=False)
        axes = grid[top:]
        top_axes = grid[0] if top else None
    def style_axis(ax, n, ylabels, last_row):
        ax.set_xlim(-0.04, 1.06)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_yticks(range(n))
        ax.set_yticklabels(ylabels, fontsize=11, style="italic")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "", "0.5", "", "1"] if last_row else [])
        ax.tick_params(axis="x", length=3, color="#9a9a9a")
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#9a9a9a")
        ax.grid(axis="x", color="#e6e6e6", lw=0.8, zorder=0)

    table = []
    for j, (ptitle, label, summs) in enumerate(columns):
        if top_axes is not None:
            # one line per column: the intact target prompt and the same prompt with its description severed,
            # pooled over every row (the drop is a property of the model, not of the row)
            tiers = [NEUTRAL_TIERS.get(t, t) if args.neutral_carriers else t for _, ls_ in ROWS for _, t in ls_]
            ceil_s = [v for t in tiers for v in values(summs, t, "none|ceiling|0.0", args.metric)]
            sev_s = [v for t in tiers for v in values(summs, t, "none|ceiling_severed|0.0", args.metric)]
            ax = top_axes[j]
            ax.plot([0, 1], [0, 0], color="#c9c9c9", lw=2.2, zorder=1, solid_capstyle="round")
            if ceil_s and sev_s:
                c, sv = st.mean(ceil_s), st.mean(sev_s)
                ax.plot([c, c], [-0.42, 0.42], color=ORANGE, ls=(0, (3, 2)), lw=1.8, zorder=2)
                if c - sv > 0.03:
                    ax.annotate("", xy=(sv, 0), xytext=(c, 0), zorder=3,
                                arrowprops=dict(arrowstyle="-|>", color="#4d4d4d", lw=1.4, shrinkA=2, shrinkB=2))
                if len(sev_s) > 1:
                    ax.plot([min(sev_s), max(sev_s)], [0, 0], color="#4d4d4d", lw=1.6, zorder=4, solid_capstyle="round")
                ax.scatter([sv], [0], marker="D", s=38, color="#4d4d4d", zorder=5)
                ax.annotate(f"{sv:.2f}", (sv, 0), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color="#4d4d4d")
            style_axis(ax, 1, [], False)
            ax.set_title(label + (f"  ({len(summs)} seeds)" if len(summs) > 1 and not args.no_seed_count else ""), fontsize=10.5, pad=8)
            if j == 0:
                ax.text(-0.22, 0.5, args.top_row_label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=11)
        for i, (row_name, lines) in enumerate(ROWS):
            ax = axes[i][j]
            for k, (fac, tier) in enumerate(lines):
                ref_tier = tier  # the Bayesian reference is keyed by the plain tier name, before any carrier remap
                if args.neutral_carriers:
                    tier = NEUTRAL_TIERS.get(tier, tier)
                y = len(lines) - 1 - k
                loo_s = values(summs, tier, f"{est}|loo|{args.alpha}", args.metric)
                loo, exact = mean(loo_s), mean(values(summs, tier, f"{est}|exact|1.0", args.metric))
                ceiling = mean(values(summs, tier, "none|ceiling|0.0", args.metric))
                severed = mean(values(summs, tier, "none|ceiling_severed|0.0", args.metric))
                unedited = mean(values(summs, tier, "none|none|0.0", args.metric))
                ax.plot([0, 1], [y, y], color="#c9c9c9", lw=2.2, zorder=1, solid_capstyle="round")
                ctrl = []
                for n, (cond, marker, _) in enumerate(controls):
                    v = mean(values(summs, tier, f"{est}|{cond}|1.0", args.metric))
                    ctrl.append(v)
                    if v is not None:
                        ax.scatter([v], [y + 0.22 - 0.15 * n], marker=marker, s=22, color=GRAY, zorder=3, linewidths=1.2)
                if ceiling is not None:
                    ax.plot([ceiling, ceiling], [y - 0.42, y + 0.42], color=ORANGE, ls=(0, (3, 2)), lw=1.8, zorder=2)
                if args.severed_ceiling and severed is not None:
                    if args.severed_style == "line":
                        ax.plot([severed, severed], [y - 0.3, y + 0.3], color="#4d4d4d", ls=":", lw=1.4, zorder=2)
                    elif args.severed_style == "band" and ceiling is not None:
                        ax.fill_betweenx([y - 0.42, y + 0.42], severed, ceiling, color=ORANGE, alpha=0.18, lw=0, zorder=1.5)
                if args.observer:
                    v = mean(values(summs, tier, "none|ceiling|0.0", args.metric + "_bayes"))
                    if v is not None:
                        ax.scatter([v], [y], marker="|", s=140, color="black", zorder=4, linewidths=1.6)
                if ref_tier in reference:
                    ax.scatter([reference[ref_tier]], [y], marker="|", s=170, color="black", zorder=4, linewidths=1.4)
                if exact is not None:
                    ax.scatter([exact], [y], s=150, facecolors="none", edgecolors=BLUE, linewidths=1.8, zorder=4)
                if loo is not None:
                    if len(loo_s) > 1:
                        ax.plot([min(loo_s), max(loo_s)], [y, y], color=DBLUE, lw=1.6, zorder=4, solid_capstyle="round")
                    ax.scatter([loo], [y], s=70, color=DBLUE, zorder=5)
                    ax.annotate(f"{loo:.2f}", (loo, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=DBLUE)
                # recovery fraction on the factor half
                h = HALF[tier]
                h_loo, h_sev, h_none = (mean(values(summs, tier, kk, h)) for kk in (f"{est}|loo|{args.alpha}", "none|ceiling_severed|0.0", "none|none|0.0"))
                # undefined when the severed prompt itself sits on the unedited floor (nothing to recover)
                recov = None if None in (h_loo, h_sev, h_none) or (h_sev - h_none) < 0.05 else (h_loo - h_none) / (h_sev - h_none)
                table.append((ptitle, label, len(summs), row_name, fac.replace("$", "").replace("\\", ""), loo,
                              (min(loo_s), max(loo_s)) if len(loo_s) > 1 else None, exact, ceiling, severed, unedited, *ctrl, recov))
            style_axis(ax, len(lines), [fac for fac, _ in reversed(lines)] if (j == 0 or args.stack) else [],
                       i == len(ROWS) - 1 and (not args.stack or j == n_cols - 1))
            if i == 0 and top_axes is None:
                title = label + (f"  ({len(summs)} seeds)" if len(summs) > 1 and not args.no_seed_count else "")
                if args.severed_ceiling and args.severed_style == "title":
                    sev_all = [mean(values(summs, NEUTRAL_TIERS.get(t, t) if args.neutral_carriers else t, "none|ceiling_severed|0.0", args.metric))
                               for _, ls_ in ROWS for _, t in ls_]
                    sev_all = [v for v in sev_all if v is not None]
                    if sev_all:
                        title += f"\n{args.severed_label}: {min(sev_all):.2f}" + (f" to {max(sev_all):.2f}" if max(sev_all) - min(sev_all) >= 0.005 else "")
                ax.set_title(title, fontsize=10.5, pad=8)
            if j == 0 or args.stack:
                ax.text(-0.22, 0.5, row_name, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=11)
    # panel titles centred over their columns
    n_models = len(specs)
    for p, (ptitle, _) in enumerate(panels):
        if ptitle and not args.stack:
            left = axes[0][p * n_models].get_position().x0
            right = axes[0][p * n_models + n_models - 1].get_position().x1
            width = max(18, int(34 * n_models))          # wrap long titles over narrow panels
            fig.text((left + right) / 2, 0.955, textwrap.fill(ptitle, width), ha="center", va="bottom",
                     fontsize=11.5, fontweight="bold", linespacing=1.25)
    what = "all designed tests pass" if args.metric.endswith("_all") else "the designed test passes"
    if args.stack:
        fig.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.14 if not args.no_legend else 0.06)
        axes[-1][-1].set_xlabel(f"joint score: {what} for the\ntarget function and the language\nreadout is the target language",
                                fontsize=8)
    else:
        if args.legend == "right":
            fig_w, fig_h = fig.get_size_inches()
            right = 1.0 - (legend_w + 0.1) / fig_w
            bottom_in = 0.62 if "\n" not in (args.xlabel or "") else 0.8   # tick labels + axis label, in inches
            fig.subplots_adjust(left=0.09 * (fig_w - legend_w) / fig_w + 0.02, right=right, top=1 - 0.45 / fig_h, bottom=bottom_in / fig_h)
            fig.text((0.09 * (fig_w - legend_w) / fig_w + 0.02 + right) / 2, 0.08 / fig_h, args.xlabel or f"fraction of trials where {what} for the target function\nand the language readout is the target language",
                     ha="center", va="bottom", fontsize=10)
        else:
            fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.24)
            fig.text(0.5, 0.15, args.xlabel or f"fraction of trials where {what} for the target function and the language readout is the target language",
                     ha="center", fontsize=10)
    ctrl_handles = {cond: plt.Line2D([], [], marker=marker, color=GRAY, ls="", ms=6 if marker == "x" else 5, mew=1.2, label=lab)
                    for cond, marker, lab in controls}
    handles = [plt.Line2D([], [], marker="o", color=DBLUE, ls="-", lw=1.6, ms=7, label="leave-one-out factor (mean, seed range)"),
               plt.Line2D([], [], marker="o", mfc="none", mec=BLUE, mew=1.8, ls="", ms=9, label="same-context-only factor"),
               plt.Line2D([], [], color=ORANGE, ls=(0, (3, 2)), lw=1.8, label=args.prompted_label),
               *ctrl_handles.values()]
    if reference:
        handles.append(plt.Line2D([], [], marker="|", color="black", ls="", ms=10, mew=1.4, label=args.reference_label))
    if args.severed_ceiling and args.severed_style == "line":
        handles.append(plt.Line2D([], [], color="#4d4d4d", ls=":", lw=1.4, label=args.severed_label))
    elif args.severed_ceiling and args.severed_style == "row":
        handles.append(plt.Line2D([], [], marker="D", color="#4d4d4d", ls="-", lw=1.6, ms=5, label=args.severed_label))
    elif args.severed_ceiling and args.severed_style == "band":
        handles.append(plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.18, lw=0, label=f"lost when the description is masked"))
    if args.observer:
        handles.append(plt.Line2D([], [], marker="|", color="black", ls="", ms=10, mew=1.6, label="exact Bayesian observer"))
    if not args.no_legend and args.legend == "right" and not args.stack:
        fig.legend(handles=handles, loc="center right", ncol=1, frameon=False, fontsize=9.5, bbox_to_anchor=(0.995, 0.5),
                   labelspacing=1.1, handlelength=2.2)
    elif not args.no_legend:
        fig.legend(handles=handles, loc="upper center", ncol=2 if args.stack else (4 if n_cols > 1 else 2), frameon=False,
                   fontsize=8 if args.stack else 9, bbox_to_anchor=(0.5, 0.08 if args.stack else 0.11), columnspacing=1.2)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {args.out} (+ .pdf); metric {args.metric}, estimator {est}, dose {args.alpha}")

    def f2(v):
        return "-" if v is None else f"{v:.2f}"
    print("\n| panel | model | seeds | tier | factor | leave-one-out (mean) | seed range | exact | prompted | prompted, severed | unedited | "
          + " | ".join(lab for _, _, lab in controls) + " | recovery of factor half |")
    print("|" + "---|" * (12 + len(controls)))
    for r in table:
        rng = "-" if r[6] is None else f"{r[6][0]:.2f} to {r[6][1]:.2f}"
        rec = "-" if r[-1] is None else f"{r[-1]:.0%}"
        print("| " + " | ".join([r[0], r[1], str(r[2]), r[3], r[4], f2(r[5]), rng, *[f2(v) for v in r[7:-1]], rec]) + " |")


if __name__ == "__main__":
    main()
