"""Recreate all six-method figures and tables using only the collected NPZ files."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import (
    BASELINES,
    case_path,
    cases,
    COLORS,
    CONFIG,
    DEFAULT_RESULTS_DIR,
    IQL,
    LABELS,
    METHODS,
    noise_name,
)
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

def load_results(results_dir):
    data, checks = {}, []
    expected = list(cases())
    assert len(expected) == 272
    actual = list((results_dir / "raw").glob("s*_a*/**/*.npz"))
    assert {p.resolve() for p in actual} == {
        case_path(results_dir, c).resolve() for c in expected
    }
    for case in expected:
        path = case_path(results_dir, case)
        meta = json.loads(path.with_suffix(".json").read_text())
        assert meta["method"] == case[2] and meta["updates"] == 1000000
        assert meta["seed_ids"] == CONFIG["seeds"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["data_sha256"]
        with np.load(path) as source:
            h = {k: source[k] for k in source.files}
        assert h["v_bias"].shape == (20, 1001)
        np.testing.assert_array_equal(h["step"], np.arange(0, 1000001, 1000))
        assert np.isfinite(h["q_final"]).all() and np.isfinite(h["v_bias"]).all()
        errors = []
        for step, key in ((100000, "checkpoint_q_100k"), (1000000, "q_final")):
            b = h[key].mean(1).max(-1).mean(-1)
            curve = h["v_bias"][:, step // 1000]
            errors.append(float(np.max(np.abs(b - curve))))
            np.testing.assert_allclose(b, curve, atol=1e-10, rtol=0)
        if case[2].startswith("df_"):
            np.testing.assert_array_equal(h["q_final"][:, 0], h["q_final"][:, 1])
        if case[2] in IQL:
            v = h["value_final"].mean((1, 2))
            np.testing.assert_allclose(v, h["value_bias"][:, -1], atol=1e-10, rtol=0)
        data[case] = h
        checks.append(
            dict(
                states=case[0],
                noise=case[1],
                method=case[2],
                D=case[3],
                tau=case[4],
                max_metric_error=max(errors),
                passed=True,
            )
        )
    return data, checks


def bias(h, step):
    key = "q_final" if step == 1000000 else "checkpoint_q_100k"
    return h[key].mean(1).max(-1).mean(-1)


def stats(data, case, step=1000000):
    y = bias(data[case], step)
    return float(y.mean()), float(y.std(ddof=1))


def formatted(data, case, step):
    mean, sd = stats(data, case, step)
    return f"{mean:+.2f} ± {sd:.2f}"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, directory, name):
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def delay_panel(ax, data, size, mode, tau, step):
    for method in BASELINES:
        mean, sd = stats(data, (size, mode, method, None, None), step)
        ax.axhline(
            mean, color=COLORS[method], ls="--", label=f"{LABELS[method]} ({mean:+.2f})"
        )
        ax.axhspan(mean - sd, mean + sd, color=COLORS[method], alpha=0.07)
    for method in IQL:
        values = [
            stats(data, (size, mode, method, d, tau), step) for d in CONFIG["delays"]
        ]
        ax.errorbar(
            CONFIG["delays"],
            [v[0] for v in values],
            yerr=[v[1] for v in values],
            color=COLORS[method],
            marker="o",
            capsize=3,
            label=LABELS[method],
        )
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set(
        xscale="log",
        xticks=CONFIG["delays"],
        xlabel="Delay D",
        ylabel="Final max-Q bias",
        title=f"{noise_name(mode)}; tau={tau}",
    )
    ax.set_xticklabels(CONFIG["delays"])
    ax.grid(axis="y", alpha=0.15)
    ax.legend(fontsize=8, loc="best")


def figures(data, directory):
    for size in CONFIG["sizes"]:
        for step, budget in ((100000, "100k"), (1000000, "1m")):
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
            for i, tau in enumerate(CONFIG["tau_values"]):
                for j, mode in enumerate(CONFIG["noise_modes"]):
                    delay_panel(axes[i, j], data, size, mode, tau, step)
            fig.suptitle(
                f"{size} states / {size} actions: "
                f"bias vs delay D after {step:,} updates\n"
                "gamma=0.999; N=2; 20 seeds; mean ± seed SD; same clean-Q metric"
            )
            save(fig, directory, f"s{size}_a{size}_bias_vs_delay_{budget}")


def latex_tables(data, directory):
    directory = directory / "latex"
    directory.mkdir(parents=True, exist_ok=True)
    labels = dict(LABELS, df_min="DF-IQL (min)", df_mean="DF-IQL (mean)")

    def cell(case, step):
        mean, sd = stats(data, case, step)
        return f"${mean:+.2f}{{\\pm}}{sd:.2f}$"

    for size in CONFIG["sizes"]:
        for step, budget in ((100000, "100k"), (1000000, "1m")):
            name = f"s{size}_a{size}_bias_vs_delay_{budget}"
            updates = f"{step:,}".replace(",", "{,}")
            lines = [
                r"% Requires \usepackage{booktabs,graphicx,wrapfig}",
                r"\begin{wraptable}{r}{0.48\textwidth}",
                r"    \vspace{-1.5em}",
                r"    \centering",
                r"    \caption{",
                f"    Max-$Q$ bias for $S=A={size}$ after ${updates}$ updates",
                f"    ($\\gamma={CONFIG['gamma']}$, "
                f"$N={CONFIG['ensemble_size']}$).",
                r"    Values are mean $\pm$ sample standard deviation across "
                f"{len(CONFIG['seeds'])} seeds; $Q^*=0$.",
                r"    Min Q and Double Q are independent of $D$ and $\tau$.",
                r"    }",
                f"    \\label{{tab:tabular_{name}}}",
                r"    \renewcommand{\arraystretch}{1.04}",
                r"    \setlength{\tabcolsep}{2.5pt}",
                r"    \scriptsize",
                r"    \resizebox{\linewidth}{!}{%",
                r"    \begin{tabular}{@{}lcccc@{}}",
                r"        \toprule",
                "        Method & "
                + " & ".join(f"$D={d}$" for d in CONFIG["delays"])
                + r" \\",
            ]
            for mode in CONFIG["noise_modes"]:
                for tau in CONFIG["tau_values"]:
                    lines.extend(
                        [
                            r"        \midrule",
                            r"        \multicolumn{5}{@{}l}{"
                            + f"{noise_name(mode)}, $\\tau={tau:g}$"
                            + r"} \\",
                        ]
                    )
                    for method in METHODS:
                        if method in BASELINES:
                            values = (
                                r"\multicolumn{4}{c}{"
                                + cell((size, mode, method, None, None), step)
                                + "}"
                            )
                        else:
                            values = " & ".join(
                                cell((size, mode, method, d, tau), step)
                                for d in CONFIG["delays"]
                            )
                        lines.append(f"        {labels[method]} & {values}" + r" \\")
            lines.extend(
                [
                    r"        \bottomrule",
                    r"    \end{tabular}%",
                    r"    }",
                    r"    \vspace{-20pt}",
                    r"\end{wraptable}",
                ]
            )
            (directory / f"{name}.tex").write_text("\n".join(lines) + "\n")


def tables(data, directory):
    results, comparison, per_seed = [], [], []
    for case, h in data.items():
        size, mode, method, d, tau = case
        base = dict(
            states=size,
            actions=size,
            noise=noise_name(mode),
            method=LABELS[method],
            D=d if d is not None else "",
            tau=tau if tau is not None else "",
        )
        b0, b1 = bias(h, 100000), bias(h, 1000000)
        q = h["q_final"].mean(1)
        row = dict(
            base,
            bias_mean=float(b1.mean()),
            bias_sd=float(b1.std(ddof=1)),
            mean_q_bias=float(q.mean()),
            q_rmse=float(np.sqrt((q * q).mean((1, 2))).mean()),
            learned_v_bias=float(h["value_final"].mean()) if method in IQL else "",
            seeds=20,
            updates=1000000,
        )
        results.append(row)
        comparison.append(
            dict(
                base,
                bias_100k_mean=float(b0.mean()),
                bias_100k_sd=float(b0.std(ddof=1)),
                bias_1m_mean=float(b1.mean()),
                bias_1m_sd=float(b1.std(ddof=1)),
                bias_change=float((b1 - b0).mean()),
                abs_mean_bias_change=float(abs(b1.mean()) - abs(b0.mean())),
            )
        )
        for i, seed in enumerate(CONFIG["seeds"]):
            per_seed.append(
                dict(
                    base,
                    seed=seed,
                    bias_100k=float(b0[i]),
                    bias_1m=float(b1[i]),
                    bias_change=float(b1[i] - b0[i]),
                )
            )
    assert len(results) == 272 and len(per_seed) == 5440
    write_csv(directory / "results_1m.csv", results)
    write_csv(directory / "comparison_100k_vs_1m.csv", comparison)
    write_csv(directory / "per_seed.csv", per_seed)
    workbook = Workbook()
    workbook.remove(workbook.active)
    sheets = []
    for size in CONFIG["sizes"]:
        rows = []
        for mode in CONFIG["noise_modes"]:
            for tau in CONFIG["tau_values"]:
                for d in CONFIG["delays"]:
                    row = dict(noise=noise_name(mode), tau=tau, D=d)
                    for method in METHODS:
                        case = (
                            (size, mode, method, None, None)
                            if method in BASELINES
                            else (size, mode, method, d, tau)
                        )
                        row[f"{LABELS[method]} 100k"] = formatted(data, case, 100000)
                        row[f"{LABELS[method]} 1M"] = formatted(data, case, 1000000)
                    rows.append(row)
        write_csv(directory / f"s{size}_a{size}_comparison.csv", rows)
        sheets.append((f"S{size}_A{size}", rows))
    sheets += [
        ("Results_1M", results),
        ("Comparison", comparison),
        ("Per_seed", per_seed),
    ]
    for name, rows in sheets:
        sheet = workbook.create_sheet(name)
        columns = list(rows[0])
        sheet.append(columns)
        for row in rows:
            sheet.append([row[k] for k in columns])
    sheet = workbook.create_sheet("Definitions")
    sheet.append(["Key", "Value"])
    sheet.append(
        [
            "Metric",
            "mean_s max_a mean_i clean Q_i(s,a); Q*=V*=0. Learned V is separate.",
        ]
    )
    sheet.append(
        [
            "Statistics",
            "Mean and sample SD across 20 matched seeds, not standard error.",
        ]
    )
    sheet.append(
        [
            "100k source",
            "Checkpoint Q tables from the same 1M NPZ; "
            "no other log folder is required.",
        ]
    )
    sheet.append(
        [
            "Scope",
            "Six algorithms, 272 conditions, 5440 seed trajectories; "
            "S=A; fixed total updates.",
        ]
    )
    sheet.append(
        [
            "Interpretation",
            "A longer finite-training checkpoint does not imply convergence; "
            "visits per entry differ by size.",
        ]
    )
    for k, v in CONFIG.items():
        sheet.append([k, json.dumps(v, ensure_ascii=False)])
    for sheet in workbook:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="24506A")
        for col in sheet.columns:
            sheet.column_dimensions[col[0].column_letter].width = min(
                35, max(12, max(len(str(c.value)) for c in col) + 2)
            )
    workbook.save(directory / "tabular_bias_1m.xlsx")
    latex_tables(data, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--check-only", action="store_true")
    outputs.add_argument("--latex-only", action="store_true")
    args = parser.parse_args()
    data, checks = load_results(args.results_dir)
    if args.latex_only:
        latex_tables(data, args.results_dir / "tables")
    elif not args.check_only:
        for name in ("figures", "tables", "provenance"):
            (args.results_dir / name).mkdir(parents=True, exist_ok=True)
        figures(data, args.results_dir / "figures")
        tables(data, args.results_dir / "tables")
        report = dict(
            conditions=272,
            seeds=20,
            seed_trajectories=5440,
            updates=1000000,
            algorithms=list(LABELS.values()),
            raw_hashes_verified=True,
            max_metric_error=max(c["max_metric_error"] for c in checks),
            requires_legacy_logs=False,
            checks=checks,
        )
        (args.results_dir / "provenance/validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(
        json.dumps(
            dict(
                conditions=len(data),
                max_metric_error=max(c["max_metric_error"] for c in checks),
                results_dir=str(args.results_dir.resolve()),
                checked_only=args.check_only,
                latex_only=args.latex_only,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
