"""Generate and verify LaTeX tables for max-Q bias, filtering bias and coupling."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from common import CODE_DIR, CONFIG, DEFAULT_RESULTS_DIR

BASELINES = ("max_min", "double_q")
METHODS = BASELINES + ("df_min", "sd_iql", "dd_iql")
LABELS = {
    "max_min": "Clipped Double Q",
    "double_q": "Double Q",
    "df_min": "DF-IQL",
    "sd_iql": "SD-IQL",
    "dd_iql": "DD-IQL",
}
METRICS = {
    "max_q_bias": ("final_bias_summary.csv", "max_q_bias", 2, True),
    "filter_bias": ("last100k_summary.csv", "operator_bias", 2, True),
    "coupling_abs": ("last100k_summary.csv", "readout_abs_corr", 3, False),
    "coupling_signed": ("last100k_summary.csv", "readout_signed_corr", 4, True),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def min_q_reference(root):
    """Recompute the unchanged clipped baseline from its verified 20-seed Q tables."""
    previous = root.parent / "tabular_bias_1m"
    previous_config = json.loads((previous / "experiment_config.json").read_text())
    for key in (
        "sizes",
        "seeds",
        "updates",
        "noise_modes",
        "gamma",
        "alpha_q",
        "sigma",
        "ensemble_size",
        "field_rho",
        "field_features",
        "field_lengthscale",
        "field_action_scale",
        "shared_fraction",
    ):
        assert previous_config[key] == CONFIG[key], key
    previous_summary = read(previous / "tables/results_1m.csv")
    rows, sources = [], []
    for size in CONFIG["sizes"]:
        for mode in CONFIG["noise_modes"]:
            noise = "IID" if mode.endswith("_iid") else "NN-like field"
            path = previous / "raw" / f"s{size}_a{size}" / mode / "baselines/max_min.npz"
            meta = json.loads(path.with_suffix(".json").read_text())
            assert meta["num_states"] == size and meta["num_actions"] == size
            assert meta["method"] == "max_min" and meta["noise_mode"] == mode
            assert meta["seed_ids"] == CONFIG["seeds"]
            assert meta["updates"] == CONFIG["updates"]
            assert meta["gamma"] == CONFIG["gamma"]
            assert meta["alpha_q"] == CONFIG["alpha_q"]
            sha = digest(path)
            assert sha == meta["data_sha256"], path
            with np.load(path) as data:
                q = data["q_final"]
            assert q.shape == (20, 2, size, size)
            bias = q.mean(1).max(-1).mean(-1)
            mean, sd = float(bias.mean()), float(bias.std(ddof=1))
            match = next(
                r
                for r in previous_summary
                if int(r["states"]) == size
                and r["noise"] == noise
                and r["method"] == "Min Q"
            )
            np.testing.assert_allclose(
                [mean, sd],
                [float(match["bias_mean"]), float(match["bias_sd"])],
                atol=1e-12,
                rtol=0,
            )
            rows.append(
                dict(
                    states=size,
                    actions=size,
                    noise=noise,
                    method="max_min",
                    D="",
                    tau="",
                    seeds=20,
                    max_q_bias_mean=mean,
                    max_q_bias_sd=sd,
                    source_npz=str(path),
                    source_sha256=sha,
                )
            )
            sources.append(
                dict(
                    source=str(path),
                    sha256=sha,
                    num_states=size,
                    noise=noise,
                    seed_count=20,
                    bias_mean=mean,
                    bias_sd=sd,
                )
            )
    filename = root / "tables/min_q_baseline_reference.csv"
    with filename.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "provenance/min_q_table_sources.json").write_text(
        json.dumps(
            dict(
                conditions=8,
                original_data_hashes_verified=True,
                all_bias_values_recomputed_from_raw_q=True,
                experiment_settings_match=True,
                source_results=sources,
            ),
            indent=2,
        )
        + "\n"
    )
    return read(filename)


def formatted(mean, sd, digits, signed):
    assert math.isfinite(mean) and math.isfinite(sd) and sd >= 0
    # Avoid printed negative zero when the signed mean rounds to zero.
    rounded = round(mean, digits)
    if rounded == 0:
        rounded = 0.0
    value = format(rounded, f"{'+' if signed else ''}.{digits}f")
    return f"${value}{{\\pm}}{sd:.{digits}f}$"


def caption(metric, size):
    prefix = rf"$S=A={size}$, $\gamma=0.999$, $N=2$. "
    suffix = r"Values are mean $\pm$ sample SD across 20 training seeds."
    if metric == "max_q_bias":
        text = (
            r"Max-$Q$ bias after $10^6$ updates, computed from clean ensemble-mean Q. "
            r"The noise-free reference is zero; "
            r"Clipped Double Q and Double Q are independent of $D$ and $\tau$. "
        )
    elif metric == "filter_bias":
        text = (
            r"Signed filtering-output bias relative to the known $F^\circ=0$. "
            r"Each seed averages 101 pre-update snapshots in the final $10^5$ "
            r"of $10^6$ updates. Clipped Double Q and Double Q "
            r"have no expectile filter (N/A). "
        )
    else:
        kind = "mean absolute" if metric == "coupling_abs" else "mean signed"
        text = (
            f"Readout coupling: {kind} temporal Pearson correlation between "
            r"filter weights and receiver readout errors. "
            r"Each fixed receiver/state/action "
            r"uses 101 snapshots in the final $10^5$ of $10^6$ updates. "
            r"Clipped Double Q and Double Q have no expectile filter (N/A). "
        )
    return prefix + text + suffix


def generate_table(metric, size, rows, relative_path):
    _, field, digits, signed = METRICS[metric]
    lines = [
        r"% Requires \usepackage{booktabs,graphicx,wrapfig}",
        "% See metric_definitions.tex and README.md for the precise estimands.",
        r"\begin{wraptable}{r}{0.48\textwidth}",
        r"    \vspace{-1.5em}",
        r"    \centering",
        r"    \caption{" + caption(metric, size) + "}",
        rf"    \label{{tab:refresh_first_s{size}_a{size}_{metric}_1m}}",
        r"    \renewcommand{\arraystretch}{1.04}",
        r"    \setlength{\tabcolsep}{2.5pt}",
        r"    \scriptsize",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{@{}lcccc@{}}",
        r"        \toprule",
        r"        Method & $D=1$ & $D=10$ & $D=100$ & $D=1000$ \\",
    ]
    cells = []
    for noise in ("IID", "NN-like field"):
        for tau in CONFIG["tau_values"]:
            lines += [
                r"        \midrule",
                rf"        \multicolumn{{5}}{{@{{}}l}}{{{noise}, $\tau={tau}$}} \\",
            ]
            for method in METHODS:
                if method in BASELINES and metric != "max_q_bias":
                    lines.append(
                        rf"        {LABELS[method]} & \multicolumn{{4}}{{c}}{{N/A}} \\",
                    )
                    continue
                delays = (None,) if method in BASELINES else CONFIG["delays"]
                values = []
                for delay in delays:
                    matches = [
                        r
                        for r in rows
                        if int(r["states"]) == size
                        and int(r["actions"]) == size
                        and r["noise"] == noise
                        and r["method"] == method
                        and (
                            method in BASELINES
                            or (int(r["D"]) == delay and float(r["tau"]) == tau)
                        )
                    ]
                    assert len(matches) == 1, (metric, size, noise, tau, method, delay)
                    row = matches[0]
                    assert int(row["seeds"]) == 20
                    if metric != "max_q_bias":
                        assert row["window"] == "900000_1000000"
                        assert int(row["diagnostic_samples_per_seed"]) == 101
                    mean, sd = float(row[field + "_mean"]), float(row[field + "_sd"])
                    value = formatted(mean, sd, digits, signed)
                    values.append(value)
                    cells.append(
                        dict(
                            metric=metric,
                            states=size,
                            noise=noise,
                            tau=tau,
                            method=method,
                            D=delay,
                            mean=mean,
                            sd=sd,
                            rendered=value,
                            file=relative_path,
                            source_csv=(
                                "min_q_baseline_reference.csv"
                                if method == "max_min"
                                else METRICS[metric][0]
                            ),
                        )
                    )
                if method in BASELINES:
                    content = r"\multicolumn{4}{c}{" + values[0] + "}"
                else:
                    content = " & ".join(values)
                lines.append(rf"        {LABELS[method]} & {content} \\")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
    ]
    if metric == "coupling_abs":
        lines += [
            r"    \par\smallskip",
            r"    \parbox{\linewidth}{\scriptsize Under independence, 101 samples can "
            r"yield mean absolute sample correlation near $0.08$. "
            r"See the signed-correlation companion table.}",
        ]
    lines += [r"    \vspace{-20pt}", r"\end{wraptable}", ""]
    content = "\n".join(lines)
    expected = Counter(c["rendered"] for c in cells)
    found = Counter(re.findall(r"\$[+-]?\d+\.\d+\{\\pm\}\d+\.\d+\$", content))
    assert found == expected, relative_path
    assert content.count(r"\multicolumn{5}") == 4
    assert content.count("{c}{N/A}") == (0 if metric == "max_q_bias" else 8)
    return content, cells


DEFINITIONS = r"""% Requires \usepackage{amsmath}
\paragraph{Measurement definitions.}
All rewards are zero and there are no terminal states, so the known noise-free
references are $Q^\circ=V^\circ=F^\circ=0$ for every state/action.
For one seed, the final learned-value metric is
\[
 B_Q=\frac{1}{S}\sum_s\max_a\left[\frac{1}{2}\sum_{i=1}^2 Q_{i,10^6}(s,a)\right].
\]
These clean learned tables exclude the injected readout noise.
For IQL methods, the pre-update diagnostic output is
\[
 F_{i,t}(s)=\frac{\sum_a w_{i,t}(s,a)Y_{i,t}(s,a)}
                         {\sum_a w_{i,t}(s,a)}.
\]
Its signed bias $B_F$ averages $F_{i,t}(s)-F^\circ(s)$ over receivers, states and
101 sampled times in the last $10^5$ updates, within each seed.
SD/DD use $Y_i=\bar Q_i+\epsilon_i$; DF uses one shared receiver
$Y=\min_i(\bar Q_i+\epsilon_i)$.
This normalized diagnostic does not replace the expectile gradient update.
The measured readout error is $e_i^{\rm ro}=\epsilon_i$ for SD/DD and
$e^{\rm ro}=Y-\min_i\bar Q_i$ for DF; DF's readout error need not have mean zero.
For each fixed receiver/state/action, let
\[
 r_{i,s,a}=\widehat{\mathrm{Corr}}_{t\in\mathcal T}
               \bigl(w_{i,t}(s,a),e^{\rm ro}_{i,t}(s,a)\bigr).
\]
The main coupling table reports the within-seed mean $C_{\rm abs}$ of
$|r_{i,s,a}|$; supplementary tables report the mean signed correlation
$C_{\rm signed}$. These are temporal correlations, not correlations across settings.
Undefined correlations from constant weights/errors are excluded; valid-unit
fractions remain available in the CSV diagnostics.
All tables then report mean $\pm$ sample SD across 20 training seeds.
At 101 observations, independent variables can still yield mean $|r|$ near 0.08.
\par
Clipped Double Q uses bootstrap
$\max_a\min_i[Q_i(s,a)+\epsilon_i(s,a)]$; Double Q uses cross-head evaluation.
Neither baseline has an expectile filter. Both have a learned max-$Q$ bias, but
their filter bias and filter coupling are N/A. Their backup operators are not
inserted into the filtering-bias table. Clipped Double Q uses existing runs with the same
MDP, noise, seeds, discount, learning rate and update count; raw Q tables and
hashes are verified before generating its table cells. Small readout coupling does not
guarantee small accumulated learned-value error. RMSE, over/under magnitudes,
and other error-proxy diagnostics remain available in the CSV results.
"""


def compile_preview(directory, paths):
    preview = directory / "preview"
    preview.mkdir(exist_ok=True)
    parts = [
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,booktabs,graphicx,wrapfig}",
        r"\begin{document}",
        DEFINITIONS,
        r"\clearpage",
    ]
    for path in paths:
        content = (
            path.read_text()
            .replace(
                r"\begin{wraptable}{r}{0.48\textwidth}",
                r"\begin{table}[p]\centering\begin{minipage}{0.48\textwidth}",
            )
            .replace(
                r"\end{wraptable}",
                r"\end{minipage}\end{table}\clearpage",
            )
        )
        parts.append(content)
    parts += [r"\end{document}", ""]
    tex = preview / "preview.tex"
    tex.write_text("\n".join(parts))
    with tempfile.TemporaryDirectory(prefix="tabular_metric_tables_") as temporary:
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                temporary,
                str(tex.resolve()),
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        shutil.copy2(Path(temporary) / "preview.pdf", preview / "preview.pdf")
        log = (Path(temporary) / "preview.log").read_text()
    (preview / "compile.log").write_text(result.stdout + result.stderr)
    assert "Overfull" not in log, "Inspect preview/compile.log for layout overflow"
    return dict(compiled=True, overfull_boxes=0, pdf=str(preview / "preview.pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--compile-preview", action="store_true")
    args = parser.parse_args()
    root = args.results_dir
    directory = root / "tables/latex"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "supplementary").mkdir(exist_ok=True)
    sources = {
        filename: read(root / "tables" / filename)
        for filename in {x[0] for x in METRICS.values()}
    }
    min_rows = min_q_reference(root)
    sources[METRICS["max_q_bias"][0]] += min_rows
    previous_validation = root / "provenance/latex_tables_validation.json"
    previous_cells = (
        json.loads(previous_validation.read_text())["cells"]
        if previous_validation.exists()
        else []
    )
    paths, cells = [], []
    for metric, (filename, _, _, _) in METRICS.items():
        for size in CONFIG["sizes"]:
            prefix = "supplementary/" if metric == "coupling_signed" else ""
            relative = f"{prefix}s{size}_a{size}_{metric}_1m.tex"
            content, values = generate_table(metric, size, sources[filename], relative)
            path = directory / relative
            path.write_text(content)
            paths.append(path)
            cells.extend(values)
    assert len(paths) == 16 and len(cells) == 800

    def cell_key(cell):
        return tuple(
            cell[k] for k in ("metric", "states", "noise", "tau", "method", "D")
        )

    current = {cell_key(cell): cell for cell in cells}
    for old in previous_cells:
        fresh = current[cell_key(old)]
        assert (old["mean"], old["sd"], old["rendered"]) == (
            fresh["mean"],
            fresh["sd"],
            fresh["rendered"],
        )
    (directory / "metric_definitions.tex").write_text(DEFINITIONS)
    lines = [
        "# LaTeX tables: bias, filtering bias, coupling",
        "",
        "주요 지표 3개 × S=A=2,4,8,16 = 12개 표입니다. "
        "Signed coupling 보조표 4개도 포함합니다.",
        "",
        "| S=A | 최종 max-Q bias | Filter bias "
        "| Coupling (mean absolute r) | Signed coupling |",
        "|---:|---|---|---|---|",
    ]
    for size in CONFIG["sizes"]:
        stem = f"s{size}_a{size}"
        lines.append(
            f"| {size} | [TeX]({stem}_max_q_bias_1m.tex) "
            f"| [TeX]({stem}_filter_bias_1m.tex) "
            f"| [TeX]({stem}_coupling_abs_1m.tex) "
            f"| [TeX](supplementary/{stem}_coupling_signed_1m.tex) |"
        )
    lines += [
        "",
        "각 표는 IID/NN-like × τ=0.9/0.99의 네 블록으로 구성합니다.",
        "Clipped Double Q와 Double Q의 max-Q bias는 "
        "D·τ와 무관하므로 네 열을 합쳤습니다.",
        "두 baseline의 expectile filter bias와 filter coupling은 N/A입니다.",
        "Clipped Double Q는 기존 동일 조건의 원시 Q·해시를 검증한 "
        "8개 결과를 재사용합니다.",
        "[Clipped Double Q 원본 대조 CSV](../min_q_baseline_reference.csv)",
        "",
        "Bias는 최종 1M clean max-Q이고, filter bias와 coupling은 "
        "마지막 100k 구간의 101개 업데이트 직전 시점에서 계산합니다.",
        "평균 ± 표본 표준편차는 20 seed 기준입니다. "
        "결합 지표는 시간축 Pearson correlation이며, "
        "주입 readout 오차에 대한 것입니다.",
        "평균 절대 correlation에는 독립인 경우에도 "
        "약 0.08의 표본 잡음이 있을 수 있습니다.",
        "Signed correlation은 보조표에서 확인할 수 있습니다. "
        "RMSE와 과대/과소평가 크기는 "
        "기존 CSV에 별도로 보존돼 있습니다.",
        "",
        "[정확한 지표 정의](metric_definitions.tex) · "
        "[PDF 미리보기](preview/preview.pdf)",
        "",
        "필요 패키지: `booktabs`, `graphicx`, `wrapfig`. "
        "지표 정의 파일은 `amsmath`도 사용합니다.",
        "",
        "재생성:",
        "",
        "```bash",
        "python utils/iclr2027_dd_iql/rebuttal/"
        "tabular_bias_refresh_first_sweep_1m/generate_latex_tables.py --compile-preview",
        "```",
        "",
    ]
    (directory / "README.md").write_text("\n".join(lines))
    validation = dict(
        primary_tables=12,
        supplementary_signed_coupling_tables=4,
        numeric_mean_sd_pairs_verified=len(cells),
        all_numeric_cells_match_csv=True,
        previous_numeric_cells_unchanged=len(previous_cells),
        min_q_source_conditions=8,
        sources={
            name: digest(root / "tables" / name)
            for name in (*sources, "min_q_baseline_reference.csv")
        },
        generated_files=[str(p.relative_to(root)) for p in paths],
        cells=cells,
    )
    if args.compile_preview:
        validation["preview"] = compile_preview(directory, paths)
    provenance = root / "provenance"
    (provenance / "latex_tables_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    marker = "\n## LaTeX metric tables\n"
    readme = root / "README.md"
    body, present, remainder = readme.read_text().partition(marker)
    following_sections = ""
    if present and "\n## " in remainder:
        following_sections = "\n## " + remainder.split("\n## ", 1)[1]
    readme.write_text(
        body + marker + "\n[Bias·filter bias·coupling LaTeX 표]"
        "(tables/latex/README.md): 주 표 12개와 signed coupling 보조표 4개입니다.\n"
        "표에는 동일 조건의 기존 Clipped Double Q baseline 8개도 추가했습니다. "
        "[원본 대조 CSV](tables/min_q_baseline_reference.csv)에서 확인할 수 있습니다.\n"
        + following_sections
    )
    snapshot = provenance / "code_snapshot/generate_latex_tables.py"
    shutil.copy2(Path(__file__), snapshot)
    manifest_path = provenance / "code_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generate_latex_tables.py"] = digest(CODE_DIR / "generate_latex_tables.py")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    artifacts = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and "raw" not in p.relative_to(root).parts
        and p.name != "artifact_hashes.json"
    }
    (provenance / "artifact_hashes.json").write_text(
        json.dumps(artifacts, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in validation.items() if k != "cells"}))


if __name__ == "__main__":
    main()
