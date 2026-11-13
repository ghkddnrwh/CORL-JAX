"""Refresh cached figures and create twelve title-free S=4 metric figures."""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import CODE_DIR, CONFIG, DEFAULT_RESULTS_DIR
from figure_presentation import (
    coupling_scatter,
    individual_plots,
    panel_plot,
    read_rows,
)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_size(args):
    root, size = args
    tables = root / "tables" / f"s{size}_a{size}"
    final = read_rows(tables / "final_bias_summary.csv")
    checkpoint = read_rows(tables / "checkpoint_100k_summary.csv")
    diagnostic = read_rows(tables / "last100k_summary.csv")
    primary = root / "figures"
    additional = root / "diagnostics_figures" / f"s{size}_a{size}"
    primary.mkdir(exist_ok=True)
    additional.mkdir(parents=True, exist_ok=True)
    files = []
    for name, rows, short in (("100k", checkpoint, True), ("1m", final, False)):
        filename = f"s{size}_a{size}_bias_vs_delay_{name}.png"
        panel_plot(
            rows,
            "max_q_bias",
            "Max-Q bias",
            filename,
            primary,
            f"after {name} updates",
            size,
            root,
            checkpoint=short,
        )
        files.append(str((primary / filename).relative_to(root)))
    for rows, metric, label, filename, scope in (
        (
            final,
            "max_q_bias",
            "Max-Q bias",
            f"s{size}_a{size}_bias_vs_delay_1m.png",
            "after 1M updates",
        ),
        (
            diagnostic,
            "operator_bias",
            "Filtering signed bias",
            "operator_bias_vs_delay.png",
            "last 100k pre-update diagnostics",
        ),
        (
            diagnostic,
            "operator_rmse",
            "Filtering RMSE",
            "operator_rmse_vs_delay.png",
            "last 100k pre-update diagnostics",
        ),
        (
            diagnostic,
            "readout_abs_corr",
            "Mean |corr(weight, readout error)|",
            "readout_coupling_vs_delay.png",
            "last 100k, 101 samples per fixed unit",
        ),
    ):
        panel_plot(rows, metric, label, filename, additional, scope, size, root)
        files.append(str((additional / filename).relative_to(root)))
    for cross in (False, True):
        coupling_scatter(diagnostic, additional, size, cross)
        filename = (
            "coupling_cross_section_vs_filtering_accuracy.png"
            if cross
            else "coupling_vs_filtering_accuracy.png"
        )
        files.append(str((additional / filename).relative_to(root)))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.results_dir
    inputs = [
        root / "tables" / name
        for name in (
            "final_bias_summary.csv",
            "checkpoint_100k_summary.csv",
            "last100k_summary.csv",
            "min_q_baseline_reference.csv",
        )
    ]
    before = {str(p.relative_to(root)): digest(p) for p in inputs}
    raw_manifest_hash = digest(root / "provenance/raw_manifest.json")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        updated = [
            path
            for files in pool.map(render_size, [(root, s) for s in CONFIG["sizes"]])
            for path in files
        ]
    records = individual_plots(
        read_rows(root / "tables/s4_a4/final_bias_summary.csv"),
        read_rows(root / "tables/s4_a4/last100k_summary.csv"),
        root,
    )
    assert before == {str(p.relative_to(root)): digest(p) for p in inputs}
    assert digest(root / "provenance/raw_manifest.json") == raw_manifest_hash
    lines = [
        "# 4-state individual figures",
        "",
        "제목 없는 단일 패널 12개입니다.",
        "범례: DF-IQL, SD-IQL, DD-IQL.",
        "Clipped Double Q와 Double Q는 개별 그림에서 제외했습니다.",
        "",
        "Coupling은 Readout × Temporal × Absolute입니다.",
        "Y축: Q-bias / Bias against noise-free filtering target / Coupling.",
        "세 IQL 방법에 mean ± seed SD 오차막대를 표시합니다.",
        "",
        "| Noise | τ | Q-bias | Bias against noise-free filtering target | Coupling |",
        "|---|---:|---|---|---|",
    ]
    for noise in ("IID", "NN-like field"):
        for tau in CONFIG["tau_values"]:
            selected = [r for r in records if r["noise"] == noise and r["tau"] == tau]
            links = [f"[PNG]({Path(r['file']).name})" for r in selected]
            lines.append(f"| {noise} | {tau} | " + " | ".join(links) + " |")
    (root / "figures_s4_individual/README.md").write_text("\n".join(lines) + "\n")
    readme = root / "README.md"
    marker = "\n## 4-state individual figures\n"
    body = readme.read_text().split(marker)[0]
    body = body.replace("DF-IQL (min)", "DF-IQL").replace("DF-min", "DF-IQL")
    body = body.replace("Min Q(Clipped Double Q)", "Clipped Double Q").replace(
        "Min Q", "Clipped Double Q"
    )
    body = body.replace(
        "# Current-first size sweep: Double Q, DF-IQL, SD, DD",
        "# Current-first size sweep: Clipped Double Q, Double Q, DF-IQL, SD, DD",
    )
    baseline_note = (
        "Clipped Double Q는 같은 조건으로 실행한 기존 baseline 결과를 사용합니다. "
        "회색 수평선은 Clipped Double Q이며, filter bias와 coupling은 N/A입니다.\n"
    )
    if baseline_note not in body:
        body = body.replace("## 크기별 그래프\n", baseline_note + "\n## 크기별 그래프\n")
    readme.write_text(
        body + marker + "\n[제목 없는 개별 그림 12개](figures_s4_individual/README.md). "
        "DF-IQL·SD-IQL·DD-IQL만 표시하며 "
        "coupling은 Readout × Temporal × Absolute입니다.\n"
    )
    report = dict(
        updated_existing_figures=updated,
        additional_figures=len(records),
        coupling="readout_temporal_absolute",
        no_titles_verified=True,
        one_axis_per_additional_figure=True,
        double_q_excluded_from_additional=True,
        clipped_double_q_excluded_from_additional=True,
        baseline_filter_metrics="not_applicable",
        input_summary_hashes=before,
        numerical_summaries_unchanged=True,
        raw_manifest_unchanged=True,
        individual_figures=records,
    )
    provenance = root / "provenance"
    (provenance / "figure_presentation_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    source_manifest = {}
    for path in CODE_DIR.iterdir():
        if path.is_file() and path.suffix in (".py", ".json", ".txt", ".md"):
            shutil.copy2(path, provenance / "code_snapshot" / path.name)
            source_manifest[path.name] = digest(path)
    (provenance / "code_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )
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
    print(
        json.dumps(
            dict(
                updated_existing_figures=len(updated),
                additional_figures=len(records),
                individual_labels=["DF-IQL", "SD-IQL", "DD-IQL"],
                numerical_summaries_unchanged=True,
            )
        )
    )


if __name__ == "__main__":
    main()
