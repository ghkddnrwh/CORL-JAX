"""Collect the four completed size reports into tables, README and manifests."""

import csv
import datetime
import hashlib
import json
import shutil

from common import CODE_DIR, CONFIG, DEFAULT_RESULTS_DIR, LABELS

def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formatted(row):
    return f"{float(row['max_q_bias_mean']):+.2f} ± {float(row['max_q_bias_sd']):.2f}"


def main():
    root = DEFAULT_RESULTS_DIR
    verification = json.loads(
        (root / "provenance/independent_validation.json").read_text()
    )
    assert verification["conditions"] == 200
    assert verification["original_double_q_conditions_bitwise_equal"] == 8
    lines = [
        "# Current-first size sweep: Double Q, DF-IQL, SD, DD",
        "",
        "S=A=2,4,8,16, 1M updates, 20 seeds (200–219). "
        "두 noise, τ=0.9/0.99, D=1/10/100/1000입니다.",
        "총 200조건·4000개 seed 결과입니다. 완료된 S=4 IQL 48조건은 재사용하고, "
        "152조건·3040개 seed를 추가 실행했습니다.",
        "기존 코드와 로그는 보존했습니다. Double Q 8조건은 이번에 새로 실행했으며 "
        "기존 baseline과 비트 단위로 일치합니다.",
        "",
        "Filter는 블록 첫 업데이트 직전에 현재 target-Q+noise와 V로 만들고 "
        "D회 사용합니다. 시차는 0..D−1입니다.",
        "Double Q는 expectile filter/V가 없고 D·τ와 무관합니다. "
        "그래프의 파란 수평선이 Double Q입니다.",
        "",
        "## 크기별 그래프",
        "",
        "| S=A | 100k | 1M | 전체 수치 |",
        "|---:|---|---|---|",
    ]
    for size in CONFIG["sizes"]:
        tag = f"s{size}_a{size}"
        lines.append(
            f"| {size} | [그래프](figures/{tag}_bias_vs_delay_100k.png) "
            f"| [그래프](figures/{tag}_bias_vs_delay_1m.png) "
            f"| [CSV](tables/{tag}/final_bias_summary.csv) |"
        )
    lines += [
        "",
        "최종 bias는 모든 알고리즘에서 mean_s max_a mean_i clean Q_i(s,a)입니다. "
        "참값은 0입니다.",
        "평균 ± 표본 표준편차는 20 seed 기준입니다. "
        "상태/행동 수에 따라 항목당 방문 횟수가 "
        "달라 1M을 수렴값으로 가정하지 않습니다.",
        "",
        "- [통합 최종 수치](tables/final_bias_summary.csv)",
        "- [통합 filtering/Q/V 마지막 100k 수치](tables/last100k_summary.csv)",
        "- [통합 readout/proxy coupling calibration]"
        "(tables/proxy_accuracy_tracking_summary.csv)",
        "- [이전 방식과 같은 max-Q 기준 비교](tables/old_vs_new_final_max_q.csv)",
        "- [검증 결과](provenance/independent_validation.json)",
        "",
    ]
    for size in CONFIG["sizes"]:
        tag = f"s{size}_a{size}"
        rows = read(root / f"tables/{tag}/final_bias_summary.csv")
        assert len(rows) == 50
        lines += [
            f"## S=A={size}: 최종 max-Q bias",
            "",
            f"[Filtering bias](diagnostics_figures/{tag}/operator_bias_vs_delay.png) · "
            f"[Filtering RMSE](diagnostics_figures/{tag}/operator_rmse_vs_delay.png) · "
            "[Readout correlation]"
            f"(diagnostics_figures/{tag}/readout_coupling_vs_delay.png)",
            "",
        ]
        for noise in ("IID", "NN-like field"):
            for tau in (0.9, 0.99):
                baseline = next(
                    r for r in rows if r["method"] == "double_q" and r["noise"] == noise
                )
                lines += [
                    f"### {noise}, τ={tau}",
                    "",
                    f"Double Q: **{formatted(baseline)}** (모든 D·τ에 공통)",
                    "",
                    "| Method | D=1 | D=10 | D=100 | D=1000 |",
                    "|---|---:|---:|---:|---:|",
                ]
                for method in ("df_min", "sd_iql", "dd_iql"):
                    values = [
                        formatted(
                            next(
                                r
                                for r in rows
                                if r["noise"] == noise
                                and r["method"] == method
                                and float(r["tau"]) == tau
                                and int(r["D"]) == delay
                            )
                        )
                        for delay in CONFIG["delays"]
                    ]
                    lines.append(
                        "| " + LABELS[method] + " | " + " | ".join(values) + " |"
                    )
                lines.append("")
    lines += [
        "## 진단 해석",
        "",
        "Filtering F=sum(wY)/sum(w)는 진단용이며 "
        "실제 expectile SGD 업데이트와 구분합니다.",
        "Double Q의 operator는 head별 cross-evaluation이고 "
        "expectile filtering이 아닙니다. "
        "Filter coupling은 N/A입니다.",
        "마지막 100k의 sparse 진단은 unit당 시점 101개입니다. "
        "독립인 경우에도 평균 |corr|가 약 0.08일 수 있습니다.",
        "Signed correlation과 covariance를 함께 기록했습니다. "
        "주입 noise와의 coupling과 누적된 Q/V 오차는 별도로 해석해야 합니다.",
        "",
        "코드: [README](../../../../utils/iclr2027_dd_iql/rebuttal/"
        "tabular_bias_refresh_first_sweep_1m/README.md)",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    for filename in (
        "final_bias_summary.csv",
        "last100k_summary.csv",
        "checkpoint_100k_summary.csv",
        "old_vs_new_final_max_q.csv",
        "proxy_accuracy_tracking_summary.csv",
    ):
        rows = []
        for size in CONFIG["sizes"]:
            rows.extend(
                dict(states=size, actions=size, **r)
                for r in read(root / f"tables/s{size}_a{size}/{filename}")
            )
        with (root / "tables" / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    raw = [
        dict(
            path=str(p.relative_to(root)),
            sha256=digest(p),
            metadata_sha256=digest(p.with_suffix(".json")),
        )
        for p in sorted((root / "raw").rglob("*.npz"))
    ]
    assert len(raw) == 200
    (root / "provenance/raw_manifest.json").write_text(json.dumps(raw, indent=2) + "\n")
    # Add this postprocessing source to the run's source snapshot.
    snapshot = root / "provenance/code_snapshot"
    manifest = json.loads((root / "provenance/code_manifest.json").read_text())
    shutil.copy2(CODE_DIR / "summarize_results.py", snapshot / "summarize_results.py")
    manifest["summarize_results.py"] = digest(CODE_DIR / "summarize_results.py")
    (root / "provenance/code_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    completion = dict(
        completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        conditions=200,
        seed_trajectories=4000,
        new_conditions=152,
        new_seed_trajectories=3040,
        reused_current_first_s4_conditions=48,
        updates_per_seed=1000000,
        double_q_conditions=8,
        primary_figures=len(list((root / "figures").glob("*.png"))),
        independent_validation_passed=True,
        snapshot_timing=CONFIG["snapshot_timing"],
    )
    (root / "provenance/completion.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    hashes = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and "raw" not in p.relative_to(root).parts
        and p.name != "artifact_hashes.json"
    }
    (root / "provenance/artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    print(json.dumps(completion))


if __name__ == "__main__":
    main()
