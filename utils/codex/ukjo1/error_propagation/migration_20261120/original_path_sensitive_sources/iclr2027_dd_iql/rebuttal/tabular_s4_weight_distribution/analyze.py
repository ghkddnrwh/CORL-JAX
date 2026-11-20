"""Read-only Q/weight distribution diagnostics for the saved 4-state experiment.

No training or RNG calls. Counterfactuals hold recorded V, mean clean Q and
readout error fixed and remove only within-state clean-Q action deviations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sizes_1m"
OUTPUT = ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_s4_weight_distribution"
METHODS = ("df_min", "sd_iql", "dd_iql")
LABELS = {"df_min": "DF-IQL", "sd_iql": "SD-IQL", "dd_iql": "DD-IQL"}
COLORS = {"df_min": "#9467bd", "sd_iql": "#ff7f0e", "dd_iql": "#1f77b4"}
NOISES = {"nn_like_field_iid": "IID", "nn_like_field": "NN-like field"}
SEEDS = np.arange(200, 220)
TIME_METRICS = (
    "mean_weight",
    "high_weight_fraction",
    "clean_q_mean",
    "source_v_mean",
    "clean_action_sd",
    "readout_action_sd",
    "clean_upper_gap",
    "v_minus_clean_mean",
    "zero_high_state_fraction",
    "state_ess_fraction",
    "flat_clean_mean_weight",
    "flat_clean_gate_flip_fraction",
)


def load_case(path, method, delay, tau):
    with np.load(path, allow_pickle=False) as data:
        h = {
            key: data[key]
            for key in (
                "diagnostic_step",
                "diagnostic_target_q",
                "diagnostic_noise",
                "diagnostic_delayed_observed_q",
                "diagnostic_delayed_noise",
                "diagnostic_delayed_value",
                "diagnostic_weights",
                "diagnostic_used_weight",
                "diagnostic_state",
                "diagnostic_action",
            )
        }
    steps = h["diagnostic_step"]
    assert steps[-1] == 1_000_000
    frozen_y = h["diagnostic_delayed_observed_q"]
    clean = frozen_y - h["diagnostic_delayed_noise"]
    # At age zero the exact current clean target table is available. At other
    # phases reconstruction has only floating-point subtraction error.
    refresh = (steps - 1) % delay == 0
    clean[:, refresh] = h["diagnostic_target_q"][:, refresh]
    value = h["diagnostic_delayed_value"]
    if method == "df_min":
        y = frozen_y.min(2, keepdims=True)
        clean = clean.min(2, keepdims=True)
    elif method == "dd_iql":
        y, clean, value = frozen_y[:, :, ::-1], clean[:, :, ::-1], value[:, :, ::-1]
    else:
        y = frozen_y
    weight = h["diagnostic_weights"]
    reconstructed = np.where(y - value[..., None] >= 0, tau, 1 - tau)
    np.testing.assert_array_equal(weight, reconstructed)
    r, t, heads, _, actions = weight.shape
    assert (r, actions) == (20, 4)
    used = weight[
        np.arange(r)[:, None, None],
        np.arange(t)[None, :, None],
        np.arange(heads)[None, None, :],
        h["diagnostic_state"][:, :, None],
        h["diagnostic_action"][:, :, None],
    ]
    np.testing.assert_array_equal(used, h["diagnostic_used_weight"])
    return dict(steps=steps, y=y, clean=clean, value=value, weight=weight, used=used)


def metrics(case, tau, mode, method):
    clean, y, value, weight = (case[k] for k in ("clean", "y", "value", "weight"))
    mu = clean.mean(-1)
    noise = y - clean
    high = weight == tau
    k = high.sum(-1)
    normalized = weight / weight.sum(-1, keepdims=True)
    state_ess = 1 / (4 * np.square(normalized).sum(-1))
    flat_y = mu[..., None] + noise
    flat_w = np.where(flat_y >= value[..., None], tau, 1 - tau)
    clean_w = np.where(clean >= value[..., None], tau, 1 - tau)
    point = {
        "mean_weight": weight.mean((2, 3, 4)),
        "high_weight_fraction": high.mean((2, 3, 4)),
        "clean_q_mean": clean.mean((2, 3, 4)),
        "source_v_mean": value.mean((2, 3)),
        "clean_action_sd": clean.std(-1).mean((2, 3)),
        "readout_action_sd": noise.std(-1).mean((2, 3)),
        "clean_upper_gap": (clean.max(-1) - mu).mean((2, 3)),
        "v_minus_clean_mean": (value - mu).mean((2, 3)),
        "zero_high_state_fraction": (k == 0).mean((2, 3)),
        "state_ess_fraction": state_ess.mean((2, 3)),
        "flat_clean_mean_weight": flat_w.mean((2, 3, 4)),
        "flat_clean_gate_flip_fraction": (flat_w != weight).mean((2, 3, 4)),
    }
    np.testing.assert_allclose(
        point["mean_weight"],
        (1 - tau) + (2 * tau - 1) * point["high_weight_fraction"],
        rtol=0,
        atol=2e-15,
    )
    late = case["steps"] > 900_000
    window = {key: values[:, late].mean(1) for key, values in point.items()}
    late_w = weight[:, late]
    window["pooled_ess_fraction"] = late_w.mean((1, 2, 3, 4)) ** 2 / np.square(
        late_w
    ).mean((1, 2, 3, 4))
    window["high_weight_mass_fraction"] = (
        tau * window["high_weight_fraction"] / window["mean_weight"]
    )
    window["used_weight_mean"] = case["used"][:, late].mean((1, 2))
    window["no_readout_mean_weight"] = clean_w[:, late].mean((1, 2, 3, 4))
    window["flat_minus_original_mean_weight"] = (
        window["flat_clean_mean_weight"] - window["mean_weight"]
    )
    window["readout_error_rms"] = np.sqrt(np.square(noise[:, late]).mean((1, 2, 3, 4)))
    # IID noise at a refresh is independent of the pre-update clean Q/V.
    # DF's clean heads are identical, so its survival probability is squared.
    expected_weight = np.full_like(weight, np.nan)
    if mode == "nn_like_field_iid":
        probability = ndtr(clean - value[..., None])
        if method == "df_min":
            probability = probability ** 2
        expected_weight = (1 - tau) + (2 * tau - 1) * probability
    window["iid_gaussian_expected_weight"] = expected_weight[:, late].mean((1, 2, 3, 4))
    for count in range(5):
        window[f"high_count_{count}_fraction"] = (k[:, late] == count).mean((1, 2, 3))
    return point, window


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )


def save(fig, stem, booklet):
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        # Tau contains a decimal point; with_suffix would remove the case name.
        fig.savefig(stem.parent / (stem.name + "." + ext), dpi=220, bbox_inches="tight")
    booklet.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def ecdf(ax, values, label, color, linestyle="-"):
    levels = np.linspace(0.001, 0.999, 401)
    ax.plot(
        np.quantile(values, levels),
        levels,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
    )


def distribution_figure(cases, windows, mode, delay, tau, out, booklet):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for row, method in enumerate(METHODS):
        case = cases[method]
        clean, y, v, w = (case[k] for k in ("clean", "y", "value", "weight"))
        index = len(case["steps"]) - 1
        snapshot = int(case["steps"][index] - (case["steps"][index] - 1) % delay)
        # Fixed illustration: seed 200, state 0, receiver 0. DD's gate source is
        # head 1, not receiver 0. Pooling is confined to the other two columns.
        c, source, gate, gate_w = (
            clean[0, index, 0, 0],
            y[0, index, 0, 0],
            v[0, index, 0, 0],
            w[0, index, 0, 0],
        )
        ax = axes[row, 0]
        ax.plot(np.arange(4), c, "s--", color="0.4", label="Clean source Q")
        ax.plot(np.arange(4), source, "o-", color="#1f77b4", label="Noisy source Q")
        ax.axhline(gate, color="#d62728", label="Source V", linewidth=1.5)
        ax.set_xticks(
            np.arange(4), [f"{a}\n" + rf"$\omega={gate_w[a]:.2f}$" for a in range(4)]
        )
        ax.set_xlabel("Action and recorded weight")
        ax.set_ylabel(LABELS[method] + "\nValue")
        ax.set_title(f"Seed 200, state 0; gate step {snapshot:,}")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.2)
        if row == 0:
            ax.legend(loc="best")
        late = case["steps"] > 900_000
        mu = clean[:, late].mean(-1)
        ax = axes[row, 1]
        ecdf(ax, clean[:, late] - mu[..., None], "Clean Q - state mean", "0.4", "--")
        ecdf(ax, y[:, late] - mu[..., None], "Noisy Q - state mean", "#1f77b4")
        ecdf(ax, v[:, late] - mu, "V - state mean clean Q", "#d62728")
        ax.set_title("Centered distributions: final 100k, 20 seeds")
        ax.set_xlabel("Value relative to within-state mean clean Q")
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        if row == 0:
            ax.legend(loc="best")
        summary = windows[method]
        ax = axes[row, 2]
        probability = [summary[f"high_count_{k}_fraction"].mean() for k in range(5)]
        ax.bar(np.arange(5), probability, color=COLORS[method], alpha=0.85)
        ax.set_title("High-weight count within four actions")
        ax.set_xlabel(rf"Number of actions with $\omega=\tau={tau}$")
        ax.set_ylabel("Fraction of head/state snapshots")
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(5))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        annotation = (
            f"Mean weight: {summary['mean_weight'].mean():.3f}\n"
            f"High-weight fraction: {summary['high_weight_fraction'].mean():.3f}\n"
            f"Pooled ESS fraction: {summary['pooled_ess_fraction'].mean():.3f}\n"
            f"Mean state ESS / 4: {summary['state_ess_fraction'].mean():.3f}"
        )
        ax.text(
            0.98,
            0.97,
            annotation,
            va="top",
            ha="right",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )
    fig.suptitle(rf"{NOISES[mode]}; $\tau={tau}$; $D={delay}$", fontsize=15)
    slug = "iid" if mode == "nn_like_field_iid" else "nn_like"
    save(fig, out / "figures" / f"{slug}_tau{tau}_d{delay}_distributions", booklet)


def time_figure(points, steps, mode, delay, tau, out, booklet):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for method in METHODS:
        data, color = points[method], COLORS[method]
        for ax, keys in (
            (axes[0, 0], ("clean_q_mean", "source_v_mean")),
            (axes[1, 0], ("clean_action_sd", "readout_action_sd")),
            (axes[1, 1], ("v_minus_clean_mean", "clean_upper_gap")),
        ):
            for j, key in enumerate(keys):
                ax.plot(
                    steps / 1e6,
                    data[key].mean(0),
                    color=color,
                    linestyle="-" if j == 0 else "--",
                    linewidth=1.3,
                    label=LABELS[method] if j == 0 else None,
                )
        values = data["mean_weight"]
        mean, sd = values.mean(0), values.std(0, ddof=1)
        ax = axes[0, 1]
        ax.plot(steps / 1e6, mean, color=color, linewidth=1.3, label=LABELS[method])
        ax.fill_between(steps / 1e6, mean - sd, mean + sd, color=color, alpha=0.1)
    titles = (
        "Mean clean Q (solid) and source V (dashed)",
        "Mean filtering weight (mean ± sample SD)",
        "Within-state action SD: clean Q (solid), readout (dashed)",
        "V - mean clean Q (solid), max clean Q - mean (dashed)",
    )
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel(r"Update ($\times10^6$)")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(rf"{NOISES[mode]}; $\tau={tau}$; $D={delay}$", fontsize=15)
    slug = "iid" if mode == "nn_like_field_iid" else "nn_like"
    save(fig, out / "figures" / f"{slug}_tau{tau}_d{delay}_time", booklet)


def delay_figure(summary, mode, tau, out, booklet):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    keys = (
        "mean_weight",
        "pooled_ess_fraction",
        "clean_action_sd",
        "v_minus_clean_mean",
    )
    titles = (
        "Mean filtering weight",
        "Pooled weight ESS fraction",
        "Within-state clean-Q action SD",
        "Source V minus mean clean Q",
    )
    group = summary[(summary.noise == mode) & (summary.tau == tau)]
    for ax, key, title in zip(axes.flat, keys, titles):
        for method in METHODS:
            data = group[group.method == method].sort_values("delay")
            ax.errorbar(
                data.delay,
                data[key + "_mean"],
                yerr=data[key + "_sd"],
                color=COLORS[method],
                marker="o",
                linewidth=1.6,
                capsize=3,
                label=LABELS[method],
            )
            if key == "mean_weight":
                ax.plot(
                    data.delay,
                    data.flat_clean_mean_weight_mean,
                    color=COLORS[method],
                    linestyle="--",
                    linewidth=1,
                )
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.set_xlabel("Filter refresh interval D")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0, 0].text(
        0.02,
        0.03,
        "Dashed: flatten clean Q across actions,\nkeep V and readout error fixed",
        transform=axes[0, 0].transAxes,
        fontsize=8,
    )
    fig.suptitle(rf"{NOISES[mode]}; $\tau={tau}$; final 100k; 20 seeds", fontsize=14)
    slug = "iid" if mode == "nn_like_field_iid" else "nn_like"
    save(fig, out / "figures" / f"{slug}_tau{tau}_delay_summary", booklet)


def examples(case, identity):
    rows = []
    for requested in (1, 100_000, 500_000, 1_000_000):
        index = int(np.abs(case["steps"] - requested).argmin())
        step = int(case["steps"][index])
        refresh = step - (step - 1) % identity["delay"]
        for head in range(case["weight"].shape[2]):
            for state in range(4):
                for action in range(4):
                    key = (0, index, head, state, action)
                    source_head = (
                        "min"
                        if identity["method"] == "df_min"
                        else (1 - head if identity["method"] == "dd_iql" else head)
                    )
                    rows.append(
                        {
                            **identity,
                            "seed": 200,
                            "requested_step": requested,
                            "actual_step": step,
                            "filter_refresh_step": refresh,
                            "receiver_head": head,
                            "source_head": source_head,
                            "state": state,
                            "action": action,
                            "source_clean_q": case["clean"][key],
                            "source_observed_q": case["y"][key],
                            "source_v": case["value"][key[:-1]],
                            "source_advantage": case["y"][key] - case["value"][key[:-1]],
                            "weight": case["weight"][key],
                        }
                    )
    return rows


def write_report(summary, out):
    text = [
        "# Four-state Q / weight distribution diagnostics",
        "",
        "기존 4-state, 1M, seed 200-219 로그만 읽었습니다.",
        "재학습과 원본 변경은 없습니다.",
        "",
        "[전체 그림 PDF](weight_q_distribution_report.pdf)",
        "",
        "## 측정 범위",
        "",
        "- DF-IQL / SD-IQL / DD-IQL, IID / NN-like, tau=.9/.99, D=1/10/100/1000.",
        "- 표는 마지막 100k의 101개 진단 시점에서 seed별 평균을 구한 뒤",
        "  20 seed의 평균과 표본 SD를 계산합니다.",
        "- 분포 그림은 tau=.99의 모든 D를, 시간 그림은 D=1/1000을 보여줍니다.",
        "- Clean learned target-Q와 noise를 포함한 source readout을 구분합니다.",
        "- weight를 만든 실제 frozen source Q/V를 사용합니다.",
        "  D>1에서 현재 receiver Q/V로 gate를 재구성하지 않습니다.",
        "- snapshot 예시는 seed=200, state=0, receiver=0, 1M 관측으로 고정했습니다.",
        "  DD의 filter source는 head=1입니다. 정확한 refresh 시점도 표시합니다.",
        "- pooled 분포는 state별 clean Q 평균을 먼저 빼서",
        "  seed/state/time 사이의 전체 값 차이를 행동별 outlier로 오인하지 않게 합니다.",
        "",
        "## Weight와 concentration",
        "",
        "omega는 tau 또는 1-tau 두 값입니다. mean(omega)=(1-tau)+(2tau-1)P(high).",
        "k=0은 네 행동 모두 낮은 weight를 받는 snapshot입니다.",
        "이 경우 행동 간 normalized weight는 균등하지만 업데이트 계수는 작습니다.",
        "따라서 다음 두 ESS를 구분합니다. 학습 업데이트는 정규화하지 않습니다.",
        "",
        "- Pooled ESS fraction: seed별 마지막 구간의 모든 time/head/state/action을",
        "  E[w]^2/E[w^2]를 계산합니다. 희소한 high weight도 반영합니다.",
        "- Mean state ESS / 4: 각 head/state/time의 네 행동에서",
        "  (sum w)^2/(4 sum w^2)를 계산한 뒤 평균합니다.",
        "  k=0 또는 4이면 1입니다. Pooled 집중도와 구분해야 합니다.",
        "",
        "## tau=.99, D=1: 마지막 100k 평균",
        "",
        "| Noise | Method | Mean w | P(high) | P(k=0) | Pooled ESS frac. | "
        "Clean Q mean | Clean action SD | V-mean Q | Flattened mean w |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary[(summary.tau == 0.99) & (summary.delay == 1)].iterrows():
        fields = [
            row[key + "_mean"]
            for key in (
                "mean_weight",
                "high_weight_fraction",
                "zero_high_state_fraction",
                "pooled_ess_fraction",
                "clean_q_mean",
                "clean_action_sd",
                "v_minus_clean_mean",
                "flat_clean_mean_weight",
            )
        ]
        text.append(
            f"| {NOISES[row.noise]} | {LABELS[row.method]} | "
            + " | ".join(f"{x:.4g}" for x in fields)
            + " |"
        )
    text.extend(
        [
            "",
            "## V를 고정한 counterfactual",
            "",
            "각 state의 clean Q를 그 state의 행동 평균으로 바꿉니다.",
            "V, 공통 Q 수준, 관측 readout error는 고정한 채 weight를 다시 계산합니다.",
            "현재 gate가 clean Q의 행동 간 차이에 얼마나 민감한지 보는 진단입니다.",
            "V와 학습 이력을 다시 학습하는 실험은 아닙니다.",
            "따라서 과거 outlier가 V를 올렸을 가능성까지 배제할 수는 없습니다.",
            "",
            "## 현재 관측에 대한 해석",
            "",
            "IID, tau=.99, D=1에서 SD의 clean Q 평균은 약 1188이지만",
            "행동 간 표준편차는 약 .083입니다. 일부 clean Q만 튀기보다",
            "대부분의 clean Q가 함께 올라간 모습에 가깝습니다.",
            "V는 그 clean Q 평균보다 약 1.40 높고, 약 8.4%의 readout만",
            "높은 weight를 받습니다. Clean Q의 행동 간 차이를 없애도",
            "mean weight는 .0923에서 .0909로 거의 유지됩니다.",
            "",
            "Q와 V를 같은 상수만큼 옮기면 weight는 불변입니다.",
            "하지만 실제 학습이 같은 양의 이동을 만드는 것은 아닙니다.",
            "값이 state/action에 걸쳐 비슷하면 Q≈gamma*V이므로",
            "V-Q≈(1-gamma)*V가 될 수 있습니다. 공통 bias가 커지는 동안",
            "이 간격도 커져서 high weight 비율이 줄어드는 경로는 가능합니다.",
            "이는 근사 설명이며 인과관계의 증명은 아닙니다.",
            "",
            "IID에서 기록된 clean Q와 V를 조건으로 Gaussian readout을 적분한",
            "예측 mean weight는 iid_gaussian_expected_weight 열에 있습니다.",
            "SD/DD: (1-tau)+(2tau-1)*mean_a Phi(Q_a-V).",
            "DF: 독립적인 두 head의 survival probability 곱을 사용합니다.",
            "NN-like에는 이 조건부 IID 식을 적용하지 않고 N/A로 저장했습니다.",
            "",
            "## Files",
            "",
            "- `last_window_summary.csv`: 전체 48조건의 seed 평균과 표본 SD.",
            "- `per_seed_metrics.csv`: 960 trajectory 집계.",
            "- `time_summary.csv.gz`: 진단 시점별 20 seed 평균과 표본 SD.",
            "- `example_snapshots.csv`: seed200의 모든 state/action/head Q,V,weight.",
            "- `source_manifest.json`: 원본 로그 경로와 SHA256.",
            "- `verification.json`: source gate와 실제 update weight 검증 결과.",
            "",
            "## 해석 범위",
            "",
            "단일 Q의 절대 크기와, 같은 state 안에서의 상대적인 돌출을 구분합니다.",
            "이 MDP의 참 Q와 V는 모두 0이므로 공통 수준 상승도 실제 bias입니다.",
            "하지만 공통 수준만으로 weight 감소의 원인을 설명할 수는 없습니다.",
            "또한 모든 정책의 return이 0이므로 정책 성능의 원인 검증은 아닙니다.",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output
    (out / "figures").mkdir(parents=True, exist_ok=True)
    style()
    seed_rows, summary_rows, time_frames, sample_rows, manifest = [], [], [], [], []
    with PdfPages(out / "weight_q_distribution_report.pdf") as booklet:
        for mode in NOISES:
            for tau in (0.9, 0.99):
                for delay in (1, 10, 100, 1000):
                    cases, points, windows = {}, {}, {}
                    for method in METHODS:
                        path = (
                            args.input
                            / "raw/s4_a4"
                            / mode
                            / f"d{delay}"
                            / f"tau{tau}"
                            / f"{method}.npz"
                        )
                        case = load_case(path, method, delay, tau)
                        point, window = metrics(case, tau, mode, method)
                        cases[method], points[method], windows[method] = (
                            case,
                            point,
                            window,
                        )
                        identity = dict(noise=mode, tau=tau, delay=delay, method=method)
                        for index, seed in enumerate(SEEDS):
                            seed_rows.append(
                                {
                                    **identity,
                                    "seed": seed,
                                    **{k: v[index] for k, v in window.items()},
                                }
                            )
                        summary = identity.copy()
                        for key, values in window.items():
                            summary[key + "_mean"] = values.mean()
                            summary[key + "_sd"] = values.std(ddof=1)
                        summary_rows.append(summary)
                        time_data = {
                            **{k: [v] * len(case["steps"]) for k, v in identity.items()},
                            "step": case["steps"],
                        }
                        for key in TIME_METRICS:
                            time_data[key + "_mean"] = point[key].mean(0)
                            time_data[key + "_sd"] = point[key].std(0, ddof=1)
                        time_frames.append(pd.DataFrame(time_data))
                        sample_rows.extend(examples(case, identity))
                        manifest.append(
                            {
                                **identity,
                                "path": str(path.relative_to(ROOT)),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "diagnostic_snapshots": len(case["steps"]),
                                "last_window_snapshots": int(
                                    (case["steps"] > 900000).sum()
                                ),
                            }
                        )
                    if tau == 0.99:
                        distribution_figure(
                            cases, windows, mode, delay, tau, out, booklet
                        )
                        if delay in (1, 1000):
                            time_figure(
                                points,
                                cases["sd_iql"]["steps"],
                                mode,
                                delay,
                                tau,
                                out,
                                booklet,
                            )
                    print(f"Read {NOISES[mode]}, tau={tau}, D={delay}", flush=True)
                    del cases, points, windows
        summary = pd.DataFrame(summary_rows)
        for mode in NOISES:
            for tau in (0.9, 0.99):
                delay_figure(summary, mode, tau, out, booklet)
    pd.DataFrame(seed_rows).to_csv(out / "per_seed_metrics.csv", index=False)
    summary.to_csv(out / "last_window_summary.csv", index=False)
    pd.concat(time_frames, ignore_index=True).to_csv(
        out / "time_summary.csv.gz", index=False
    )
    pd.DataFrame(sample_rows).to_csv(out / "example_snapshots.csv", index=False)
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verification = dict(
        conditions=len(manifest),
        trajectories=len(seed_rows),
        exact_reconstructed_source_weights=True,
        exact_recorded_used_weights=True,
        binary_weight_probability_identity=True,
        new_training_runs=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    write_report(summary, out)
    print(summary[(summary.tau == 0.99) & (summary.delay == 1)].to_string(index=False))


if __name__ == "__main__":
    main()
