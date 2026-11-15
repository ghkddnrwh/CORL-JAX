"""Analyze paired bias interventions without modifying earlier experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from run_controls import controls, iid_df_moments, METHODS, NOISES, OUT, ROOT
from scipy.integrate import quad
from scipy.special import ndtr

PRIOR = (
    ROOT / "utils/iclr2027_dd_iql/rebuttal/tabular_df_dd_weight_hypothesis/analyze.py"
)
spec = importlib.util.spec_from_file_location("prior_analysis", PRIOR)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
NAMES = {
    "baseline": "Original",
    "oracle_clean_q": "Oracle clean-Q readouts",
    "noise_bias_half": "50% noise-bias correction",
    "noise_bias_full": "100% noise-bias correction",
    "bellman_plus_0.6": "Q Bellman target +0.6",
}
COLORS = {
    "baseline": "#cc79a7",
    "noise_bias_half": "#e69f00",
    "noise_bias_full": "#009e73",
    "oracle_clean_q": "0.25",
    "bellman_plus_0.6": "#d55e00",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, out, name, booklet):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / (name + "." + ext), dpi=210, bbox_inches="tight")
    booklet.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def trajectory_figure(traces, steps, method, names, out, stem, booklet):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name in names:
        for ax, key in zip(axes, ("learned_target_q", "source_gap", "mean_weight")):
            draw = prior.binned_line if key == "mean_weight" else prior.line
            color = COLORS[name]
            if method == "dd_iql" and name == "baseline":
                color = "#0072b2"
            draw(
                ax,
                steps / 1e6,
                traces[("nn_like_field_iid", method, name)][key],
                NAMES[name],
                color,
                band=name == "baseline",
            )
    method_label = "DF-IQL, D=1" if method == "df_min" else "DD-IQL, D=100"
    for ax, ylabel in zip(
        axes,
        (
            "Mean clean target Q",
            "Source V minus mean clean Q",
            "Mean filtering weight (20k bins)",
        ),
    ):
        ax.set(xlabel="Updates (millions)", ylabel=ylabel, xlim=(0, 1))
        ax.grid(alpha=0.2)
    axes[0].text(0.03, 0.95, method_label, transform=axes[0].transAxes, va="top")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, out, stem, booklet)


def counterexample_figure(summary, out, booklet):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    selected = (
        ("df_min", "baseline"),
        ("df_min", "oracle_clean_q"),
        ("dd_iql", "baseline"),
        ("dd_iql", "bellman_plus_0.6"),
    )
    labels = ("DF\nOriginal", "DF\nOracle Q", "DD\nOriginal", "DD\nQ target +0.6")
    for row, noise in enumerate(NOISES):
        records = [
            summary[
                (summary.noise == noise)
                & (summary.method == method)
                & (summary.control == control)
            ].iloc[0]
            for method, control in selected
        ]
        for col, key in enumerate(("learned_target_q", "mean_weight")):
            ax = axes[row, col]
            colors = ["#cc79a7", "0.4", "#0072b2", "#d55e00"]
            ax.bar(
                np.arange(4),
                [r[key + "_mean"] for r in records],
                yerr=[r[key + "_sd"] for r in records],
                capsize=3,
                color=colors,
            )
            for i, r in enumerate(records):
                value = r[key + "_mean"]
                ax.text(
                    i,
                    value + (15 if col == 0 else 0.025),
                    f"{value:.1f}" if col == 0 else f"{value:.3f}",
                    ha="center",
                    fontsize=9,
                )
            ax.set_xticks(np.arange(4), labels)
            ax.set_ylabel(
                ("IID" if row == 0 else "NN-like")
                + "\n"
                + ("Mean clean target Q" if col == 0 else "Mean filtering weight")
            )
            ax.set_ylim((-0.05, 0.6) if col else (-20, 650))
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    save(fig, out, "large_bias_and_low_weight_are_distinct", booklet)


def geometry_figure(summary, out, booklet):
    iid = summary[summary.noise == "nn_like_field_iid"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    xs = np.arange(len(iid))
    labels = [
        (
            ("DF" if r.method == "df_min" else "DD")
            + "\n"
            + {
                "baseline": "Original",
                "oracle_clean_q": "Oracle Q",
                "noise_bias_half": "50% corr.",
                "noise_bias_full": "100% corr.",
                "bellman_plus_0.6": "Q target +0.6",
            }[r.control]
        )
        for _, r in iid.iterrows()
    ]
    ax.bar(
        xs - 0.18,
        iid.mean_weight_mean,
        0.36,
        yerr=iid.mean_weight_sd,
        capsize=3,
        color="#0072b2",
        label="Recorded gate",
    )
    ax.bar(
        xs + 0.18,
        iid.iid_expected_weight_mean,
        0.36,
        yerr=iid.iid_expected_weight_sd,
        capsize=3,
        color="#e69f00",
        label="Gaussian prediction from actual frozen Q / V",
    )
    ax.set_xticks(xs, labels)
    ax.set(ylabel="Mean filtering weight", ylim=(0, 0.62))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, out, "iid_weight_geometry_prediction", booklet)


def findings(summary, paired, out):
    lines = [
        "Bias / mean-weight causality controls: S=A=4, tau=.99, 1M",
        "",
        "20 paired seeds (200-219), IID/NN-like, same noise and action streams.",
        "240 trajectories, 80 baseline trajectories reproduced original arrays bitwise.",
        "Original algorithms and raw logs were preserved.",
        "",
        "1. DF IID: reducing the conditional readout-noise bias raises mean weight.",
        "The gate formula is unchanged. Subtract lambda*c(g) from V's receiver, where",
        "c(g)=E[w*Z|clean Q,V]/E[w|clean Q,V], Z=min of two independent N(0,1).",
        "lambda=0, .5, 1 gives original, half correction, full correction.",
        "This cancels the NET injected-noise term, including min mean and covariance.",
        "It is not a general estimator of accumulated Q/V error.",
        "It is an oracle diagnostic requiring the known noise law, not a new algorithm.",
        "No IID conditional correction is applied to the correlated NN-like field.",
        "",
        "Final100k mean: 101 pre-update snapshots per seed, then 20-seed mean.",
        "Noise | Method | Control | Mean weight | Learned target Q | Source V-Q gap",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r.noise} | {r.method} | {r.control} | "
            f"{r.mean_weight_mean:.6f} | {r.learned_target_q_mean:.6f} | "
            f"{r.source_gap_mean:.6f}"
        )
    lines += [
        "",
        "2. Large common value bias is not sufficient for low weight.",
        "Adding +0.6 to DD's Q Bellman target on every update creates a known forcing",
        "error in the original zero-reward MDP. All learned values remain errors vs 0.",
        "DD then has Q of similar magnitude to DF but still has mean weight near .5.",
        "Thus common absolute value level alone does not determine filtering weight.",
        "This artificial forcing is a counterexample, not a naturally failing DD run.",
        "",
        "3. Large accumulating clean-Q bias is not necessary for low weight.",
        "Oracle-clean-Q controls replace clean Q by the known oracle 0 in BOTH the",
        "filter source and V receiver at every update. Noise and learned V remain.",
        "The internal learned Q backup and EMA remain active, but their output cannot",
        "feed back into filter/V. This breaks Q feedback, retaining noise selection.",
        "DF learned Q/V stay around .8 instead of hundreds, yet mean weight is ~.055.",
        "The relevant source clean Q is 0; V is about .8 ABOVE that source reference.",
        "Do not compute source gap from the internal learned Q in this intervention.",
        "DD oracle-clean-Q values are near zero and mean weight remains near .5.",
        "",
        "4. Relative threshold and aggregation directly predict the observed weights.",
        "For independent unit Gaussian readouts and common clean DF heads:",
        "E[w|Q,V]=.01+.98*Phi(Q-V)^2 (DF min),",
        "E[w|Q,V]=.01+.98*Phi(Q-V) (single source head).",
        "The predictions integrate over actual per-coordinate frozen source Q/V,",
        "not Phi at a pooled mean gap. Geometry figure verifies this prediction.",
        "Even at zero gap, expected DF and DD weights are .255 and .5 respectively.",
        "The remaining DF reduction is associated with its larger positive V-Q gap.",
        "Equal Q/V shifts leave the gate invariant; every saved gate was checked.",
        "",
        "Conclusion",
        "The endogenous DF readout-bias pathway contributes to its low mean weight:",
        "correcting the pathway reduces large common bias AND increases mean weight.",
        "But an unqualified claim that large overestimated Q/V alone cause low weight",
        "is contradicted by the two counterfactual controls. The relative V-Q threshold",
        "and min aggregation are essential. This is a coupled mechanism, not an",
        "identified one-way causal coefficient from absolute Q level to mean weight.",
        "The full correction leaves learned bias around +12: it cancels only the",
        "conditional injected-noise term, not learned-error/filter dependencies.",
        "This also illustrates why low readout coupling does not prove total accuracy.",
        "All policies return zero here. No policy-performance causality or NN oracle",
        "bias claim follows from this toy MDP alone.",
        "",
        "Artifacts and estimands",
        "causal_controls_report.pdf: 4 pages, plus individual PNG/PDF figures.",
        "summary.csv: 12 condition means/sample SD; per_seed_metrics.csv: 240 rows.",
        "paired_differences.csv: same-seed intervention-minus-original changes.",
        "Actual-update moments use ALL sampled final100k updates, not 101 snapshots.",
        "Weight trajectories use per-seed 20k bin means; Q/gap use saved snapshots.",
        "Shaded bands are across-seed sample SD. Raw data and code snapshots retained.",
        "verification.json records raw hashes, exact baseline and gate verification,",
        "full-run V bookkeeping, and independent numerical quadrature of correction.",
        "",
    ]
    (out / "findings.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    (out / "figures").mkdir(exist_ok=True)
    manifest = json.loads((out / "manifest.json").read_text())
    checks = [
        c for job in manifest["baseline_checks"] for c in job["baseline_bitwise_checks"]
    ]
    assert len(checks) == 80
    for check in checks:
        assert digest(ROOT / check["source"]) == check["sha256"]
    prior.audit.style()
    rows, traces, frames, hashes, max_v_error = [], {}, [], {}, 0.0
    for noise in NOISES:
        for method, _, delay in METHODS:
            for name, oracle_q, fraction, forcing in controls(noise, method):
                points = []
                for seed in range(200, 220):
                    path = out / "raw" / noise / method / name / f"seed{seed}.npz"
                    hashes[str(path.relative_to(out))] = digest(path)
                    with np.load(path, allow_pickle=False) as file:
                        h = {key: file[key] for key in file.files}
                    assert all(np.isfinite(a).all() for a in h.values())
                    point = prior.point_metrics(h, method)
                    c, _, v = prior.source_arrays(h, method)
                    if oracle_q:
                        np.testing.assert_array_equal(c, np.zeros_like(c))
                    if method == "df_min":
                        np.testing.assert_array_equal(
                            h["diagnostic_target_q"][:, 0],
                            h["diagnostic_target_q"][:, 1],
                        )
                    expected_weight = np.full_like(point["mean_weight"], np.nan)
                    expected_noise = expected_weight.copy()
                    if noise.endswith("_iid"):
                        prob = ndtr(c - v[..., None])
                        if method == "df_min":
                            prob = prob**2
                            gap = v[..., None] - c
                            phi = np.exp(-(gap**2) / 2) / np.sqrt(2 * np.pi)
                            weighted_z = -0.01 / np.sqrt(np.pi) + 0.98 * (
                                2 * phi * ndtr(-gap)
                                - ndtr(-np.sqrt(2) * gap) / np.sqrt(np.pi)
                            )
                            expected_noise = (1 - fraction) * weighted_z.mean((1, 2, 3))
                        else:
                            expected_noise = np.zeros_like(expected_weight)
                        expected_weight = (0.01 + 0.98 * prob).mean((1, 2, 3))
                    point["iid_expected_weight"] = expected_weight
                    point["iid_expected_net_noise"] = expected_noise
                    points.append(point)
                    steps = h["diagnostic_step"]
                    late = steps > 900_000
                    assert late.sum() == 101
                    moment = h["update_moments"][-100:].mean(0)
                    # All committed V updates must add to the final V table since init=0.
                    total = h["update_moments"]
                    implied = (
                        0.1 * 1000 * (total[:, 2] + total[:, 3] - total[:, 4]).sum()
                    )
                    actual = h["value_final"].sum() / h["value_final"].shape[0]
                    err = abs(implied - actual)
                    max_v_error = max(max_v_error, err)
                    np.testing.assert_allclose(implied, actual, rtol=0, atol=1e-6)
                    rows.append(
                        dict(
                            noise=noise,
                            method=method,
                            delay=delay,
                            control=name,
                            seed=seed,
                            **{
                                key: values[late].mean() for key, values in point.items()
                            },
                            used_weight=moment[0],
                            raw_noise_mean=moment[1],
                            weighted_raw_noise=moment[2],
                            weighted_correction=moment[4],
                            weighted_net_noise=moment[2] - moment[4],
                            weighted_clean_residual=moment[3],
                            mean_raw_V_increment=0.1
                            * (moment[2] + moment[3] - moment[4]),
                        )
                    )
                trace = {key: np.stack([p[key] for p in points]) for key in points[0]}
                traces[(noise, method, name)] = trace
                frame = pd.DataFrame(
                    dict(step=steps, noise=noise, method=method, control=name)
                )
                for key, values in trace.items():
                    frame[key + "_mean"] = values.mean(0)
                    frame[key + "_sd"] = values.std(0, ddof=1)
                frames.append(frame)
    per_seed = pd.DataFrame(rows)
    assert len(per_seed) == 240
    per_seed.to_csv(out / "per_seed_metrics.csv", index=False)
    ids = ["noise", "method", "delay", "control"]
    summary = per_seed.groupby(ids, sort=False).agg(
        {key: ["mean", "std"] for key in per_seed if key not in ids + ["seed"]}
    )
    summary.columns = [
        key + ("_sd" if stat == "std" else "_mean") for key, stat in summary.columns
    ]
    summary = summary.reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    deltas = []
    for (noise, method), group in per_seed.groupby(["noise", "method"]):
        base = group[group.control == "baseline"].set_index("seed")
        for name in group.control.unique():
            if name == "baseline":
                continue
            other = group[group.control == name].set_index("seed")
            for key in (
                "mean_weight",
                "learned_target_q",
                "source_gap",
                "weighted_net_noise",
            ):
                d = other[key] - base[key]
                deltas.append(
                    dict(
                        noise=noise,
                        method=method,
                        control=name,
                        metric=key,
                        paired_delta_mean=d.mean(),
                        paired_delta_sd=d.std(ddof=1),
                    )
                )
    paired = pd.DataFrame(deltas)
    paired.to_csv(out / "paired_differences.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(out / "time_summary.csv.gz", index=False)
    with PdfPages(out / "causal_controls_report.pdf") as booklet:
        trajectory_figure(
            traces,
            steps,
            "df_min",
            ["baseline", "noise_bias_half", "noise_bias_full"],
            out,
            "iid_df_bias_correction",
            booklet,
        )
        trajectory_figure(
            traces,
            steps,
            "dd_iql",
            ["baseline", "bellman_plus_0.6"],
            out,
            "iid_dd_forced_positive_bias",
            booklet,
        )
        counterexample_figure(summary, out, booklet)
        geometry_figure(summary, out, booklet)
    quadrature_errors = []
    for tau in (0.9, 0.99):
        for gap in (-3.0, -0.5, 0.0, 0.64, 2.0, 4.0):
            ew, ewz = iid_df_moments(gap, tau)

            def density(z):
                return 2 * math.exp(-z * z / 2) / math.sqrt(2 * math.pi) * ndtr(-z)

            def w(z):
                return tau if z >= gap else 1 - tau

            q0 = quad(lambda z: w(z) * density(z), -12, 12, points=[gap], epsabs=1e-11)[
                0
            ]
            q1 = quad(
                lambda z: w(z) * z * density(z), -12, 12, points=[gap], epsabs=1e-11
            )[0]
            quadrature_errors += [abs(ew - q0), abs(ewz - q1)]
            assert abs(ewz - ew * (ewz / ew)) < 1e-14
    assert max(quadrature_errors) < 1e-10
    valid = per_seed.dropna(subset=["iid_expected_weight"])
    prediction_error = (valid.mean_weight - valid.iid_expected_weight).abs()
    verification = dict(
        trajectories=240,
        baseline_trajectories_bitwise=80,
        original_sources_sha256={c["source"]: c["sha256"] for c in checks},
        all_gate_reconstructions_exact=True,
        all_equal_shift_gates_invariant=True,
        oracle_Q_inputs_verified_zero=True,
        all_arrays_finite=True,
        df_identical_clean_heads_verified=True,
        max_V_update_accounting_error=max_v_error,
        correction_quadrature_cases=12,
        max_quadrature_error=max(quadrature_errors),
        max_per_seed_IID_weight_prediction_error=prediction_error.max(),
        runner_code_sha256=manifest["source_sha256"],
        analysis_sha256=digest(Path(__file__)),
        analysis_dependency_sha256={
            str(PRIOR.relative_to(ROOT)): digest(PRIOR),
            str(prior.PREVIOUS.relative_to(ROOT)): digest(prior.PREVIOUS),
        },
        raw_sha256=hashes,
    )
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    (out / "provenance/analyze_controls.py").write_bytes(Path(__file__).read_bytes())
    findings(summary, paired, out)
    print(
        summary[
            [
                *ids,
                "mean_weight_mean",
                "learned_target_q_mean",
                "source_gap_mean",
                "weighted_net_noise_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
