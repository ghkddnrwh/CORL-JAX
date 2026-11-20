"""Export IID baseline update observations for DF D=1 and DD D=100.

Read existing recordings only. No training, oracle correction, or intervention.
Run from any directory with Python and NumPy installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "logs/iclr2027_dd_iql/archive/history_20261115/tabular_weight_bias_causality"
OUT = ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_update_observation"
NOISE = "nn_like_field_iid"
MOMENTS = (
    "weight",
    "raw_receiver_noise",
    "weight_times_raw_noise",
    "weight_times_clean_minus_v",
    "weight_times_correction",
    "v_before",
    "clean_receiver_q",
)
METRICS = {
    "mean_weight": "Mean filtering weight",
    "high_weight_fraction": "Fraction receiving weight 0.99",
    "raw_noise_mean": "Mean receiver readout noise",
    "weighted_clean_residual": "Mean w (clean Q - V)",
    "weighted_noise": "Mean w epsilon",
    "unscaled_update": "Sum before learning rate",
    "actual_v_increment": "Mean visited-state V increment per head",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    previous = json.loads((SOURCE / "verification.json").read_text())
    assert tuple(manifest["moments"]) == MOMENTS
    assert manifest["tau"] == 0.99 and manifest["updates"] == 1_000_000
    assert manifest["seeds"] == list(range(200, 220))
    for relative, expected in manifest["source_sha256"].items():
        assert digest(ROOT / relative) == expected, relative

    records, inputs, accounting_errors = [], {}, []
    for method, delay in (("df_min", 1), ("dd_iql", 100)):
        baseline = next(
            c for c in manifest["controls"][NOISE][method] if c[0] == "baseline"
        )
        assert baseline == ["baseline", False, 0.0, 0.0]
        original_path = (
            ROOT
            / "logs/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sizes_1m"
            / "raw/s4_a4"
            / NOISE
            / f"d{delay}/tau0.99/{method}.npz"
        )
        original_hash = digest(original_path)
        assert (
            original_hash
            == previous["original_sources_sha256"][str(original_path.relative_to(ROOT))]
        )
        inputs[str(original_path.relative_to(ROOT))] = original_hash
        with np.load(original_path, allow_pickle=False) as file:
            reference = {k: file[k] for k in file.files if k != "diagnostic_noise"}

        for seed in range(200, 220):
            path = SOURCE / "raw" / NOISE / method / "baseline" / f"seed{seed}.npz"
            sha256 = digest(path)
            assert sha256 == previous["raw_sha256"][str(path.relative_to(SOURCE))]
            inputs[str(path.relative_to(ROOT))] = sha256
            with np.load(path, allow_pickle=False) as file:
                h = {k: file[k] for k in file.files}
            assert all(np.isfinite(value).all() for value in h.values())
            # Recheck original baseline trajectories, not just a cached success flag.
            for key in (
                "diagnostic_q",
                "diagnostic_target_q",
                "diagnostic_value",
                "diagnostic_delayed_observed_q",
                "diagnostic_delayed_value",
                "diagnostic_weights",
                "q_final",
                "q_target_final",
                "value_final",
            ):
                np.testing.assert_array_equal(h[key], reference[key][seed - 200])
            np.testing.assert_array_equal(
                h["moment_end_step"], np.arange(1000, 1_000_001, 1000)
            )
            moments = h["update_moments"]
            assert moments.shape == (1000, len(MOMENTS))
            np.testing.assert_array_equal(moments[:, 4], np.zeros(1000))
            mask = h["moment_end_step"] > 900_000
            assert mask.sum() == 100
            m = moments[mask].mean(axis=0)
            unscaled = m[2] + m[3]
            records.append(
                dict(
                    method=method,
                    delay=delay,
                    seed=seed,
                    noise=NOISE,
                    first_update=900001,
                    last_update=1000000,
                    mean_weight=m[0],
                    high_weight_fraction=(m[0] - 0.01) / 0.98,
                    raw_noise_mean=m[1],
                    weighted_clean_residual=m[3],
                    weighted_noise=m[2],
                    unscaled_update=unscaled,
                    actual_v_increment=0.1 * unscaled,
                )
            )
            # Sum over states, average over V heads: each update visits one state.
            reconstructed = 0.1 * 1000 * (moments[:, 2] + moments[:, 3]).sum()
            actual = h["value_final"].sum(axis=-1).mean()
            np.testing.assert_allclose(reconstructed, actual, atol=1e-6, rtol=0)
            accounting_errors.append(float(abs(reconstructed - actual)))

    summary = []
    for method, delay in (("df_min", 1), ("dd_iql", 100)):
        rows = [r for r in records if r["method"] == method]
        for key, description in METRICS.items():
            values = np.array([r[key] for r in rows])
            summary.append(
                dict(
                    method=method,
                    delay=delay,
                    metric=key,
                    description=description,
                    n_seeds=len(values),
                    mean=values.mean(),
                    seed_sd=values.std(ddof=1),
                    seed_min=values.min(),
                    seed_max=values.max(),
                )
            )

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "per_seed.csv", records)
    write_csv(OUT / "summary.csv", summary)
    verification = dict(
        scope="IID noise, unmodified baseline only",
        trajectories=40,
        original_trajectories_equal=40,
        first_update=900001,
        last_update=1000000,
        bins_per_seed=100,
        updates_per_bin=1000,
        alpha_v=0.1,
        dd_head_aggregation="mean of two updated V heads, not their sum",
        all_weighted_corrections_zero=True,
        max_full_run_v_accounting_error=max(accounting_errors),
        source_code_sha256=manifest["source_sha256"],
        input_sha256=inputs,
        extraction_sha256=digest(Path(__file__)),
        csv_sha256={
            name: digest(OUT / name) for name in ("per_seed.csv", "summary.csv")
        },
    )
    (OUT / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    for row in summary:
        print(
            f"{row['method']:7} {row['metric']:26} "
            f"{row['mean']:+.12f} +/- {row['seed_sd']:.12f}"
        )
    print(
        f"Verified 40 baseline trajectories; V accounting error <= "
        f"{max(accounting_errors):.3g}. Saved to {OUT}"
    )


if __name__ == "__main__":
    main()
