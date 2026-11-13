"""Check oracle edge cases and every stored result against the original replay."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import case_path, cases, DEFAULT_RESULTS_DIR
from measure import coupling, derive

def oracle_checks():
    checks = []
    for method in ("df_min", "sd_iql", "dd_iql"):
        for offset in (0.0, 7.0):
            vn = 1 if method == "df_min" else 2
            shape = (1, 2, 2, 4, 4)
            h = {
                "diagnostic_q": np.full(shape, offset),
                "diagnostic_target_q": np.full(shape, offset),
                "diagnostic_noise": np.zeros(shape),
                "diagnostic_delayed_observed_q": np.full(shape, offset),
                "diagnostic_delayed_noise": np.zeros(shape),
                "diagnostic_value": np.full((1, 2, vn, 4), offset),
                "diagnostic_delayed_value": np.full((1, 2, vn, 4), offset),
                "diagnostic_used_weight": np.full((1, 2, vn), 0.99),
                "diagnostic_used_backup": np.full((1, 2, 2), 0.999 * offset),
                "diagnostic_state": np.array([[0, 1]]),
                "diagnostic_action": np.array([[1, 2]]),
                "diagnostic_step": np.array([1, 2]),
            }
            d = derive(h, method, 1, 0.99)
            np.testing.assert_allclose(d["operator"], offset, atol=1e-14, rtol=0)
            np.testing.assert_allclose(
                d["operator_accumulated"], offset, atol=1e-14, rtol=0
            )
            np.testing.assert_array_equal(d["operator_readout"], 0)
            for cross_section in (False, True):
                stats = coupling(
                    d["weights"],
                    d["proxies"]["oracle"],
                    cross_section=cross_section,
                )
                assert np.isnan(stats["abs_corr"]).all()
                np.testing.assert_array_equal(stats["valid_fraction"], 0)
            checks.append(
                dict(
                    method=method,
                    reference=0,
                    clean_offset=offset,
                    filtering_error=offset,
                    constant_weight_correlation="undefined",
                )
            )
    w = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 1.0, 3.0, 2.0]).reshape(1, 2, 1, 2, 2)
    e = np.concatenate([2 * w[:, :1] + 7, -3 * w[:, 1:] + 7], axis=1)
    affine = coupling(w, e, cross_section=True)
    np.testing.assert_allclose(affine["abs_corr"], 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(affine["signed_corr"], 0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(affine["covariance"], -0.625, atol=1e-14, rtol=0)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR.parent / "tabular_bias_diagnostics_s4_1m",
    )
    args = parser.parse_args()
    checks = oracle_checks()
    matched_arrays = 0
    gates = 0
    final_refresh_distinguished = 0
    for case in cases([4], methods=("max_min", "df_min", "sd_iql", "dd_iql")):
        path = case_path(args.results_dir, case)
        meta = json.loads(path.with_suffix(".json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["data_sha256"]
        original_path = case_path(DEFAULT_RESULTS_DIR, case)
        original_meta = json.loads(original_path.with_suffix(".json").read_text())
        assert (
            hashlib.sha256(original_path.read_bytes()).hexdigest()
            == original_meta["data_sha256"]
            == meta["original_sha256"]
        )
        with np.load(path) as f:
            h = {k: f[k] for k in f.files}
        with np.load(original_path) as original:
            for key in original.files:
                np.testing.assert_array_equal(h[key], original[key])
                matched_arrays += 1
        method, delay, tau = case[2], case[3] or 1, case[4] or 0.9
        d = derive(h, method, delay, tau)
        for field in ("operator", "operator_accumulated", "operator_readout"):
            np.testing.assert_array_equal(h["diagnostic_" + field], d[field])
        if method != "max_min":
            np.testing.assert_array_equal(h["diagnostic_weights"], d["weights"])
            age = (h["diagnostic_step"] - 1) % delay
            np.testing.assert_array_equal(h["diagnostic_snapshot_age"], age)
            assert len(np.unique(age)) == delay
            gates += h["diagnostic_used_weight"].size
            if delay > 1:
                assert not np.array_equal(
                    h["diagnostic_delayed_observed_q"][:, -1],
                    h["q_delayed_observed_final"],
                )
                final_refresh_distinguished += 1
            else:
                np.testing.assert_array_equal(d["proxies"]["temporal_observed"], 0)
        np.testing.assert_allclose(
            d["operator"],
            d["operator_accumulated"] + d["operator_readout"],
            atol=2e-12,
            rtol=0,
        )
    report = dict(
        oracle_edge_checks=checks,
        original_arrays_bitwise_verified=matched_arrays,
        actual_update_weights_verified=gates,
        pre_final_gate_distinguished_from_post_final_snapshot=final_refresh_distinguished,
        all_instantaneous_outputs_verified=True,
        all_original_file_hashes_unchanged=True,
    )
    path = args.results_dir / "provenance/independent_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
