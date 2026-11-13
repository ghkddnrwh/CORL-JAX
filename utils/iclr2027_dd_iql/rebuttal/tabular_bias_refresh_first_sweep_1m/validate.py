"""Validate saved gates and original Double-Q/D=1 references."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import case_path, cases, CONFIG, DEFAULT_RESULTS_DIR, REPO_ROOT
from measure import derive

def check(case):
    path = case_path(DEFAULT_RESULTS_DIR, case)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["data_sha256"]
    with np.load(path) as stored:
        h = {key: stored[key] for key in stored.files}
    size, mode, method, delay, tau = case
    d = derive(h, method, delay or 1, tau or 0.9)
    for field in ("operator", "operator_accumulated", "operator_readout"):
        np.testing.assert_array_equal(h["diagnostic_" + field], d[field])
    if method != "double_q":
        age = (h["diagnostic_step"] - 1) % delay
        np.testing.assert_array_equal(h["diagnostic_noise_lag"], age)
        assert len(np.unique(age)) == delay
        np.testing.assert_array_equal(h["refresh_count"], CONFIG["updates"] // delay)
        fresh = age == 0
        np.testing.assert_array_equal(
            h["diagnostic_delayed_noise"][:, fresh], h["diagnostic_noise"][:, fresh]
        )
        np.testing.assert_array_equal(
            h["diagnostic_delayed_observed_q"][:, fresh],
            h["diagnostic_target_q"][:, fresh] + h["diagnostic_noise"][:, fresh],
        )
        np.testing.assert_array_equal(
            h["diagnostic_delayed_value"][:, fresh], h["diagnostic_value"][:, fresh]
        )
    old_root = REPO_ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_bias_1m"
    previous = case_path(old_root, case)
    reference_match = None
    if method == "double_q" or delay == 1:
        with np.load(previous) as old:
            keys = ["q_final", "v_bias", "mean_q_bias", "q_rmse", "checkpoint_q_100k"]
            if method != "double_q":
                keys += [
                    "value_final",
                    "value_table",
                    "q_target_final",
                    "checkpoint_value_100k",
                ]
            for key in keys:
                np.testing.assert_array_equal(h[key], old[key])
        reference_match = True
    return dict(
        size=size,
        mode=mode,
        method=method,
        D=delay,
        tau=tau,
        actual_operator_verified=True,
        current_filter_at_refresh_verified=method != "double_q",
        original_reference_bitwise_equal=reference_match,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(check, cases()))
    report = dict(
        conditions=len(rows),
        seed_trajectories=len(rows) * 20,
        original_double_q_conditions_bitwise_equal=sum(
            r["method"] == "double_q" and bool(r["original_reference_bitwise_equal"])
            for r in rows
        ),
        original_d1_iql_conditions_bitwise_equal=sum(
            r["method"] != "double_q"
            and r["D"] == 1
            and bool(r["original_reference_bitwise_equal"])
            for r in rows
        ),
        condition_reports=rows,
    )
    (DEFAULT_RESULTS_DIR / "provenance/independent_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "condition_reports"}))


if __name__ == "__main__":
    main()
