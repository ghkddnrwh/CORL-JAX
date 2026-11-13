"""Verify saved current-first gates, actual backups, and unchanged D=1 learning."""

import hashlib
import json

import numpy as np
from common import case_path, cases, CONFIG, DEFAULT_RESULTS_DIR, REPO_ROOT
from measure import derive

def main():
    reports = []
    original = REPO_ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_bias_diagnostics_s4_1m"
    for case in cases():
        path = case_path(DEFAULT_RESULTS_DIR, case)
        meta = json.loads(path.with_suffix(".json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["data_sha256"]
        with np.load(path) as stored:
            h = {key: stored[key] for key in stored.files}
        _, _, method, delay, tau = case
        d = derive(h, method, delay, tau)
        for field in ("operator", "operator_accumulated", "operator_readout"):
            np.testing.assert_array_equal(h["diagnostic_" + field], d[field])
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
        same_d1 = None
        previous = case_path(original, case)
        if delay == 1 and previous.exists():
            with np.load(previous) as old:
                for key in (
                    "q_final",
                    "value_final",
                    "q_target_final",
                    "v_bias",
                    "mean_q_bias",
                    "q_rmse",
                    "value_table",
                    "checkpoint_q_100k",
                    "checkpoint_value_100k",
                    "diagnostic_noise",
                ):
                    np.testing.assert_array_equal(h[key], old[key])
            same_d1 = True
        reports.append(
            dict(
                path=str(path),
                actual_filter_and_backup_verified=True,
                zero_lag_at_refresh_verified=True,
                noise_lags_covered=delay,
                d1_matches_previous=same_d1,
            )
        )
    report = dict(
        conditions=len(reports),
        seed_trajectories=20 * len(reports),
        training_updates_per_seed=1000000,
        all_filters_and_backups_verified=True,
        all_refreshes_use_current_values=True,
        condition_reports=reports,
    )
    (DEFAULT_RESULTS_DIR / "provenance/independent_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "condition_reports"}))


if __name__ == "__main__":
    main()
