"""Check the collected data and compiled updates against the local frozen reference."""

import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

from common import CODE_DIR, CONFIG, DEFAULT_RESULTS_DIR

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    for name, digest in CONFIG["reference_hashes"].items():
        assert (
            hashlib.sha256((CODE_DIR / "reference" / name).read_bytes()).hexdigest()
            == digest
        )
    import sys

    from engine import validate_compiled
    from plot_results import load_results

    for name in ("tabular_dd_iql", "tabular_q_backups", "stress_iid_min_ddq_bias"):
        assert (
            Path(sys.modules[name].__file__).resolve().parent == CODE_DIR / "reference"
        )
    checks = validate_compiled()
    assert len(checks) == 72 and all(c["passed"] for c in checks)
    data, data_checks = load_results(args.results_dir)
    report = dict(
        reference_checks=checks,
        collected_conditions=len(data),
        maximum_reference_error=max(c["max_abs_error"] for c in checks),
        maximum_metric_error=max(c["max_metric_error"] for c in data_checks),
        local_reference_imports_verified=True,
        requires_legacy_logs=False,
    )
    (args.results_dir / "provenance").mkdir(parents=True, exist_ok=True)
    (args.results_dir / "provenance/engine_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "reference_checks"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
