from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPTS = [
    "01_delayed_interval_coupling.py",
    "02_receiver_td_coupling.py",
    "03_source_receiver_decoupling.py",
    "04_value_stability_performance.py",
    "05_mechanism_controls.py",
]


def main() -> None:
    """
    Run all five diagnostics in one Python process.

    This lets orl_diag_common._load_history_cached reuse already parsed
    history.jsonl DataFrames across analyses 01--05.  Pass only arguments
    shared by the common parser, e.g. --log-root, --out-root,
    --early-start, --early-end, --diag-points, --primary-discount,
    --primary-tau, and --last-evals.
    """

    root = Path(__file__).resolve().parent
    forwarded = sys.argv[1:]
    original_argv = list(sys.argv)

    try:
        for script_name in SCRIPTS:
            script_path = root / script_name
            print("\n" + "=" * 80)
            print(f"[RUN] {script_name}")
            print("=" * 80)

            sys.argv = [str(script_path), *forwarded]
            runpy.run_path(
                str(script_path),
                run_name="__main__",
            )
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
