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

    Run all analyses in one process so the compact W&B history cache is reused
    across analyses 02--05.  The 2026-09-25 inputs use different formats:
    IQL is a W&B JSONL download, while DD-IQL is a local NPZ checkpoint tree.

    Typical invocation from the project root:
      python 00_run_all_diagnostics.py \
        --iql-log-root logs/wandb_logs_2026_09_25_orl_smooth_144 \
        --ddiql-log-root logs/iclr2027/iql_based/ogbench/diagnostics_dd_iql \
        --out-root paper_diagnostic_tables_20260925
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
