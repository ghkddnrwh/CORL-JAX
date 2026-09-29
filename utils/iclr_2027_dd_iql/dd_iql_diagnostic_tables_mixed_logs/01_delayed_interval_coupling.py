from __future__ import annotations

import numpy as np
import pandas as pd

from orl_diag_common import (
    common_parser,
    condition_sort_key,
    condition_setting_label,
    discover_runs_from_args,
    display_env,
    evenly_spaced_mask,
    float_label,
    format_pm,
    group_by_condition,
    load_metric_frame,
    make_latex_row,
    summarize_seed_values,
    window_frame,
    write_csv,
    write_latex_rows,
)


def infer_default_ages(runs, delay_override=None):
    if delay_override is not None:
        d = int(delay_override)
        return [0, d // 4, d // 2, (3 * d) // 4]

    periods = sorted({int(r.delay_period) for r in runs if r.algo == "DD-IQL" and r.delay_period})
    if len(periods) == 1:
        d = periods[0]
        return [0, d // 4, d // 2, (3 * d) // 4]
    if not periods:
        raise RuntimeError("Could not infer DD-IQL delayed_update_period from configs.")
    raise RuntimeError(
        "Multiple DD-IQL delay periods were found. Please pass --delay-period or explicit --ages. "
        f"Observed periods: {periods}"
    )


def per_seed_delay_stats(
    run,
    metric,
    start,
    end,
    ages,
    absolute=False,
    delay_override=None,
    diag_points=0,
):
    frame = window_frame(load_metric_frame(run, metric, prefer_delayed=True), start, end)
    if frame.empty or metric not in frame.columns:
        return None

    delay_period = int(delay_override) if delay_override is not None else run.delay_period
    if delay_period is None or delay_period <= 0:
        return None

    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    timestep = frame["timestep"].to_numpy(dtype=int)

    # Prefer the logged delay age.  Fall back to timestep % D for older logs.
    if "diag/delay_age" in frame.columns:
        delay_age = pd.to_numeric(frame["diag/delay_age"], errors="coerce").to_numpy(dtype=float)
    else:
        delay_age = (timestep % delay_period).astype(float)

    valid = np.isfinite(values) & np.isfinite(delay_age)
    values = values[valid]
    timestep = timestep[valid]
    delay_age = delay_age[valid].astype(int)
    if absolute:
        values = np.abs(values)
    if len(values) == 0:
        return None

    # Subsample separately by age so all phases remain represented even though
    # the high-frequency delayed log and the common bias log use different cadences.
    if diag_points is not None and int(diag_points) > 0:
        keep = np.zeros(len(timestep), dtype=bool)
        for observed_age in np.unique(delay_age):
            idx = np.flatnonzero(delay_age == observed_age)
            local_keep = evenly_spaced_mask(timestep[idx], int(diag_points))
            keep[idx[local_keep]] = True
        timestep = timestep[keep]
        values = values[keep]
        delay_age = delay_age[keep]

    result = {"delay_period": delay_period}
    for age in ages:
        mask = delay_age == int(age)
        result[f"age_{age}"] = float(np.mean(values[mask])) if np.any(mask) else np.nan

    nonzero = delay_age != 0
    result["delayed_mean"] = float(np.mean(values[nonzero])) if np.any(nonzero) else np.nan
    result["interval_mean"] = float(np.mean(values))
    return result


def summarize_stat(seed_stats, key):
    return summarize_seed_values(
        item[key] for item in seed_stats if item is not None and key in item
    )


def make_table(*, args, runs, metric, absolute, file_stem, title):
    out_dir = args.out_root / "01_delayed_interval_coupling"
    dd_groups = group_by_condition([r for r in runs if r.algo == "DD-IQL"])

    rows = []
    records = []
    for condition in sorted(dd_groups, key=condition_sort_key):
        env, gamma, tau = condition
        seed_stats = [
            per_seed_delay_stats(
                run,
                metric,
                args.early_start,
                args.early_end,
                args.ages,
                absolute=absolute,
                delay_override=args.delay_period,
                diag_points=args.diag_points,
            )
            for run in dd_groups[condition]
        ]
        seed_stats = [x for x in seed_stats if x is not None]
        if not seed_stats:
            continue

        periods = sorted({int(x["delay_period"]) for x in seed_stats})
        delay_label = str(periods[0]) if len(periods) == 1 else "/".join(map(str, periods))

        cells = [
            display_env(env),
            condition_setting_label(condition),
            float_label(gamma),
            float_label(tau),
            delay_label,
        ]
        rec = {
            "environment": env,
            "setting": condition_setting_label(condition),
            "discount": gamma,
            "expectile": tau,
            "delay_period": delay_label,
            "metric": metric,
            "absolute": absolute,
        }

        for age in args.ages:
            mean, std, n = summarize_stat(seed_stats, f"age_{age}")
            cells.append(format_pm(mean, std, digits=3))
            rec[f"dd_age_{age}_mean"] = mean
            rec[f"dd_age_{age}_std"] = std
            rec[f"dd_age_{age}_n"] = n

        delayed_mean, delayed_std, delayed_n = summarize_stat(seed_stats, "delayed_mean")
        interval_mean, interval_std, interval_n = summarize_stat(seed_stats, "interval_mean")
        cells += [
            format_pm(delayed_mean, delayed_std, digits=3),
            format_pm(interval_mean, interval_std, digits=3),
        ]
        rec.update(
            dd_delayed_mean=delayed_mean,
            dd_delayed_std=delayed_std,
            dd_delayed_n=delayed_n,
            dd_interval_mean=interval_mean,
            dd_interval_std=interval_std,
            dd_interval_n=interval_n,
        )
        rows.append(make_latex_row(cells))
        records.append(rec)

    columns = [
        "Environment",
        "Setting",
        r"$\gamma$",
        r"$\tau$",
        "$D$",
        *[f"DD age {age}" for age in args.ages],
        "DD delayed mean",
        "DD interval mean",
    ]
    comments = [
        "This table uses the high-frequency delayed_diagnostics.npz log.",
        (
            "The current DD-IQL reruns log receiver TD correlation only in the coarse "
            "bias_diagnostics.npz stream; with D=100 and diagnostic_freq=2500 those TD "
            "samples all occur at delay age 0. Therefore the old delay-age TD table "
            "cannot be reconstructed at ages 25/50/75 from these files."
        ),
        (
            "Instead, this analysis reports the logged temporal filter--error coupling "
            "diag/filter_receiver_temporal_{corr,cov}, which is available every 25 steps."
        ),
        f"Diagnostic window: {args.early_start}--{args.early_end}.",
        f"At most {args.diag_points} points per seed are sampled independently within each delay age.",
        "Each value is first averaged within a seed and then summarized as mean +- std across seeds.",
    ]

    for suffix in ("all_rows", "table_rows", "primary_rows"):
        write_latex_rows(
            out_dir / f"latex_{file_stem}_{suffix}.txt",
            title,
            columns,
            rows,
            comments,
        )
    write_csv(out_dir / f"{file_stem}_summary.csv", records)


def main():
    parser = common_parser("Delayed-filter interval analysis for the mixed-format 2026-09-25 logs.")
    parser.add_argument(
        "--ages",
        nargs="+",
        type=int,
        default=None,
        help="Delay ages to report. Default: 0, D/4, D/2, 3D/4 inferred from DD-IQL configs.",
    )
    parser.add_argument(
        "--delay-period",
        type=int,
        default=None,
        help="Optional delay-period override. Otherwise use delayed_update_period from each DD-IQL config.",
    )
    args = parser.parse_args()
    runs = discover_runs_from_args(args)
    if args.ages is None:
        args.ages = infer_default_ages(runs, args.delay_period)
        print(f"[Delay ages] inferred {args.ages}")

    make_table(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_temporal_corr",
        absolute=False,
        file_stem="delay_temporal_corr",
        title="Delay-age receiver temporal correlation",
    )
    make_table(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_temporal_corr",
        absolute=True,
        file_stem="delay_temporal_corr_abs",
        title="Delay-age absolute receiver temporal correlation",
    )
    make_table(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_temporal_cov",
        absolute=False,
        file_stem="delay_temporal_cov",
        title="Delay-age receiver temporal covariance",
    )
    make_table(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_temporal_cov",
        absolute=True,
        file_stem="delay_temporal_cov_abs",
        title="Delay-age absolute receiver temporal covariance",
    )


if __name__ == "__main__":
    main()
