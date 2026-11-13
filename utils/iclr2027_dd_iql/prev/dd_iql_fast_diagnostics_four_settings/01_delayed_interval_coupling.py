from __future__ import annotations

import numpy as np

from orl_diag_common import (
    common_parser,
    condition_sort_key,
    condition_setting_label,
    discover_runs,
    display_env,
    float_label,
    format_pm,
    group_by_condition,
    load_diag,
    make_latex_row,
    seed_metric_mean,
    summarize_seed_values,
    evenly_spaced_mask,
    window_frame,
    write_csv,
    write_latex_rows,
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
    frame = window_frame(
        load_diag(run),
        start,
        end,
    )

    if metric not in frame.columns:
        return None

    delay_period = (
        int(delay_override)
        if delay_override is not None
        else run.delay_period
    )

    if delay_period is None or delay_period <= 0:
        return None

    timestep = (
        frame["timestep"]
        .to_numpy(dtype=int)
    )

    values = (
        frame[metric]
        .to_numpy(dtype=float)
    )

    valid = np.isfinite(values)

    timestep = timestep[valid]
    values = values[valid]

    if absolute:
        values = np.abs(values)

    if len(values) == 0:
        return None

    delay_age = timestep % delay_period

    # Subsample independently inside each delay-age group so that
    # age 0/250/500/750 remain balanced instead of being accidentally
    # removed by a global training-step stride.
    if diag_points is not None and int(diag_points) > 0:
        keep = np.zeros(len(timestep), dtype=bool)

        for observed_age in np.unique(delay_age):
            idx = np.flatnonzero(delay_age == observed_age)
            local_keep = evenly_spaced_mask(
                timestep[idx],
                int(diag_points),
            )
            keep[idx[local_keep]] = True

        timestep = timestep[keep]
        values = values[keep]
        delay_age = delay_age[keep]

    result = {
        "delay_period": delay_period,
    }

    for age in ages:
        mask = delay_age == age

        result[f"age_{age}"] = (
            float(np.mean(values[mask]))
            if np.any(mask)
            else np.nan
        )

    delayed_mask = delay_age != 0

    result["delayed_mean"] = (
        float(
            np.mean(
                values[delayed_mask]
            )
        )
        if np.any(delayed_mask)
        else np.nan
    )

    result["interval_mean"] = (
        float(np.mean(values))
    )

    return result


def summarize_stat(
    seed_stats,
    key,
):
    values = [
        item[key]
        for item in seed_stats
        if item is not None
        and key in item
    ]

    return summarize_seed_values(values)


def make_table(
    *,
    args,
    runs,
    metric,
    absolute,
    file_stem,
    title,
):
    out_dir = (
        args.out_root
        / "01_delayed_interval_coupling"
    )

    dd_groups = group_by_condition(
        [
            r for r in runs
            if r.algo == "DD-IQL"
        ]
    )

    iql_groups = group_by_condition(
        [
            r for r in runs
            if r.algo == "IQL"
        ]
    )

    all_rows = []
    csv_records = []

    for condition in sorted(
        dd_groups,
        key=condition_sort_key,
    ):
        env, gamma, tau = condition

        dd_runs = dd_groups[condition]
        iql_runs = iql_groups.get(
            condition,
            [],
        )

        dd_seed_stats = [
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
            for run in dd_runs
        ]

        dd_seed_stats = [
            item
            for item in dd_seed_stats
            if item is not None
        ]

        if not dd_seed_stats:
            continue

        delay_periods = sorted(
            {
                int(item["delay_period"])
                for item in dd_seed_stats
            }
        )

        delay_label = (
            str(delay_periods[0])
            if len(delay_periods) == 1
            else "/".join(
                str(x)
                for x in delay_periods
            )
        )

        iql_values = [
            seed_metric_mean(
                run,
                metric,
                args.early_start,
                args.early_end,
                absolute=absolute,
                max_points=args.diag_points,
            )
            for run in iql_runs
        ]

        (
            iql_mean,
            iql_std,
            iql_n,
        ) = summarize_seed_values(
            iql_values
        )

        cells = [
            display_env(env),
            condition_setting_label(condition),
            float_label(gamma),
            float_label(tau),
            delay_label,
            format_pm(
                iql_mean,
                iql_std,
                digits=3,
            ),
        ]

        record = {
            "environment": env,
            "setting": condition_setting_label(condition),
            "discount": gamma,
            "expectile": tau,
            "delay_period": delay_label,
            "metric": metric,
            "absolute": absolute,
            "iql_mean": iql_mean,
            "iql_std": iql_std,
            "iql_n": iql_n,
        }

        for age in args.ages:
            mean, std, n = summarize_stat(
                dd_seed_stats,
                f"age_{age}",
            )

            cells.append(
                format_pm(
                    mean,
                    std,
                    digits=3,
                )
            )

            record[
                f"dd_age_{age}_mean"
            ] = mean
            record[
                f"dd_age_{age}_std"
            ] = std
            record[
                f"dd_age_{age}_n"
            ] = n

        (
            delayed_mean,
            delayed_std,
            delayed_n,
        ) = summarize_stat(
            dd_seed_stats,
            "delayed_mean",
        )

        (
            interval_mean,
            interval_std,
            interval_n,
        ) = summarize_stat(
            dd_seed_stats,
            "interval_mean",
        )

        cells.extend(
            [
                format_pm(
                    delayed_mean,
                    delayed_std,
                    digits=3,
                ),
                format_pm(
                    interval_mean,
                    interval_std,
                    digits=3,
                ),
            ]
        )

        record.update(
            {
                "dd_delayed_mean": delayed_mean,
                "dd_delayed_std": delayed_std,
                "dd_delayed_n": delayed_n,
                "dd_interval_mean": interval_mean,
                "dd_interval_std": interval_std,
                "dd_interval_n": interval_n,
            }
        )

        row = make_latex_row(cells)

        all_rows.append(row)
        csv_records.append(record)

    columns = [
        "Environment",
        "Setting",
        r"$\gamma$",
        r"$\tau$",
        "$D$",
        "IQL mean",
        *[
            f"DD age {age}"
            for age in args.ages
        ],
        "DD delayed mean",
        "DD interval mean",
    ]

    comments = [
        (
            "Row order within each environment: Original/tau=0.9, "
            "Original/tau=0.99, High-gamma/tau=0.9, High-gamma/tau=0.99."
        ),
        (
            f"Diagnostic window: "
            f"{args.early_start}--{args.early_end}."
        ),
        (
            f"Diagnostic checkpoints: at most {args.diag_points} points "
            "per seed (applied independently within each DD delay age)."
        ),
        (
            "Each value is first averaged within each seed, "
            "then reported as mean +- std across seeds."
        ),
        (
            "Age 0 denotes the diagnostic immediately after "
            "the delayed snapshot refresh."
        ),
        (
            "Delayed mean averages diagnostic samples with "
            "delay_age != 0; interval mean averages all ages."
        ),
    ]

    write_latex_rows(
        out_dir / f"latex_{file_stem}_all_rows.txt",
        title + " -- all settings",
        columns,
        all_rows,
        comments,
    )

    write_latex_rows(
        out_dir / f"latex_{file_stem}_table_rows.txt",
        title + " -- four settings per environment",
        columns,
        all_rows,
        comments,
    )

    # Backward-compatible legacy filename. It now contains all four settings.
    write_latex_rows(
        out_dir / f"latex_{file_stem}_primary_rows.txt",
        title + " -- four settings per environment (legacy filename)",
        columns,
        all_rows,
        comments,
    )

    write_csv(
        out_dir / f"{file_stem}_summary.csv",
        csv_records,
    )


def main():
    parser = common_parser(
        "Delayed-filter interval analysis."
    )

    parser.add_argument(
        "--ages",
        nargs="+",
        type=int,
        default=[0, 250, 500, 750],
    )

    parser.add_argument(
        "--delay-period",
        type=int,
        default=None,
        help=(
            "Optional override. If omitted, use "
            "delayed_update_period from each run config."
        ),
    )

    args = parser.parse_args()

    runs = discover_runs(
        args.log_root
    )

    make_table(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_td_corr",
        absolute=False,
        file_stem="delay_td_corr",
        title="Delay-age receiver TD correlation",
    )

    make_table(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_td_corr",
        absolute=True,
        file_stem="delay_td_corr_abs",
        title="Delay-age absolute receiver TD correlation",
    )

    make_table(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_td_cov",
        absolute=False,
        file_stem="delay_td_cov",
        title="Delay-age receiver TD covariance",
    )

    make_table(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_td_cov",
        absolute=True,
        file_stem="delay_td_cov_abs",
        title="Delay-age absolute receiver TD covariance",
    )


if __name__ == "__main__":
    main()