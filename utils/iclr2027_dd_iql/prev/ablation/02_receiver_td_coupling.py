from __future__ import annotations

import numpy as np

from orl_diag_common import (
    common_parser,
    condition_sort_key,
    discover_runs,
    display_env,
    float_label,
    format_percent,
    format_pm,
    group_by_condition,
    is_primary_condition,
    make_latex_row,
    seed_metric_mean,
    summarize_seed_values,
    write_csv,
    write_latex_rows,
)


def analyze_metric(
    *,
    args,
    runs,
    metric,
    stem,
    title,
):
    out_dir = (
        args.out_root
        / "02_receiver_td_coupling"
    )

    iql_groups = group_by_condition(
        [
            r for r in runs
            if r.algo == "IQL"
        ]
    )

    dd_groups = group_by_condition(
        [
            r for r in runs
            if r.algo == "DD-IQL"
        ]
    )

    common_conditions = sorted(
        set(iql_groups)
        & set(dd_groups),
        key=condition_sort_key,
    )

    all_rows = []
    primary_rows = []
    csv_records = []

    num_dd_abs_smaller = 0
    num_valid_conditions = 0

    for condition in common_conditions:
        env, gamma, tau = condition

        iql_signed = [
            seed_metric_mean(
                run,
                metric,
                args.early_start,
                args.early_end,
                absolute=False,
            )
            for run in iql_groups[condition]
        ]

        dd_signed = [
            seed_metric_mean(
                run,
                metric,
                args.early_start,
                args.early_end,
                absolute=False,
            )
            for run in dd_groups[condition]
        ]

        iql_abs = [
            seed_metric_mean(
                run,
                metric,
                args.early_start,
                args.early_end,
                absolute=True,
            )
            for run in iql_groups[condition]
        ]

        dd_abs = [
            seed_metric_mean(
                run,
                metric,
                args.early_start,
                args.early_end,
                absolute=True,
            )
            for run in dd_groups[condition]
        ]

        (
            iql_mean,
            iql_std,
            iql_n,
        ) = summarize_seed_values(
            iql_signed
        )

        (
            dd_mean,
            dd_std,
            dd_n,
        ) = summarize_seed_values(
            dd_signed
        )

        (
            iql_abs_mean,
            iql_abs_std,
            _,
        ) = summarize_seed_values(
            iql_abs
        )

        (
            dd_abs_mean,
            dd_abs_std,
            _,
        ) = summarize_seed_values(
            dd_abs
        )

        if (
            np.isfinite(iql_abs_mean)
            and np.isfinite(dd_abs_mean)
        ):
            num_valid_conditions += 1

            if dd_abs_mean < iql_abs_mean:
                num_dd_abs_smaller += 1

        if (
            np.isfinite(iql_abs_mean)
            and np.isfinite(dd_abs_mean)
            and iql_abs_mean > 1e-12
        ):
            abs_reduction_pct = (
                100.0
                * (
                    iql_abs_mean
                    - dd_abs_mean
                )
                / iql_abs_mean
            )
        else:
            abs_reduction_pct = np.nan

        row = make_latex_row(
            [
                display_env(env),
                float_label(gamma),
                float_label(tau),
                format_pm(
                    iql_mean,
                    iql_std,
                    digits=3,
                ),
                format_pm(
                    dd_mean,
                    dd_std,
                    digits=3,
                ),
                format_pm(
                    iql_abs_mean,
                    iql_abs_std,
                    digits=3,
                ),
                format_pm(
                    dd_abs_mean,
                    dd_abs_std,
                    digits=3,
                ),
                format_percent(
                    abs_reduction_pct,
                    digits=1,
                ),
            ]
        )

        all_rows.append(row)

        if is_primary_condition(
            condition,
            args.primary_discount,
            args.primary_tau,
        ):
            primary_rows.append(row)

        csv_records.append(
            {
                "environment": env,
                "discount": gamma,
                "expectile": tau,
                "metric": metric,
                "iql_signed_mean": iql_mean,
                "iql_signed_std": iql_std,
                "iql_n": iql_n,
                "ddiql_signed_mean": dd_mean,
                "ddiql_signed_std": dd_std,
                "ddiql_n": dd_n,
                "iql_abs_mean": iql_abs_mean,
                "iql_abs_std": iql_abs_std,
                "ddiql_abs_mean": dd_abs_mean,
                "ddiql_abs_std": dd_abs_std,
                "absolute_reduction_percent": (
                    abs_reduction_pct
                ),
            }
        )

    comments = [
        (
            f"Diagnostic window: "
            f"{args.early_start}--{args.early_end}."
        ),
        (
            "Signed and absolute metrics are first averaged "
            "within each seed, then summarized across seeds."
        ),
        (
            "Reduction uses mean absolute coupling: "
            "100 * (IQL - DD-IQL) / IQL."
        ),
        (
            f"DD-IQL has smaller mean absolute coupling in "
            f"{num_dd_abs_smaller}/{num_valid_conditions} "
            f"matched conditions."
        ),
    ]

    columns = [
        "Environment",
        r"$\gamma$",
        r"$\tau$",
        "IQL signed",
        "DD-IQL signed",
        "IQL absolute",
        "DD-IQL absolute",
        "Abs. reduction",
    ]

    write_latex_rows(
        out_dir / f"latex_{stem}_all_rows.txt",
        title + " -- all settings",
        columns,
        all_rows,
        comments,
    )

    write_latex_rows(
        out_dir / f"latex_{stem}_primary_rows.txt",
        title + " -- primary bias-stress setting",
        columns,
        primary_rows,
        comments,
    )

    write_csv(
        out_dir / f"{stem}_summary.csv",
        csv_records,
    )


def main():
    parser = common_parser(
        "IQL vs DD-IQL receiver-side "
        "filter-error coupling."
    )

    args = parser.parse_args()

    runs = discover_runs(
        args.log_root
    )

    analyze_metric(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_td_corr",
        stem="receiver_td_corr",
        title="Filter--receiver TD correlation",
    )

    analyze_metric(
        args=args,
        runs=runs,
        metric="diag/filter_receiver_td_cov",
        stem="receiver_td_cov",
        title="Filter--receiver TD covariance",
    )


if __name__ == "__main__":
    main()