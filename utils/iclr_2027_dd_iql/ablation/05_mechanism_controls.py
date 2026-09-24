from __future__ import annotations

from orl_diag_common import (
    common_parser,
    condition_sort_key,
    discover_runs,
    display_env,
    float_label,
    format_pm,
    group_by_condition,
    is_primary_condition,
    make_latex_row,
    seed_metric_mean,
    summarize_seed_values,
    write_csv,
    write_latex_rows,
)


def summarize_metric(
    group,
    metric,
    args,
):
    values = [
        seed_metric_mean(
            run,
            metric,
            args.early_start,
            args.early_end,
        )
        for run in group
    ]

    return summarize_seed_values(
        values
    )


def main():
    parser = common_parser(
        "Mechanism-control diagnostics."
    )

    args = parser.parse_args()

    runs = discover_runs(
        args.log_root
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

    conditions = sorted(
        set(iql_groups)
        & set(dd_groups),
        key=condition_sort_key,
    )

    out_dir = (
        args.out_root
        / "05_mechanism_controls"
    )

    error_rows_all = []
    error_rows_primary = []

    signal_rows_all = []
    signal_rows_primary = []

    csv_records = []

    for condition in conditions:
        env, gamma, tau = condition

        summaries = {}

        for algo, group in (
            ("iql", iql_groups[condition]),
            ("ddiql", dd_groups[condition]),
        ):
            summaries[
                f"{algo}_td_corr"
            ] = summarize_metric(
                group,
                "diag/filter_receiver_td_corr",
                args,
            )

            summaries[
                f"{algo}_td_rmse"
            ] = summarize_metric(
                group,
                "diag/td_residual_rmse",
                args,
            )

            summaries[
                f"{algo}_source_adv_corr"
            ] = summarize_metric(
                group,
                "diag/filter_source_adv_corr",
                args,
            )

            summaries[
                f"{algo}_filter_weight"
            ] = summarize_metric(
                group,
                "diag/filter_weight_mean",
                args,
            )

        error_row = make_latex_row(
            [
                display_env(env),
                float_label(gamma),
                float_label(tau),
                format_pm(
                    summaries[
                        "iql_td_corr"
                    ][0],
                    summaries[
                        "iql_td_corr"
                    ][1],
                    digits=3,
                ),
                format_pm(
                    summaries[
                        "ddiql_td_corr"
                    ][0],
                    summaries[
                        "ddiql_td_corr"
                    ][1],
                    digits=3,
                ),
                format_pm(
                    summaries[
                        "iql_td_rmse"
                    ][0],
                    summaries[
                        "iql_td_rmse"
                    ][1],
                    digits=3,
                ),
                format_pm(
                    summaries[
                        "ddiql_td_rmse"
                    ][0],
                    summaries[
                        "ddiql_td_rmse"
                    ][1],
                    digits=3,
                ),
            ]
        )

        signal_row = make_latex_row(
            [
                display_env(env),
                float_label(gamma),
                float_label(tau),
                format_pm(
                    summaries[
                        "iql_source_adv_corr"
                    ][0],
                    summaries[
                        "iql_source_adv_corr"
                    ][1],
                    digits=3,
                ),
                format_pm(
                    summaries[
                        "ddiql_source_adv_corr"
                    ][0],
                    summaries[
                        "ddiql_source_adv_corr"
                    ][1],
                    digits=3,
                ),
                format_pm(
                    summaries[
                        "iql_filter_weight"
                    ][0],
                    summaries[
                        "iql_filter_weight"
                    ][1],
                    digits=3,
                ),
                format_pm(
                    summaries[
                        "ddiql_filter_weight"
                    ][0],
                    summaries[
                        "ddiql_filter_weight"
                    ][1],
                    digits=3,
                ),
            ]
        )

        error_rows_all.append(
            error_row
        )
        signal_rows_all.append(
            signal_row
        )

        if is_primary_condition(
            condition,
            args.primary_discount,
            args.primary_tau,
        ):
            error_rows_primary.append(
                error_row
            )
            signal_rows_primary.append(
                signal_row
            )

        record = {
            "environment": env,
            "discount": gamma,
            "expectile": tau,
        }

        for key, (
            mean,
            std,
            n,
        ) in summaries.items():
            record[f"{key}_mean"] = mean
            record[f"{key}_std"] = std
            record[f"{key}_n"] = n

        csv_records.append(record)

    comments = [
        (
            f"Diagnostic window: "
            f"{args.early_start}--{args.early_end}."
        ),
        (
            "TD correlation measures coupling; TD-RMSE measures "
            "raw Bellman-residual magnitude."
        ),
        (
            "A small filter-error correlation does not require "
            "a small TD residual magnitude."
        ),
        (
            "Source-advantage correlation tests whether the "
            "intended filtering signal remains present."
        ),
    ]

    write_latex_rows(
        out_dir
        / "latex_coupling_vs_error_scale_all_rows.txt",
        "Filter-error coupling versus TD-error scale -- all settings",
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "IQL TD corr.",
            "DD-IQL TD corr.",
            "IQL TD-RMSE",
            "DD-IQL TD-RMSE",
        ],
        error_rows_all,
        comments,
    )

    write_latex_rows(
        out_dir
        / "latex_coupling_vs_error_scale_primary_rows.txt",
        (
            "Filter-error coupling versus TD-error scale "
            "-- primary bias-stress setting"
        ),
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "IQL TD corr.",
            "DD-IQL TD corr.",
            "IQL TD-RMSE",
            "DD-IQL TD-RMSE",
        ],
        error_rows_primary,
        comments,
    )

    write_latex_rows(
        out_dir
        / "latex_signal_weight_all_rows.txt",
        "Filtering-signal and weight controls -- all settings",
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "IQL source-adv. corr.",
            "DD-IQL source-adv. corr.",
            "IQL mean weight",
            "DD-IQL mean weight",
        ],
        signal_rows_all,
        comments,
    )

    write_latex_rows(
        out_dir
        / "latex_signal_weight_primary_rows.txt",
        (
            "Filtering-signal and weight controls "
            "-- primary bias-stress setting"
        ),
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "IQL source-adv. corr.",
            "DD-IQL source-adv. corr.",
            "IQL mean weight",
            "DD-IQL mean weight",
        ],
        signal_rows_primary,
        comments,
    )

    write_csv(
        out_dir
        / "mechanism_controls_summary.csv",
        csv_records,
    )


if __name__ == "__main__":
    main()