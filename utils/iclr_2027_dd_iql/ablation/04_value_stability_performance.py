from __future__ import annotations

from orl_diag_common import (
    common_parser,
    condition_sort_key,
    discover_runs,
    display_env,
    final_eval_seed_value,
    float_label,
    format_pm,
    group_by_condition,
    is_primary_condition,
    make_latex_row,
    summarize_seed_values,
    write_csv,
    write_latex_rows,
)


def summarize_eval_metric(
    group,
    metric,
    last_evals,
):
    values = [
        final_eval_seed_value(
            run,
            metric,
            last_evals=last_evals,
        )
        for run in group
    ]

    return summarize_seed_values(
        values
    )


def main():
    parser = common_parser(
        "Evaluation rollout value stability "
        "and policy performance."
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
        / "04_value_stability_performance"
    )

    main_all = []
    main_primary = []

    calibration_all = []
    calibration_primary = []

    csv_records = []

    for condition in conditions:
        env, gamma, tau = condition

        iql = iql_groups[condition]
        dd = dd_groups[condition]

        metrics = {}

        for algo, group in (
            ("iql", iql),
            ("ddiql", dd),
        ):
            for metric_name, metric_key in (
                (
                    "success",
                    "eval/success_rate",
                ),
                (
                    "q_gap",
                    "eval/q_return_gap_mean",
                ),
                (
                    "q_rmse",
                    "eval/q_return_gap_rmse",
                ),
                (
                    "q_over_fraction",
                    "eval/q_over_return_fraction",
                ),
                (
                    "v_gap",
                    "eval/v_return_gap_mean",
                ),
                (
                    "v_rmse",
                    "eval/v_return_gap_rmse",
                ),
            ):
                metrics[
                    f"{algo}_{metric_name}"
                ] = summarize_eval_metric(
                    group,
                    metric_key,
                    args.last_evals,
                )

        main_row = make_latex_row(
            [
                display_env(env),
                float_label(gamma),
                float_label(tau),
                format_pm(
                    metrics["iql_success"][0],
                    metrics["iql_success"][1],
                    digits=3,
                ),
                format_pm(
                    metrics["ddiql_success"][0],
                    metrics["ddiql_success"][1],
                    digits=3,
                ),
                format_pm(
                    metrics["iql_q_gap"][0],
                    metrics["iql_q_gap"][1],
                    digits=2,
                ),
                format_pm(
                    metrics["ddiql_q_gap"][0],
                    metrics["ddiql_q_gap"][1],
                    digits=2,
                ),
            ]
        )

        calibration_row = make_latex_row(
            [
                display_env(env),
                float_label(gamma),
                float_label(tau),
                format_pm(
                    metrics["iql_q_rmse"][0],
                    metrics["iql_q_rmse"][1],
                    digits=2,
                ),
                format_pm(
                    metrics["ddiql_q_rmse"][0],
                    metrics["ddiql_q_rmse"][1],
                    digits=2,
                ),
                format_pm(
                    metrics[
                        "iql_q_over_fraction"
                    ][0],
                    metrics[
                        "iql_q_over_fraction"
                    ][1],
                    digits=3,
                ),
                format_pm(
                    metrics[
                        "ddiql_q_over_fraction"
                    ][0],
                    metrics[
                        "ddiql_q_over_fraction"
                    ][1],
                    digits=3,
                ),
            ]
        )

        main_all.append(main_row)
        calibration_all.append(
            calibration_row
        )

        if is_primary_condition(
            condition,
            args.primary_discount,
            args.primary_tau,
        ):
            main_primary.append(
                main_row
            )
            calibration_primary.append(
                calibration_row
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
        ) in metrics.items():
            record[f"{key}_mean"] = mean
            record[f"{key}_std"] = std
            record[f"{key}_n"] = n

        csv_records.append(record)

    comments = [
        (
            f"Within each seed, the final "
            f"{args.last_evals} evaluation(s) are averaged."
        ),
        (
            "Reported uncertainty is std across training seeds."
        ),
        (
            "Q-return gap = predicted Q minus realized "
            "discounted return along the evaluation rollout."
        ),
        (
            "These are empirical rollout calibration/stability "
            "diagnostics, not oracle Q* bias."
        ),
    ]

    write_latex_rows(
        out_dir
        / "latex_success_qgap_all_rows.txt",
        "Policy success and rollout Q-return gap -- all settings",
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "IQL success",
            "DD-IQL success",
            "IQL Q-return gap",
            "DD-IQL Q-return gap",
        ],
        main_all,
        comments,
    )

    write_latex_rows(
        out_dir
        / "latex_success_qgap_primary_rows.txt",
        (
            "Policy success and rollout Q-return gap "
            "-- primary bias-stress setting"
        ),
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "IQL success",
            "DD-IQL success",
            "IQL Q-return gap",
            "DD-IQL Q-return gap",
        ],
        main_primary,
        comments,
    )

    write_latex_rows(
        out_dir
        / "latex_qcalibration_all_rows.txt",
        "Rollout Q calibration -- all settings",
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "IQL Q-RMSE",
            "DD-IQL Q-RMSE",
            "IQL over-return fraction",
            "DD-IQL over-return fraction",
        ],
        calibration_all,
        comments,
    )

    write_latex_rows(
        out_dir
        / "latex_qcalibration_primary_rows.txt",
        (
            "Rollout Q calibration "
            "-- primary bias-stress setting"
        ),
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "IQL Q-RMSE",
            "DD-IQL Q-RMSE",
            "IQL over-return fraction",
            "DD-IQL over-return fraction",
        ],
        calibration_primary,
        comments,
    )

    write_csv(
        out_dir
        / "value_stability_performance_summary.csv",
        csv_records,
    )


if __name__ == "__main__":
    main()