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


METRICS = {
    "source_loo": "diag/filter_source_loo_corr",
    "receiver_loo": "diag/filter_receiver_loo_corr",
    "source_loo_abs": "diag/filter_source_loo_corr",
    "receiver_loo_abs": "diag/filter_receiver_loo_corr",
    "source_adv": "diag/filter_source_adv_corr",
    "receiver_adv": "diag/filter_receiver_adv_corr",
    "source_receiver_error_abs": (
        "diag/source_receiver_error_corr_abs"
    ),
    "filter_weight": "diag/filter_weight_mean",
}


def seed_value(
    run,
    key,
    args,
):
    absolute = key.endswith("_abs")

    metric = METRICS[key]

    return seed_metric_mean(
        run,
        metric,
        args.early_start,
        args.early_end,
        absolute=absolute,
    )


def main():
    parser = common_parser(
        "DD-IQL source/receiver decoupling diagnostics."
    )

    args = parser.parse_args()

    runs = discover_runs(
        args.log_root
    )

    dd_groups = group_by_condition(
        [
            r for r in runs
            if r.algo == "DD-IQL"
        ]
    )

    out_dir = (
        args.out_root
        / "03_source_receiver_decoupling"
    )

    decoupling_all = []
    decoupling_primary = []

    signal_all = []
    signal_primary = []

    csv_records = []

    for condition in sorted(
        dd_groups,
        key=condition_sort_key,
    ):
        env, gamma, tau = condition
        group = dd_groups[condition]

        summaries = {}

        for key in METRICS:
            values = [
                seed_value(
                    run,
                    key,
                    args,
                )
                for run in group
            ]

            summaries[key] = (
                summarize_seed_values(
                    values
                )
            )

        decoupling_row = make_latex_row(
            [
                display_env(env),
                float_label(gamma),
                float_label(tau),
                format_pm(
                    summaries["source_loo"][0],
                    summaries["source_loo"][1],
                    digits=3,
                ),
                format_pm(
                    summaries["receiver_loo"][0],
                    summaries["receiver_loo"][1],
                    digits=3,
                ),
                format_pm(
                    summaries["source_loo_abs"][0],
                    summaries["source_loo_abs"][1],
                    digits=3,
                ),
                format_pm(
                    summaries["receiver_loo_abs"][0],
                    summaries["receiver_loo_abs"][1],
                    digits=3,
                ),
                format_pm(
                    summaries[
                        "source_receiver_error_abs"
                    ][0],
                    summaries[
                        "source_receiver_error_abs"
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
                    summaries["source_adv"][0],
                    summaries["source_adv"][1],
                    digits=3,
                ),
                format_pm(
                    summaries["receiver_adv"][0],
                    summaries["receiver_adv"][1],
                    digits=3,
                ),
                format_pm(
                    summaries["filter_weight"][0],
                    summaries["filter_weight"][1],
                    digits=3,
                ),
            ]
        )

        decoupling_all.append(
            decoupling_row
        )

        signal_all.append(
            signal_row
        )

        if is_primary_condition(
            condition,
            args.primary_discount,
            args.primary_tau,
        ):
            decoupling_primary.append(
                decoupling_row
            )

            signal_primary.append(
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

    common_comments = [
        (
            f"Diagnostic window: "
            f"{args.early_start}--{args.early_end}."
        ),
        (
            "LOO = leave-one-out ensemble disagreement: "
            "Q_i minus the mean of all other ensemble critics."
        ),
        (
            "LOO is an estimator-specific disagreement proxy, "
            "not an oracle Q-error."
        ),
        (
            "Values are averaged within each seed first, "
            "then reported as mean +- seed std."
        ),
    ]

    write_latex_rows(
        out_dir
        / "latex_source_receiver_loo_all_rows.txt",
        "DD-IQL source vs receiver LOO coupling -- all settings",
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "Source LOO corr.",
            "Receiver LOO corr.",
            "Source |LOO corr.|",
            "Receiver |LOO corr.|",
            "Source--receiver error |corr.|",
        ],
        decoupling_all,
        common_comments,
    )

    write_latex_rows(
        out_dir
        / "latex_source_receiver_loo_primary_rows.txt",
        (
            "DD-IQL source vs receiver LOO coupling "
            "-- primary bias-stress setting"
        ),
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "Source LOO corr.",
            "Receiver LOO corr.",
            "Source |LOO corr.|",
            "Receiver |LOO corr.|",
            "Source--receiver error |corr.|",
        ],
        decoupling_primary,
        common_comments,
    )

    write_latex_rows(
        out_dir
        / "latex_filter_signal_all_rows.txt",
        "DD-IQL filtering-signal preservation -- all settings",
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "Source-adv. corr.",
            "Receiver-adv. corr.",
            "Mean filter weight",
        ],
        signal_all,
        common_comments,
    )

    write_latex_rows(
        out_dir
        / "latex_filter_signal_primary_rows.txt",
        (
            "DD-IQL filtering-signal preservation "
            "-- primary bias-stress setting"
        ),
        [
            "Environment",
            r"$\gamma$",
            r"$\tau$",
            "Source-adv. corr.",
            "Receiver-adv. corr.",
            "Mean filter weight",
        ],
        signal_primary,
        common_comments,
    )

    write_csv(
        out_dir
        / "source_receiver_decoupling_summary.csv",
        csv_records,
    )


if __name__ == "__main__":
    main()