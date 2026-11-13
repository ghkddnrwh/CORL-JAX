"""Report every pilot trial when comparing IID and NN-like noise.

Only field correlation parameters and the fixed DD delay are searched. The
MDP, data, N=2, sigma, alpha, gamma and budget remain fixed across trials.
Use separate seed IDs for confirmation after choosing a pilot configuration.
"""

import argparse
import json
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE_DIR))

import numpy as np
from stress_iid_min_ddq_bias import make_noise, run_batch, save_csv
from tabular_dd_iql import validate_dd_iql_config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=30_000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dd-method", choices=["dd_iql", "dd_q"], default="dd_iql")
    parser.add_argument("--expectile-tau", type=float, default=0.7)
    parser.add_argument("--alpha-v", type=float, default=None)
    parser.add_argument("--q-target-tau", type=float, default=0.005)
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.9, 0.99, 0.999])
    parser.add_argument(
        "--action-scales", nargs="+", type=float, default=[0.2, 0.5, 1.0]
    )
    parser.add_argument("--delays", nargs="+", type=int, default=[1, 10, 100, 1000])
    args = parser.parse_args()
    if args.alpha_v is None:
        args.alpha_v = args.alpha
    if args.dd_method == "dd_iql":
        validate_dd_iql_config(
            args.alpha, args.alpha_v, args.expectile_tau, args.q_target_tau
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_path = CORE_DIR / "compare_tabular_methods.py"
    source = source_path.read_text()
    seed_ids = list(range(args.num_seeds))
    actions = np.stack(
        [
            np.random.default_rng(2_000_003 + seed).integers(4, size=args.updates)
            for seed in seed_ids
        ]
    )
    metadata = dict(
        updates=args.updates,
        seed_ids=seed_ids,
        gamma=args.gamma,
        alpha=args.alpha,
        sigma=1.0,
        field_features=64,
        dd_method=args.dd_method,
        expectile_tau=args.expectile_tau,
        alpha_v=args.alpha_v,
        q_target_tau=args.q_target_tau,
        field_lengthscale=0.35,
        field_shared_fraction=0.0,
        rhos=args.rhos,
        action_scales=args.action_scales,
        delays=args.delays,
        role="pilot: report all trials",
        environment="one continuing state, four zero-reward self-loop actions",
        behavior="same fixed uniform action dataset for every trial",
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "compare_tabular_methods_snapshot.py").write_text(source)
    (args.output_dir / "tabular_dd_iql.py").write_text(
        source_path.with_name("tabular_dd_iql.py").read_text(),
    )
    rows = []

    def evaluate(noise, mode, rho, scale):
        trials = [("clipped_double_q", 1), ("double_q", 1)]
        trials += [(args.dd_method, delay) for delay in args.delays]
        for method, delay in trials:
            history = run_batch(
                method,
                noise,
                actions,
                args.gamma,
                args.alpha,
                1000,
                delay,
                expectile_tau=args.expectile_tau,
                alpha_v=args.alpha_v,
                q_target_tau=args.q_target_tau,
            )
            tail = history["step"] > 0.75 * args.updates
            tail_means = history["v_bias"][:, tail].mean(axis=1)
            final = history["v_bias"][:, -1]
            row = dict(
                noise_mode=mode,
                rho=rho,
                action_scale=scale,
                dd_delay_steps=delay if method == args.dd_method else 0,
                method=method,
                tail_bias_mean=float(tail_means.mean()),
                tail_bias_sd=float(tail_means.std(ddof=1)),
                final_bias_mean=float(final.mean()),
                final_bias_sd=float(final.std(ddof=1)),
            )
            rows.append(row)
            save_csv(args.output_dir / "all_trials.csv", rows)
            print(json.dumps(row), flush=True)

    iid = make_noise(
        source, source_path, seed_ids, args.updates, 1.0, "nn_like_field_iid"
    )
    evaluate(iid, "nn_like_field_iid", 0.0, 0.0)
    for scale in args.action_scales:
        for rho in args.rhos:
            field = make_noise(
                source,
                source_path,
                seed_ids,
                args.updates,
                1.0,
                "nn_like_field",
                rho,
                64,
                0.35,
                scale,
            )
            evaluate(field, "nn_like_field", rho, scale)


if __name__ == "__main__":
    main()
