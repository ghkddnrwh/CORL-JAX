"""Measure accumulated IID/NN-like noise bias in a continuing zero-reward MDP.

There are one or more states and 2, 4, 8, or 16 real actions. For multiple states,
action a moves to state a % num_states. Reward is zero and there is no
termination. Thus Q*=V*=0 for gamma<1.
Uniform behavior uses the same fixed action dataset for all methods.
Epsilon-greedy behavior shares random proposals and exploration draws;
actual actions depend on each method's current noisy Q. All methods share
Gaussian error draws. N=2 and max/min order are preserved.

The DD method defaults to tabular DD-IQL with learned expectile V tables.
Use --dd-method dd_q to reproduce the previous delayed-argmax experiments.
Use --dd-method sd_iql for self-delayed filtering or df_iql for two clipped
critics and one shared V (paper Appendix D.3). DF at D=1 is standard IQL;
At D=1, all IQL variants share the current Q observation between filtering
and regression (with the selected self/peer index). For D>1, the observed
noisy function stays frozen until the delayed snapshot refresh.
The comparison curves still report max_a mean_i Q_i; learned V is saved
separately in learned_value_curves.csv and the per-method NPZ files.

This environment and behavior dataset differ from the episodic GridWorld.
The batched implementation is checked against the original train() using
replayed errors and transitions before running the full experiment.

With --num-actions 2, only original actions 0 and 1 are available. Their
noise functions are retained exactly from the four-action field. Uniform
actions use the original four-action draws modulo the action count, preserving
the visited-state sequence in the two-state MDP when comparing 2 versus 4.
"""

import argparse
import csv
import hashlib
import json
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from utils.iclr2027_dd_iql.tabular_dd_iql import (
        dd_iql_filter_indices,
        dd_iql_update,
        validate_dd_iql_config,
    )
    from utils.iclr2027_dd_iql.tabular_q_backups import (
        CLIPPED_Q_BACKUPS,
        clipped_q_bootstrap,
    )
except ModuleNotFoundError:
    from tabular_dd_iql import (
        dd_iql_filter_indices,
        dd_iql_update,
        validate_dd_iql_config,
    )
    from tabular_q_backups import CLIPPED_Q_BACKUPS, clipped_q_bootstrap


def get_methods(dd_method="dd_iql"):
    labels = {
        "dd_iql": "DD-IQL tabular (N=2)",
        "sd_iql": "SD-IQL tabular (N=2)",
        "df_iql": "DF-IQL tabular (N=2, shared V)",
        "dd_q": "DD-Q (N=2)",
    }
    return {
        "q_learning": "Q-learning (N=1)",
        "clipped_double_q": "Min Q (N=2)",
        "double_q": "Double Q (N=2)",
        dd_method: labels[dd_method],
    }


METHODS = get_methods()
IQL_METHODS = ("dd_iql", "sd_iql", "df_iql")
GAUSSIAN_MAXMIN_FOUR = 0.2742687674321522


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_batch(
    method,
    noise,
    actions,
    gamma,
    alpha,
    log_every,
    dd_delay_steps=1,
    behavior="uniform",
    explore_mask=None,
    expectile_tau=0.7,
    alpha_v=None,
    q_target_tau=0.005,
    clipped_q_backup="max_min",
    df_reduction="min",
):
    if method not in (*get_methods("dd_iql"), "dd_q", "sd_iql", "df_iql"):
        raise ValueError(f"Unknown method: {method}")
    if method == "clipped_double_q" and clipped_q_backup not in CLIPPED_Q_BACKUPS:
        raise ValueError(f"clipped_q_backup must be one of {CLIPPED_Q_BACKUPS}")
    if dd_delay_steps < 1 or log_every < 1:
        raise ValueError("dd_delay_steps and log_every must be >= 1")
    if alpha_v is None:
        alpha_v = alpha
    if method in IQL_METHODS:
        validate_dd_iql_config(alpha, alpha_v, expectile_tau, q_target_tau)
    if method == "df_iql" and df_reduction not in ("min", "mean"):
        raise ValueError("df_reduction must be 'min' or 'mean'")
    df_reduce = np.min if df_reduction == "min" else np.mean
    seeds, updates = actions.shape
    num_states = noise.shape[3] if noise.ndim == 5 else 1
    num_actions = noise.shape[-1]
    if np.any(actions < 0) or np.any(actions >= num_actions):
        raise ValueError("actions must index the available noise/Q action axis")
    if noise.ndim == 4:
        noise = noise[:, :, :, None, :]
    n = 1 if method == "q_learning" else 2
    q = np.zeros((seeds, n, num_states, num_actions))
    index = np.arange(seeds)
    state = np.zeros(seeds, dtype=int)
    delayed = noise[:, 0, :n].copy() if method == "dd_q" else None
    steps, v_curves, q_curves, rmse_curves = (
        [0],
        [np.zeros(seeds)],
        [np.zeros(seeds)],
        [np.zeros(seeds)],
    )
    if method in IQL_METHODS:
        value = np.zeros((seeds, 1 if method == "df_iql" else n, num_states))
        q_target = q.copy()
        delayed = q_target + noise[:, 0, :n]
        value_delayed = value.copy()
        filter_indices = (
            dd_iql_filter_indices(n, 0) if method == "dd_iql" else np.arange(n)
        )
        value_tables = [value.mean(axis=1)]
    for t in range(updates):
        action = actions[:, t]
        if behavior == "epsilon_greedy":
            current = q[index, :, state] + noise[index, t, :n, state]
            action = np.where(
                explore_mask[:, t], action, current.mean(axis=1).argmax(axis=1)
            )
        next_state = action % num_states
        view = q[index, :, next_state] + noise[index, t, :n, next_state]
        if method in IQL_METHODS:
            observed_target_q = (
                q_target[index, :, state, action] + noise[index, t, :n, state, action]
            )
            if method == "df_iql":
                # Appendix D.3: clipped critics regress one shared V.
                # The JAX DF implementation uses current weights at D=1.
                observed_target_q = df_reduce(observed_target_q, axis=1, keepdims=True)
                delayed_adv = (
                    observed_target_q - value[index, :, state]
                    if dd_delay_steps == 1
                    else df_reduce(
                        delayed[index, :, state, action], axis=1, keepdims=True
                    )
                    - value_delayed[index, :, state]
                )
            else:
                # At D=1, delayed parameters equal current target-Q/V.
                # Reuse epsilon_t too; freezing epsilon_{t-1} here would
                # introduce an extra noise-only delay absent from DF/JAX.
                delayed_adv = (
                    observed_target_q - value[index, :, state]
                    if dd_delay_steps == 1
                    else delayed[index, :, state, action]
                    - value_delayed[index, :, state]
                )[:, filter_indices]
            q_new, v_new, _ = dd_iql_update(
                q[index, :, state, action],
                value[index, :, state],
                value[index, :, next_state],
                observed_target_q,
                delayed_adv,
                0.0,
                False,
                gamma,
                alpha,
                alpha_v,
                expectile_tau,
            )
            q[index, :, state, action] = q_new
            value[index, :, state] = v_new
            q_target = (1.0 - q_target_tau) * q_target + q_target_tau * q
            if (t + 1) % dd_delay_steps == 0:
                delayed = q_target + noise[:, t, :n]
                value_delayed = value.copy()
                if method == "dd_iql":
                    filter_indices = dd_iql_filter_indices(n, (t + 1) // dd_delay_steps)
        elif method == "q_learning":
            targets = gamma * view[:, 0].max(axis=1)[:, None]
        elif method == "clipped_double_q":
            targets = gamma * clipped_q_bootstrap(view, clipped_q_backup)[:, None]
        elif method == "double_q":
            selected = view.argmax(axis=2)
            targets = gamma * np.stack(
                (
                    view[index, 1, selected[:, 0]],
                    view[index, 0, selected[:, 1]],
                ),
                axis=1,
            )
        else:  # Legacy DD-Q: delayed peer argmax, current own evaluation.
            selected = delayed[index, :, next_state].argmax(axis=2)
            targets = gamma * np.stack(
                (
                    view[index, 0, selected[:, 1]],
                    view[index, 1, selected[:, 0]],
                ),
                axis=1,
            )
        if method not in IQL_METHODS:
            q[index, :, state, action] += alpha * (targets - q[index, :, state, action])
        if method == "dd_q" and (t + 1) % dd_delay_steps == 0:
            delayed = q + noise[:, t, :n]
        state = next_state
        if (t + 1) % log_every == 0 or t + 1 == updates:
            estimate = q.mean(axis=1)
            steps.append(t + 1)
            v_curves.append(estimate.max(axis=2).mean(axis=1))
            q_curves.append(estimate.mean(axis=(1, 2)))
            rmse_curves.append(np.sqrt(np.mean(estimate**2, axis=(1, 2))))
            if method in IQL_METHODS:
                value_tables.append(value.mean(axis=1))
    history = {
        "step": np.asarray(steps),
        "v_bias": np.stack(v_curves, axis=1),
        "mean_q_bias": np.stack(q_curves, axis=1),
        "q_rmse": np.stack(rmse_curves, axis=1),
        "q_final": q[:, :, 0] if num_states == 1 else q,
    }
    if method in IQL_METHODS:
        history.update(
            value_table=np.stack(value_tables, axis=1),
            value_final=value,
            value_delayed_final=value_delayed,
            q_target_final=q_target,
            q_delayed_observed_final=delayed,
            greedy_policy_final=q.mean(axis=1).argmax(axis=-1),
        )
        history["value_bias"] = history["value_table"].mean(axis=-1)
        history["value_rmse"] = np.sqrt(np.mean(history["value_table"] ** 2, axis=-1))
    return history


def check_against_original(
    source,
    source_path,
    noise,
    actions,
    gamma,
    alpha,
    dd_delay_steps=1,
    noise_mode="nn_like_field_iid",
    behavior="uniform",
    explore_mask=None,
    dd_method="dd_iql",
    expectile_tau=0.7,
    alpha_v=None,
    q_target_tau=0.005,
    clipped_q_backup="max_min",
    df_reduction="min",
):
    """Check every target/update against train() on the same recorded data.

    For other action counts, adapt only train's Q-table action-axis allocation.
    The transition/noise replay and optimal-value mock use the same action
    count. All algorithm update expressions in the reference remain intact.
    """
    num_actions = noise.shape[-1]
    if num_actions != 4:
        allocation = "Qs = np.zeros((n, grid_height, grid_width, 4), dtype=float)"
        if source.count(allocation) != 1:
            raise ValueError("Cannot locate reference train's Q-table allocation")
        source = source.replace(
            allocation,
            f"Qs = np.zeros((n, grid_height, grid_width, {num_actions}), dtype=float)",
            1,
        )
    module = types.ModuleType("original_continuing_mdp_reference")
    exec(compile(source, str(source_path), "exec"), vars(module))
    count = min(max(1000, 2 * dd_delay_steps + 10), actions.shape[1])
    replay = noise[0, :count]
    num_states = replay.shape[2] if replay.ndim == 4 else 1
    if replay.ndim == 3:
        replay = replay[:, :, None, :]
    action_replay = actions[0, :count]

    class ReplayField:
        def __init__(self, n, *args, **kwargs):
            self.n, self.t = n, 0

        def step(self):
            self.t += 1

        def at(self, state):
            return (
                replay[self.t, : self.n, state[1]]
                if state[1] < num_states
                else np.zeros((self.n, num_actions))
            )

        def full(self):
            values = np.zeros((self.n, 1, num_states + 1, num_actions))
            values[:, 0, :num_states] = replay[self.t, : self.n]
            return values

    module.NNLikeErrorField = ReplayField
    module.env_step = lambda state, action, *a, **kw: (
        (0, action % num_states),
        0.0,
        False,
    )
    module.get_optimal_q = lambda height, width, *a, **kw: (
        np.zeros((height, width, num_actions)),
        np.zeros((height, width)),
    )
    checks = []
    for method in get_methods(dd_method):
        cursor = [0]

        def fixed_action(q_values, *args):
            t = cursor[0]
            action = int(action_replay[t])
            if behavior == "epsilon_greedy" and not explore_mask[0, t]:
                values = np.asarray(q_values)
                action = int(
                    np.argmax(values.max(axis=1) if values.ndim == 2 else values)
                )
            cursor[0] += 1
            return action

        module.choose_action = fixed_action
        reference = module.train(
            method,
            grid_height=1,
            grid_width=num_states + 1,
            num_updates=count,
            alpha=alpha,
            gamma=gamma,
            noise_mode=noise_mode,
            noise_sigma=1.0,
            double_num_q=2,
            dd_num_q=2,
            dd_delay_steps=dd_delay_steps,
            log_every=1,
            seed=0,
            expectile_tau=expectile_tau,
            alpha_v=alpha_v,
            q_target_tau=q_target_tau,
            clipped_q_backup=clipped_q_backup,
            df_reduction=df_reduction,
        )
        batched = run_batch(
            method,
            noise[:1, :count],
            actions[:1, :count],
            gamma,
            alpha,
            log_every=1,
            dd_delay_steps=dd_delay_steps,
            behavior=behavior,
            explore_mask=None if explore_mask is None else explore_mask[:1, :count],
            expectile_tau=expectile_tau,
            alpha_v=alpha_v,
            q_target_tau=q_target_tau,
            clipped_q_backup=clipped_q_backup,
            df_reduction=df_reduction,
        )
        for key in ["v_bias", "q_rmse"]:
            np.testing.assert_allclose(
                batched[key][0, 1:], reference[key], rtol=1e-12, atol=1e-12
            )
        final = (
            reference["Q_final"][0, 0]
            if num_states == 1
            else reference["Q_final"][0, :num_states]
        )
        np.testing.assert_allclose(
            batched["q_final"].mean(axis=1)[0], final, rtol=1e-12, atol=1e-12
        )
        if method in IQL_METHODS:
            for key in ("value_bias", "value_rmse"):
                np.testing.assert_allclose(
                    batched[key][0, 1:], reference[key], rtol=1e-12, atol=1e-12
                )
            np.testing.assert_allclose(
                batched["value_table"][0, 1:],
                reference["value_table"][:, 0, :num_states],
                rtol=1e-12,
                atol=1e-12,
            )
            for batch_key, reference_key in (
                ("value_final", "value_ensemble_final"),
                ("value_delayed_final", "value_delayed_final"),
                ("q_target_final", "Q_target_final"),
                ("q_delayed_observed_final", "Q_delayed_observed_final"),
            ):
                np.testing.assert_allclose(
                    batched[batch_key][0],
                    reference[reference_key][:, 0, :num_states],
                    rtol=1e-12,
                    atol=1e-12,
                )
        checks.append({"method": method, "updates": count, "passed": True})
    return checks


def make_noise(
    source,
    source_path,
    seed_ids,
    updates,
    sigma,
    noise_mode,
    field_rho=0.999,
    field_features=64,
    field_lengthscale=0.35,
    field_action_scale=0.2,
    num_states=1,
    num_actions=4,
):
    """Project the original Gaussian latent field, then apply its AR(1) filter.

    Linear projection commutes with the AR(1) recurrence. Drawing each shared
    and private innovation in the original order also preserves the random
    stream. Cross-action correlation is exactly that of NNLikeErrorField.
    With two actions, retain outputs 0 and 1 of the same four-action field;
    the RFF features, latent stream, and retained noise outputs are unchanged.
    """
    if num_actions not in (2, 4, 8, 16):
        raise ValueError("num_actions must be 2, 4, 8, or 16")
    field_actions = max(4, num_actions)
    if noise_mode == "nn_like_field_iid":
        values = np.stack(
            [
                sigma
                * np.random.default_rng(100_003 + seed).standard_normal(
                    (updates, 2, num_states, field_actions)
                )
                for seed in seed_ids
            ]
        )
        values = values[..., :num_actions]
        return values[:, :, :, 0] if num_states == 1 else values
    from scipy.signal import lfilter

    module = types.ModuleType("original_nn_like_noise")
    exec(compile(source, str(source_path), "exec"), vars(module))
    fields = []
    for seed in seed_ids:
        field = module.NNLikeErrorField(
            n=2,
            height=1,
            width=num_states + 1,
            sigma=sigma,
            rho=field_rho,
            feature_dim=field_features,
            shared_fraction=0.0,
            lengthscale=field_lengthscale,
            action_scale=field_action_scale,
            seed=100_003 + seed,
            spatial_correlation=True,
            **({"num_actions": field_actions} if field_actions != 4 else {}),
        )
        innovations = np.empty((updates, 2, num_states, field_actions))
        innovations[0] = field.full()[:, 0, :num_states]
        if updates > 1:
            latent = field.rng.standard_normal((updates - 1, 3, field_features))
            projected = np.einsum(
                "tnd,sad->tnsa", latent[:, 1:], field.phi[0, :num_states], optimize=True
            )
            innovations[1:] = sigma * np.sqrt(1.0 - field_rho**2) * projected
        fields.append(lfilter([1.0], [1.0, -field_rho], innovations, axis=0))
    values = np.stack(fields)
    values = values[..., :num_actions]
    return values[:, :, :, 0] if num_states == 1 else values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.99, 0.999, 0.9999])
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument(
        "--noise-mode",
        choices=["nn_like_field_iid", "nn_like_field"],
        default="nn_like_field_iid",
    )
    parser.add_argument("--field-rho", type=float, default=0.999)
    parser.add_argument("--field-features", type=int, default=64)
    parser.add_argument("--field-lengthscale", type=float, default=0.35)
    parser.add_argument("--field-action-scale", type=float, default=0.2)
    parser.add_argument("--dd-delay-steps", type=int, default=1)
    parser.add_argument(
        "--dd-method", choices=["dd_iql", "sd_iql", "df_iql", "dd_q"], default="dd_iql"
    )
    parser.add_argument("--expectile-tau", type=float, default=0.7)
    parser.add_argument(
        "--alpha-v", type=float, default=None, help="V step size; defaults to --alpha"
    )
    parser.add_argument("--q-target-tau", type=float, default=0.005)
    parser.add_argument(
        "--clipped-q-backup", choices=CLIPPED_Q_BACKUPS, default="max_min"
    )
    parser.add_argument("--df-reduction", choices=["min", "mean"], default="min")
    parser.add_argument(
        "--behavior", choices=["uniform", "epsilon_greedy"], default="uniform"
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--num-states", type=int, choices=[1, 2, 4, 8, 16], default=1)
    parser.add_argument("--num-actions", type=int, choices=[2, 4, 8, 16], default=4)
    args = parser.parse_args()
    if (
        args.updates < 1000
        or args.num_seeds < 2
        or args.log_every < 1
        or args.sigma <= 0
        or not 0 < args.alpha <= 1
        or args.seed_start < 0
        or not 0 <= args.field_rho < 1
        or args.field_features < 2
        or args.field_lengthscale <= 0
        or args.field_action_scale < 0
        or args.dd_delay_steps < 1
        or not 0 <= args.epsilon <= 1
        or any(not 0 < gamma < 1 for gamma in args.gammas)
    ):
        parser.error(
            "Need >=1000 updates, >=2 seeds, positive sigma/log interval, "
            "and alpha/gamma in (0,1]"
        )
    if args.num_states > args.num_actions:
        parser.error("num_states must be <= num_actions so every state is reachable")
    if args.alpha_v is None:
        args.alpha_v = args.alpha
    if args.dd_method in IQL_METHODS:
        try:
            validate_dd_iql_config(
                args.alpha, args.alpha_v, args.expectile_tau, args.q_target_tau
            )
        except ValueError as error:
            parser.error(str(error))
    methods = get_methods(args.dd_method)
    methods["clipped_double_q"] += f", backup={args.clipped_q_backup}"
    if args.dd_method == "df_iql":
        methods["df_iql"] += f", critic {args.df_reduction}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_path = Path(__file__).with_name("compare_tabular_methods.py")
    source = source_path.read_text()
    (args.output_dir / "compare_tabular_methods_snapshot.py").write_text(source)
    helper_source = source_path.with_name("tabular_dd_iql.py").read_text()
    (args.output_dir / "tabular_dd_iql.py").write_text(helper_source)
    backup_source = source_path.with_name("tabular_q_backups.py").read_text()
    (args.output_dir / "tabular_q_backups.py").write_text(backup_source)
    (args.output_dir / "driver_snapshot.py").write_text(Path(__file__).read_text())
    seed_ids = list(range(args.seed_start, args.seed_start + args.num_seeds))
    noise = make_noise(
        source,
        source_path,
        seed_ids,
        args.updates,
        args.sigma,
        args.noise_mode,
        args.field_rho,
        args.field_features,
        args.field_lengthscale,
        args.field_action_scale,
        args.num_states,
        args.num_actions,
    )
    actions = np.stack(
        [
            np.random.default_rng(2_000_003 + seed).integers(
                max(4, args.num_actions),
                size=args.updates,
            )
            % args.num_actions
            for seed in seed_ids
        ]
    )
    explore_mask = np.stack(
        [
            np.random.default_rng(3_000_003 + seed).random(args.updates) < args.epsilon
            for seed in seed_ids
        ]
    )
    checks = check_against_original(
        source,
        source_path,
        noise,
        actions,
        args.gammas[0],
        args.alpha,
        args.dd_delay_steps,
        args.noise_mode,
        args.behavior,
        explore_mask,
        args.dd_method,
        args.expectile_tau,
        args.alpha_v,
        args.q_target_tau,
        args.clipped_q_backup,
        args.df_reduction,
    )
    print("REFERENCE_CHECKS", json.dumps(checks), flush=True)
    metadata = {
        "environment": (
            f"Continuing zero-reward MDP; {args.num_actions} real actions; "
            "a moves to a % num_states; Q*=V*=0"
        ),
        "num_states": args.num_states,
        "num_actions": args.num_actions,
        "action_coupling": (
            "Uniform draws over max(4, num_actions), modulo num_actions"
        ),
        "noise_action_subset": (
            "First num_actions outputs of a max(4, num_actions)-action field"
        ),
        "behavior": args.behavior,
        "epsilon": args.epsilon,
        "data_protocol": "Fixed uniform dataset"
        if args.behavior == "uniform"
        else (
            "Same random proposals and exploration draws; "
            "actual actions depend on each method's noisy Q"
        ),
        "noise_mode": args.noise_mode,
        "noise": "IID Gaussian"
        if args.noise_mode == "nn_like_field_iid"
        else "Original NNLikeErrorField Gaussian projection with AR(1) persistence",
        "field_rho": args.field_rho,
        "effective_rho": 0.0
        if args.noise_mode == "nn_like_field_iid"
        else args.field_rho,
        "field_features": args.field_features,
        "field_lengthscale": args.field_lengthscale,
        "field_action_scale": args.field_action_scale,
        "field_shared_fraction": 0.0,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "gammas": args.gammas,
        "updates": args.updates,
        "num_seeds": args.num_seeds,
        "seed_ids": seed_ids,
        "dd_delay_steps": args.dd_delay_steps,
        "dd_method": args.dd_method,
        "expectile_tau": args.expectile_tau,
        "alpha_v": args.alpha_v,
        "q_target_tau": args.q_target_tau,
        "clipped_q_backup": args.clipped_q_backup,
        "df_reduction": args.df_reduction,
        "backup_helper_sha256": hashlib.sha256(backup_source.encode()).hexdigest(),
        "iql_filter": {
            "dd_iql": "cross; delayed peer Q_target - V",
            "sd_iql": "self; delayed own Q_target - V",
            "df_iql": (
                f"shared V; delayed critic {args.df_reduction} Q_target - V; "
                "current gate at D=1"
            ),
            "dd_q": "legacy delayed peer argmax",
        }[args.dd_method],
        "dd_iql_noise": "Q observations only; V learns from noisy target Q",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "helper_sha256": hashlib.sha256(helper_source.encode()).hexdigest(),
        "reference_checks": checks,
        "metrics": (
            "v_bias=max_a mean_i Q_i; value_bias=mean_i learned V_i; "
            "true Q and V are zero"
        ),
        "limitations": [
            "Environment and behavior dataset differ from the episodic GridWorld.",
            "Gamma<1; large accumulated bias is not a proof of unbounded divergence.",
            "No guarantee that the DD method or Double Q have zero learned bias.",
        ],
        "min_lower_bound": (
            "Applies only to one state, four actions, uniform behavior and "
            "nn_like_field_iid: "
            "E[mean_a Q_t] >= gamma*0.2742687674321522*sigma/(1-gamma) "
            "* (1-(1-alpha*(1-gamma)/4)^t)"
        ),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    curve_rows, seed_rows, summary_rows, bound_rows = [], [], [], []
    value_rows = []
    results = {}
    for gamma in args.gammas:
        results[gamma] = {}
        for method in methods:
            history = run_batch(
                method,
                noise,
                actions,
                gamma,
                args.alpha,
                args.log_every,
                args.dd_delay_steps,
                args.behavior,
                explore_mask,
                args.expectile_tau,
                args.alpha_v,
                args.q_target_tau,
                args.clipped_q_backup,
                args.df_reduction,
            )
            results[gamma][method] = history
            np.savez_compressed(
                args.output_dir / f"{method}_gamma{gamma:g}.npz", **history
            )
            if method in IQL_METHODS:
                for seed in range(args.num_seeds):
                    for index, step in enumerate(history["step"]):
                        value_rows.append(
                            dict(
                                gamma=gamma,
                                method=method,
                                seed=seed_ids[seed],
                                step=int(step),
                                value_bias=float(history["value_bias"][seed, index]),
                                value_rmse=float(history["value_rmse"][seed, index]),
                            )
                        )
            final_v = history["v_bias"][:, -1]
            summary = dict(
                gamma=gamma,
                method=method,
                final_v_bias_mean=float(final_v.mean()),
                final_v_bias_sd=float(final_v.std(ddof=1)),
                final_mean_q_bias=float(history["mean_q_bias"][:, -1].mean()),
                final_q_rmse=float(history["q_rmse"][:, -1].mean()),
            )
            summary_rows.append(summary)
            print(json.dumps(summary), flush=True)
            for seed in range(args.num_seeds):
                seed_rows.append(
                    dict(
                        gamma=gamma,
                        method=method,
                        seed=seed_ids[seed],
                        final_v_bias=float(final_v[seed]),
                        final_mean_q_bias=float(history["mean_q_bias"][seed, -1]),
                        final_q_rmse=float(history["q_rmse"][seed, -1]),
                    )
                )
                for index, step in enumerate(history["step"]):
                    curve_rows.append(
                        dict(
                            gamma=gamma,
                            method=method,
                            seed=seed_ids[seed],
                            step=int(step),
                            v_bias=float(history["v_bias"][seed, index]),
                            mean_q_bias=float(history["mean_q_bias"][seed, index]),
                        )
                    )
        for step in (
            results[gamma]["clipped_double_q"]["step"]
            if args.noise_mode == "nn_like_field_iid"
            and args.behavior == "uniform"
            and args.num_states == 1
            and args.num_actions == 4
            else []
        ):
            bound = (
                gamma
                * GAUSSIAN_MAXMIN_FOUR
                * args.sigma
                / (1.0 - gamma)
                * (1.0 - (1.0 - args.alpha * (1.0 - gamma) / 4.0) ** step)
            )
            bound_rows.append(
                dict(
                    gamma=gamma,
                    step=int(step),
                    min_expected_mean_q_lower_bound=float(bound),
                )
            )
    save_csv(args.output_dir / "summary.csv", summary_rows)
    save_csv(args.output_dir / "per_seed.csv", seed_rows)
    save_csv(args.output_dir / "curves.csv", curve_rows)
    if value_rows:
        save_csv(args.output_dir / "learned_value_curves.csv", value_rows)
    if bound_rows:
        save_csv(args.output_dir / "min_lower_bound.csv", bound_rows)
    for plot_name, plot_methods in [
        ("all_methods", list(methods)),
        ("n2_comparison", list(methods)[1:]),
    ]:
        fig, axes = plt.subplots(
            1, len(args.gammas), figsize=(5.2 * len(args.gammas), 4.4), squeeze=False
        )
        for ax, gamma in zip(axes[0], args.gammas):
            for method in plot_methods:
                history = results[gamma][method]
                curves = history["v_bias"]
                mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
                label = methods[method]
                if method in ("dd_q", *IQL_METHODS):
                    label += f", D={args.dd_delay_steps}"
                (line,) = ax.plot(history["step"], mean, label=label)
                ax.fill_between(
                    history["step"],
                    mean - sd,
                    mean + sd,
                    color=line.get_color(),
                    alpha=0.18,
                )
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
            xlabel = (
                "Updates on the same fixed dataset"
                if args.behavior == "uniform"
                else "Environment updates (epsilon-greedy)"
            )
            ax.set(xlabel=xlabel, title=f"gamma = {gamma:g}")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axes[0, 0].set_ylabel("max_a mean_i Q_i - true V (=0)")
        axes[0, -1].legend(fontsize=9)
        fig.suptitle(
            f"Continuing zero-reward MDP: {args.num_states} states, "
            f"{args.num_actions} actions, "
            f"{args.noise_mode}, {args.behavior}, sigma={args.sigma:g}, "
            f"{args.num_seeds} seeds (mean +/- 1 SD)"
        )
        fig.tight_layout()
        for extension in ["png", "pdf"]:
            fig.savefig(args.output_dir / f"{plot_name}.{extension}", dpi=200)
        plt.close(fig)
    print("Results:", args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
