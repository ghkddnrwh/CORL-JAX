"""Freeze value checkpoints and refit an identical, pre-collapse IQL actor.

This is a mechanism experiment, not a candidate with the original training budget.
The native actor update and its existing optimizer/schedule are preserved. Arms
share the initial actor, optimizer, minibatch stream and evaluation protocol; only
their frozen Q/V checkpoint differs. Outputs live outside the full-run table.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.uk_offline.basic import iql_jax as base
from algorithms.uk_offline.basic import iql_residual_control as research
from utils.propagation_research.diagnose_checkpoint import CheckpointReader, conditional_reference
import optax


def load_checkpoint(run, name, allow_parameter_only=False):
    payload = (run / name).read_bytes()
    saved = CheckpointReader(io.BytesIO(payload)).load()
    if saved.get("analysis_only") and not allow_parameter_only:
        raise ValueError("A full optimizer checkpoint is required for this controlled refit")
    return saved, hashlib.sha256(payload).hexdigest()


def tree_hash(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_run", type=Path)
    parser.add_argument("values_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--values-checkpoint", required=True)
    parser.add_argument("--allow-parameter-values", action="store_true",
                        help="Allow archived Q/V parameters as frozen values; never for the initial actor or V refitting.")
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--replay-seed", type=int, default=57321)
    parser.add_argument("--refit-value-updates", type=int, default=0,
                        help="First fit V to the frozen target Q, then freeze the refined V.")
    parser.add_argument("--refit-value-batch-size", type=int,
                        help="V-only batch size; actor batches remain at the native size.")
    parser.add_argument("--refit-value-step-scale", type=float, default=1.,
                        help="Multiply V-only parameter steps without resetting Adam moments.")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Use an empty analysis directory: {output}")
    source_bytes = Path(__file__).read_bytes()
    started = time.monotonic()
    initial_config = json.loads((args.initial_run / "config.json").read_text())
    values_config = json.loads((args.values_run / "config.json").read_text())
    cfg = base.TrainConfig(**initial_config["identity"]["config"])
    value_cfg = values_config["identity"]["config"]
    # Profiles may change Q parameterization, but actor/data/schedule must match.
    for key in ("env", "seed", "normalize", "hidden_dim", "n_hidden", "actor_lr",
                "actor_dropout", "iql_deterministic", "max_timesteps", "beta"):
        if getattr(cfg, key) != value_cfg[key]:
            raise ValueError(f"Actor counterfactual requires matching {key}")
    initial, initial_sha = load_checkpoint(args.initial_run, args.initial_checkpoint)
    values, values_sha = load_checkpoint(args.values_run, args.values_checkpoint,
                                        allow_parameter_only=args.allow_parameter_values)
    if values.get("analysis_only") and args.refit_value_updates:
        raise ValueError("V refitting requires a full value optimizer, not a parameter archive")
    initial_core = initial["state"].core
    frozen_core = jax.device_put(values["state"].core, jax.devices("cpu")[0])
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(
            (initial_core.actor_params, initial_core.actor_opt_state, frozen_core.q_target_params,
             frozen_core.v_params))):
        raise ValueError("Controlled actor fitting requires finite actor/optimizer and frozen Q/V parameters")
    # The base convenience loader returns a dataset-family string as its third
    # result. Use OGBench directly when retaining the real validation split.
    env, dataset, validation = base.ogbench.make_env_and_datasets(cfg.env)
    mean, std = base.compute_mean_std(dataset["observations"], 1e-3) if cfg.normalize else (0., 1.)
    env = base.wrap_env(env, state_mean=mean, state_std=std)
    base.set_seed(cfg.seed, env)
    observations = base.normalize_states(np.asarray(dataset["observations"], np.float32), mean, std)
    actions = np.asarray(dataset["actions"], np.float32)
    trainer, _, _, _ = research.initialize(cfg,
        research.PROFILES[values_config["arguments"]["profile"]],
        observations.shape[-1], actions.shape[-1], jax.devices("cpu")[0],
        values_config["arguments"].get("initialization", "iql"))
    actor_state = jax.device_put(base.ActorState(initial_core.actor_params,
        initial_core.actor_opt_state, initial_core.actor_key), jax.devices("cpu")[0])
    original_values_hash = tree_hash((frozen_core.q_params, frozen_core.q_target_params,
                                     frozen_core.v_params))
    value_refit_result = None
    if args.refit_value_updates:
        # This replay stream is independent of the subsequent, matched actor arms.
        value_rng = np.random.RandomState(args.replay_seed+14011)
        vi = value_rng.randint(len(validation["actions"]), size=4096)
        val_obs = base.normalize_states(np.asarray(validation["observations"][vi], np.float32), mean, std)
        val_actions = np.asarray(validation["actions"][vi], np.float32)
        q_val = np.asarray(jnp.min(trainer.q_def.apply(
            {"params": frozen_core.q_target_params}, jnp.asarray(val_obs), jnp.asarray(val_actions)), axis=0))
        before_v = np.asarray(trainer.v_def.apply({"params": frozen_core.v_params}, jnp.asarray(val_obs)))
        bank = value_rng.randint(len(actions), size=4096)
        reference = conditional_reference(trainer, frozen_core, val_obs[:512], observations[bank], actions[bank])

        @jax.jit
        def value_chunk(params, optimizer, batch_sequence):
            def step(carry, batch):
                p, opt = carry
                obs, act = batch["observations"], batch["actions"]
                target = jnp.min(trainer.q_def.apply(
                    {"params": frozen_core.q_target_params}, obs, act), axis=0)
                def loss(candidate):
                    advantage = target-trainer.v_def.apply({"params": candidate}, obs)
                    return jnp.mean(jnp.where(advantage >= 0, cfg.iql_tau,
                                             1-cfg.iql_tau)*advantage**2)
                value, gradient = jax.value_and_grad(loss)(p)
                update, opt = trainer.v_tx.update(gradient, opt, p)
                if args.refit_value_step_scale != 1.:
                    update = jax.tree_util.tree_map(lambda x: args.refit_value_step_scale*x, update)
                return (optax.apply_updates(p, update), opt), value
            return jax.lax.scan(step, (params, optimizer), batch_sequence)

        vp, vo = frozen_core.v_params, frozen_core.v_opt_state
        for begin in range(0, args.refit_value_updates, args.chunk):
            length = min(args.chunk, args.refit_value_updates-begin)
            indices = value_rng.randint(len(actions), size=(length, args.refit_value_batch_size or cfg.batch_size))
            (vp, vo), losses = value_chunk(vp, vo,
                {"observations": observations[indices], "actions": actions[indices]})
            if (begin+length) % 1000 == 0 or begin+length == args.refit_value_updates:
                research.append_json(output / "value_refit_training.jsonl", {
                    "additional_V_updates": begin+length,
                    "expectile_loss": float(np.asarray(losses).mean()),
                    "wall_seconds": time.monotonic()-started})
        frozen_core = frozen_core.replace(v_params=vp, v_opt_state=vo)
        after_v = np.asarray(trainer.v_def.apply({"params": vp}, jnp.asarray(val_obs)))
        loss_before = np.where(q_val-before_v >= 0, cfg.iql_tau, 1-cfg.iql_tau)*(q_val-before_v)**2
        loss_after = np.where(q_val-after_v >= 0, cfg.iql_tau, 1-cfg.iql_tau)*(q_val-after_v)**2
        change = np.asarray(loss_after-loss_before, np.float64)
        value_refit_result = {"additional_V_updates": args.refit_value_updates,
            "V_batch_size": args.refit_value_batch_size or cfg.batch_size,
            "V_parameter_step_scale": args.refit_value_step_scale,
            "validation_queries": len(vi), "validation_loss_before": float(loss_before.mean()),
            "validation_loss_after": float(loss_after.mean()),
            "validation_loss_change": float(change.mean()),
            "validation_loss_change_standard_error": float(change.std(ddof=1)/np.sqrt(len(change))),
            "local_action_residual_before_rms": float(np.sqrt(np.mean((reference["roots"]-before_v[:512])**2))),
            "local_action_residual_after_rms": float(np.sqrt(np.mean((reference["roots"]-after_v[:512])**2))),
            "local_reference_caveat": "Same fixed Q and query actions, but neighbor behavior approximation is not an oracle",
            "V_mean_before": float(before_v.mean()), "V_mean_after": float(after_v.mean())}
        (output / "value_refit_result.json").write_text(json.dumps(value_refit_result, indent=2))
        research.save_atomic(output / "refined_values_analysis_only.pkl", {
            "state": values["state"].replace(core=jax.device_get(frozen_core)),
            "analysis_only": True, "step": values["step"],
            "additional_V_updates": args.refit_value_updates})
    frozen_hash = tree_hash((frozen_core.q_params, frozen_core.q_target_params,
                             frozen_core.v_params))
    manifest = {"kind": "actor_counterfactual_not_budget_matched_candidate",
        "initial_run": str(args.initial_run), "initial_checkpoint": args.initial_checkpoint,
        "initial_checkpoint_sha256": initial_sha, "initial_step": int(initial["step"]),
        "initial_actor_and_optimizer_sha256": tree_hash(actor_state),
        "initial_actor_update_count": int(actor_state.opt_state[0].count),
        "initial_actor_schedule_count": int(actor_state.opt_state[1].count),
        "values_run": str(args.values_run), "values_checkpoint": args.values_checkpoint,
        "values_checkpoint_sha256": values_sha, "values_step": int(values["step"]),
        "values_from_parameter_archive": bool(values.get("analysis_only")),
        "original_values_sha256": original_values_hash, "value_refit": value_refit_result,
        "frozen_values_sha256": frozen_hash, "arguments": {k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()}, "actor_schedule": "Native IQL 1M schedule; original optimizer count retained",
        "training_device": "cpu", "evaluation_device": "cpu", "config": initial_config["identity"]["config"],
        "source_sha256": hashlib.sha256(source_bytes).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "source.py").write_bytes(source_bytes)

    @jax.jit
    def train_chunk(state, batches):
        def step(carry, batch):
            return trainer._actor_refit_step(carry, frozen_core, batch)
        return jax.lax.scan(step, state, batches)

    @jax.jit
    def policy(params, observation):
        result = trainer._apply_actor(params, observation, training=False)
        return jnp.clip(result if cfg.iql_deterministic else result[0], -1., 1.)

    def evaluate(updates):
        successes, rewards = [], []
        for episode in range(args.episodes):
            obs = base.reset_env(env, seed=cfg.seed if episode == 0 else None)
            done, reward, success = False, 0., np.nan
            while not done:
                action = np.asarray(policy(actor_state.params, np.asarray(obs[None], np.float32)))[0]
                obs, r, done, info = base.step_env(env, action)
                reward += r
                value = base.extract_success(info)
                if np.isfinite(value):
                    success = value
            successes.append(success)
            rewards.append(reward)
        result = {"additional_actor_updates": updates, "success_rate": float(np.mean(successes)),
            "episodes": args.episodes, "episode_successes": successes, "episode_rewards": rewards,
            "actor_params_sha256": tree_hash(actor_state.params), "wall_seconds": time.monotonic()-started}
        research.append_json(output / "evaluations.jsonl", result)
        print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}), flush=True)

    evaluate(0)
    rng = np.random.RandomState(args.replay_seed)
    replay_hash = hashlib.sha256()
    for begin in range(0, args.updates, args.chunk):
        length = min(args.chunk, args.updates-begin)
        indices = rng.randint(len(actions), size=(length, cfg.batch_size))
        replay_hash.update(indices.tobytes())
        batches = {"observations": observations[indices], "actions": actions[indices]}
        actor_state, metrics = train_chunk(actor_state, batches)
        metrics = jax.device_get(metrics)
        updates = begin+length
        if updates % 1000 == 0 or updates == args.updates:
            research.append_json(output / "training.jsonl", {"additional_actor_updates": updates,
                "metrics": {k: float(np.mean(v)) for k, v in metrics.items()},
                "wall_seconds": time.monotonic()-started})
        if updates % args.eval_every == 0 or updates == args.updates:
            evaluate(updates)
    assert tree_hash((frozen_core.q_params, frozen_core.q_target_params,
                      frozen_core.v_params)) == frozen_hash, "Frozen values changed"
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(actor_state))
    assert int(actor_state.opt_state[0].count) == manifest["initial_actor_update_count"]+args.updates
    assert int(actor_state.opt_state[1].count) == manifest["initial_actor_schedule_count"]+args.updates
    research.save_atomic(output / "actor_analysis_only.pkl", {"actor_state": jax.device_get(actor_state),
        "analysis_only": True, "kind": manifest["kind"], "additional_actor_updates": args.updates})
    result = {"additional_actor_updates": args.updates, "frozen_values_preserved": True,
        "actual_actor_update_count": int(actor_state.opt_state[0].count),
        "actual_actor_schedule_count": int(actor_state.opt_state[1].count),
        "replay_indices_sha256": replay_hash.hexdigest(), "final_actor_state_sha256": tree_hash(actor_state),
        "wall_seconds": time.monotonic()-started}
    (output / "completed.json").write_text(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
