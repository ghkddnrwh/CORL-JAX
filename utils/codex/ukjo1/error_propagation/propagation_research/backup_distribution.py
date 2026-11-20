"""Locate physical-bound violations in frozen Bellman backups on actual data.

Unlike the online post-V-update metric, this uses a fixed checkpoint V throughout.
No model, optimizer, or training RNG is changed. Adjacent-state labels are used
only when next_observations[i] exactly matches observations[i+1].
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(values.mean()),
            "rms": float(np.sqrt(np.mean(values**2))), "minimum": float(values.min()),
            "maximum": float(values.max()),
            "quantiles": dict(zip(("p01", "p50", "p99", "p999"),
                                  np.quantile(values, [.01, .5, .99, .999]).tolist()))}


def describe(target, rewards, masks, next_success, adjacency_valid, lower, upper):
    projected = np.clip(target, lower, upper)
    active = target != projected
    groups = {}
    for label, member in (("all", np.ones(len(target), dtype=bool)),
                          ("above_upper", target > upper), ("below_lower", target < lower)):
        n = int(member.sum())
        groups[label] = {"count": n, "fraction_of_sample": n/len(target),
            "raw_backup": statistics(target[member]),
            "absolute_projection_change": statistics(np.abs(target-projected)[member]),
            "terminal_fraction": float(np.mean(masks[member] == 0)) if n else None,
            "adjacent_state_match_fraction": float(adjacency_valid[member].mean()) if n else None,
            "next_state_success_fraction_among_matches": float(next_success[member & adjacency_valid].mean())
                if (member & adjacency_valid).any() else None,
            "reward_counts": {str(float(r)): int(np.sum(member & (rewards == r)))
                              for r in np.unique(rewards)}}
    return {"return_interval": [lower, upper], "projection_active_fraction": float(active.mean()),
            "projection_change_rms_all_transitions": statistics(target-projected)["rms"],
            "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", default="checkpoint_1000000.pkl")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=18213)
    args = parser.parse_args()
    if args.samples <= 0 or args.batch <= 0:
        parser.error("Sample and batch sizes must be positive")
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise ValueError("Use a new diagnostic output path")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    from algorithms.uk_offline.basic import iql_jax as base
    from algorithms.uk_offline.basic import iql_residual_control as research
    from utils.propagation_research.diagnose_checkpoint import CheckpointReader
    from utils.propagation_research.reward_bounds import return_bounds
    import jax
    import jax.numpy as jnp

    config = json.loads((args.run / "config.json").read_text())
    cfg = base.TrainConfig(**config["identity"]["config"])
    path = args.run / args.checkpoint
    with path.open("rb") as stream:
        saved = CheckpointReader(stream).load()
    core = jax.device_put(saved["state"].core, jax.devices("cpu")[0])
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(core.v_params)):
        raise ValueError("Cannot inspect a nonfinite V")
    env, data, _ = base.ogbench.make_env_and_datasets(cfg.env)
    env.close()
    mean, std = base.compute_mean_std(data["observations"], 1e-3) if cfg.normalize else (0., 1.)
    rng = np.random.RandomState(args.seed)
    # Sampling without replacement makes each diagnostic event a distinct row.
    indices = rng.choice(len(data["actions"]), size=min(args.samples, len(data["actions"])), replace=False)
    nxt = np.asarray(data["next_observations"][indices], np.float32)
    after = np.minimum(indices+1, len(data["actions"])-1)
    exact_adjacent = (indices+1 < len(data["actions"])) & np.all(
        nxt == np.asarray(data["observations"][after], np.float32), axis=-1)
    rewards = np.asarray(data["rewards"][indices], np.float32).reshape(-1)
    masks = np.asarray(data["masks"][indices], np.float32).reshape(-1)
    next_success = np.asarray(data["rewards"][after] == 0).reshape(-1)
    observations = base.normalize_states(nxt, mean, std)
    trainer, _, _, _ = research.initialize(cfg, config["identity"]["profile"],
        observations.shape[-1], data["actions"].shape[-1], jax.devices("cpu")[0],
        config["arguments"].get("initialization", "iql"))
    @jax.jit
    def predict(params, query, reward, mask):
        value = trainer.v_def.apply({"params": params}, query)
        return value, reward+cfg.discount*mask*value
    predictions, backups = [], []
    for i in range(0, len(indices), args.batch):
        value, backup = predict(core.v_params, jnp.asarray(observations[i:i+args.batch]),
                               jnp.asarray(rewards[i:i+args.batch]), jnp.asarray(masks[i:i+args.batch]))
        predictions.append(np.asarray(value))
        backups.append(np.asarray(backup))
    predicted = np.concatenate(predictions).astype(np.float64)
    if not np.isfinite(predicted).all():
        raise ValueError("V predictions are not finite on the diagnostic rows")
    target = np.concatenate(backups).astype(np.float64)
    lower, upper = return_bounds(cfg.env, cfg.discount, float(np.min(data["rewards"])),
                                float(np.max(data["rewards"])), "ogbench")
    result = {"kind": "frozen_backup_distribution_not_online_post_V_update",
        "environment": cfg.env, "profile": config["arguments"]["profile"], "training_seed": cfg.seed,
        "checkpoint": args.checkpoint, "checkpoint_step": int(saved["step"]),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "diagnostic_seed": args.seed, "samples_without_replacement": len(indices),
        "snapshot_V_next": statistics(predicted),
        "backup": describe(target, rewards, masks, next_success, exact_adjacent, lower, upper),
        "caveats": ["Uses frozen checkpoint V, not the fresh two-step V used on each training minibatch.",
                    "Success in an adjacent stored state is a descriptive label; unmatched adjacency is unknown.",
                    "Violation locations do not prove a causal mechanism for the full training trajectory."],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), indices=indices, rewards=rewards,
                        masks=masks, V_next=predicted, raw_backup=target,
                        adjacent_state_exact_match=exact_adjacent, next_state_success=next_success)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
