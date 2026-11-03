"""Evaluate a frozen actor with explicit inference device and provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from algorithms.uk_offline.basic import iql_jax as base
from utils.propagation_research.diagnose_checkpoint import CheckpointReader

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("output")
    parser.add_argument("--checkpoint", default="latest.pkl")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeded-action-space", action="store_true",
                        help="Separate frozen-evaluation robustness protocol; does not change training")
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())["identity"]["config"]
    checkpoint = run / args.checkpoint
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (args.expected_checkpoint_sha256 is not None
            and checkpoint_digest != args.expected_checkpoint_sha256):
        raise ValueError("Frozen checkpoint bytes differ from the audited model")
    if args.seeded_action_space and Path(args.output).exists():
        raise FileExistsError("Do not overwrite an existing seeded evaluation")
    with checkpoint.open("rb") as f:
        saved = CheckpointReader(f).load()
    if not all(np.isfinite(np.asarray(x)).all() for x in
               jax.tree_util.tree_leaves(saved["state"].core.actor_params)):
        raise ValueError("Cannot evaluate a nonfinite actor")
    env, dataset, _ = base.load_env_and_dataset(cfg["env"])
    mean, std = base.compute_mean_std(dataset["observations"], 1e-3) if cfg["normalize"] else (0., 1.)
    env = base.wrap_env(env, state_mean=mean, state_std=std)
    reset_metadata = None
    if args.seeded_action_space:
        from utils.propagation_research.seeded_reset import prepare_seeded_action_space
        reset_metadata = prepare_seeded_action_space(env, cfg["seed"])
    base.set_seed(cfg["seed"], env)
    cls = base.DeterministicPolicy if cfg["iql_deterministic"] else base.GaussianPolicy
    actor = cls(action_dim=env.action_space.shape[0], hidden_dim=cfg["hidden_dim"],
                n_hidden=cfg["n_hidden"], dropout=cfg["actor_dropout"])
    params = jax.device_put(saved["state"].core.actor_params, jax.devices(args.device)[0])
    actor_digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(saved["state"].core.actor_params):
        value = np.asarray(leaf)
        actor_digest.update(str((value.shape, value.dtype)).encode())
        actor_digest.update(value.tobytes())
    def policy(p, observation):
        out = actor.apply({"params": p}, observation, training=False)
        return jnp.clip(out if cfg["iql_deterministic"] else out[0], -1, 1)
    fn = jax.jit(policy, backend=args.device)
    rewards, successes = [], []
    inference_seconds, environment_seconds, n_steps = 0., 0., 0
    start = time.monotonic()
    for episode in range(args.episodes):
        obs = base.reset_env(env, seed=cfg["seed"] if episode == 0 else None)
        done, total_reward, success = False, 0., np.nan
        while not done:
            before = time.monotonic()
            action = np.asarray(fn(params, np.asarray(obs[None], dtype=np.float32)))[0]
            inference_seconds += time.monotonic()-before
            before = time.monotonic()
            obs, reward, done, info = base.step_env(env, action)
            environment_seconds += time.monotonic()-before
            total_reward += reward
            current = base.extract_success(info)
            if np.isfinite(current):
                success = current
            n_steps += 1
        rewards.append(total_reward)
        successes.append(success)
    result = {"run": str(run), "checkpoint": args.checkpoint,
              "checkpoint_sha256": checkpoint_digest,
              "actor_parameter_sha256": actor_digest.hexdigest(),
              "step": int(saved["step"]), "device": args.device, "episodes": args.episodes,
              "seed": cfg["seed"], "success_rate": float(np.mean(successes)),
              "episode_successes": successes, "episode_rewards": rewards,
              "environment_steps": n_steps, "wall_seconds": time.monotonic()-start,
              "inference_seconds": inference_seconds, "environment_seconds": environment_seconds}
    result["evaluator_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["seeded_action_space"] = reset_metadata
    if args.seeded_action_space:
        from utils.propagation_research.seeded_reset import PROTOCOL
        result["evaluation_protocol"] = PROTOCOL
    if (not np.isin(successes, [0., 1.]).all() or not np.isfinite(rewards).all()
            or len(successes) != args.episodes):
        raise ValueError("Frozen evaluation did not return finite binary episode outcomes")
    target = Path(args.output)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(target)
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
