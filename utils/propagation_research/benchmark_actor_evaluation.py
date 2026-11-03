"""Compare identical deterministic actor inference on CPU and a busy GPU."""

import argparse
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
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--synthetic-hidden", type=int)
    args = parser.parse_args()
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())["identity"]["config"]
    dataset = json.loads((run / "dataset.json").read_text())
    cls = base.DeterministicPolicy if config["iql_deterministic"] else base.GaussianPolicy
    actor = cls(action_dim=dataset["action_dim"], hidden_dim=args.synthetic_hidden or config["hidden_dim"],
                n_hidden=config["n_hidden"], dropout=config["actor_dropout"])
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    if args.synthetic_hidden:
        with jax.default_device(cpu):
            params = actor.init(jax.random.PRNGKey(0),
                jnp.zeros((1, dataset["state_dim"]), dtype=jnp.float32), training=False)["params"]
    else:
        with (run / "latest.pkl").open("rb") as f:
            params = CheckpointReader(f).load()["state"].core.actor_params
    observations = np.random.RandomState(576).normal(size=(args.iterations, dataset["state_dim"])).astype(np.float32)
    def policy(p, obs):
        out = actor.apply({"params": p}, obs, training=False)
        return jnp.clip(out if config["iql_deterministic"] else out[0], -1, 1)
    timing, outputs = {}, {}
    for name, device in (("gpu", gpu), ("cpu", cpu)):
        fn = jax.jit(policy, backend=name)
        placed = jax.device_put(params, device)
        np.asarray(fn(placed, observations[:1]))
        start = time.monotonic()
        predictions = [np.asarray(fn(placed, observation[None]))[0] for observation in observations]
        timing[name+"_seconds"] = time.monotonic()-start
        outputs[name] = np.stack(predictions)
    delta = np.abs(outputs["gpu"]-outputs["cpu"])
    result = {"run": str(run), "iterations": args.iterations,
              "hidden_dim": args.synthetic_hidden or config["hidden_dim"],
              "synthetic_parameters": bool(args.synthetic_hidden),
              **timing, "gpu_over_cpu_time": timing["gpu_seconds"]/timing["cpu_seconds"],
              "maximum_action_difference": float(delta.max()),
              "mean_action_difference": float(delta.mean())}
    np.testing.assert_allclose(outputs["cpu"], outputs["gpu"], atol=1e-5, rtol=1e-5)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
