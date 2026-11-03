"""Describe Q-action drift and V-dependent state weighting on fixed queries.

Borrowed actions are a controlled probe set, not true conditional behavior
samples or oracle action values. This diagnostic does not modify training.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))


def log_normalize(x):
    shifted = x-np.max(x, axis=-1, keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def weight_summary(advantage, beta):
    z = np.asarray(beta*advantage, dtype=np.float32)
    with np.errstate(over="ignore", under="ignore"):
        w = np.minimum(np.exp(z), np.float32(100.))
    w = np.asarray(w, dtype=np.float64)
    return {"advantage_mean": float(np.mean(advantage)), "advantage_rms": rms(advantage),
            "weight_mean": float(w.mean()), "weight_cap_fraction": float(np.mean(w >= 100.)),
            "float32_weight_zero_fraction": float(np.mean(w == 0.)),
            "weight_ess_fraction": float(w.sum()**2/(len(w)*np.square(w).sum()))
                if np.square(w).sum() else 0.}


def describe(q_old, q_late, v_old, v_late, beta):
    q_delta = q_late-q_old
    state_q_shift = q_delta.mean(axis=-1)
    action_drift = q_delta-state_q_shift[:, None]
    state_adv_shift = state_q_shift-(v_late-v_old)
    advantage_delta = (q_late-v_late[:, None])-(q_old-v_old[:, None])
    reconstructed = state_adv_shift[:, None]+action_drift
    if not np.allclose(advantage_delta, reconstructed, atol=1e-10, rtol=1e-12):
        raise AssertionError("Advantage decomposition does not reconstruct the original difference")
    old_logp, late_logp = log_normalize(beta*q_old), log_normalize(beta*q_late)
    old_p, late_p = np.exp(old_logp), np.exp(late_logp)
    # With no cap and an unconstrained separate policy at each state, subtracting
    # V changes only the state multiplier, not normalized action preferences.
    cancellation_error = max(float(np.max(np.abs(np.exp(log_normalize(beta*(q-v[:, None])))
                                                 -np.exp(log_normalize(beta*q)))))
                             for q, v in ((q_old, v_old), (q_late, v_late)))
    a, b = np.triu_indices(q_old.shape[1], k=1)
    old_diff, late_diff = q_old[:, a]-q_old[:, b], q_late[:, a]-q_late[:, b]
    untied = (np.abs(old_diff) > 1e-6) & (np.abs(late_diff) > 1e-6)
    order_changed = old_diff*late_diff < 0
    clip_old = np.exp(log_normalize(np.minimum(beta*(q_old-v_old[:, None]), np.log(100.))))
    clip_late = np.exp(log_normalize(np.minimum(beta*(q_late-v_late[:, None]), np.log(100.))))
    return {"query_count": len(q_old), "actions_per_query": q_old.shape[1],
        "Q_global_shift": float(state_q_shift.mean()),
        "V_global_shift": float((v_late-v_old).mean()),
        "advantage_global_shift": float(state_adv_shift.mean()),
        "advantage_state_shift_rms": rms(state_adv_shift),
        "advantage_state_shift_centered_rms": rms(state_adv_shift-state_adv_shift.mean()),
        "within_state_action_drift_rms": rms(action_drift),
        "total_advantage_drift_rms": rms(advantage_delta),
        "decomposition_max_absolute_error": float(np.max(np.abs(advantage_delta-reconstructed))),
        "decomposition_energy_error": float(np.mean(advantage_delta**2)
              -np.mean(state_adv_shift**2)-np.mean(action_drift**2)),
        "within_state_Q_order_reversal_fraction": float(order_changed[untied].mean()) if untied.any() else None,
        "untied_pair_fraction": float(untied.mean()),
        "Q_argmax_agreement": float(np.mean(q_old.argmax(axis=-1) == q_late.argmax(axis=-1))),
        "uncapped_normalized_action_TV_mean": float((.5*np.abs(old_p-late_p).sum(axis=-1)).mean()),
        "native_capped_normalized_action_TV_mean": float((.5*np.abs(clip_old-clip_late).sum(axis=-1)).mean()),
        "uncapped_V_cancellation_max_absolute_error": cancellation_error,
        "old_borrowed_action_weights": weight_summary((q_old-v_old[:, None]).ravel(), beta),
        "late_borrowed_action_weights": weight_summary((q_late-v_late[:, None]).ravel(), beta)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_run", type=Path)
    parser.add_argument("late_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--late-checkpoint", required=True)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--bank", type=int, default=8192)
    parser.add_argument("--neighbors", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--seed", type=int, default=88123)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    from algorithms.uk_offline.basic import iql_jax as base
    from algorithms.uk_offline.basic import iql_residual_control as research
    from utils.propagation_research.diagnose_checkpoint import CheckpointReader, conditional_reference
    import jax
    import jax.numpy as jnp

    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise ValueError("Use a new output path to preserve existing diagnostics")
    configurations = [json.loads((p / "config.json").read_text()) for p in (args.old_run, args.late_run)]
    configs = [c["identity"]["config"] for c in configurations]
    for key in ("env", "normalize", "beta", "hidden_dim", "n_hidden"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Fixed-query comparisons require matching {key}")
    env, train, validation = base.ogbench.make_env_and_datasets(configs[0]["env"])
    env.close()
    mean, std = base.compute_mean_std(train["observations"], 1e-3) if configs[0]["normalize"] else (0., 1.)
    rng = np.random.RandomState(args.seed)
    qi = rng.randint(len(validation["actions"]), size=args.queries)
    bi = rng.randint(len(train["actions"]), size=args.bank)
    query = base.normalize_states(np.asarray(validation["observations"][qi], np.float32), mean, std)
    bank = base.normalize_states(np.asarray(train["observations"][bi], np.float32), mean, std)
    bank_actions = np.asarray(train["actions"][bi], np.float32)
    actual_actions = np.asarray(validation["actions"][qi], np.float32)
    references, actual_qs, hashes, snapshots = [], [], [], []
    for run, checkpoint, config, cfg_dict in zip((args.old_run, args.late_run),
            (args.old_checkpoint, args.late_checkpoint), configurations, configs):
        path = run / checkpoint
        with path.open("rb") as stream:
            saved = CheckpointReader(stream).load()
        core = jax.device_put(saved["state"].core, jax.devices("cpu")[0])
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(
                (core.q_target_params, core.v_params))):
            raise ValueError("Cannot describe nonfinite Q/V parameters")
        trainer, _, _, _ = research.initialize(base.TrainConfig(**cfg_dict),
            config["identity"]["profile"], query.shape[-1], actual_actions.shape[-1],
            jax.devices("cpu")[0], config["arguments"].get("initialization", "iql"))
        reference = conditional_reference(trainer, core, query, bank, bank_actions, neighbors=max(args.neighbors))
        references.append(reference)
        @jax.jit
        def predict(qp, observations, actions):
            return jnp.min(trainer.q_def.apply({"params": qp}, observations, actions), axis=0)
        actual_qs.append(np.asarray(predict(core.q_target_params, jnp.asarray(query),
                                           jnp.asarray(actual_actions)), dtype=np.float64))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        snapshots.append({"run": str(run), "checkpoint": checkpoint, "step": int(saved["step"]),
                          "profile": config["arguments"]["profile"]})
    old, late = references
    if not np.array_equal(old["distance"], late["distance"]):
        raise AssertionError("The two value models received different action probe sets")
    # argpartition returns an unordered nearest set. Sort the common set once
    # before using smaller prefixes for the probe-count sensitivity analysis.
    order = np.argsort(old["distance"], axis=-1)
    q_old = np.take_along_axis(old["q"], order, axis=-1)
    q_late = np.take_along_axis(late["q"], order, axis=-1)
    results = {str(k): describe(q_old[:, :k], q_late[:, :k], old["v"], late["v"], configs[0]["beta"])
               for k in args.neighbors}
    output = {"kind": "fixed_query_advantage_drift_not_oracle", "snapshots": snapshots,
        "checkpoint_sha256": hashes, "query_seed": args.seed, "bank_size": args.bank,
        "probe_counts": results,
        "actual_validation_action_weights": {name: weight_summary(q-ref["v"], configs[0]["beta"])
            for name, q, ref in zip(("old", "late"), actual_qs, references)},
        "caveats": ["Borrowed actions need not lie in the true conditional behavior support at each query state.",
            "Order changes describe learned-Q drift; they do not identify which ranking is correct.",
            "V cancels from uncapped state-normalized exponential weights only for a separate unconstrained policy at each state. Native weight clipping, a shared neural policy, and optimization can preserve a direct V effect.",
            "This frozen-model description does not establish the causal mediation of a full training trajectory."],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), q_old=q_old, q_late=q_late,
                        v_old=old["v"], v_late=late["v"], actual_q_old=actual_qs[0],
                        actual_q_late=actual_qs[1], query_indices=qi, bank_indices=bi)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False))
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
