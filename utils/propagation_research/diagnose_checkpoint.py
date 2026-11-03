"""Independent validation-data residuals and differentiable critic-fit responses.

Local action references borrow actions from nearest dataset states and re-evaluate
Q at the query state. They approximate behavior-conditional expectiles; they are
not oracle residual labels. JVP differentiates the critic fitting algorithm with
respect to the supplied target, including Adam, without finite-difference noise.
"""

import argparse
import hashlib
import io
import json
import pickle
import time
from pathlib import Path

from algorithms.uk_offline.basic import iql_jax as base
from algorithms.uk_offline.basic import iql_residual_control as research

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze
from optax._src import numerics, transform


@jax.custom_jvp
def adam_sqrt(x):
    return jnp.sqrt(x)


@adam_sqrt.defjvp
def adam_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    value = jnp.sqrt(x)
    # At an inactive Adam coordinate mu=nu=0, the composite update has a finite
    # target derivative although a naive sqrt' produces 0/0. Keep the exact
    # forward sqrt and use the zero tangent of the inactive second moment.
    denominator = jnp.where(x > 0, 2*value, 1.)
    return value, jnp.where(x > 0, dx/denominator, 0.)


def differentiable_adam(grads, state, learning_rate):
    adam = state[0]
    mu = transform.update_moment(grads, adam.mu, .9, 1)
    nu = transform.update_moment_per_elem_norm(grads, adam.nu, .999, 2)
    count = numerics.safe_int32_increment(adam.count)
    mu_hat = transform.bias_correction(mu, .9, count)
    nu_hat = transform.bias_correction(nu, .999, count)
    normalized = jax.tree_util.tree_map(lambda m, v: m/(adam_sqrt(v)+1e-8), mu_hat, nu_hat)
    updates = jax.tree_util.tree_map(lambda x: -learning_rate*x, normalized)
    return updates, (adam._replace(count=count, mu=mu, nu=nu), state[1])


def build_jvp_controller(trainer, q_lr, horizon=20):
    """Compare a small Q-step grid on fixed independent probes, without FD.

    This is a directional finite-horizon critic-fit response, not the operator
    norm or spectral radius of the entire Q/V/target/Adam training loop.
    """
    qa, va = trainer.q_def.apply, trainer.v_def.apply
    @jax.jit
    def measure(core, fit, heldout, step_scale):
        target = fit["rewards"][..., 0]+trainer.discount*(1-fit["dones"][..., 0])*va(
            {"params": core.v_params}, fit["next_observations"])
        def output(y):
            def step(carry, _):
                qp, qo = carry
                def loss(p):
                    qs = qa({"params": p}, fit["observations"], fit["actions"])
                    return jnp.mean((qs-y[None])**2)
                grad = jax.grad(loss)(qp)
                update, qo = differentiable_adam(grad, qo, q_lr)
                update = jax.tree_util.tree_map(lambda x: step_scale*x, update)
                return (optax.apply_updates(qp, update), qo), None
            (qp, _), _ = jax.lax.scan(step, (core.q_params, core.q_opt_state), None, length=horizon)
            return jnp.mean(qa({"params": qp}, heldout["observations"], heldout["actions"]), axis=0)
        directions = (jnp.ones_like(target),
            jax.random.rademacher(jax.random.PRNGKey(573), target.shape, dtype=target.dtype))
        gains = []
        for direction in directions:
            response = jax.jvp(output, (target,), (direction,))[1]
            gains.append(jnp.sqrt(jnp.mean(response**2)))
        return jnp.stack(gains)
    return measure


class CheckpointReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "ResearchState":
            return research.ResearchState
        return super().find_class(module, name)


def empirical_expectile(values, tau, iterations=32):
    lo, hi = values.min(-1), values.max(-1)
    for _ in range(iterations):
        mid = .5*(lo+hi)
        adv = values-mid[..., None]
        score = np.mean(np.where(adv >= 0, tau, 1-tau)*adv, axis=-1)
        lo, hi = np.where(score > 0, mid, lo), np.where(score > 0, hi, mid)
    return .5*(lo+hi)


def conditional_reference(trainer, core, query, bank, bank_actions, neighbors=64):
    distance = np.mean((query[:, None]-bank[None])**2, axis=-1)
    indices = np.argpartition(distance, neighbors-1, axis=-1)[:, :neighbors]
    indices = np.take_along_axis(indices, np.argsort(np.take_along_axis(distance, indices, -1), axis=-1), -1)
    actions = bank_actions[indices]
    observations = np.repeat(query[:, None], neighbors, axis=1)
    @jax.jit
    def predict(qp, vp, obs, act):
        qs = trainer.q_def.apply({"params": qp}, obs.reshape(-1, obs.shape[-1]),
                                 act.reshape(-1, act.shape[-1]))
        return jnp.min(qs, axis=0).reshape(obs.shape[:2]), trainer.v_def.apply({"params": vp}, obs[:, 0])
    q, v = jax.device_get(predict(core.q_target_params, core.v_params,
                                 jnp.asarray(observations), jnp.asarray(actions)))
    q, v = np.asarray(q, dtype=np.float64), np.asarray(v, dtype=np.float64)
    roots = empirical_expectile(q, trainer.iql_tau)
    adv = q-v[:, None]
    weights = np.where(adv >= 0, trainer.iql_tau, 1-trainer.iql_tau)
    score = np.mean(weights*adv, axis=-1)
    displacement = score/np.mean(weights, axis=-1)
    return {"q": q, "v": v, "roots": roots, "G_over_kappa": displacement,
            "distance": np.take_along_axis(distance, indices, axis=-1)}


def critic_response(trainer, core, fit, probe, horizons=(1, 5, 20), q_lr=3e-4, q_scale=1.):
    qa = trainer.q_def.apply
    target = fit["rewards"][..., 0] + trainer.discount*(1-fit["dones"][..., 0])*trainer.v_def.apply(
        {"params": core.v_params}, fit["next_observations"])
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))
    scale = max(rms(target), 1.)
    result = {}
    for horizon in horizons:
        def output(y):
            def fit_step(carry, _):
                qp, qo = carry
                # y must remain differentiable to the OUTER JVP; it is constant
                # with respect to q_params in this INNER gradient by closure.
                def loss(params):
                    q = qa({"params": params}, fit["observations"], fit["actions"])
                    return jnp.mean((q-y[None])**2)
                grads = jax.grad(loss)(qp)
                update, qo = differentiable_adam(grads, qo, q_lr)
                update = jax.tree_util.tree_map(lambda x: q_scale*x, update)
                return (optax.apply_updates(qp, update), qo), None
            (qp, _), _ = jax.lax.scan(fit_step, (core.q_params, core.q_opt_state), None, length=horizon)
            return jnp.mean(qa({"params": qp}, probe["observations"], probe["actions"]), axis=0)
        value = jax.jit(output)
        @jax.jit
        def response(y, direction):
            return jax.jvp(output, (y,), (direction,))[1]
        directions = {"constant": jnp.ones_like(target),
                      "random": jax.random.rademacher(jax.random.PRNGKey(573), target.shape, dtype=target.dtype)}
        for name, direction in directions.items():
            derivative = np.asarray(response(target, direction))
            entry = {"jvp_gain_rms": rms(derivative),
                     "jvp_finite": bool(np.isfinite(derivative).all()),
                     "jvp_common_response": float(derivative.mean()),
                     "jvp_centered_gain_rms": rms(derivative-derivative.mean())}
            for relative_eps in (.001, .01, .1):
                eps = relative_eps*scale
                fd = np.asarray((value(target+eps*direction)-value(target-eps*direction))/(2*eps))
                entry[f"fd_{relative_eps}_gain"] = rms(fd)
                entry[f"fd_{relative_eps}_relative_error_to_jvp"] = rms(fd-derivative)/max(rms(derivative), 1e-12)
            result[f"h{horizon}_{name}"] = entry
    return result


def refit_value(trainer, core, batches, probe):
    qa, va = trainer.q_def.apply, trainer.v_def.apply
    @jax.jit
    def run(core, batches):
        def step(carry, batch):
            vp, vo = carry
            q = jnp.min(qa({"params": core.q_target_params}, batch["observations"], batch["actions"]), axis=0)
            def loss(params):
                adv = q-va({"params": params}, batch["observations"])
                w = jnp.where(adv >= 0, trainer.iql_tau, 1-trainer.iql_tau)
                return jnp.mean(w*adv**2)
            value, grad = jax.value_and_grad(loss)(vp)
            update, vo = trainer.v_tx.update(grad, vo, vp)
            return (optax.apply_updates(vp, update), vo), value
        (vp, vo), losses = jax.lax.scan(step, (core.v_params, core.v_opt_state), batches)
        return core.replace(v_params=vp, v_opt_state=vo), losses
    return run(core, batches)[0]


def value_loop_response(trainer, head, htx, state, profile, bounds, batches, heldout,
                        q_lr, v_lr, control_scale=1., horizons=(20, 100, 500)):
    update = research.build_update(trainer, head, htx, profile, bounds,
                                    hypergradient=True, q_lr=q_lr, v_lr=v_lr)
    if profile.get("jvp_control"):
        batches = dict(batches, control_scale=jnp.full(len(batches["actions"]), control_scale))
    def shifted_v(amplitude):
        vp = unfreeze(state.core.v_params)
        name = f"Dense_{trainer.n_hidden}"
        vp[name]["bias"] = vp[name]["bias"]+amplitude
        return trainer.v_def.apply({"params": freeze(vp)}, heldout["observations"])
    initial_v_tangent = jax.jvp(shifted_v, (jnp.asarray(0., dtype=jnp.float32),),
                                (jnp.asarray(1., dtype=jnp.float32),))[1]
    input_rms = float(np.sqrt(np.mean(np.asarray(initial_v_tangent, dtype=np.float64)**2)))
    results = {}
    for horizon in horizons:
        def output(amplitude):
            vp = unfreeze(state.core.v_params)
            name = f"Dense_{trainer.n_hidden}"
            vp[name]["bias"] = vp[name]["bias"]+amplitude
            initial = state.replace(core=state.core.replace(v_params=freeze(vp)))
            def body(carry, batch):
                carry, _ = update(carry, batch)
                return carry, None
            final, _ = jax.lax.scan(body, initial, {k: v[:horizon] for k, v in batches.items()})
            q = jnp.mean(trainer.q_def.apply({"params": final.core.q_params},
                          heldout["observations"], heldout["actions"]), axis=0)
            v = trainer.v_def.apply({"params": final.core.v_params}, heldout["observations"])
            return q, v
        @jax.jit
        def measure():
            return jax.jvp(output, (jnp.asarray(0., dtype=jnp.float32),),
                            (jnp.asarray(1., dtype=jnp.float32),))
        primal, response = jax.device_get(measure())
        rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))
        results[f"h{horizon}"] = {"q_response_rms": rms(response[0]),
                                  "v_response_rms": rms(response[1]),
                                  "q_common_response": float(np.mean(response[0])),
                                  "v_common_response": float(np.mean(response[1])),
                                  "primal_q_rms": rms(primal[0]),
                                  "primal_v_rms": rms(primal[1]),
                                  "initial_v_perturbation_rms": input_rms,
                                  "q_response_per_initial_v_rms": rms(response[0])/max(input_rms, 1e-12),
                                  "v_response_per_initial_v_rms": rms(response[1])/max(input_rms, 1e-12),
                                  "finite": bool(all(np.isfinite(x).all() for x in (*primal, *response)))}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--checkpoint", default="latest.pkl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("default", "highest"), default="default")
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--bank", type=int, default=4096)
    parser.add_argument("--skip-b", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--refit-head", type=int, default=0,
                        help="Additional moment-head fits with Q/V frozen; analysis only.")
    parser.add_argument("--v-trace", action="store_true",
                        help="Compare post-update V losses for repeated versus fresh minibatches.")
    args = parser.parse_args()
    diagnostic_source = Path(__file__).read_bytes()
    diagnostic_source_hash = hashlib.sha256(diagnostic_source).hexdigest()
    jax.config.update("jax_default_matmul_precision", args.precision)
    start = time.monotonic()
    run = Path(args.run)
    cfg = base.TrainConfig(**json.loads((run / "config.json").read_text())["identity"]["config"])
    profile_name = json.loads((run / "config.json").read_text())["arguments"]["profile"]
    checkpoint_bytes = (run / args.checkpoint).read_bytes()
    saved = CheckpointReader(io.BytesIO(checkpoint_bytes)).load()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    del checkpoint_bytes
    if saved.get("analysis_only") and not saved.get("optimizer_states_preserved"):
        raise ValueError("Optimizer diagnostics require a full checkpoint; parameter-only snapshots support frozen actor evaluation")
    if saved.get("optimizer_states_preserved") and not all(
            np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(saved["state"])):
        raise ValueError("Nonfinite terminal states cannot support optimizer sensitivity diagnostics")
    device = base.select_jax_device(args.device)
    trainer, head, htx, _ = research.initialize(cfg, research.PROFILES[profile_name],
        json.loads((run / "dataset.json").read_text())["state_dim"],
        json.loads((run / "dataset.json").read_text())["action_dim"], device,
        json.loads((run / "config.json").read_text())["arguments"].get("initialization", "iql"))
    state = jax.device_put(saved["state"], device)
    env, train, val = base.ogbench.make_env_and_datasets(cfg.env)
    mean, std = base.compute_mean_std(train["observations"], 1e-3) if cfg.normalize else (0., 1.)
    rng = np.random.RandomState(cfg.seed+79172)
    def sample(dataset, shape):
        indices = rng.randint(len(dataset["actions"]), size=shape)
        result = {k: np.asarray(dataset[k][indices], dtype=np.float32) for k in
                  ("observations", "next_observations", "actions")}
        for key in ("observations", "next_observations"):
            result[key] = base.normalize_states(result[key], mean, std)
        result["rewards"] = np.asarray(dataset["rewards"][indices], dtype=np.float32)[..., None]
        dones = 1-dataset["masks"] if "masks" in dataset else dataset["terminals"]
        result["dones"] = np.asarray(dones[indices], dtype=np.float32)[..., None]
        return result
    reference = sample(train, args.bank)
    heldout = sample(val, args.queries)
    batches = sample(train, (20, cfg.batch_size))
    value_before = conditional_reference(trainer, state.core, heldout["observations"],
                                          reference["observations"], reference["actions"])
    core_after = refit_value(trainer, state.core, jax.device_put(batches, device),
                             jax.device_put(heldout, device))
    v_after = np.asarray(trainer.v_def.apply({"params": core_after.v_params}, jnp.asarray(heldout["observations"])))
    roots, before, q = value_before["roots"], value_before["v"], value_before["q"]
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))
    q_heldout = np.asarray(jnp.min(trainer.q_def.apply({"params": state.core.q_target_params},
        jnp.asarray(heldout["observations"]), jnp.asarray(heldout["actions"])), axis=0), dtype=np.float64)
    advantage_before, advantage_after = q_heldout-before, q_heldout-v_after
    actual_weights = np.where(advantage_before >= 0, cfg.iql_tau, 1-cfg.iql_tau)
    loss_before = actual_weights*advantage_before**2
    loss_after = np.where(advantage_after >= 0, cfg.iql_tau, 1-cfg.iql_tau)*advantage_after**2
    loss_delta = loss_after-loss_before
    known_bounds = json.loads((run / "dataset.json").read_text())
    upper_bound = known_bounds["return_upper"]
    result = {"step": saved["step"], "environment": cfg.env, "profile": profile_name,
              "diagnostic_source_sha256": diagnostic_source_hash,
              "checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_sha256,
              "device": args.device, "matmul_precision": args.precision,
              "reference": "neighbor actions re-evaluated at heldout query states; approximate conditional",
              "query_dataset": "OGBench validation split", "queries": args.queries,
              "neighbor_bank_size": args.bank,
              "neighbors": q.shape[-1], "neighbor_state_distance_mean": float(value_before["distance"].mean()),
              "local_action_residual_before_rms": rms(roots-before),
              "local_action_residual_after_20_fresh_v_updates_rms": rms(roots-v_after),
              "local_action_G_over_kappa_rms": rms(value_before["G_over_kappa"]),
              "newton_linearization_error_rms": rms(value_before["G_over_kappa"]-(roots-before)),
              "reference_16_vs_64_rms": rms(empirical_expectile(q[:, :16], cfg.iql_tau)-roots),
              "validation_expectile_loss_before": float(loss_before.mean()),
              "validation_expectile_loss_after_20_fresh_v_updates": float(loss_after.mean()),
              "validation_loss_change": float(loss_delta.mean()),
              "validation_loss_change_standard_error": float(loss_delta.std(ddof=1)/np.sqrt(len(loss_delta))),
              "validation_target_q_above_return_upper_rms": rms(np.maximum(q_heldout-upper_bound, 0.)),
              "validation_V_above_return_upper_rms": rms(np.maximum(before-upper_bound, 0.)),
              "validation_V_after_refit_above_return_upper_rms": rms(np.maximum(v_after-upper_bound, 0.)),
              "refit_objective": "standard expectile loss, frozen target Q, independent fresh training minibatches",
              "v_change_rms": rms(v_after-before), "expectile": cfg.iql_tau, "discount": cfg.discount}
    if args.v_trace:
        train_first = {k: v[0] for k, v in batches.items()}
        train_obs = jnp.asarray(train_first["observations"])
        train_q = np.asarray(jnp.min(trainer.q_def.apply(
            {"params": state.core.q_target_params}, train_obs,
            jnp.asarray(train_first["actions"])), axis=0), dtype=np.float64)
        # Use the identical evaluation path for every trace point. The fused
        # local-reference predictor can differ by a few ulps from eager V calls.
        trace_initial_v = np.asarray(trainer.v_def.apply({"params": state.core.v_params},
            jnp.asarray(heldout["observations"])), dtype=np.float64)
        trace_initial_advantage = q_heldout-trace_initial_v
        trace_initial_loss = np.where(trace_initial_advantage >= 0, cfg.iql_tau,
                                      1-cfg.iql_tau)*trace_initial_advantage**2
        next_reference = conditional_reference(trainer, state.core, heldout["next_observations"],
                                               reference["observations"], reference["actions"])
        bootstrap_mask = 1-heldout["dones"][..., 0]

        def fit_metrics(core, mode, updates):
            train_v = np.asarray(trainer.v_def.apply({"params": core.v_params}, train_obs))
            val_v = np.asarray(trainer.v_def.apply({"params": core.v_params},
                jnp.asarray(heldout["observations"])), dtype=np.float64)
            next_v = np.asarray(trainer.v_def.apply({"params": core.v_params},
                jnp.asarray(heldout["next_observations"])), dtype=np.float64)
            train_adv, val_adv = train_q-train_v, q_heldout-val_v
            train_loss = np.where(train_adv >= 0, cfg.iql_tau, 1-cfg.iql_tau)*train_adv**2
            val_loss = np.where(val_adv >= 0, cfg.iql_tau, 1-cfg.iql_tau)*val_adv**2
            delta = val_loss-trace_initial_loss
            raw_backup = heldout["rewards"][..., 0]+cfg.discount*bootstrap_mask*next_v
            return {"minibatch_mode": mode, "V_updates": updates,
                "first_training_batch_post_fit_loss": float(train_loss.mean()),
                "validation_post_fit_loss": float(val_loss.mean()),
                "validation_loss_change": float(delta.mean()),
                "validation_loss_change_standard_error": float(delta.std(ddof=1)/np.sqrt(len(delta))),
                "local_action_residual_rms": rms(roots-val_v),
                "next_bootstrap_local_reference_displacement_rms":
                    cfg.discount*rms(bootstrap_mask*(next_reference["roots"]-next_v)),
                "next_bootstrap_V_above_upper_rms":
                    cfg.discount*rms(bootstrap_mask*np.maximum(next_v-upper_bound, 0.)),
                "raw_backup_above_upper_rms": rms(np.maximum(raw_backup-upper_bound, 0.))}

        trace = [fit_metrics(state.core, "initial", 0)]
        for mode in ("repeat_first_training_batch", "fresh_training_batches"):
            for horizon in (1, 2, 4, 20):
                selected = ({k: np.repeat(v[:1], horizon, axis=0) for k, v in batches.items()}
                            if mode == "repeat_first_training_batch"
                            else {k: v[:horizon] for k, v in batches.items()})
                fitted = (core_after if mode == "fresh_training_batches" and horizon == 20
                          else refit_value(trainer, state.core, jax.device_put(selected, device),
                                           jax.device_put(heldout, device)))
                trace.append(fit_metrics(fitted, mode, horizon))
        result["frozen_Q_post_update_V_trace"] = {
            "scope": "Analysis-only native expectile fits with Q frozen; post-update losses are evaluated explicitly",
            "online_metric_caveat": "Live value_loss_first=L(V0) and value_loss=L(V1) for a two-fit profile; neither is L(V2)",
            "reference_caveat": "Neighbor-action conditional reference remains approximate",
            "next_reference_16_vs_64_rms": rms(empirical_expectile(next_reference["q"][:, :16], cfg.iql_tau)-next_reference["roots"]),
            "next_reference_state_distance_mean": float(next_reference["distance"].mean()),
            "measurements": trace}
    for name in ("q_opt_state", "v_opt_state", "actor_opt_state"):
        arrays = [np.asarray(x) for x in jax.tree_util.tree_leaves(getattr(state.core, name))]
        result[name+"_finite_fraction"] = float(sum(np.isfinite(x).sum() for x in arrays)/sum(x.size for x in arrays))
    if research.PROFILES[profile_name].get("precondition") or "residual_target" in research.PROFILES[profile_name]:
        raw = head.apply({"params": state.kappa_params}, jnp.asarray(heldout["observations"]))
        moment_head = "residual_target" in research.PROFILES[profile_name]
        probability = np.asarray(jax.nn.sigmoid(raw[..., 1] if moment_head else raw))
        labels = np.mean(q-before[:, None] >= 0, axis=-1)
        result["kappa_probability_local_reference_rmse"] = rms(probability-labels)
        result["kappa_probability_mean"] = float(probability.mean())
        result["local_reference_positive_probability_mean"] = float(labels.mean())
        kappa_hat = (1-cfg.iql_tau)+(2*cfg.iql_tau-1)*probability
        result["head_kappa_global_relative_gap"] = float(abs(kappa_hat.mean()-actual_weights.mean())/max(actual_weights.mean(), 1e-12))
        result["head_probability_brier_sampled_validation_action"] = float(np.mean((probability-(advantage_before >= 0))**2))
        if moment_head:
            g_hat = np.asarray(raw[..., 0], dtype=np.float64)
            score = actual_weights*advantage_before
            estimated = g_hat/kappa_hat
            reference_delta = roots-before
            result["head_G_sample_label_rmse"] = rms(g_hat-score)
            result["sample_score_rms"] = rms(score)
            result["head_G_global_normalized_gap"] = float(abs(g_hat.mean()-score.mean())/max(np.mean(np.abs(score)), 1e-12))
            result["head_displacement_rms"] = rms(estimated)
            result["head_vs_local_action_displacement_rmse"] = rms(estimated-reference_delta)
            result["head_local_reference_direction_cosine"] = float(np.mean(estimated*reference_delta)/max(rms(estimated)*rms(reference_delta), 1e-12))
    if args.refit_head:
        if "residual_target" not in research.PROFILES[profile_name]:
            raise ValueError("Frozen moment-head refit requires a residual_target profile")
        hp, ho = state.kappa_params, state.kappa_opt

        @jax.jit
        def fit_head_chunk(params, optimizer, batch_sequence):
            def step(carry, batch):
                p, opt = carry
                obs, actions = batch["observations"], batch["actions"]
                target_q = jnp.min(trainer.q_def.apply(
                    {"params": state.core.q_target_params}, obs, actions), axis=0)
                v = trainer.v_def.apply({"params": state.core.v_params}, obs)
                advantage = target_q-v
                label = jax.lax.stop_gradient(jnp.where(advantage >= 0,
                    cfg.iql_tau, 1-cfg.iql_tau)*advantage)
                scale = jnp.maximum(jnp.sqrt(jnp.mean(label**2)), .01)
                def loss(candidate):
                    raw = head.apply({"params": candidate}, obs)
                    return jnp.mean(((raw[..., 0]-label)/scale)**2) + jnp.mean(
                        optax.sigmoid_binary_cross_entropy(raw[..., 1],
                            (advantage >= 0).astype(jnp.float32)))
                value, grad = jax.value_and_grad(loss)(p)
                update, opt = htx.update(grad, opt, p)
                return (optax.apply_updates(p, update), opt), value
            return jax.lax.scan(step, (params, optimizer), batch_sequence)

        raw_before = np.asarray(head.apply({"params": hp}, jnp.asarray(heldout["observations"])))
        for begin in range(0, args.refit_head, 100):
            head_batches = sample(train, (min(100, args.refit_head-begin), cfg.batch_size))
            (hp, ho), _ = fit_head_chunk(hp, ho, jax.device_put(head_batches, device))
        raw_after = np.asarray(head.apply({"params": hp}, jnp.asarray(heldout["observations"])))
        sampled_score = actual_weights*advantage_before
        score_mse_before = (raw_before[..., 0]-sampled_score)**2
        score_mse_after = (raw_after[..., 0]-sampled_score)**2
        score_mse_change = score_mse_after-score_mse_before
        p_after = np.asarray(jax.nn.sigmoid(raw_after[..., 1]))
        kappa_after = (1-cfg.iql_tau)+(2*cfg.iql_tau-1)*p_after
        displacement_after = raw_after[..., 0]/kappa_after
        reference_delta = roots-before
        result["frozen_Q_V_head_refit"] = {
            "additional_head_updates": args.refit_head,
            "scope": "Only moment-head parameters and optimizer; value and actor networks frozen",
            "validation_G_mse_before": float(score_mse_before.mean()),
            "validation_G_mse_after": float(score_mse_after.mean()),
            "validation_G_zero_prediction_mse": float(np.mean(sampled_score**2)),
            "validation_G_mse_change": float(score_mse_change.mean()),
            "validation_G_mse_change_standard_error": float(score_mse_change.std(ddof=1)/np.sqrt(len(score_mse_change))),
            "validation_G_prediction_rms_after": rms(raw_after[..., 0]),
            "probability_brier_sampled_validation_action_after": float(np.mean((p_after-(advantage_before >= 0))**2)),
            "local_reference_displacement_rmse_after": rms(displacement_after-reference_delta),
            "local_reference_direction_cosine_after": float(np.mean(displacement_after*reference_delta)/
                max(rms(displacement_after)*rms(reference_delta), 1e-12)),
            "reference_caveat": "Neighbor-action displacement is approximate; sampled score MSE includes irreducible action noise"}
    if not args.skip_b:
        fit = {k: v[0] for k, v in batches.items()}
        result["critic_fit_response"] = critic_response(trainer, state.core,
            jax.device_put(fit, device), jax.device_put(heldout, device), q_lr=cfg.qf_lr,
            q_scale=research.PROFILES[profile_name].get("q_step_scale", 1.))
    if args.loop:
        loop_batches = jax.device_put(sample(train, (500, cfg.batch_size)), device)
        if research.PROFILES[profile_name].get("fresh_v_batch"):
            extra = sample(train, (500, cfg.batch_size))
            loop_batches["extra_v_observations"] = jax.device_put(extra["observations"], device)
            loop_batches["extra_v_actions"] = jax.device_put(extra["actions"], device)
        bound_info = json.loads((run / "dataset.json").read_text())
        result["value_loop_response"] = value_loop_response(trainer, head, htx, state,
            research.PROFILES[profile_name], (bound_info["return_lower"], bound_info["return_upper"]),
            loop_batches, jax.device_put(heldout, device), cfg.qf_lr, cfg.vf_lr,
            saved.get("control_scale", 1.))
    result["wall_seconds"] = time.monotonic()-start
    suffix = "_loop" if args.loop else ""
    if args.queries != 128 or args.bank != 4096:
        suffix += f"_q{args.queries}_b{args.bank}"
    if args.skip_b:
        suffix += "_noB"
    if args.refit_head:
        suffix += f"_headrefit{args.refit_head}"
    if args.v_trace:
        suffix += "_vtrace"
    output = run / f"deep_diagnostics_{saved['step']}_{args.device}_{args.precision}{suffix}.json"
    (run / f"diagnostic_source_{diagnostic_source_hash[:12]}.py").write_bytes(diagnostic_source)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
