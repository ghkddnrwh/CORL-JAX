"""Controlled residual/propagation experiments, sharing the original IQL models.

The baseline calls the original update verbatim. All profiles share data sampling,
initialization, actor schedule, evaluation seeds and gradient-step budgets. Auxiliary
diagnostics use independent RNGs and do not consume the training minibatch stream.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.uk_offline.basic import iql_jax as base
from utils.propagation_research.reward_bounds import return_bounds

import optax
import yaml
from flax import linen as nn, struct


PROFILES = {
    "baseline": {},
    "numerical_control": {"numerical_control": True},
    "v4": {"v_steps": 4},
    "v4_fresh_backup": {"v_steps": 4, "fresh_backup": True},
    "v2_fresh_backup": {"v_steps": 2, "fresh_backup": True},
    "v2_old_backup": {"v_steps": 2},
    "fresh_backup": {"fresh_backup": True},
    "bounded_fresh": {"bound_backup": True, "fresh_backup": True},
    "bounded_v2_old": {"bound_backup": True, "v_steps": 2},
    "kappa": {"precondition": True},
    "layernorm": {"layernorm": True},
    "bounded": {"bound_backup": True},
    "target_v": {"target_v_rate": 0.005},
    "joint": {"v_steps": 2, "precondition": True, "layernorm": True,
              "target_v_rate": 0.005},
    "ln_kappa": {"precondition": True, "layernorm": True},
    "ln_v4": {"v_steps": 4, "layernorm": True},
    "ln_target_v": {"layernorm": True, "target_v_rate": 0.005},
    "q_slow": {"q_step_scale": 0.5},
    "target_slow": {"q_target_rate": 0.001},
    "residual_target": {"residual_target": 0.25},
    "residual_target_full": {"residual_target": 1.0},
    "residual_actor": {"residual_target": 0.25, "correct_actor": True},
    "ln_residual": {"residual_target": 0.25, "layernorm": True},
    "lagged_weights": {"weight_target_rate": 0.005},
    "lagged_weights_v2": {"weight_target_rate": 0.005, "v_steps": 2},
    "lagged_joint": {"weight_target_rate": 0.005, "target_v_rate": 0.005,
                     "layernorm": True},
    "jvp_fit_control": {"jvp_control": True},
    "jvp_residual": {"jvp_control": True, "residual_target": 0.25},
    "bounded_v2": {"bound_backup": True, "v_steps": 2, "fresh_backup": True},
    "bounded_v2_independent": {"bound_backup": True, "v_steps": 2,
                                "fresh_backup": True, "fresh_v_batch": True},
    "bounded_residual": {"bound_backup": True, "residual_target": 0.25},
    "bounded_ln": {"bound_backup": True, "layernorm": True},
    "negative_v": {"negative_value": True},
    "negative_v2": {"negative_value": True, "v_steps": 2, "fresh_backup": True},
    "ln_negative_v": {"negative_value": True, "layernorm": True},
}


class NormalizedQ(nn.Module):
    hidden_dim: int
    n_hidden: int

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        for _ in range(self.n_hidden):
            x = nn.relu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        return nn.Dense(1)(x)[..., 0]


class NormalizedTwinQ(nn.Module):
    hidden_dim: int
    n_hidden: int
    num_critics: int = 2

    @nn.compact
    def __call__(self, state, action):
        return jnp.stack([NormalizedQ(self.hidden_dim, self.n_hidden,
                                     name=f"q{i+1}")(state, action)
                          for i in range(self.num_critics)])


class KappaHead(nn.Module):
    @nn.compact
    def __call__(self, state):
        x = nn.relu(nn.Dense(128)(state))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(1)(x)[..., 0]


class MomentHead(nn.Module):
    @nn.compact
    def __call__(self, state):
        x = nn.relu(nn.Dense(128, name="g1")(state))
        x = nn.relu(nn.Dense(128, name="g2")(x))
        g = nn.Dense(1, kernel_init=nn.initializers.zeros,
                     bias_init=nn.initializers.zeros, name="g_out")(x)[..., 0]
        probability_logit = KappaHead(name="kappa")(state)
        return jnp.stack([g, probability_logit], axis=-1)


class NonpositiveValue(nn.Module):
    hidden_dim: int
    n_hidden: int

    @nn.compact
    def __call__(self, state):
        x = state
        for _ in range(self.n_hidden):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        raw = nn.Dense(1)(x)[..., 0]
        # Identity-like at negative cost values; enforces the known zero upper
        # return bound smoothly. Only used after checking nonpositive rewards.
        return -jax.nn.softplus(-raw)


@struct.dataclass
class ResearchState:
    core: base.IQLState
    v_target: object
    kappa_params: object
    kappa_opt: object


def initialize(config, profile, state_dim, action_dim, device, initialization="iql"):
    keys = ("qf_lr", "vf_lr", "actor_lr", "discount", "tau", "beta", "iql_tau",
            "iql_deterministic", "actor_dropout", "hidden_dim", "n_hidden",
            "num_critics", "seed")
    trainer = base.IQLJAX(max_action=1.0, state_dim=state_dim, action_dim=action_dim,
                          max_steps=config.max_timesteps, device=device,
                          **{k: getattr(config, k) for k in keys})
    key_q = jax.random.split(jax.random.PRNGKey(config.seed), 4)[1]
    if initialization == "diagnostic":
        # PD splits five keys; original IQL splits four. Same integer seed does
        # not give the same model initialization in those two programs.
        ka, key_q, kv, _, ak = jax.random.split(jax.random.PRNGKey(config.seed), 5)
        obs, act = jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim))
        ap = trainer.actor_def.init(ka, obs, training=False)["params"]
        qp = trainer.q_def.init(key_q, obs, act)["params"]
        vp = trainer.v_def.init(kv, obs)["params"]
        trainer.state = trainer.state.replace(q_params=qp, q_target_params=qp,
            q_opt_state=trainer.q_tx.init(qp), v_params=vp, v_opt_state=trainer.v_tx.init(vp),
            actor_params=ap, actor_opt_state=trainer.actor_tx.init(ap), actor_key=ak)
        trainer.initial_actor_params = ap
        trainer.initial_actor_opt_state = trainer.actor_tx.init(ap)
        trainer.initial_actor_key = ak
    if profile.get("layernorm"):
        trainer.q_def = NormalizedTwinQ(config.hidden_dim, config.n_hidden,
                                        config.num_critics)
        qp = trainer.q_def.init(key_q, jnp.zeros((1, state_dim)),
                               jnp.zeros((1, action_dim)))["params"]
        trainer.state = trainer.state.replace(q_params=qp, q_target_params=qp,
                                               q_opt_state=trainer.q_tx.init(qp))
    if profile.get("negative_value"):
        trainer.v_def = NonpositiveValue(config.hidden_dim, config.n_hidden)
    trainer._train_step = trainer._build_train_step()
    trainer._actor_refit_step = trainer._build_actor_refit_step()
    head = MomentHead() if "residual_target" in profile else KappaHead()
    hp = head.init(jax.random.PRNGKey(config.seed + 4831),
                   jnp.zeros((1, state_dim)))["params"]
    htx = optax.adam(3e-4)
    state = ResearchState(trainer.state, trainer.state.v_params, hp, htx.init(hp))
    return trainer, head, htx, state


def build_update(trainer, head, htx, profile, reward_bounds, hypergradient=False,
                 q_lr=3e-4, v_lr=3e-4):
    # Keep exact original update for the baseline, including the old-V backup.
    if not profile and not hypergradient:
        def baseline(state, batch):
            core, logs = trainer._train_step(state.core, batch)
            return state.replace(core=core), logs
        return baseline

    qa, va = trainer.q_def.apply, trainer.v_def.apply
    sg = (lambda x: x) if hypergradient else jax.lax.stop_gradient
    q_optimizer, v_optimizer, h_optimizer = trainer.q_tx.update, trainer.v_tx.update, htx.update
    if hypergradient:
        from utils.propagation_research.diagnose_checkpoint import differentiable_adam
        # Targets remain closure constants for each INNER parameter gradient,
        # while OUTER differentiation must follow their dependence on prior state.
        q_optimizer = lambda g, o, p: differentiable_adam(g, o, q_lr)
        v_optimizer = lambda g, o, p: differentiable_adam(g, o, v_lr)
        h_optimizer = lambda g, o, p: differentiable_adam(g, o, 3e-4)
    gamma, expectile = trainer.discount, trainer.iql_tau
    v_steps = profile.get("v_steps", 1)
    fresh_v_batch = profile.get("fresh_v_batch", False)
    if fresh_v_batch and (v_steps != 2 or profile.get("precondition") or profile.get("residual_target")):
        raise ValueError("Independent V batches currently support two unweighted expectile updates")
    lower, upper = reward_bounds

    def update(state, batch):
        c = state.core
        obs, act, nxt = batch["observations"], batch["actions"], batch["next_observations"]
        rew, done = batch["rewards"][..., 0], batch["dones"][..., 0]
        tq = jnp.min(qa({"params": c.q_target_params}, obs, act), axis=0)
        old_v = va({"params": c.v_params}, obs)
        adv = tq - old_v
        hp, ho = state.kappa_params, state.kappa_opt
        precondition = profile.get("precondition", False)
        residual_target = profile.get("residual_target", 0.)
        residual_clip_fraction = jnp.asarray(0.)
        residual_applied_rms = jnp.asarray(0.)
        correction_cap = max((upper-lower)*(1-gamma)*.5, .5)

        def moment_correction(params, observations):
            raw = head.apply({"params": params}, observations)
            kappa = (1-expectile)+(2*expectile-1)*jax.nn.sigmoid(raw[..., 1])
            delta = raw[..., 0]/jnp.maximum(kappa, 1-expectile)
            return jnp.clip(residual_target*delta, -correction_cap, correction_cap), delta

        if residual_target:
            value_weight = jnp.where(adv >= 0, expectile, 1-expectile)
            label = sg(value_weight*adv)
            label_scale = sg(jnp.maximum(jnp.sqrt(jnp.mean(label**2)), .01))
            def moment_loss(p):
                raw = head.apply({"params": p}, obs)
                g_loss = jnp.mean(((raw[..., 0]-label)/label_scale)**2)
                k_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(
                    raw[..., 1], sg((adv >= 0).astype(jnp.float32))))
                return g_loss+k_loss
            hl, hg = jax.value_and_grad(moment_loss)(hp)
            hu, ho = h_optimizer(hg, ho, hp)
            hp = optax.apply_updates(hp, hu)
            scaling = jnp.ones_like(old_v)
            if profile.get("correct_actor"):
                correction, _ = moment_correction(state.kappa_params, obs)
                adv = adv-sg(correction)
        elif precondition:
            def head_loss(p):
                logits = head.apply({"params": p}, obs)
                return jnp.mean(optax.sigmoid_binary_cross_entropy(
                    logits, sg((adv >= 0).astype(jnp.float32))))
            hl, hg = jax.value_and_grad(head_loss)(hp)
            hu, ho = h_optimizer(hg, ho, hp)
            hp = optax.apply_updates(hp, hu)
            probability = jax.nn.sigmoid(head.apply({"params": state.kappa_params}, obs))
            khat = (1-expectile) + (2*expectile-1)*probability
            # State-dependent, detached curvature scaling preserves the conditional
            # expectile root only when kappa is accurately estimated. It is a proxy.
            invk = 1 / jnp.maximum(khat, max(1-expectile, 0.02))
            scaling = sg(invk / jnp.mean(invk))
        else:
            hl = jnp.asarray(0.)
            scaling = jnp.ones_like(old_v)

        lagged_weight = None
        if "weight_target_rate" in profile:
            lagged_adv = tq-va({"params": state.v_target}, obs)
            lagged_weight = sg(jnp.where(lagged_adv >= 0, expectile, 1-expectile))

        if fresh_v_batch:
            extra_observations = batch["extra_v_observations"]
            extra_q = jnp.min(qa({"params": c.q_target_params}, extra_observations,
                                 batch["extra_v_actions"]), axis=0)
            v_data = (jnp.stack([obs, extra_observations]), jnp.stack([tq, extra_q]))
        else:
            v_data = None

        def v_fit(carry, data):
            vp, vo = carry
            v_obs, v_target = data if fresh_v_batch else (obs, tq)
            def loss(p):
                residual = sg(v_target) - va({"params": p}, v_obs)
                weight = jnp.where(residual >= 0, expectile, 1-expectile)
                if lagged_weight is not None:
                    weight = lagged_weight
                return jnp.mean(scaling * weight * residual**2)
            vl, vg = jax.value_and_grad(loss)(vp)
            vu, vo = v_optimizer(vg, vo, vp)
            return (optax.apply_updates(vp, vu), vo), vl

        (vp, vo), vls = jax.lax.scan(v_fit, (c.v_params, c.v_opt_state), v_data,
                                      length=v_steps)
        backup_params = vp if profile.get("fresh_backup") else c.v_params
        if "target_v_rate" in profile:
            backup_params = state.v_target
        target = rew + (1-done)*gamma*va({"params": backup_params}, nxt)
        if residual_target:
            correction, raw_correction = moment_correction(state.kappa_params, nxt)
            applied = sg((1-done)*gamma*correction)
            target = target+applied
            residual_applied_rms = jnp.sqrt(jnp.mean(applied**2))
            residual_clip_fraction = jnp.mean((jnp.abs(residual_target*raw_correction) > correction_cap).astype(jnp.float32))
        unclipped_target = target
        if profile.get("bound_backup"):
            target = jnp.clip(target, lower, upper)

        def q_loss(p):
            qs = qa({"params": p}, obs, act)
            return jnp.mean((qs-sg(target)[None])**2), qs
        (ql, qs), qg = jax.value_and_grad(q_loss, has_aux=True)(c.q_params)
        qu, qo = q_optimizer(qg, c.q_opt_state, c.q_params)
        if "q_step_scale" in profile:
            qu = jax.tree_util.tree_map(lambda x: profile["q_step_scale"]*x, qu)
        if profile.get("jvp_control"):
            qu = jax.tree_util.tree_map(lambda x: batch["control_scale"]*x, qu)
        qp = optax.apply_updates(c.q_params, qu)
        qt = base.soft_update(qp, c.q_target_params, profile.get("q_target_rate", trainer.tau))
        ak, dk = jax.random.split(c.actor_key)
        weights = jnp.exp(jnp.minimum(trainer.beta*sg(adv), jnp.log(base.EXP_ADV_MAX)))

        def actor_loss(p):
            out = trainer._apply_actor(p, obs, training=True, rng=dk)
            if trainer.iql_deterministic:
                bc = jnp.sum((out-act)**2, axis=-1)
            else:
                mean, log_std = out
                bc = jnp.sum(0.5*(((act-mean)/jnp.exp(log_std))**2 +
                                  2*log_std+jnp.log(2*jnp.pi)), axis=-1)
            return jnp.mean(weights*bc)
        al, ag = jax.value_and_grad(actor_loss)(c.actor_params)
        au, ao = trainer.actor_tx.update(ag, c.actor_opt_state, c.actor_params)
        ap = optax.apply_updates(c.actor_params, au)
        core = c.replace(total_it=c.total_it+1, q_params=qp, q_target_params=qt,
                         q_opt_state=qo, v_params=vp, v_opt_state=vo,
                         actor_params=ap, actor_opt_state=ao, actor_key=ak)
        vt = base.soft_update(vp, state.v_target,
                              profile.get("target_v_rate", profile.get("weight_target_rate", 1.0)))
        logs = {"q_loss": ql, "value_loss": vls[-1], "value_loss_first": vls[0],
                "actor_loss": al, "q_mean": jnp.mean(qs), "v_mean": jnp.mean(old_v),
                "kappa_head_loss": hl, "precondition_min": jnp.min(scaling),
                "precondition_max": jnp.max(scaling),
                "backup_clip_fraction": jnp.mean((target != unclipped_target).astype(jnp.float32)),
                "residual_applied_rms": residual_applied_rms,
                "residual_clip_fraction": residual_clip_fraction,
                "q_update_scale": batch["control_scale"] if profile.get("jvp_control") else
                    jnp.asarray(profile.get("q_step_scale", 1.)),
                "lagged_weight_disagreement": jnp.asarray(0.) if lagged_weight is None else
                    jnp.mean((lagged_weight != jnp.where(tq-old_v >= 0, expectile, 1-expectile)).astype(jnp.float32)),
                "exp_adv_mean": jnp.mean(weights)}
        return ResearchState(core, vt, hp, ho), logs
    return update


def make_scan(update):
    @jax.jit
    def run(state, batches):
        return jax.lax.scan(update, state, batches)
    return run


def build_probe(trainer):
    @jax.jit
    def probe(c, batch):
        obs, act = batch["observations"], batch["actions"]
        qs = trainer.q_def.apply({"params": c.q_params}, obs, act)
        tq = trainer.q_def.apply({"params": c.q_target_params}, obs, act)
        v = trainer.v_def.apply({"params": c.v_params}, obs)
        vn = trainer.v_def.apply({"params": c.v_params}, batch["next_observations"])
        target = batch["rewards"][..., 0]+trainer.discount*(1-batch["dones"][..., 0])*vn
        moment_finite = {}
        for name in ("q_opt_state", "v_opt_state", "actor_opt_state"):
            leaves = jax.tree_util.tree_leaves(getattr(c, name))
            finite_count = sum(jnp.sum(jnp.isfinite(x)) for x in leaves)
            moment_finite[name+"_finite_fraction"] = finite_count/sum(x.size for x in leaves)
            moment_finite[name+"_all_finite"] = jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(x)) for x in leaves]))
        return qs, tq, v, target, moment_finite
    return probe


def probe_summary(values, expectile, lower, upper, beta):
    qs, tq, v, target = [np.asarray(x, dtype=np.float64) for x in values[:4]]
    moment_finite = {k: float(x) for k, x in values[4].items()}
    q = qs.mean(0)
    adv = tq.min(0)-v
    w = np.where(adv >= 0, expectile, 1-expectile)
    score = w*adv
    actor_weights = np.exp(np.minimum(beta*adv, np.log(base.EXP_ADV_MAX)))
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    scale = max(abs(lower), abs(upper), 1.)
    return {**moment_finite, "q_rms": rms(q), "v_rms": rms(v), "q_centered_rms": rms(q-q.mean()),
            "q_abs_p99": float(np.quantile(np.abs(q), .99)),
            "q_mean": float(q.mean()), "v_mean": float(v.mean()),
            "q_disagreement_rms": rms(qs[0]-qs[1]), "td_rms": rms(qs-target[None]),
            "q_target_lag_rms": rms(qs-tq), "score_mean": float(score.mean()),
            "score_rms": rms(score), "kappa_batch": float(w.mean()),
            "global_score_over_kappa": float(score.mean()/max(w.mean(), 1e-12)),
            "positive_adv_fraction": float(np.mean(adv >= 0)),
            "weight_ess_fraction": float(w.mean()**2/max(np.mean(w*w), 1e-12)),
            "adv_p05": float(np.quantile(adv, .05)), "adv_p95": float(np.quantile(adv, .95)),
            "actor_weight_clip_fraction": float(np.mean(beta*adv >= np.log(base.EXP_ADV_MAX))),
            "actor_weight_ess_fraction": float(actor_weights.mean()**2/max(np.mean(actor_weights**2), 1e-300)),
            "actor_weight_underflow_fraction": float(np.mean(beta*adv < -87)),
            "q_bound_violation_fraction": float(np.mean((qs < lower-1)|(qs > upper+1))),
            "q_above_return_upper_rms": rms(np.maximum(qs-upper, 0.)),
            "q_min_above_return_upper_rms": rms(np.maximum(qs.min(0)-upper, 0.)),
            "v_above_return_upper_rms": rms(np.maximum(v-upper, 0.)),
            "q_below_return_lower_rms": rms(np.maximum(lower-qs, 0.)),
            "q_rms_over_return_bound": rms(q)/scale,
            "finite": bool(all(np.isfinite(x).all() for x in (qs, tq, v, target))
                           and all(value == 1. for key, value in moment_finite.items()
                                   if key.endswith("_all_finite")))}


def append_json(path, item):
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    with path.open("a") as f:
        f.write(json.dumps(finite_json(item), allow_nan=False)+"\n")


def save_atomic(path, payload):
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)


def main():
    setup_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initialization", choices=("iql", "diagnostic"), default="iql")
    parser.add_argument("--expectile", type=float, default=.9)
    parser.add_argument("--discount", type=float)
    parser.add_argument("--bound-source", choices=("empirical", "ogbench"), default="empirical")
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--stop-at", type=int, default=None,
                        help="Screening endpoint; actor schedule remains --steps.")
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--probe-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100000)
    parser.add_argument("--promote", action="store_true",
                        help="Allow increasing evaluation episodes when resuming a screened model.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = base.TrainConfig(env=args.env, seed=args.seed, log_wandb=False)
    with open("hyperparams/ogbench/iql_jax.yml") as f:
        hp = yaml.safe_load(f)[args.env]
    for k, val in hp.items():
        if k in cfg.__dataclass_fields__:
            setattr(cfg, k, val)
    cfg.max_timesteps, cfg.iql_tau = args.steps, args.expectile
    if args.discount is not None:
        cfg.discount = args.discount
    for field in ("max_timesteps", "batch_size", "hidden_dim", "n_hidden"):
        setattr(cfg, field, int(getattr(cfg, field)))
    profile = PROFILES[args.profile]
    identity = {"config": asdict(cfg), "profile": profile,
                "protocol": {"chunk": args.chunk, "episodes": args.episodes,
                             "eval_every": args.eval_every, "probe_every": args.probe_every}}
    if args.initialization != "iql":
        identity["protocol"]["initialization"] = args.initialization
    if args.eval_device != "gpu":
        identity["protocol"]["evaluation_device"] = args.eval_device
    if args.bound_source != "empirical":
        identity["protocol"]["return_bound_source"] = args.bound_source
    config_path = out / "config.json"
    if config_path.exists():
        previous_identity = json.loads(config_path.read_text())["identity"]
        if previous_identity != identity:
            before, after = json.loads(json.dumps(previous_identity)), json.loads(json.dumps(identity))
            before["protocol"].pop("episodes", None)
            after["protocol"].pop("episodes", None)
            if not args.promote or before != after:
                raise ValueError("Existing run has different scientific configuration")
            append_json(out / "promotions.jsonl", {"time": time.time(),
                "previous_protocol": previous_identity["protocol"], "new_protocol": identity["protocol"],
                "arguments": vars(args)})
    else:
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # Source-only exports have no .git directory. The exact source hash
        # remains available even when a checkout revision cannot be queried.
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                               stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        provenance = {"identity": identity, "arguments": vars(args),
                      "git_head": revision,
                      "source_sha256": source_hash, "jax_version": jax.__version__}
        config_path.write_text(json.dumps(provenance, indent=2))
        (out / "algorithm_source.py").write_bytes(Path(__file__).read_bytes())
    if (out / "completed.json").exists():
        print("Already complete", flush=True)
        return
    device = base.select_jax_device("gpu")
    env, dataset, _ = base.load_env_and_dataset(args.env)
    mean, std = base.compute_mean_std(dataset["observations"], 1e-3) if cfg.normalize else (0., 1.)
    env = base.wrap_env(env, state_mean=mean, state_std=std)
    data = {k: np.asarray(dataset[k], dtype=np.float32) for k in
            ("observations", "actions", "next_observations")}
    data["observations"] = base.normalize_states(data["observations"], mean, std)
    data["next_observations"] = base.normalize_states(data["next_observations"], mean, std)
    data["rewards"] = np.asarray(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
    dones = 1-dataset["masks"] if "masks" in dataset else dataset["terminals"]
    data["dones"] = np.asarray(dones, dtype=np.float32).reshape(-1, 1)
    rmin, rmax = float(data["rewards"].min()), float(data["rewards"].max())
    if profile.get("negative_value") and rmax > 0:
        raise ValueError("Nonpositive V requires a nonpositive reward range")
    # Include zero for terminal returns. Dataset extrema are empirical bounds,
    # not an assumed oracle; this is explicitly logged for the projection profile.
    lower, upper = return_bounds(args.env, cfg.discount, rmin, rmax, args.bound_source)
    (out / "dataset.json").write_text(json.dumps({"size": len(data["actions"]),
        "reward_min": rmin, "reward_max": rmax, "return_lower": lower,
        "return_upper": upper, "return_bound_source": args.bound_source,
        "state_dim": data["observations"].shape[-1],
        "action_dim": data["actions"].shape[-1], "mask_mean": float(1-data["dones"].mean())}))
    base.set_seed(args.seed, env)
    trainer, head, htx, state = initialize(cfg, profile, data["observations"].shape[-1],
                                          data["actions"].shape[-1], device, args.initialization)
    scan = make_scan(build_update(trainer, head, htx, profile, (lower, upper)))
    probe = build_probe(trainer)
    rng = np.random.RandomState(args.seed)
    extra_v_rng = np.random.RandomState(args.seed+54331)
    probe_rng = np.random.RandomState(args.seed+19401)
    def sample(indices):
        return {k: jax.device_put(x[indices], device) for k, x in data.items()}
    fixed_probe = sample(probe_rng.randint(len(data["actions"]), size=1024))
    checkpoint = out / "latest.pkl"
    start_step = 0
    control_scale = 1.
    if checkpoint.exists():
        with checkpoint.open("rb") as f:
            saved = pickle.load(f)
        if saved.get("analysis_only"):
            raise ValueError("Analysis-only snapshots cannot resume training")
        state = jax.device_put(saved["state"], device)
        rng.set_state(saved["rng"])
        if profile.get("fresh_v_batch"):
            if "extra_v_rng" not in saved:
                raise ValueError("Independent V-batch checkpoint lacks its replay RNG")
            extra_v_rng.set_state(saved["extra_v_rng"])
        start_step = saved["step"]
        control_scale = saved.get("control_scale", 1.)
        resume_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        append_json(out / "resume_history.jsonl", {"step": start_step,
            "source_sha256": resume_hash, "arguments": vars(args), "time": time.time()})
        (out / f"algorithm_resume_{resume_hash[:12]}.py").write_bytes(Path(__file__).read_bytes())
        # Discard records newer than the recoverable model, including partial evals.
        for name in ("diagnostics.jsonl", "evaluations.jsonl"):
            path = out / name
            if path.exists():
                records = [json.loads(x) for x in path.read_text().splitlines()]
                path.write_text("".join(json.dumps(x)+"\n" for x in records if x["step"] <= start_step))
        print(f"Resuming at {start_step}", flush=True)
    if profile.get("jvp_control"):
        from utils.propagation_research.diagnose_checkpoint import build_jvp_controller
        controller = build_jvp_controller(trainer, cfg.qf_lr)
        control_rng = np.random.RandomState(args.seed+61597)
        control_fit = sample(control_rng.randint(len(data["actions"]), size=cfg.batch_size))
        control_heldout = {k: x[:256] for k, x in fixed_probe.items()}
    # Compile deterministic evaluation once instead of dispatching individual layers.
    def policy(params, observation):
        action = trainer._apply_actor(params, observation, training=False)
        return jnp.clip(action if cfg.iql_deterministic else action[0], -1., 1.)
    policy = jax.jit(policy, backend=args.eval_device)
    eval_device = jax.devices(args.eval_device)[0]
    def evaluate(step):
        rewards, successes = [], []
        eval_params = jax.device_put(state.core.actor_params, eval_device)
        for episode in range(args.episodes):
            # Reproduce original evaluation: seed first reset only.
            obs = base.reset_env(env, seed=args.seed if episode == 0 else None)
            done, reward, success = False, 0., np.nan
            while not done:
                action = np.asarray(policy(eval_params, np.asarray(obs[None], dtype=np.float32)))[0]
                obs, r, done, info = base.step_env(env, action)
                reward += r
                value = base.extract_success(info)
                if np.isfinite(value):
                    success = value
            rewards.append(reward)
            successes.append(success)
        record = {"step": step, "success_rate": float(np.mean(successes)),
                  "reward_mean": float(np.mean(rewards)), "episode_successes": successes,
                  "episode_rewards": rewards, "n_episodes": args.episodes,
                  "eval_device": args.eval_device}
        append_json(out / "evaluations.jsonl", record)
        print(json.dumps({k: v for k, v in record.items() if not isinstance(v, list)}), flush=True)
    start_time = time.monotonic()
    timing = {"setup_seconds": start_time-setup_started, "batch_preparation_seconds": 0.,
              "train_seconds": 0., "probe_seconds": 0., "evaluation_seconds": 0.,
              "checkpoint_seconds": 0., "controller_seconds": 0.}
    window_logs, window_steps = {}, 0
    endpoint = args.stop_at or args.steps
    last_step = start_step
    numerical_termination = False
    for step in range(start_step, endpoint, args.chunk):
        length = min(args.chunk, endpoint-step)
        if profile.get("jvp_control") and step >= 10000 and step % 10000 == 0:
            controller_started = time.monotonic()
            candidates = []
            for scale in (1., .5, .25, .1):
                gains = np.asarray(controller(state.core, control_fit, control_heldout,
                                               jnp.asarray(scale, dtype=jnp.float32)))
                candidates.append({"scale": scale, "constant_gain": float(gains[0]),
                                   "random_gain": float(gains[1]),
                                   "gamma_max_gain": float(cfg.discount*np.max(gains)),
                                   "finite": bool(np.isfinite(gains).all())})
                if candidates[-1]["finite"] and candidates[-1]["gamma_max_gain"] <= .8:
                    break
            valid = [x for x in candidates if x["finite"]]
            safe = [x for x in valid if x["gamma_max_gain"] <= .8]
            chosen = safe[0] if safe else min(valid, key=lambda x: x["gamma_max_gain"]) if valid else None
            control_scale = chosen["scale"] if chosen else 1.
            append_json(out / "controller.jsonl", {"step": step, "scale": control_scale,
                        "candidates": candidates, "valid": bool(valid), "margin_met": bool(safe)})
            timing["controller_seconds"] += time.monotonic()-controller_started
        sample_started = time.monotonic()
        idx = rng.randint(len(data["actions"]), size=(length, cfg.batch_size))
        batches = sample(idx)
        if profile.get("fresh_v_batch"):
            extra_idx = extra_v_rng.randint(len(data["actions"]), size=(length, cfg.batch_size))
            batches["extra_v_observations"] = jax.device_put(data["observations"][extra_idx], device)
            batches["extra_v_actions"] = jax.device_put(data["actions"][extra_idx], device)
        if profile.get("jvp_control"):
            batches["control_scale"] = jax.device_put(np.full(length, control_scale, np.float32), device)
        timing["batch_preparation_seconds"] += time.monotonic()-sample_started
        train_started = time.monotonic()
        state, logs = scan(state, batches)
        host_logs = jax.device_get(logs)
        timing["train_seconds"] += time.monotonic()-train_started
        if step == start_step:
            timing["first_chunk_seconds_including_compile"] = time.monotonic()-train_started
        for key, values in host_logs.items():
            window_logs[key] = window_logs.get(key, 0.)+float(np.asarray(values, dtype=np.float64).sum())
        window_steps += length
        last_step = step+length
        if last_step % args.probe_every == 0:
            probe_started = time.monotonic()
            stats = probe_summary(jax.device_get(probe(state.core, fixed_probe)),
                                   cfg.iql_tau, lower, upper, cfg.beta)
            stats.update({"step": last_step, "wall_seconds": time.monotonic()-start_time,
                          "updates_per_second": (last_step-start_step)/max(time.monotonic()-start_time, 1e-9),
                          "train_metric_window_steps": window_steps,
                          "train": {k: value/window_steps for k, value in window_logs.items()}})
            window_logs, window_steps = {}, 0
            timing["probe_seconds"] += time.monotonic()-probe_started
            stats["timing"] = dict(timing)
            append_json(out / "diagnostics.jsonl", stats)
            if last_step % 10000 == 0:
                print(json.dumps({k: stats[k] for k in ("step", "q_rms", "v_rms", "td_rms", "updates_per_second")}), flush=True)
            if not stats["finite"] or stats["q_rms_over_return_bound"] > 100:
                terminal_started = time.monotonic()
                host_state = jax.device_get(state)
                terminal = out / f"terminal_checkpoint_{last_step}.pkl"
                save_atomic(terminal, {"step": last_step, "state": host_state,
                    "rng": rng.get_state(), "extra_v_rng": extra_v_rng.get_state(),
                    "control_scale": control_scale, "analysis_only": True,
                    "optimizer_states_preserved": True, "kind": "terminal_full_state"})
                stats["terminal_checkpoint"] = terminal.name
                stats["terminal_actor_parameters_finite"] = bool(all(
                    np.isfinite(np.asarray(x)).all() for x in
                    jax.tree_util.tree_leaves(host_state.core.actor_params)))
                timing["checkpoint_seconds"] += time.monotonic()-terminal_started
                stats["timing"] = dict(timing)
                (out / "diverged.json").write_text(json.dumps(stats, indent=2))
                numerical_termination = True
                break
        if last_step % args.eval_every == 0 or last_step == endpoint:
            eval_started = time.monotonic()
            evaluate(last_step)
            timing["evaluation_seconds"] += time.monotonic()-eval_started
        if last_step % args.checkpoint_every == 0 or last_step == endpoint:
            checkpoint_started = time.monotonic()
            save_atomic(checkpoint, {"step": last_step, "state": jax.device_get(state),
                                     "rng": rng.get_state(), "extra_v_rng": extra_v_rng.get_state(),
                                     "control_scale": control_scale})
            if last_step in (100000, 300000, args.steps):
                snapshot = out / f"checkpoint_{last_step}.pkl"
                if not snapshot.exists():
                    # Future latest checkpoints use atomic replacement, so this
                    # hard link preserves the exact model without a second copy.
                    os.link(checkpoint, snapshot)
            timing["checkpoint_seconds"] += time.monotonic()-checkpoint_started
    status = ("diverged" if numerical_termination else "complete" if last_step == args.steps
              else "screened" if last_step == endpoint else "diverged")
    (out / ("completed.json" if status == "complete" else "status.json")).write_text(
        json.dumps({"status": status, "step": last_step, "wall_seconds": time.monotonic()-start_time,
                    "timing": timing}))
    env.close()


if __name__ == "__main__":
    main()
