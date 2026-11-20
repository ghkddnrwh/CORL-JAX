"""Real-checkpoint optimizer/continuation checks before the stabilization runs."""
import argparse
import json
from pathlib import Path
import pickle

import flax.serialization
import jax
import numpy as np
import ogbench

from utils.codex.ukjo1.error_propagation.flow_iql_research import critic_stabilization_core_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as ref
from utils.codex.ukjo1.error_propagation.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.codex.ukjo1.error_propagation.flow_iql_research.diagnose_sample_budget_components_v1 import atomic, read_source


def diff(a, b):
    aa, ta = jax.tree_util.tree_flatten(a)
    bb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    values = [np.max(np.abs(np.asarray(x, np.float64)-np.asarray(y, np.float64))) for x, y in zip(aa, bb) if np.asarray(x).size]
    return max(values, default=0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    results = []
    for case in [c for c in plan['cases'] if c['seed'] == 1]:
        parent = Path(case['parent_run'])
        source = dict(run=str(parent), checkpoint=str(parent/'checkpoint_800000.pkl'), step=800000,
            config_sha256=case['config_sha256'], checkpoint_sha256=case['parent_checkpoints']['800000'])
        _, state = read_source(source)
        with (parent/'checkpoint_800000.pkl').open('rb') as stream:
            saved = pickle.load(stream)
        env, data, validation = ogbench.make_env_and_datasets(case['environment'])
        env.close()
        del validation
        rng = np.random.RandomState(case['seed'])
        rng.set_state(saved['replay_rng'])
        indices = rng.randint(len(data['actions']), size=(8, 256))
        batches = {k: jax.device_put(np.asarray(data[k][indices], np.float32)) for k in
                   ('observations', 'actions', 'next_observations', 'rewards', 'masks')}
        batches['actions'] = np.clip(batches['actions'], -1+1e-5, 1-1e-5)
        del data
        dataset = json.loads((parent/'dataset.json').read_text())
        bounds = dataset['return_lower'], dataset['return_upper']
        original, _ = ref.make_scan(ref.build_update(case['recipe'], bounds))(state, batches)
        control, _ = ref.make_scan(core.build_update(case['recipe'], bounds, 'control'))(state, batches)
        difference = diff(flax.serialization.to_state_dict(original), flax.serialization.to_state_dict(control))
        assert difference == 0., difference
        batch = {k: v[0] for k,v in batches.items()}
        a, _ = jax.jit(ref.build_update(case['recipe'], bounds))(state, batch)
        b, _ = jax.jit(core.build_update(case['recipe'], bounds, 'q_lr_0p1'))(state, batch)
        old_q = state.base.agent.network.params['modules_critic']
        aq, bq = a.base.agent.network.params['modules_critic'], b.base.agent.network.params['modules_critic']
        expected = jax.tree_util.tree_map(lambda p,q:p + .1*(q-p), old_q, aq)
        q_error = diff(bq, expected)
        # The reference delta is recovered from already-rounded fp32 weights.
        # Bound that subtraction/addition check in weight ULPs, not a fixed
        # absolute epsilon (some LayerNorm weights are larger than one).
        ulp_ratios=[]
        for old,full,reduced,want in zip(*[jax.tree_util.tree_leaves(x) for x in (old_q,aq,bq,expected)]):
            scale=np.maximum.reduce([np.abs(np.asarray(x,np.float32)) for x in (old,full,reduced)])
            bound=2.*np.spacing(scale).astype(np.float64)
            error=np.abs(np.asarray(reduced,np.float64)-np.asarray(want,np.float64))
            ulp_ratios.append(float(np.max(error/bound)))
        assert max(ulp_ratios) <= 1., (q_error,max(ulp_ratios))
        flow_error = diff(a.base.agent.network.params['modules_actor_flow'], b.base.agent.network.params['modules_actor_flow'])
        actor_error = diff(a.actor, b.actor)
        moment_error = diff(a.base.agent.network.opt_state, b.base.agent.network.opt_state)
        target_error = diff(a.base.agent.network.params['modules_target_critic'], b.base.agent.network.params['modules_target_critic'])
        assert flow_error == actor_error == moment_error == target_error == 0.
        assert np.array_equal(np.asarray(a.base.agent.rng), np.asarray(b.base.agent.rng))
        result = dict(case=case['id'], real_checkpoint_and_minibatches=True, continuation_updates=8,
            control_full_state_max_abs_difference=difference, scaled_Q_max_abs_error=q_error,
            scaled_Q_max_fraction_of_two_weight_ULP_bound=max(ulp_ratios),
            first_update_BC_Flow_difference=flow_error, first_update_actor_difference=actor_error,
            first_update_Adam_moments_difference=moment_error, first_update_target_Q_difference=target_error)
        results.append(result)
        print(json.dumps(result), flush=True)
    endpoints = {m: [float(core.lr_multiplier(m,s)) for s in (800000, 899999, 999999)] for m in core.METHODS}
    assert endpoints['q_cosine'][0] == 1. and endpoints['q_cosine'][-1] == 0.
    atomic(args.plan.parent/'analysis/implementation_validation_v1.json',
           dict(status='PASS', checks=results, lr_multipliers=endpoints,
                validation_source=str(Path(__file__).resolve()),
                validation_note='v1 used a fixed absolute tolerance smaller than one ULP of some actual weights; v2 checks two fp32 weight ULPs. Learner code unchanged.',
                optimizer_update_scaling=True, gradient_scaling=False))


if __name__ == '__main__':
    main()
