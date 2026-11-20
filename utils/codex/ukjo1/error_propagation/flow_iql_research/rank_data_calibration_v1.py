"""Observed-action ranks against BC proposals on independent train/val rows.

Uniform augmented ranks are an ideal conditional-generator calibration reference,
not an oracle test identifying critic error or a formal independence test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import rank_flow_core_v1 as core
from utils.flow_iql_research import data_rank_flow_core_v1 as weighting
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=1024)
    args = parser.parse_args(); assert not args.output.exists()
    config = json.loads((args.run/'config.json').read_text()); identity = config['identity']
    assert identity['kind'] == 'rank_flow'
    for source, digest in config['source_sha256'].items():
        assert sha(source) == digest, source
    checkpoint = args.run/f'checkpoint_{args.step}.pkl'
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['step'] == int(saved['state']['q_updates']) == args.step
    assert json.loads(json.dumps(saved['identity'])) == identity
    assert saved['source_sha256'] == config['source_sha256']
    assert int(saved['state']['v_updates']) == int(saved['state']['v_opt']['0']['count']) == 2*args.step
    assert int(saved['state']['agent']['network']['opt_state']['0']['count']) == args.step
    cfg = dict(identity['config'])
    for field in ('actor_hidden_dims','value_hidden_dims'):
        cfg[field] = tuple(cfg[field])
    _, training, validation = ogbench.make_env_and_datasets(identity['environment'])
    template = core.create(identity['seed'], jnp.zeros_like(training['observations'][:1]),
                           jnp.zeros_like(training['actions'][:1]), cfg)
    state = flax.serialization.from_state_dict(template, saved['state']); agent = state.agent

    @jax.jit
    def inspect(observations, actions):
        repeated = jnp.broadcast_to(observations, (9,*observations.shape))
        noise = jax.random.normal(jax.random.PRNGKey(81921),(9,*actions.shape))
        proposals = agent.compute_flow_actions(repeated,noise)
        net = agent.network
        proposal_q = net.select('critic')(repeated,actions=proposals).min(0)
        data_q = net.select('critic')(observations,actions=actions).min(0)
        ties = jax.random.uniform(jax.random.PRNGKey(81922),(10,*actions.shape[:-1]))
        weights, counts = weighting.data_rank_weights(data_q,proposal_q,ties[0],ties[1:])
        p0 = jax.random.normal(jax.random.PRNGKey(81923),actions.shape)
        t = jax.random.uniform(jax.random.PRNGKey(81924),(*actions.shape[:-1],1))
        xt = (1-t)*p0+t*actions
        velocity = actions-p0
        policy = net.select('actor_policy_flow')(observations,xt,t)
        bc = net.select('actor_flow')(observations,xt,t)
        return dict(weights=weights,augmented_rank_counts=counts,data_Q=data_q,
            comparator_Q=proposal_q,comparator_action_std=jnp.std(proposals,axis=0),
            exact_Q_tie_fraction=jnp.mean((data_q == proposal_q).astype(jnp.float32)),
            policy_data_FM_per_example=jnp.mean((policy-velocity)**2,axis=-1),
            BC_data_FM_per_example=jnp.mean((bc-velocity)**2,axis=-1))

    results = {}; arrays = {}
    for name,data in [('independent_train',training),('validation',validation)]:
        indices = np.random.RandomState(81920).randint(len(data['actions']),size=args.queries)
        observations = np.asarray(data['observations'][indices],np.float32)
        actions = np.clip(np.asarray(data['actions'][indices],np.float32),-1+1e-5,1-1e-5)
        found = {key:np.asarray(value)for key,value in jax.device_get(inspect(jnp.asarray(observations),jnp.asarray(actions))).items()}
        assert all(np.isfinite(value).all()for value in found.values())
        w = found['weights'].astype(np.float64)
        histogram = np.bincount(found['augmented_rank_counts'],minlength=10)/args.queries
        result = dict(queries=args.queries,rank_count_histogram=histogram.tolist(),
            rank_histogram_L1_distance_from_ideal_uniform=float(np.abs(histogram-.1).sum()),
            weight_mean=float(w.mean()),weight_ESS_fraction=float(w.mean()**2/max(np.mean(w*w),1e-20)),
            zero_weight_fraction=float(np.mean(w==0)),one_weight_fraction=float(np.mean(w==1)),nine_weight_fraction=float(np.mean(w==9)),
            zero_total_mass=bool(w.sum()==0),
            data_Q_minus_same_state_comparator_mean=float(np.mean(found['data_Q']-found['comparator_Q'].mean(0))),
            comparator_Q_std_mean=float(found['comparator_Q'].std(0).mean()),
            comparator_action_std_mean=float(found['comparator_action_std'].mean()),
            exact_Q_tie_fraction=float(found['exact_Q_tie_fraction']),
            index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
            observations_actions_sha256=hashlib.sha256(observations.tobytes()+actions.tobytes()).hexdigest())
        for prefix,key in [('policy','policy_data_FM_per_example'),('BC','BC_data_FM_per_example')]:
            result[prefix+'_unweighted_data_FM_loss']=float(found[key].mean())
            result[prefix+'_raw_rank_weighted_data_FM_loss']=float(np.mean(w*found[key]))
            result[prefix+'_weight_normalized_auxiliary_FM_loss']=float(np.sum(w*found[key])/w.sum())if w.sum()>0 else None
        results[name]=result
        arrays.update({name+'_'+key:value for key,value in found.items()})
        arrays[name+'_indices']=indices
    result = dict(time=time.time(),run=args.run.name,step=args.step,environment=identity['environment'],
        profile=identity['profile'],actual_policy_objective=cfg['policy_objective'],sets=results,
        checkpoint_sha256=sha(checkpoint),config_sha256=sha(args.run/'config.json'),
        diagnostic_source_sha256=sha(__file__),rank_weight_source_sha256=sha(weighting.__file__),
        ideal_reference=dict(rank_probabilities=[.1]*10,weight_mean=1.,weight_ESS_fraction=1/8.2),
        limits=[
            'Uniform ranks require independent conditional actions from the same behavior law and a fixed scoring function; neither is certified here.',
            'Training queries were not reserved from model fitting. Their comparison with validation is descriptive, not an independent critic-error test.',
            'A shifted rank histogram can reflect BC misspecification, train-validation shift or critic overfitting; it does not identify one cause.',
            'All Q contrasts are learned-model quantities. These frozen dataset probes do not evaluate environment returns.',
            'For selected-label rank flow this data-weighted FM loss is a counterfactual reference, not its training objective.',
            'Weight-normalized FM is auxiliary reporting only; the data-rank runner trains the raw batch mean without normalization.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(args.output)
    print(json.dumps(dict(run=args.run.name,step=args.step,sets=results)),flush=True)


if __name__ == '__main__':
    main()
