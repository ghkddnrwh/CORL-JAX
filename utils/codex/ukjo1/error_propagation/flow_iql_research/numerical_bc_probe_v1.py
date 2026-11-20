"""Same-input BC arithmetic under different joint Q/V update graphs.

This is a short numerical diagnostic, never an independent training seed or a
performance run. It copies one donor's parameters into fresh zero Adam states;
no donor optimizer is resumed, no source checkpoint is changed, and no episode
is evaluated. Scheduler output intentionally has no benchmark config.json.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core as core
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import optax


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(tree):
    return [np.asarray(value).copy() for value in jax.device_get(jax.tree_util.tree_leaves(tree))]


def delta(first, second):
    assert len(first) == len(second)
    return max(float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) for a,b in zip(first, second))


def digest_arrays(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env', choices=core.ENVIRONMENTS, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=('gpu','cpu'), default='gpu')
    parser.add_argument('--width', type=int)
    parser.add_argument('--steps', type=int, default=32)
    args = parser.parse_args(); assert args.steps >= 2
    args.output.mkdir(exist_ok=True)
    assert not (args.output/'completed.json').exists()
    if args.device == 'gpu':
        assert os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] == '0.1'
    cfg = core.config_for(args.env, .999, .9, args.width)
    cfg.update(agent_name='ifql', policy_type='ifql', num_samples=9)
    donor = None
    if args.checkpoint:
        config_path = args.checkpoint.parent/'config.json'
        donor_config = json.loads(config_path.read_text())
        for source, digest in donor_config['source_sha256'].items():
            assert sha(source) == digest, source
        with args.checkpoint.open('rb') as stream:
            saved = pickle.load(stream)
        identity = donor_config['identity']
        assert identity['environment'] == args.env and identity['kind'] == 'ifql'
        assert json.loads(json.dumps(saved['identity'])) == identity
        assert saved['source_sha256'] == donor_config['source_sha256']
        cfg = dict(identity['config'])
        assert args.width is None
        donor = dict(checkpoint=str(args.checkpoint), sha256=sha(args.checkpoint), parameter_training_step=saved['step'])
    for key in ('actor_hidden_dims','value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    _, raw, _ = ogbench.make_env_and_datasets(args.env)
    fields = ('observations','actions','next_observations','rewards','masks')
    data = {key:np.asarray(raw[key], np.float32) for key in fields}
    data['actions'] = np.clip(data['actions'], -1+1e-5, 1-1e-5)
    indices = np.random.RandomState(63111).randint(len(data['actions']), size=(args.steps,cfg['batch_size']))
    device = jax.devices(args.device)[0]
    initial = core.create(31, jax.device_put(data['observations'][:1],device),
                           jax.device_put(data['actions'][:1],device), cfg, 'ifql')
    if donor:
        params = jax.device_put(saved['state']['agent']['network']['params'], device)
        network = initial.agent.network.replace(params=params, opt_state=initial.agent.network.tx.init(params))
        initial = initial.replace(agent=initial.agent.replace(network=network),
                                   v_opt=optax.adam(cfg['lr']).init(params['modules_value']))
        del saved
    batches = {key:jax.device_put(value[indices],device) for key,value in data.items()}
    first = jax.tree_util.tree_map(lambda value:value[0],batches)
    remaining = jax.tree_util.tree_map(lambda value:value[1:],batches)
    bounds = (-1000.,0.)
    profiles = dict(native=core.build_update('official',bounds,'ifql'),
                    V2=core.build_update('bounded_v2',bounds,'ifql'))

    def bc_only(state,batch):
        agent = state.agent
        next_rng,rng = jax.random.split(agent.rng)
        _,actor_rng = jax.random.split(rng)
        network,logs = agent.network.apply_loss_fn(lambda params:agent.actor_loss(batch,params,actor_rng))
        return state.replace(agent=agent.replace(network=network,rng=next_rng)), logs

    profiles['BC_only'] = bc_only
    summaries, arrays, hlo = {}, {}, {}
    for precision in ('default','float32'):
        context = contextlib.nullcontext() if precision == 'default' else jax.default_matmul_precision('float32')
        with context:
            results = {}
            for name,update in profiles.items():
                start = time.monotonic(); function = jax.jit(update)
                one, _ = function(initial,first)
                repeat, _ = function(initial,first)
                one_params = flatten(one.agent.network.params['modules_actor_flow'])
                one_moments = flatten(one.agent.network.opt_state[0].mu['modules_actor_flow'])
                repeat_params = flatten(repeat.agent.network.params['modules_actor_flow'])
                repeat_moments = flatten(repeat.agent.network.opt_state[0].mu['modules_actor_flow'])
                final, _ = core.make_scan(update)(one,remaining)
                final_params = flatten(final.agent.network.params['modules_actor_flow'])
                assert all(np.isfinite(array).all() for array in one_params+one_moments+final_params)
                results[name] = dict(one_params=one_params,one_moments=one_moments,final_params=final_params,
                    repeat_parameter_max_abs=delta(one_params,repeat_params),
                    repeat_moment_max_abs=delta(one_moments,repeat_moments),
                    final_BC_sha256=digest_arrays(final_params),seconds=time.monotonic()-start)
                try:
                    text = function.lower(initial,first).compiler_ir(dialect='hlo').as_hlo_text()
                    hlo[precision+'_'+name] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(),
                        dot_precision_lines=[line.strip() for line in text.splitlines() if 'operand_precision' in line][:8])
                except (AttributeError,TypeError) as error:
                    hlo[precision+'_'+name] = dict(unavailable=repr(error))
                for index,array in enumerate(final_params):
                    arrays[f'{precision}_{name}_{index}'] = array
                del one,repeat,final
            pairs = {}
            for first_name,second_name in [('native','V2'),('native','BC_only'),('V2','BC_only')]:
                first_result,second_result = results[first_name],results[second_name]
                pairs[first_name+'_vs_'+second_name] = dict(
                    first_BC_parameter_max_abs=delta(first_result['one_params'],second_result['one_params']),
                    first_BC_Adam_mu_max_abs=delta(first_result['one_moments'],second_result['one_moments']),
                    final_BC_parameter_max_abs=delta(first_result['final_params'],second_result['final_params']))
            summaries[precision] = dict(comparisons=pairs,per_graph={name:{key:value for key,value in result.items() if not isinstance(value,list)} for name,result in results.items()})
            del results
    record = dict(time=time.time(),kind='numerical_BC_arithmetic_probe_not_a_training_run',
        device=str(device),jax_version=jax.__version__,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        memory_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),diagnostic_updates_per_graph=args.steps,
        donor=donor,optimizer_initialization='AllAdamstatesfreshzero;donorparametersonly,notdonortrainingresume.',
        batch_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),results=summaries,HLO=hlo,
        source_sha256={str(Path(__file__)):sha(__file__),str(Path(core.__file__)):sha(core.__file__)},
        limits=['Noepisodeorbenchmarkscoreisgenerated;thesearearithmeticchecks,notnewtrainingreplicates.',
                'Ashortprobeononecommoninputcannotprovewhatcausedhistoricalmillion-steptrajectorydifferences.',
                'Requestedfloat32matmulprecisionandHLOsignaturesdonotbythemselvesidentifytheactualCUDAkernelorcertifyanumericalbug.',
                'Repeatedsame-graphandcross-graphdifferencesmustbereportedseparately.'])
    np.savez_compressed(args.output/'BC_final_arrays.npz',**arrays)
    (args.output/'probe_result.json').write_text(json.dumps(record,indent=2))
    (args.output/'completed.json').write_text(json.dumps(dict(step=0,kind=record['kind'],diagnostic_updates=args.steps),indent=2))
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
