"""Audit known zero-bootstrap labels and frozen Q/V errors at success states."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import ogbench.relabel_utils
import ogbench.locomaze.maze


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''): digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.root.resolve()
    environments = policy_core.ENVIRONMENTS
    counts, rows, sources = [], [], []
    for environment in environments:
        env, train, validation = ogbench.make_env_and_datasets(environment)
        for split, data in [('training', train), ('validation', validation)]:
            masks, rewards = np.asarray(data['masks']), np.asarray(data['rewards'])
            assert np.isin(masks, [0, 1]).all()
            terminal = np.flatnonzero(masks == 0)
            assert np.all(rewards[terminal] == 0)
            counts.append(dict(environment=environment, split=split, transitions=len(masks),
                               zero_bootstrap_count=len(terminal), zero_bootstrap_fraction=float(len(terminal)/len(masks)),
                               expected_in_uniform256_batch=float(256*len(terminal)/len(masks)),
                               zero_bootstrap_rewards_all_zero=True))
        if 'humanoidmaze' not in environment:
            env.close()
            continue
        short = 'hmmed' if 'medium' in environment else 'hmlarge'
        runs = dict(V2=root/'runs'/f'p1_{short}_ifql_c0_bounded_v2_g999_t90_s0',
                    finite=root/'runs'/f'eb1_{short}_bounded_v2_empirical_backup_g999_t90_s0')
        for source, run in runs.items():
            conf = json.loads((run/'config.json').read_text())
            ident, config = conf['identity'], dict(conf['identity']['config'])
            assert ident['environment'] == environment
            for name, digest in conf['source_sha256'].items(): assert sha(name) == digest
            for name in ('actor_hidden_dims', 'value_hidden_dims'): config[name] = tuple(config[name])
            template = policy_core.create(ident['seed'], jnp.zeros_like(jnp.asarray(train['observations'][:1])),
                                         jnp.zeros_like(jnp.asarray(train['actions'][:1])), config, 'ifql')
            cp = run/'checkpoint_500000.pkl'
            if not cp.exists(): cp = root/'analysis/checkpoint_archive'/run.name/'checkpoint_500000.pkl'
            with cp.open('rb') as stream: saved = pickle.load(stream)
            assert saved['step'] == 500000
            assert json.loads(json.dumps(saved['identity'])) == ident
            assert saved['source_sha256'] == conf['source_sha256']
            state = flax.serialization.from_state_dict(template, saved['state'])
            assert int(state.q_updates) == 500000 and int(state.v_updates) == 1000000
            sources.append(dict(run=run.name, checkpoint=str(cp), sha256=sha(cp), config_sha256=sha(run/'config.json')))
            @jax.jit
            def predict(obs, actions):
                net = state.agent.network
                return (net.select('critic')(obs, actions=actions),
                        net.select('target_critic')(obs, actions=actions).min(0),
                        net.select('value')(obs))
            for split, data in [('training', train), ('validation', validation)]:
                indices = np.flatnonzero(np.asarray(data['masks']) == 0)
                full_count = len(indices)
                if not full_count:
                    rows.append(dict(environment=environment, source=source, split=split, terminal_count=0))
                    continue
                if len(indices) > 4096:
                    indices = np.random.RandomState(89131).choice(indices, 4096, replace=False)
                outputs = []
                for start in range(0, len(indices), 1024):
                    chosen = indices[start:start+1024]
                    obs = jnp.asarray(data['observations'][chosen], jnp.float32)
                    actions = jnp.clip(jnp.asarray(data['actions'][chosen], jnp.float32), -1+1e-5, 1-1e-5)
                    outputs.append(jax.device_get(predict(obs, actions)))
                q = np.concatenate([out[0] for out in outputs], axis=1).astype(np.float64)
                target = np.concatenate([out[1] for out in outputs]).astype(np.float64)
                v = np.concatenate([out[2] for out in outputs]).astype(np.float64)
                assert np.isfinite(q).all() and np.isfinite(v).all()
                displacement = target-v
                score = np.where(displacement >= 0, config['expectile'], 1-config['expectile'])*displacement
                rms = lambda x:float(np.sqrt(np.mean(x*x)))
                rows.append(dict(environment=environment, source=source, split=split, terminal_count=full_count,
                    inspected_terminal_count=len(indices), inspected_all_terminal_transitions=len(indices) == full_count,
                    index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(), Q_mean=float(q.mean()), Q_RMS_to_known_zero=rms(q),
                    minQ_RMS_to_known_zero=rms(q.min(0)), V_mean=float(v.mean()), V_RMS_to_known_zero=rms(v),
                    target_minQ_minus_V_RMS=rms(displacement), expectile_score_proxy_RMS=rms(score),
                    both_Q_heads_within1_fraction=float(np.mean((np.abs(q) <= 1).all(0))),
                    V_within1_fraction=float(np.mean(np.abs(v) <= 1)), Q_abs_max=float(np.max(np.abs(q)))))
            del state, saved, template
        env.close()
    library = {str(Path(module.__file__)):sha(module.__file__) for module in (ogbench.relabel_utils, ogbench.locomaze.maze)}
    result = dict(time=time.time(), kind='known_terminal_boundary_value_audit', counts=counts, boundary_errors=rows,
                  checkpoints=sources, source_sha256=sha(__file__), library_source_sha256=library,
                  interpretation='Atzero-bootstraprows,themaskedBellmantargetisexactlyr=0regardlessofV. OGBenchsingle-taskrelabelusesstate-basedsuccess,andinstalledlocomazeusessuccess_timingpre. Q/Vzeroareknownboundaryconditionsofthemaskedtask,notoracleinteriorvalues.',
                  limits=['Fixed500Kmodelsandobserveddatasetgoalstatesonly;notreturnimprovementorall-stateaccuracy.',
                          'Thisdoesnotidentifythefirstharmfulupdateorproveboundarysupervisionwillimprovethebenchmark.',
                          'ExpectilescorestilluseslearnedtargetQ;smallscorecannotcertifycorrectboundaryV.',
                          'Traininggoalsarepartofthetrainingdataset,notanindependenttraining-seedconfirmation.',
                          'Zero-bootstrapinterpretationisfortheimplementedmaskedepisodictask;externaldatasetterminationsemanticsmustberecheckedbeforegeneralizing.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)); tmp.replace(args.output)
    print(json.dumps(dict(output=str(args.output), counts=len(counts), boundary_groups=len(rows))), flush=True)


if __name__ == '__main__':
    main()
