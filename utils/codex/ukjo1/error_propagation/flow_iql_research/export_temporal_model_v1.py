"""Export an existing trained IFQL model and its causal critic history.

The bundle is for deployment, not a resumable or newly trained checkpoint.
Original model files and primary evaluations are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    parser.add_argument('--step', type=int, default=1000000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    conf = json.loads((args.run/'config.json').read_text()); ident = conf['identity']
    assert ident['kind'] == 'ifql' and ident['profile'] == 'bounded_v2'
    assert ident['config']['discount'] == .999 and ident['config']['expectile'] == .9
    assert args.step in (800000, 900000, 1000000)
    root = Path.cwd().resolve()
    training_sources = {}
    for source, digest in conf['source_sha256'].items():
        assert sha(source) == digest, source
        path = Path(source).resolve().relative_to(root)
        training_sources[str(path)] = digest
    history, evidence, current = [], [], None
    for step in (args.step-200000, args.step-100000, args.step):
        checkpoint = args.run/f'checkpoint_{step}.pkl'
        if not checkpoint.exists():
            checkpoint = args.run.parent.parent/'analysis/checkpoint_archive'/args.run.name/f'checkpoint_{step}.pkl'
        with checkpoint.open('rb') as file:
            saved = pickle.load(file)
        assert json.loads(json.dumps(saved['identity'])) == ident
        assert saved['source_sha256'] == conf['source_sha256']
        assert saved['step'] == int(saved['state']['q_updates']) == step
        assert int(saved['state']['v_updates']) == 2*step
        history.append(dict(step=step, params=saved['state']['agent']['network']['params']['modules_critic']))
        evidence.append(dict(step=step, checkpoint=str(checkpoint), sha256=sha(checkpoint)))
        current = saved
    cfg = dict(ident['config'])
    cfg.update(num_samples=9, critic_history_size=3, critic_snapshot_stride=100000)
    sources = ['utils/flow_iql_research/temporal_core_v1.py',
               'utils/flow_iql_research/temporal_selection_v1.py',
               'utils/flow_iql_research/selection_v2.py',
               'utils/flow_iql_research/evaluate_temporal_model_v1.py',
               'utils/flow_iql_research/train_temporal_policy_v1.py']
    bundle = dict(kind='temporal_IFQL_deployment_bundle_v1', created_at=time.time(),
        resumable_training_checkpoint=False, step=args.step, state=current['state'],
        critic_history=history, training_identity=ident, inference_config=cfg,
        source_checkpoints=evidence, source_config_sha256=sha(args.run/'config.json'),
        training_source_sha256=training_sources,
        deployment_source_sha256={p: sha(p) for p in sources}, exporter_source_sha256=sha(__file__),
        recipe=dict(samples=9, critic_members=3, critic_combination='mean_of_per_member_minimum_twin_Q',
                    snapshot_stride=100000, current_behavior_actor=True),
        notes=['This exports actual retained parameters; no training, optimizer updates or checkpoint selection occurred.',
               'Reported experiments used the original learning runner and separately evaluated K3 deployment.',
               'The integrated runner has the same learning equations and short CPU equivalence validation.'])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    temporary = args.output.with_suffix('.tmp')
    with temporary.open('wb') as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output),
                         step=args.step, member_steps=[h['step'] for h in history])), flush=True)


if __name__ == '__main__':
    main()
