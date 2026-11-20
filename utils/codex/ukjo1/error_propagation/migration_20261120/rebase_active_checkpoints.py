"""Rebase only path metadata; preserve model, optimizer and RNG values exactly."""
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path
import time

REPO = Path('/home/ukjo1/research/ssd1/offline_rl/insampling')
CODE = REPO/'utils/codex/ukjo1/error_propagation'
AUDIT = CODE/'migration_20261120'
ROOT = (REPO/'logs/codex/ukjo1/error_propagation/diagnosis/finite_mean_stability_4task_20261120_v1').resolve()
BACKUP = ROOT/'migration_backup_20261120'
FIELDS = {'source_sha256', 'checkpoint_sha256', 'config_sha256'}
changes = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def backup(path, relative):
    target = BACKUP/relative
    assert not target.exists(), target
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(path, target)
    return sha(path)


def commit_json(path, value):
    tmp = path.with_suffix('.migration.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def semantic_digest(value):
    retained = {k:v for k,v in value.items() if k not in FIELDS}
    return hashlib.sha256(pickle.dumps(retained, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def commit_model(path, value, original_semantics, before):
    assert semantic_digest(value) == original_semantics
    tmp = path.with_suffix('.migration.tmp')
    with tmp.open('wb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    with tmp.open('rb') as f:
        restored = pickle.load(f)
    assert semantic_digest(restored) == original_semantics
    tmp.replace(path)
    changes.append(dict(path=str(path), before_sha256=before, after_sha256=sha(path),
                        non_path_payload_sha256=original_semantics, exactly_preserved=True))


def main():
    BACKUP.mkdir(exist_ok=False)
    sources = json.loads((AUDIT/'current_trainer_sources.json').read_text())
    plan = json.loads((ROOT/'registration_v2.json').read_text())
    resumes = []
    for job in plan['jobs']:
        run = Path(job['output']).resolve()
        config_path = run/'config.json'
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        assert {str(Path(k).resolve()):v for k,v in config['source_sha256'].items()} == sources
        before_config = backup(config_path, Path(job['id'])/'config.json')
        config['source_sha256'] = sources
        config['arguments']['output'] = str(run)
        commit_json(config_path, config)
        changes.append(dict(path=str(config_path), before_sha256=before_config, after_sha256=sha(config_path)))
        checkpoints = sorted(run.glob('checkpoint_*.pkl'))
        if (run/'latest.pkl').exists():
            checkpoints.append(run/'latest.pkl')
        for path in checkpoints:
            before = backup(path, Path(job['id'])/path.name)
            with path.open('rb') as f:
                saved = pickle.load(f)
            assert saved['identity'] == config['identity']
            assert {str(Path(k).resolve()):v for k,v in saved['source_sha256'].items()} == sources
            semantics = semantic_digest(saved)
            saved['source_sha256'] = sources
            if path.name == 'latest.pkl':
                resumes.append(dict(id=job['id'], step=saved['step'], checkpoint=str(path),
                    state_optimizer_rng_preserved=True, semantic_sha256=semantics))
            commit_model(path, saved, semantics, before)
        for path in sorted(run.glob('actor_*.pkl')):
            before = backup(path, Path(job['id'])/path.name)
            with path.open('rb') as f:
                artifact = pickle.load(f)
            assert artifact['identity'] == config['identity']
            semantics = semantic_digest(artifact)
            artifact['source_sha256'] = sources
            artifact['config_sha256'] = sha(config_path)
            artifact['checkpoint_sha256'] = sha(run/f"checkpoint_{artifact['step']}.pkl")
            commit_model(path, artifact, semantics, before)
        for path in sorted(Path(job['evaluation_directory']).glob('*.json')):
            result = json.loads(path.read_text())
            if 'checkpoint_sha256' not in result:
                continue
            before = backup(path, Path(job['id'])/'evaluations'/path.name)
            original_scores = copy.deepcopy({k:v for k,v in result.items() if k not in FIELDS})
            result['config_sha256'] = sha(config_path)
            result['checkpoint_sha256'] = sha(run/f"checkpoint_{result['step']}.pkl")
            result['actor_artifact_sha256'] = sha(run/f"actor_{result['step']}.pkl")
            for k in ('episode_successes', 'episode_returns', 'success_rate', 'n_episodes'):
                if k in original_scores:
                    assert original_scores[k] == result[k]
            commit_json(path, result)
            changes.append(dict(path=str(path), before_sha256=before, after_sha256=sha(path), scores_unchanged=True))
    result = dict(time=time.time(), status='PASS', training_runs=len(resumes), resume_checkpoints=resumes,
                  changes=changes, original_artifacts=str(BACKUP),
                  statement='Only source/output path metadata and dependent integrity hashes changed. All numerical payloads preserved exactly.')
    (AUDIT/'checkpoint_relocation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', runs=len(resumes), artifacts=len(changes),
                         resume_steps={r['id']:r['step'] for r in resumes})), flush=True)


if __name__ == '__main__':
    main()
