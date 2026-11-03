"""Replay CPU50 from a compact bundle; exact against its fixed1M primary row."""
import argparse
import json
from pathlib import Path
import time
from unittest import mock
from utils.flow_iql_research.mean_backup_bundle_v1 import load,sha
from utils.flow_iql_research import train_mean_empirical_backup_v1 as training
from utils.flow_iql_research import tau_matched_extraction_v1 as helper
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import ogbench


def main():
    parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path)
    parser.add_argument('--reference-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();original=Path.open;blocked=[];updates=[]
    def guarded(path,*positional,**keywords):
        if '/runs/'in str(path)or'/checkpoint_archive/'in str(path):blocked.append(str(path));raise AssertionError('Inference accessed source learner directory')
        return original(path,*positional,**keywords)
    def no_update(*args,**kwargs):updates.append(True);raise AssertionError('Inference attempted optimization')
    with mock.patch.object(Path,'open',guarded),mock.patch.object(TrainState,'apply_loss_fn',no_update):
        bundle,state,policy=load(args.bundle)
        env=ogbench.make_env_and_datasets(bundle['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
        result=training.evaluate(state,env,policy,1000000,50);env.close()
    assert not blocked and not updates
    run=args.reference_run;conf=json.loads((run/'config.json').read_text())
    assert sha(run/'config.json')==bundle['provenance']['config_sha256']and conf['identity']==bundle['identity']
    reference=helper.primary_record(run,1000000);assert reference is not None
    exact={field:reference[field]==result[field]for field in('episode_successes','episode_rewards','episode_lengths')}
    assert all(exact.values()),exact
    result.update(kind='compact_mean_backup_bundle_CPU50_equivalence',time=time.time(),environment=bundle['environment'],training_seed=bundle['training_seed'],
        bundle=str(args.bundle),bundle_sha256=sha(args.bundle),source_checkpoint_sha256=bundle['provenance']['checkpoint_sha256'],
        reference_run=run.name,reference_evaluations_sha256=sha(run/'evaluations.jsonl'),reference_episode_arrays_exact=exact,
        no_source_directory_reads_during_load_or_rollout=True,no_optimizer_updates=True,only_actor_and_online_critic_params=True,
        primary_evaluations_modified=False,new_training_replication=False,source_sha256=sha(__file__),reset_metadata=reset,
        evaluation_helper_source_sha256=sha(training.__file__),goal_complete=False,
        note='Replay of one fixed1M stored model, not a new statistical observation or the800K/900K/1M metric.')
    helper.atomic(args.output,result);print(json.dumps(dict(status='exact',success=result['success_rate'],environment=bundle['environment'],episode_arrays=exact)),flush=True)


if __name__=='__main__':main()
