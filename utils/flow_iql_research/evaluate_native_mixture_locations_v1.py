"""Read-only posthoc common-decoder comparison, distinct from primary results."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as lazy
from utils.flow_iql_research import standalone_mixture_actor_v1 as original
import jax
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--decoder',choices=(*lazy.DECODERS,'original_sample'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()and a.step in (800000,900000,1000000)
    assert all(d.platform=='cpu'for d in jax.devices())
    conf=json.loads((a.run/'config.json').read_text());identity=conf['identity'];assert identity['recipe']=='mdn9_bc'
    actor=a.run/f'actor_{a.step}.pkl';digest=sha(actor)
    loaded=original.load_actor(actor)if a.decoder=='original_sample'else lazy.load_actor(actor,a.decoder)
    namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=loaded.params)))
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
    result=evaluate(namespace,env,loaded.policy,a.step,50);env.close()
    result.update(time=time.time(),protocol='posthoc_native_mixture_common_decoder_CPU50_v1',decoder=a.decoder,
        run=a.run.name,recipe='mdn9_bc',environment=identity['environment'],training_seed=identity['seed'],
        actor_artifact_sha256=digest,config_sha256=sha(a.run/'config.json'),
        reset_metadata=reset,learned_model_trees_loaded=['actor_params'],optimizer_updates_performed=0,
        no_training_or_primary_evaluation_changes=True,counts_toward_native_primary=False,
        candidate_bank=False,critic_loaded=False,Flow_loaded=False,action_heads_evaluated=1,
        scope='Posthoc same-native-model global-decoder study on completed tasks, not full-nine/fresh/isolatedspeed confirmation.')
    if a.decoder=='original_sample':
        reference=a.run.parent.parent/'analysis/evaluations_v1'/f'{a.run.name}_{a.step}_primary_n50.json'
        expected=json.loads(reference.read_text());assert expected['actor_artifact_sha256']==digest
        exact={k:result[k]==expected[k]for k in ['episode_successes','episode_rewards','episode_lengths']}
        assert all(exact.values());result['primary_reference_episode_arrays_exact']=exact
    assert sha(actor)==digest
    result['source_sha256']={str(Path(path).resolve()):sha(path)for path in [__file__,lazy.__file__]}
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(a.output)
    print(json.dumps(dict(run=a.run.name,step=a.step,decoder=a.decoder,success=result['success_rate'])))


if __name__=='__main__':main()
