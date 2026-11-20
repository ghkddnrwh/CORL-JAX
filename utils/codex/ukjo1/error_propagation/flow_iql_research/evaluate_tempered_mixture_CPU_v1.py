"""Read-only common temperature ablation, separate from registered primary."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from utils.flow_iql_research import train_mean_empirical_backup_v1 as evaluation
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
from utils.flow_iql_research import lazy_tempered_mixture_actor_v1 as tempered
import jax
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--temperature',type=float,choices=[.25,.5,1.],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    assert a.step in [800000,900000,1000000]and all(d.platform=='cpu'for d in jax.devices())
    conf=json.loads((a.run/'config.json').read_text());identity=conf['identity']
    assert identity['recipe']=='mdn9_bc'and identity['seed']==0
    path=a.run/f'actor_{a.step}.pkl';digest=sha(path);loaded=tempered.load_actor(path,a.temperature)
    state=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=loaded.params)))
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
    row=evaluation.evaluate(state,env,loaded.policy,a.step,50);env.close()
    row.update(time=time.time(),protocol='posthoc_native_mixture_global_temperature_CPU50_v1',
        run=a.run.name,native_training_recipe='mdn9_bc',environment=identity['environment'],training_seed=0,
        deployment_rule='sample_component_then_tempered_Gaussian',temperature=a.temperature,
        actor_artifact_sha256=digest,config_sha256=sha(a.run/'config.json'),reset_metadata=reset,
        learned_model_trees_loaded=['actor_params'],optimizer_updates_performed=0,
        computed_action_heads=1,realized_actions=1,candidate_action_bank=False,critic_loaded=False,Flow_loaded=False,
        deployment_metadata=loaded.metadata,counts_toward_production_primary=False,production_state_or_scores_modified=False,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,tempered.__file__,evaluation.__file__]},
        scope='Global seed0 temperature ablation; all800K900K1MCPU50. No per-environment selection, fresh-seed claim, or isolated speed claim. Temperature1 is a selected-head numerical control distinct from the original full-output MDN client.')
    assert sha(path)==digest
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(row,indent=2,allow_nan=False));tmp.replace(a.output)
    print(json.dumps(dict(run=a.run.name,step=a.step,temperature=a.temperature,success=row['success_rate'])))


if __name__=='__main__':main()
