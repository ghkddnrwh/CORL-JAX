"""CPU50 rollout of the explicit single routed-location actor artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from utils.flow_iql_research import train_mean_empirical_backup_v1 as evaluation
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
from utils.flow_iql_research import export_mixture_location_actor_v1 as exporter
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as portable
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as lazy
import jax
import ogbench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--actor-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path);a=p.parse_args();assert not a.output.exists()
    assert a.step in (800000,900000,1000000) and all(d.platform=='cpu' for d in jax.devices())
    conf=json.loads((a.run/'config.json').read_text());identity=conf['identity'];assert identity['recipe']=='mdn9_bc'
    native=a.run/f'actor_{a.step}.pkl';native_digest=sha(native)
    if not a.actor_output.exists():exporter.export(native,a.actor_output)
    loaded=portable.load_actor(a.actor_output);digest=sha(a.actor_output)
    state=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=loaded.params)))
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True)
    reset=prepare_seeded_action_space(env,42)
    row=evaluation.evaluate(state,env,loaded.policy,a.step,50);env.close()
    row.update(time=time.time(),protocol='common_routed_mixture_location_last3_CPU50_v1',
        run=a.run.name,environment=identity['environment'],training_seed=identity['seed'],
        native_training_recipe='mdn9_bc',deployment_candidate='mdn9_component_location_v1',
        deployment_rule='sampled_component_location',actor_path=str(a.actor_output.resolve()),
        actor_artifact_sha256=digest,native_actor_artifact_sha256=native_digest,
        config_sha256=sha(a.run/'config.json'),reset_metadata=reset,
        learned_model_trees_loaded=['actor_params'],deployment_metadata=loaded.metadata,
        optimizer_updates_performed=0,requires_critic=False,requires_BC_flow=False,
        candidate_action_bank=False,computed_action_heads=1,no_production_results_replaced=True,
        isolated_latency_claim=False)
    if a.reference:
        expected=json.loads(a.reference.read_text())
        assert expected['actor_artifact_sha256']==native_digest
        exact={key:row[key]==expected[key] for key in ('episode_successes','episode_rewards','episode_lengths')}
        assert all(exact.values())
        row.update(reference=str(a.reference),reference_sha256=sha(a.reference),reference_episode_arrays_exact=exact)
    assert sha(native)==native_digest and sha(a.actor_output)==digest
    row['source_sha256']={str(Path(path).resolve()):sha(path) for path in
        [__file__,exporter.__file__,portable.__file__,lazy.__file__,evaluation.__file__]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2,allow_nan=False));temp.replace(a.output)
    print(json.dumps(dict(run=a.run.name,step=a.step,success=row['success_rate'],
                         portable_actor_only=True,reference_exact=bool(a.reference))))


if __name__=='__main__':
    main()
