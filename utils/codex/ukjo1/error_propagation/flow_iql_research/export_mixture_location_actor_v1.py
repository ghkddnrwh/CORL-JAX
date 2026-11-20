"""Export an explicit deployment identity without relabelling the learner."""
import argparse
import hashlib
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as portable
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as lazy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(native_path,output):
    assert not output.exists()
    digest=sha(native_path)
    with native_path.open('rb') as stream:
        native=pickle.load(stream)
    assert native['schema']=='single_actor_deployment_v1'
    assert native['identity']['recipe']=='mdn9_bc'
    assert native['actor_updates']==native['Q_updates']==native['step']
    original=native['metadata']
    assert original['architecture']=='clipped_diagonal_mixture_MLP'
    assert original['flow_integration_steps']==original['generated_candidate_actions']==0
    metadata=dict(original,architecture='routed_component_location_MLP',
        deployment_decoder='sampled_component_location',computed_action_heads=1,
        Dense_dot_operations=6,within_component_action_noise=False,
        output='sample_one_component_by_learned_probabilities_then_clip_its_location',
        parameter_layout='Native MDN weights retained exactly; unused scale-head columns are not evaluated.')
    result=dict(schema='routed_mixture_location_actor_v1',time=time.time(),
        actor_params=native['actor_params'],metadata=metadata,
        training_identity=native['identity'],training_source_sha256=native['source_sha256'],
        training_step=native['step'],actor_updates=native['actor_updates'],Q_updates=native['Q_updates'],
        native_training_metadata=original,no_new_training_updates=True,
        native_actor_path=str(native_path.resolve()),native_actor_sha256=digest,
        training_checkpoint_sha256=native['checkpoint_sha256'],
        deployment_source_sha256={str(Path(p).resolve()):sha(p) for p in [__file__,portable.__file__,lazy.__file__]},
        scope='Same native jointly trained neural parameters with one globally fixed deployment decoder. Not a new SGD run or evidence of full-nine/fresh-seed/isolated-latency success.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        pickle.dump(result,stream,protocol=pickle.HIGHEST_PROTOCOL)
    assert sha(native_path)==digest
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-actor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.native_actor,a.output)
    print(str(a.output))


if __name__=='__main__':
    main()
