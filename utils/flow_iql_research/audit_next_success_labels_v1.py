"""Check exact next-success certification on all nine train/validation datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research import policy_core
from utils.flow_iql_research import next_success_labels_v1 as helper
import numpy as np
import ogbench
import ogbench.utils
import ogbench.relabel_utils


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    # Endpoint states can share observation vectors with another trajectory.
    # The explicit trajectory-cut flag must still prevent a false next label.
    obs=np.asarray([[0.,0.],[1.,1.],[0.,0.],[1.,1.]])
    synthetic=dict(observations=obs,next_observations=np.asarray([[1.,1.],[0.,0.],[1.,1.],[2.,2.]]),
        terminals=np.asarray([0,1,0,1]),masks=np.asarray([1,0,1,0]),rewards=np.asarray([-1,0,-1,0]))
    success,meta=helper.labels(synthetic,chunk=1);assert success.tolist()==[1,0,1,0] and meta['unknown_next_count']==2
    changed={**synthetic,'next_observations':synthetic['next_observations'].copy()};changed['next_observations'][0,0]=123.
    try:helper.labels(changed)
    except AssertionError:pass
    else:raise AssertionError('Unsafe adjacency accepted')
    rows=[]
    for environment in policy_core.ENVIRONMENTS:
        env,train,validation=ogbench.make_env_and_datasets(environment)
        for split,data in [('train',train),('validation',validation)]:
            success,metadata=helper.labels(data)
            rows.append(dict(environment=environment,split=split,**metadata))
            print(json.dumps(rows[-1]),flush=True)
        env.close()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result=dict(time=time.time(),kind='certified_next_success_adjacency_audit',rows=rows,
        unit_cases=dict(cross_trajectory_repeated_observations_rejected=True,noncut_mismatch_rejected=True,endpoint_unknown=True),
        source_sha256={str(Path(m.__file__)):sha(m.__file__)for m in (helper,ogbench.utils,ogbench.relabel_utils)},
        audit_source_sha256=sha(__file__),
        interpretation='Anexactlyadjacentretainedsuccessstatehasr0/mask0andthereforeknowncontinuationV0intheimplementedstate-basedmaskedtask. Thebootstrapoverrideaddsnotunablecoefficient.',
        limits=['Onlycertifiednoncutnextstatesarecovered;trajectoryendpointlabelsremainunknown.',
            'Thelabelisfornextstateandmustnotreplacecurrentmasksorapplytocross-trajectoryrows.',
            'Reducingknownboundarytargeterroraloneisnotapolicyreturnguarantee.'],goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(args.output)


if __name__=='__main__':main()
