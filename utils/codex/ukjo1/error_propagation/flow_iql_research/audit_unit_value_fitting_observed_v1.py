"""Independently recompute original-unit diagnostics from saved query arrays."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from utils.flow_iql_research import tau_matched_extraction_v1 as helper


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();plan=json.loads(args.registration.read_text())
    helper.check_sources(plan['source_sha256']);rows=[];metadata={};metrics_checked=0
    for job in plan['jobs']:
        path=Path(job['output'])
        if not path.exists():continue
        result=json.loads(path.read_text());assert result['job']==job and result['status']=='complete'
        assert helper.digest(result['raw_arrays'])==result['raw_arrays_sha256']
        assert helper.digest(result['checkpoint'])==result['checkpoint_sha256']
        conf=json.loads((Path(plan['root'])/'runs'/job['run_id']/'config.json').read_text())
        assert helper.digest(Path(plan['root'])/'runs'/job['run_id']/'config.json')==result['config_sha256']
        env=conf['identity']['environment'];tau=conf['identity']['config']['expectile']
        if env in metadata:assert metadata[env]==result['data']
        metadata[env]=result['data'];values={}
        with np.load(result['raw_arrays'],allow_pickle=False)as arrays:
            n=len(arrays['validation_uniform_indices']);goals=len(arrays['validation_goal_indices']);assert n==8192
            assert all(np.isfinite(arrays[key]).all()for key in arrays.files)
            assert np.all(arrays['masks'][n:]==0)and np.all(arrays['rewards'][n:]==0)
            fit=arrays['target_fit_Q_cost'];q=arrays['online_Q_cost']
            for key in arrays.files:
                if not key.endswith('_V_cost'):continue
                v=arrays[key];table={}
                for stratum,indices in [('uniform_validation',slice(0,n)),('all_validation_success_boundary',slice(n,None))]:
                    if stratum=='all_validation_success_boundary'and not goals:continue
                    error=fit[indices]-v[indices];weight=np.where(error>=0,tau,1-tau)
                    numbers=dict(fitted_expectile_MSE=float(np.average(weight*error*error)),
                        sample_score_mean=float(np.average(weight*error)),sample_score_RMS=float(np.linalg.norm(weight*error)/np.sqrt(error.size)))
                    if stratum=='all_validation_success_boundary':
                        numbers.update(V_boundary_RMSE=float(np.linalg.norm(v[indices])/np.sqrt(goals)),
                            target_Q_min_boundary_RMSE=float(np.linalg.norm(fit[indices])/np.sqrt(goals)),
                            online_Q_min_boundary_RMSE=float(np.linalg.norm(q[:,indices].min(axis=0))/np.sqrt(goals)))
                    recorded=result['statistics'][stratum][key]
                    for name,value in numbers.items():
                        assert np.isclose(value,recorded[name],rtol=1e-12,atol=1e-12),(job['id'],key,stratum,name)
                        metrics_checked+=1
                    table[stratum]=numbers
                values[key]=table
        rows.append(dict(job=job['id'],run=job['run_id'],step=job['step'],environment=env,unit_model=result['unit_return_model'],
            success_boundary_rows=goals,source_checkpoint_sha256=result['checkpoint_sha256'],
            result_sha256=helper.digest(path),raw_arrays_sha256=result['raw_arrays_sha256'],recomputed=values))
    result=dict(time=time.time(),status='observed_snapshot_audited',observed_jobs=len(rows),planned_jobs=len(plan['jobs']),
        metrics_independently_recomputed=metrics_checked,rows=rows,goal_complete=False,
        registration_sha256=helper.digest(args.registration),audit_source_sha256=helper.digest(__file__),
        limits=['Only zero-labelled success-boundary rows have exact Q/V labels. Uniform sample fitting and score moments are not conditional residual or B_k estimates.',
            'Compare counterfit objectives within one source frozen teacher; different trained teachers change the fitting target.',
            'Private 2/8/32-step curves are diagnostic, not policy evaluations or a basis for tuning GPU training.',
            'Rows observed after this snapshot remain pending; no selection of a best checkpoint, counterfit step or environment.'])
    helper.atomic(args.output,result)
    print(json.dumps({k:result[k]for k in ('status','observed_jobs','metrics_independently_recomputed')}),flush=True)


if __name__=='__main__':main()
