"""Describe logit-score attenuation on already saved, audited fixed queries.

Analytic derivatives, not neural parameter gradients or a new training trial.
Relative attenuation between strata is unchanged by a global loss multiplier.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from utils.flow_iql_research import tau_matched_extraction_v1 as helper


def main():
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();audit=json.loads(args.audit.read_text());root=args.audit.parent.parent
    registration=root/'unit_value_fitting_diagnostic_registration_v1.json';plan=json.loads(registration.read_text())
    assert helper.digest(registration)==audit['registration_sha256'];helper.check_sources(plan['source_sha256'])
    jobs={j['id']:j for j in plan['jobs']};rows=[]
    for observed in audit['rows']:
        if not observed['unit_model']:continue
        job=jobs[observed['job']];result=json.loads(Path(job['output']).read_text())
        assert helper.digest(job['output'])==observed['result_sha256']
        assert helper.digest(result['raw_arrays'])==observed['raw_arrays_sha256']
        cfg=plan['expected_identities'][job['run_id']]['config'];scale=cfg['return_affine_scale'];tau=cfg['expectile']
        strata={}
        with np.load(result['raw_arrays'],allow_pickle=False)as arrays:
            n=len(arrays['validation_uniform_indices']);goals=len(arrays['validation_goal_indices'])
            # Saved costs were decoded from float32 probabilities. This inverse
            # recovers their values; finite-precision loss code is not rerun.
            q=1+scale*arrays['target_fit_Q_cost'];v=1+scale*arrays['initial_V_cost']
            assert np.all((q>=0)&(q<=1))and np.all((v>=0)&(v<=1))
            dkl=np.where(q>=v,tau,1-tau)*(v-q);dmse=2*v*(1-v)*dkl
            for label,indices in [('uniform_validation',slice(0,n)),('success_boundary',slice(n,None))]:
                if label=='success_boundary'and not goals:continue
                rms=lambda x:float(np.linalg.norm(x)/np.sqrt(x.size))
                kl=rms(dkl[indices]);mse=rms(dmse[indices]);attenuation=2*v[indices]*(1-v[indices])
                strata[label]=dict(rows=len(v[indices]),KL_V_logit_score_RMS=kl,MSE_V_logit_score_RMS=mse,
                    KL_to_MSE_score_RMS_ratio=kl/mse if mse else None,
                    MSE_extra_sigmoid_attenuation_median=float(np.median(attenuation)),
                    MSE_extra_sigmoid_attenuation_min=float(attenuation.min()),
                    MSE_extra_sigmoid_attenuation_max=float(attenuation.max()),
                    V_probability_endpoint_rows=int(np.sum((v[indices]==0)|(v[indices]==1))))
        rel=None
        if 'success_boundary'in strata:
            a=strata['success_boundary']['KL_to_MSE_score_RMS_ratio'];b=strata['uniform_validation']['KL_to_MSE_score_RMS_ratio']
            if a is not None and b:rel=a/b
        rows.append(dict(job=job['id'],environment=observed['environment'],step=job['step'],
            original_checkpoint_sha256=observed['source_checkpoint_sha256'],strata=strata,
            boundary_vs_uniform_relative_KL_boost=rel))
    result=dict(time=time.time(),status='fixed_observed_query_analysis_complete',rows=rows,
        audited_input_sha256=helper.digest(args.audit),source_sha256=helper.digest(__file__),
        formula_KL='w_tau*(pV-pQ)',formula_MSE='2*pV*(1-pV)*w_tau*(pV-pQ)',
        gradient_scope='Analytic per-example derivatives with respect to V raw logits, evaluated from saved probabilities. Not gradients with respect to neural network parameters.',
        global_scaling_limit='Absolute gradient size depends on loss scaling. Boundary/uniform relative boost is invariant to a global multiplier; Adam dynamics still require actual experiments.',
        new_training_updates=0,new_policy_evaluations=0,goal_complete=False,
        limits=['Both losses have the same population expectile root with fixed teacher and unrestricted scalar predictions.',
            'Different source teachers and decoded initializations prohibit reading training-performance causality from this table.',
            'Success-boundary rows and uniform validation rows are separate strata. They are not pooled as a population distribution.',
            'Neural Jacobians, sample-gradient cancellation, optimizer moments, minibatch frequency and later bootstrapping are not measured here.',
            'This descriptive posthoc diagnostic neither selects a checkpoint nor promotes a training recipe.'])
    helper.atomic(args.output,result)
    print(json.dumps(dict(status=result['status'],observed_unit_models=len(rows),relative_boosts=[dict(job=r['job'],boost=r['boundary_vs_uniform_relative_KL_boost'])for r in rows])),flush=True)


if __name__=='__main__':main()
