"""Oracle counterexample: residual geometry matters; traces do not remove bias.

Uses the return-based preconditioner from Munos et al. (arxiv:1606.02647),
and the IDQL implicit actor identity (arxiv:2304.10573). This is an analysis
of a candidate direction, not an OGBench algorithm or new novelty claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def expectile(q,mu,tau):
    hi=np.argmax(q,axis=1);lo=1-hi;i=np.arange(len(q))
    weights_hi=tau*mu[i,hi];weights_lo=(1-tau)*mu[i,lo]
    return (weights_hi*q[i,hi]+weights_lo*q[i,lo])/(weights_hi+weights_lo)


def setup(discount,tau,geometry):
    states,actions=3,2
    transition=np.zeros((states*actions,states))
    transition[0,1]=transition[1,2]=1
    transition[2:4,1]=1;transition[4:6,2]=1
    base=.1;gap=.001;mu=np.full((states,actions),.5)
    reward=np.array([-1-base,-1-base,-base,-base,-base-gap,-base-gap])
    good=-base/(1-discount);bad=-(base+gap)/(1-discount)
    oracle=np.array([reward[0]+discount*good,reward[1]+discount*bad,good,good,bad,bad])
    b=np.array([0.,gap,gap]) if geometry=='common_successor_offset' else np.array([0.,-(1-tau)/tau*gap,gap])
    biased=oracle.copy()
    biased[:2]+=discount*b[1:]/(1-discount)
    biased[2:4]+=discount*b[1]/(1-discount);biased[4:]+=discount*b[2]/(1-discount)
    return transition,reward,mu,b,oracle,biased


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();rows=[]
    for discount in (.99,.999):
        for tau in (.9,.99):
            paired=[]
            for geometry in ('common_successor_offset','heterogeneous_successor_bias'):
                p,reward,mu,b,oracle,biased=setup(discount,tau,geometry)
                def backup(q):return reward+discount*p@(expectile(q.reshape(3,2),mu,tau)+b)
                np.testing.assert_allclose(backup(biased),biased,atol=3e-12,rtol=0)
                v=expectile(biased.reshape(3,2),mu,tau)+b
                difference=biased.reshape(3,2)-v[:,None]
                w=np.where(difference>=0,tau,1-tau)
                score=np.sum(mu*w*difference,axis=1)
                z=np.sum(mu*w,axis=1)
                tilted=mu*w/z[:,None]
                c=w/tau
                assert np.all(c<=tilted/mu+1e-14) and np.all(c<=1)
                np.testing.assert_allclose(np.sum(tilted*biased.reshape(3,2),axis=1)-v,score/z,atol=3e-13,rtol=0)
                pc=(p[:,:,None]*(mu*c)[None,:,:]).reshape(6,6)
                assert np.max(pc.sum(1))<=1+1e-14
                lower=reward.min()/(1-discount)
                assert biased.min()>lower and biased.max()<0
                for horizon in (1,5,20):
                    kernel=np.eye(6);power=np.eye(6)
                    for _ in range(1,horizon):
                        power=power@(discount*pc);kernel+=power
                    # The sum of powers is invertible because spectral radius(gamma Pc)<1.
                    assert np.linalg.svd(kernel,compute_uv=False).min()>0
                    root_residual=kernel@(backup(biased)-biased)
                    assert np.max(np.abs(root_residual))<2e-10
                    q=np.zeros(6);history=[]
                    for iteration in range(100001):
                        error=float(np.max(np.abs(q-biased)))
                        if iteration in (0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000):
                            history.append(dict(iteration=iteration,error_to_biased_fixedpoint=error))
                        if error<1e-8:break
                        q=q+kernel@(backup(q)-q)
                        assert q.min()>=lower-1e-9 and q.max()<=1e-9
                    else:raise AssertionError('Iteration did not converge')
                    selected=int(np.argmax(q[:2]));true_return=float(oracle[selected])
                    rows.append(dict(discount=discount,expectile=tau,geometry=geometry,horizon=horizon,
                        iterations_to_1e8=iteration,trace_coefficient=c.tolist(),
                        final_Q=q.tolist(),analytic_biased_Q=biased.tolist(),oracle_Q=oracle.tolist(),
                        value_residual_b=b.tolist(),conditional_expectile_score=score.tolist(),
                        value_residual_sup=float(np.max(np.abs(b))),
                        conditional_score_sup=float(np.max(np.abs(score))),
                        propagated_Q_bias_sup=float(np.max(np.abs(biased-oracle))),
                        actual_fitted_Bellman_residual_sup=float(np.max(np.abs(backup(q)-q))),
                        start_greedy_action=selected,true_start_return=true_return,
                        true_start_regret=float(oracle[:2].max()-true_return),
                        bounded_targets_inactive=True,history=history))
                paired.append(rows[-1])
            for field in ('value_residual_sup','conditional_score_sup','propagated_Q_bias_sup'):
                np.testing.assert_allclose(paired[0][field],paired[1][field],atol=3e-12,rtol=0)
            assert paired[0]['start_greedy_action']==0 and paired[1]['start_greedy_action']==1
    result=dict(time=time.time(),status='complete',rows=rows,oracle_models=8,trace_iterations=24,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=['https://arxiv.org/html/1606.02647v2','https://arxiv.org/html/2304.10573v2'],
        derivation='With V(Q)=e_tau(Q)+b, T_b Q=r+gamma P V(Q), and any substochastic Pc, '
            'the finite trace update is Q + A_H(Q)(T_b Q-Q), A_H=sum_{j=0}^{H-1}(gamma Pc)^j. '
            'A_H is invertible at every Q when gamma<1 and c<=1, so trace preconditioning '
            'alone has exactly the same roots T_b Q=Q. It need not remove systematic V residual bias. '
            'The tabular fixedpoint identity does not assert neural/global convergence. '
            'For exact V, IDQL pi=mu*w/Z and c=w/tau<=pi/mu because Z<=tau. '
            'With approximate V the actor expectation discrepancy is rho/Z, not zero.',
        decision='Do not add trace-only GPU pilots as a systematic residual-bias fix based on gamma**H intuition. '
            'Prioritize already-running fresh-data V fits and finite conditional-action expectile backups. '
            'Trace acceleration remains a separate possible optimization hypothesis, not ruled out for stochastic/neural training.',
        limits=['Synthetic known three-state/two-action MDP, not evidence of OGBench policy gains or its actual causal failure.',
            'The injected V-minus-population-expectile error is held fixed; real fitting error changes with learning.',
            'Both compared geometries have equal b sup, conditional score sup and propagated Q error sup, but different greedy policy return.',
            'No active projection in this example. Outer projection with active constraints can alter fixedpoint arguments.',
            'Finite empirical action expectiles remove only their finite-set score, not true population residual or flow-model error.',
            'No GPU training, extra OGBench evaluations, source edits to registered algorithms, or novelty claim.'],goal_complete=False)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status='complete',oracle_models=8,trace_iterations=24,
        example=[{k:r[k]for k in ('geometry','horizon','value_residual_sup','conditional_score_sup','propagated_Q_bias_sup','true_start_regret')}for r in rows if r['discount']==.999 and r['expectile']==.9])),flush=True)


if __name__=='__main__':main()
