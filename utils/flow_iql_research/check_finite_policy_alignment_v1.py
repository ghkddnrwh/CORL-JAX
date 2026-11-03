"""Exact finite-MDP checks of backup/extraction mismatch, not RL experiments."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np


def moments(probabilities,samples,tau):
    count=np.arange(samples+1,dtype=np.float64)
    fraction=tau*count/(tau*count+(1-tau)*(samples-count))
    coefficients=np.empty_like(probabilities,dtype=np.float64)
    for state in range(len(probabilities)):
        for high in range(2):
            p=probabilities[state,high]
            mass=np.array([math.comb(samples,int(k))*p**k*(1-p)**(samples-k)for k in count])
            coefficients[state,high]=mass@fraction
    return coefficients,1-(1-probabilities)**samples


def weights_for(orientation,coefficient):
    weights=np.zeros_like(coefficient)
    for state,high in enumerate(orientation):
        weights[state,high]=coefficient[state,high]
        weights[state,1-high]=1-coefficient[state,high]
    return weights


def solve_policy(reward,transition,weights,gamma):
    states,actions=reward.shape
    matrix=np.zeros((states,states*actions))
    for state in range(states):matrix[state,state*actions:(state+1)*actions]=weights[state]
    system=np.eye(states*actions)-gamma*transition.reshape(-1,states)@matrix
    return np.linalg.solve(system,reward.ravel()).reshape(reward.shape),system


def solve_fixed_point(reward,transition,coefficient,gamma):
    for orientation in itertools.product(range(2),repeat=len(reward)):
        weights=weights_for(orientation,coefficient)
        value,_=solve_policy(reward,transition,weights,gamma)
        if all(value[state,high]>=value[state,1-high]-1e-9 for state,high in enumerate(orientation)):
            return value
    raise AssertionError('No consistent orientation: implementation error')


def state_backup(value,coefficient):
    high=np.argmax(value,axis=-1);low=1-high;state=np.arange(len(value))
    return value[state,low]+coefficient[state,high]*(value[state,high]-value[state,low])


def bisection_expectile(values,tau):
    low=values.min(-1);high=values.max(-1)
    for _ in range(70):
        middle=(low+high)/2;difference=values-middle[:,None]
        score=np.mean(np.where(difference>=0,tau,1-tau)*difference,axis=-1)
        low=np.where(score>0,middle,low);high=np.where(score<=0,middle,high)
    return (low+high)/2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rng=np.random.RandomState(60337);bounds=[]
    for samples in (1,2,9,99):
        for tau in (.9,.99):
            coefficient=(samples-1)*(1-tau)/(tau+(samples-1)*(1-tau))
            draws=rng.normal(size=(128,samples))
            expectile=bisection_expectile(draws,tau)
            gap=draws.max(-1)-expectile
            assert np.min(gap)>=-1e-12
            assert np.max(gap-coefficient*np.ptp(draws,axis=-1))<1e-12
            extreme=np.zeros((1,samples));extreme[0,-1]=1
            extreme_gap=float(1-bisection_expectile(extreme,tau)[0])
            assert abs(extreme_gap-coefficient)<1e-12
            bounds.append(dict(N=samples,tau=tau,max_gap_over_sample_range_bound=coefficient,
                               extreme_case_attains_bound=extreme_gap))
    checks=[];max_identity_error=0.;max_residual=0.;max_general_identity_error=0.
    for case in range(6):
        reward=-rng.uniform(size=(2,2))
        transition=rng.uniform(size=(2,2,2));transition/=transition.sum(-1,keepdims=True)
        if case%2:
            transition*=rng.uniform(.4,1,size=(2,2,1))  # terminal/masked mass
        probabilities=rng.uniform(.05,.95,size=(2,1));probabilities=np.concatenate([probabilities,1-probabilities],axis=1)
        for samples in (1,2,9):
            for gamma in (.9,.99,.999):
                previous=None
                for tau in (.9,.99):
                    coefficient,maximum=moments(probabilities,samples,tau)
                    value=solve_fixed_point(reward,transition,coefficient,gamma)
                    best=solve_fixed_point(reward,transition,maximum,gamma)
                    weights=weights_for(np.argmax(value,axis=-1),maximum)
                    deployed,system=solve_policy(reward,transition,weights,gamma)
                    defect=state_backup(value,maximum)-state_backup(value,coefficient)
                    right=gamma*transition.reshape(-1,2)@defect
                    predicted=np.linalg.solve(system,right).reshape(value.shape)
                    error=float(np.max(np.abs((deployed-value)-predicted)))
                    residual=float(np.max(np.abs(value-(reward+gamma*(transition@state_backup(value,coefficient))))))
                    max_identity_error=max(max_identity_error,error);max_residual=max(max_residual,residual)
                    assert error<1e-7 and residual<1e-9
                    assert np.min(deployed-value)>=-1e-8 and np.min(best-value)>=-1e-8
                    norm_bound=gamma/(1-gamma)*np.max(defect)
                    assert np.max(deployed-value)<=norm_bound+1e-7
                    c=(samples-1)*(1-tau)/(tau+(samples-1)*(1-tau))
                    assert np.max(defect-c*np.ptp(value,axis=-1))<1e-9
                    # Independently chosen approximate Q, V and deployed policy:
                    # verify the signed value-fit/extraction decomposition.
                    qhat=rng.normal(size=(2,2));vhat=rng.normal(size=2)
                    phat=rng.uniform(size=(2,2));phat/=phat.sum(-1,keepdims=True)
                    policy_value,policy_system=solve_policy(reward,transition,phat,gamma)
                    f_tau=state_backup(qhat,coefficient);f_max=state_backup(qhat,maximum)
                    b_train=reward+gamma*(transition@vhat)-qhat
                    e=vhat-f_tau;delta=f_max-np.sum(phat*qhat,axis=-1);d=f_max-f_tau
                    combined=b_train+gamma*(transition@(d-e-delta))
                    recovered=np.linalg.solve(policy_system,combined.ravel()).reshape(qhat.shape)
                    general_error=float(np.max(np.abs(policy_value-qhat-recovered)))
                    assert general_error<1e-7
                    max_general_identity_error=max(max_general_identity_error,general_error)
                    if previous is not None:assert np.min(value-previous)>=-1e-8
                    previous=value
                    checks.append(dict(case=case,N=samples,gamma=gamma,tau=tau,
                        policy_Q_minus_finite_expectile_Q_sup=float(np.max(deployed-value)),
                        exact_resolvent_identity_error=error,propagation_upper_bound=float(norm_bound)))
    # One state: action rewards -1/0, uniform behavior. The selected policy is
    # identical at both taus, while its exact value differs from each backup's.
    reward=np.array([[-1.,0.]])
    transition=np.ones((1,2,1));probabilities=np.array([[.5,.5]]);examples=[]
    for tau in (.9,.99):
        coefficient,maximum=moments(probabilities,9,tau)
        value=solve_fixed_point(reward,transition,coefficient,.999)
        weights=weights_for(np.argmax(value,axis=-1),maximum)
        deployed,_=solve_policy(reward,transition,weights,.999)
        examples.append(dict(tau=tau,finite_expectile_Q=value.tolist(),deployed_policy_Q=deployed.tolist(),
            deployed_action_probabilities=weights.tolist(),exact_policy_value_gap=(deployed-value).tolist()))
    np.testing.assert_allclose(examples[0]['deployed_policy_Q'],examples[1]['deployed_policy_Q'],atol=1e-9)
    result=dict(time=time.time(),kind='mathematical_finite_operator_check_not_benchmark_data',
        sample_bound_checks=bounds,finite_MDP_conditions_checked=len(checks),
        cases_include_continuing_and_masked_transition_kernels=True,
        maximum_exact_resolvent_identity_error=max_identity_error,maximum_Bellman_residual=max_residual,
        maximum_arbitrary_Q_V_policy_decomposition_error=max_general_identity_error,
        one_state_example=examples,checks=checks,
        scope='ExactscalarQ,fixedbehavior,smallfiniteMDPs;notneuraltraining,generatoraccuracyorOGBenchperformance.',
        note='utils/flow_iql_research/FINITE_BACKUP_POLICY_ALIGNMENT.txt',
        checker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(conditions=len(checks),max_identity_error=max_identity_error,
        max_general_identity_error=max_general_identity_error,max_Bellman_residual=max_residual,one_state_example=examples)),flush=True)


if __name__=='__main__':main()
