"""Check when a constant return-floor critic has zero nonterminal TD target gap."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research import policy_core
from utils.propagation_research.reward_bounds import return_bounds
import numpy as np
import ogbench
import ogbench.relabel_utils


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();gamma=.999;rows=[]
    for environment in policy_core.ENVIRONMENTS:
        env,train,validation=ogbench.make_env_and_datasets(environment)
        for split,data in [('training',train),('validation',validation)]:
            rewards=np.asarray(data['rewards'],np.float64);masks=np.asarray(data['masks'],np.float64)
            lower,upper=return_bounds(environment,gamma,float(rewards.min()),float(rewards.max()),'ogbench')
            assert upper==0 and np.isin(masks,[0,1]).all()
            component_cost=-lower*(1-gamma);terminal=masks==0;nonterminal=masks==1
            assert np.all(rewards[terminal]==0)
            raw=rewards+gamma*masks*lower;target=np.clip(raw,lower,upper);residual=target-lower
            np.testing.assert_allclose(residual[nonterminal],(rewards+component_cost)[nonterminal],atol=1e-9)
            np.testing.assert_allclose(residual[terminal],-lower,atol=1e-9)
            counts=np.unique(rewards,return_counts=True)
            rows.append(dict(environment=environment,split=split,transitions=len(rewards),return_lower=lower,
                component_cost=component_cost,reward_histogram={str(float(value)):int(count)for value,count in zip(*counts)},
                full_success_fraction=float(terminal.mean()),
                nonterminal_zero_floor_TD_fraction_of_dataset=float(np.mean(nonterminal&(np.abs(residual)<1e-8))),
                nonterminal_progress_floor_TD_fraction_of_dataset=float(np.mean(nonterminal&(residual>1e-8))),
                conditional_nonterminal_progress_fraction=float(np.mean(residual[nonterminal]>1e-8)),
                conditional_nonterminal_floor_TD_RMS=float(np.sqrt(np.mean(residual[nonterminal]**2))),
                constant_floor_all_transition_TD_RMS=float(np.sqrt(np.mean(residual**2))),
                floor_target_projection_fraction=float(np.mean(np.abs(raw-target)>1e-8))))
        env.close()
    audit_path=args.root/'analysis/terminal_boundary_audit_v1/summary.json';audit=json.loads(audit_path.read_text());bounds=[]
    for group in audit['boundary_errors']:
        if not group['inspected_all_terminal_transitions']:continue
        count=next(row for row in audit['counts']if row['environment']==group['environment']and row['split']==group['split'])
        f=count['zero_bootstrap_fraction'];error=group['Q_RMS_to_known_zero']
        bounds.append(dict(environment=group['environment'],source=group['source'],split=group['split'],goal_fraction=f,
            empirical_dataset_both_head_Q_error_RMS_lower_bound=float(np.sqrt(f)*error),
            frozen_full_dataset_critic_sum_head_MSE_lower_bound=float(2*f*error**2)))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result=dict(time=time.time(),discount=gamma,rows=rows,certified_boundary_contributions=bounds,
        source_sha256=sha(__file__),relabeler_sha256=sha(ogbench.relabel_utils.__file__),boundary_audit_sha256=sha(audit_path),
        interpretation='Withmask1andminimumreward-C,theconstantL=-C/(1-gamma)satisfiesthelocalTDtargetexactly. Goalboundariesandpartial-progressrewardsbreakthatself-consistency. ThisisNOTaglobalBellmanfixedpointclaim.',
        limits=['Constant-floor diagnostics describe a constructed scalar critic, not an actual neural network or a convergence proof.',
                'Known-boundary lower bounds concern the empirical dataset measure, not an unknown deployment occupancy.',
                'Critic loss lower bounds apply to the frozen exact masked target and the sum over two heads, not a moving training-window statistic.',
                'Rare-goal frequency and dense reward differences do not by themselves establish a causal explanation or a performance-improvement theorem.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);temp=args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(args.output)
    print(json.dumps(dict(output=str(args.output),dataset_rows=len(rows),certified_boundary_bounds=len(bounds))),flush=True)


if __name__=='__main__':main()
