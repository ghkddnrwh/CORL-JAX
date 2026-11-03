"""Actual saved native/V2 model checks; no new training or RL episodes."""
import argparse
import copy
import json
from pathlib import Path
import time
from utils.flow_iql_research import policy_core, selection_v2, tau_matched_selection_v1 as selection
from utils.flow_iql_research import tau_matched_extraction_v1 as audit, train_policy_v2
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    run_ids=['ifql_control_ant_official_g999_t90_s0','ieconfirm_s3_pz3_ifql_bounded_v2_g999_t90_N9']
    rows=[]
    for run_id in run_ids:
        run=args.root/'runs'/run_id; conf=json.loads((run/'config.json').read_text())
        native=conf['identity']['profile']=='official'
        plan=dict(root=str(args.root),expected_identities={run_id:conf['identity']},
            expected_training_sources={run_id:conf['source_sha256']},expected_final_clocks={run_id:
                dict(Q_Adam=1000000,V_total=1000000 if native else 2000000,separate_V_Adam=0 if native else 2000000)})
        _,saved,checkpoint,sha=audit.load_parent(plan,run_id,1000000)
        cfg=copy.deepcopy(conf['identity']['config'])
        for field in ('actor_hidden_dims','value_hidden_dims'):cfg[field]=tuple(cfg[field])
        env=ogbench.make_env_and_datasets(conf['identity']['environment'],env_only=True)
        template=policy_core.create(conf['identity']['seed'],jnp.zeros((1,env.observation_space.shape[-1])),
            jnp.zeros((1,env.action_space.shape[-1])),cfg,'ifql')
        state=flax.serialization.from_state_dict(template,saved['state'])
        observations=np.random.RandomState(91837).normal(size=(32,env.observation_space.shape[-1])).astype(np.float32)
        params=state.agent.network.params
        official=train_policy_v2.make_policy(state.agent.replace(config=state.agent.config.copy({'num_samples':9})))
        prefix=selection_v2.make_policy(state.agent,'max',9,'behavior')
        normal=selection.make_decision(state.agent)
        high=selection.make_decision(state.agent.replace(config=state.agent.config.copy({'expectile':.99})))
        gains=[]; overrides=0
        for seed in range(6):
            for i, observation in enumerate(observations):
                key=jax.random.fold_in(jax.random.PRNGKey(seed+91837),i)
                old=np.asarray(official(params,observation,key))
                before=np.asarray(prefix(params,observation,key))
                action,q9,q9new,index=normal(params,observation,key)
                assert np.array_equal(old,before)
                assert np.array_equal(before,np.asarray(action))
                assert np.array_equal(np.asarray(q9),np.asarray(q9new))
                action,a,b,index=high(params,observation,key)
                difference=float(b-a);assert np.isfinite(difference) and difference>=0
                gains.append(difference);overrides+=int(index>=9)
        rejections=[]
        for field in ('identity','sources','clocks'):
            bad=copy.deepcopy(plan)
            if field=='identity':bad['expected_identities'][run_id]['seed']+=1
            elif field=='sources':bad['expected_training_sources'][run_id][next(iter(conf['source_sha256']))]='0'*64
            else:bad['expected_final_clocks'][run_id]['V_total']+=1000000
            try:audit.load_parent(bad,run_id,1000000)
            except AssertionError:rejections.append(field)
            else:raise AssertionError(field)
        assert audit.digest(checkpoint)==sha
        rows.append(dict(run=run_id,checkpoint_sha256=sha,native=native,
            random_query_count=192,exact_N9_prefix_actions=True,
            exact_official_N9_action_comparison_queries=192,
            N99_predicted_Q_gap_min=min(gains),N99_predicted_Q_gap_mean=float(np.mean(gains)),
            N99_extra_action_selected_count=overrides,negative_checks_rejected=rejections))
        env.close()
    assert selection.proposal_count(.9)==9 and selection.proposal_count(.99)==99
    assert not audit.finite_tree({'bad':np.nan})
    result=dict(time=time.time(),status='complete',rows=rows,total_random_queries=384,
        inference_only_modified_expectile_config_for_N99_test=True,new_GPU_training=0,new_evaluation_episodes=0,
        sources={p:audit.digest(p) for p in (str(Path(__file__)),selection.__file__,audit.__file__)},
        limitations=['Random queries validate inference and provenance behavior; they do not estimate policy quality.',
            'The N99 test changes only the extraction expectile config on old actual tau.9 models.',
            'The native N32 saved model is evaluated with extraction-only config N9 for the prefix check.',
            'All queries have the unbatched observation shape used by the actual evaluator.'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
