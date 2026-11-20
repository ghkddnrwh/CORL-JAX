"""Check affine Bellman identity, expectile scoring root, and real update clocks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research import probability_value_core_v1 as proposed, policy_core as original
from utils.propagation_research.reward_bounds import REWARD_COMPONENTS
import flax
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def equal(a,b):
    aa=jax.tree_util.tree_leaves(a);bb=jax.tree_util.tree_leaves(b)
    return len(aa)==len(bb) and all(np.array_equal(x,y)for x,y in zip(aa,bb))


def root(values,tau):
    left,right=float(values.min()),float(values.max())
    for _ in range(90):
        middle=(left+right)/2;score=np.mean(np.where(values>=middle,tau,1-tau)*(values-middle))
        if score>0:left=middle
        else:right=middle
    return (left+right)/2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();rng=np.random.RandomState(52031)
    bellman=[]
    for env in proposed.ENVIRONMENTS:
        for gamma in (.99,.999):
            cfg=proposed.config_for(env,gamma,.9);c=REWARD_COMPONENTS[env];a=cfg['return_affine_scale'];lower=-1/a
            reward=-c*rng.uniform(size=1000);mask=rng.randint(2,size=1000);v=rng.uniform(lower,0,size=1000)
            target=reward+gamma*mask*v
            transformed=proposed.normalized_reward(reward,mask,gamma,c)+gamma*mask*(1+a*v)
            expected=1+a*target
            np.testing.assert_allclose(transformed,expected,atol=5e-15,rtol=0)
            assert transformed.min()>=-1e-14 and transformed.max()<=1+1e-14
            np.testing.assert_allclose(np.clip(transformed,0,1),1+a*np.clip(target,lower,0),atol=5e-15,rtol=0)
            q=rng.uniform(lower,0,size=19);e=root(q,.9);scaled=root(1+a*q,.9)
            np.testing.assert_allclose(1+a*e,scaled,atol=5e-14,rtol=0)
            bellman.append(dict(environment=env,discount=gamma,declared_reward_components=c,
                original_return_lower=lower,max_affine_error=float(np.max(np.abs(transformed-expected)))))
    assert proposed.normalized_reward(-3.,1.,.999,9)>0 # Puzzle shaping is not a binary success reward.
    assert proposed.config_for('puzzle-4x4-play-singletask-v0',.999)['return_reward_components']==16
    gradient=jax.jit(jax.grad(lambda logit,target,tau:proposed.expectile_loss(logit,target,tau,'bernoulli_kl')))
    maximum_gradient_error=0.;maximum_root_gradient=0.
    for tau in (.5,.9,.99):
        for _ in range(100):
            targets=rng.uniform(size=17).astype(np.float32);targets[:2]=[0,1]
            p=float(rng.uniform(.001,.999));logit=np.float32(np.log(p/(1-p)))
            predicted=float(jax.nn.sigmoid(logit));expected=float(np.mean(np.where(targets>=predicted,tau,1-tau)*(predicted-targets)))
            observed=float(gradient(logit,targets,tau));error=abs(observed-expected)
            assert error<2e-7;maximum_gradient_error=max(maximum_gradient_error,error)
            e=root(targets.astype(np.float64),tau);logit=np.float32(np.log(e/(1-e)))
            g=abs(float(gradient(logit,targets,tau)));assert g<2e-7;maximum_root_gradient=max(maximum_root_gradient,g)
    endpoint_losses=[]
    for target in (0.,1.):
        for logit in (-1000.,1000.):
            value=float(proposed.expectile_loss(jnp.asarray(logit),jnp.asarray(target),.99,'bernoulli_kl'))
            grad=float(gradient(jnp.asarray(logit),jnp.asarray(target),.99))
            assert np.isfinite([value,grad]).all();endpoint_losses.append(dict(target=target,logit=logit,loss=value,gradient=grad))
    environment='puzzle-3x3-play-singletask-v0';env,data,_=ogbench.make_env_and_datasets(environment)
    indices=np.r_[rng.randint(len(data['actions']),size=31),np.flatnonzero(data['masks']==0)[0]]
    batch={k:jnp.asarray(np.asarray(data[k][indices],np.float32))for k in ('observations','actions','next_observations','rewards','masks')}
    batch['actions']=jnp.clip(batch['actions'],-1+1e-5,1-1e-5)
    configs={};states={};updates={}
    for profile in ('original','bounded_v2_unit_mse','bounded_v2_unit_kl'):
        module=original if profile=='original'else proposed
        cfg=module.config_for(environment,.999,.9);cfg.update(num_samples=9,agent_name='ifql',policy_type='ifql')
        if profile!='original':cfg['value_scoring_rule']=proposed.PROFILES[profile]['geometry']
        configs[profile]=cfg;states[profile]=module.create(3,batch['observations'][:1],batch['actions'][:1],cfg,'ifql')
        updates[profile]=jax.jit(module.build_update('bounded_v2'if profile=='original'else profile,(-9000.,0.),'ifql'))
    assert equal(states['original'].agent.network.params,states['bounded_v2_unit_mse'].agent.network.params)
    assert equal(states['bounded_v2_unit_mse'].agent.network.params,states['bounded_v2_unit_kl'].agent.network.params)
    for _ in range(4):
        for profile in states:states[profile],logs=updates[profile](states[profile],batch)
        reference=states['original'];net=reference.agent.network
        for profile,state in states.items():
            n=state.agent.network
            assert equal(n.params['modules_actor_flow'],net.params['modules_actor_flow'])
            assert equal(n.opt_state[0].mu['modules_actor_flow'],net.opt_state[0].mu['modules_actor_flow'])
            assert equal(n.opt_state[0].nu['modules_actor_flow'],net.opt_state[0].nu['modules_actor_flow'])
            assert equal(state.agent.rng,reference.agent.rng)
            assert int(n.opt_state[0].count)==int(state.q_updates)
            assert int(state.v_opt[0].count)==int(state.v_updates)==2*int(state.q_updates)
            assert all(np.all(np.asarray(x)==0)for x in jax.tree_util.tree_leaves(n.opt_state[0].mu['modules_value']))
    for profile in ('bounded_v2_unit_mse','bounded_v2_unit_kl'):
        values=jax.device_get(proposed.probe(states[profile],batch));assert bool(values[4])
        for value in values[:3]:assert np.min(value)>=-9000.-.001 and np.max(value)<=.001
        summary=proposed.probe_summary(values,.9,(-9000.,0.),None);assert summary['finite']
    assert not equal(states['bounded_v2_unit_mse'].agent.network.params['modules_critic'],states['bounded_v2_unit_kl'].agent.network.params['modules_critic'])
    result=dict(time=time.time(),status='passed',affine_checks=bellman,scalar_distributions=300,
        maximum_KL_gradient_identity_error=maximum_gradient_error,maximum_root_gradient=maximum_root_gradient,
        endpoint_checks=endpoint_losses,real_dataset_update_test=dict(environment=environment,batch=32,
            forced_terminal_examples=1,profiles=3,updates_each=4,identical_initial_raw_networks=True,
            actor_parameters_Adam_moments_and_RNG_exact=True,Q_V_Adam_counts_correct=True,
            joint_V_moments_stay_zero=True,inverse_cost_values_within_known_bounds=True,
            KL_and_MSE_critic_updates_differ=True),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (Path(__file__),Path(proposed.__file__))},
        new_GPU_training=0,new_OGBench_policy_evaluations=0,goal_complete=False,
        limits=['FourCPUupdates and exact scalar equations do not establish neural convergence or benchmark improvements.',
            'Both new profiles use the same raw-logit initialization; transformed original-cost initialization differs from the original linear-cost method.',
            'The sigmoid-MSE versus sigmoid-KL comparison isolates loss geometry within the new shared representation.',
            'Normalized return is not solely success probability for shaped manipulation rewards.'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False));env.close()
    print(json.dumps(dict(status='passed',affine_checks=len(bellman),scalar_distributions=300,actual_cpu_updates=12,actor_exact=True)),flush=True)


if __name__=='__main__':main()
