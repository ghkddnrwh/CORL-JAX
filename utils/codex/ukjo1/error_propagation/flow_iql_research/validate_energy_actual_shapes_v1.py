"""Actual-size CPU8update runner/resume/export/evaluator integration fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np


ENVIRONMENTS = {'ant':'antmaze-large-navigate-singletask-v0',
                'cube':'cube-single-play-singletask-v0','pz3':'puzzle-3x3-play-singletask-v0'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def leaves(tree,prefix=''):
    if isinstance(tree,dict):
        for key in sorted(tree):
            yield from leaves(tree[key],prefix+'/'+str(key))
    elif isinstance(tree,(tuple,list)):
        for index,value in enumerate(tree):
            yield from leaves(value,prefix+'/'+str(index))
    elif tree is not None:
        yield prefix,np.asarray(tree)


def equal(first,second):
    left,right = dict(leaves(first)),dict(leaves(second))
    return left.keys() == right.keys() and all(np.array_equal(value,right[key]) for key,value in left.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-plan',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=tuple(ENVIRONMENTS),required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) == 6
    plan = json.loads(args.source_plan.read_text())
    def check_sources():
        for path,digest in plan['source_sha256'].items():
            assert sha(path) == digest,path
    check_sources()
    for tag in args.cases:
        output = args.output_root / tag
        output.mkdir(parents=True,exist_ok=False)
        def execute(label,command):
            receipt = dict(time=time.time(),command=command,scope='CPU implementation fixture, not policy performance.')
            write(output / f'{label}_request.json',receipt)
            with (output / f'{label}.log').open('xb') as log:
                child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True,close_fds=True)
            receipt.update(pid=child.pid,birth=Path(f'/proc/{child.pid}/stat').read_text().rsplit(') ',1)[1].split()[19])
            write(output / f'{label}_handle.json',receipt)
            code = child.wait()
            assert code == 0,(label,code,(output / f'{label}.log').read_text()[-4000:])
            check_sources()
        def training(destination,stop=None):
            command = [sys.executable,'-u','-m','utils.flow_iql_research.train_energy_onestep_v1',
                '--env',ENVIRONMENTS[tag],'--recipe','energy_bc','--seed','0','--steps','8',
                '--chunk','2','--probe-every','2','--train-device','cpu','--output',str(output / destination)]
            if stop:
                command += ['--stop-at',str(stop)]
            return command
        execute('full8',training('full'))
        execute('resume4',training('resume',4))
        execute('resume8',training('resume'))
        with (output / 'full/checkpoint_8.pkl').open('rb') as stream:
            full = pickle.load(stream)
        with (output / 'resume/checkpoint_8.pkl').open('rb') as stream:
            resumed = pickle.load(stream)
        assert full['step'] == resumed['step'] == 8
        assert full['identity'] == resumed['identity']
        assert equal(full['state'],resumed['state']) and equal(full['replay_rng'],resumed['replay_rng'])
        state = full['state']
        assert all(np.isfinite(v).all() for _,v in leaves(state))
        assert int(state['base']['q_updates']) == int(state['base']['agent']['network']['opt_state']['0']['count']) == 8
        assert int(state['actor_updates']) == int(state['actor']['opt_state']['0']['count']) == 8
        assert int(state['base']['v_updates']) == int(state['base']['v_opt']['0']['count']) == 0
        config = json.loads((output / 'full/config.json').read_text())
        assert config['source_sha256'] == plan['training_source_sha256']
        assert config['identity']['training_device'] == 'cpu'
        artifact = output / 'full/actor_8.pkl'
        execute('portable_isolation',[sys.executable,'-u','-m',
            'utils.flow_iql_research.validate_standalone_mean_teacher_actor_v1',
            '--actor',str(artifact),'--output',str(output / 'portable_isolation_result.json')])
        evaluations = []
        for mode in ('primary','teacher'):
            target = output / f'{mode}_API_result.json'
            execute(f'{mode}_API',[sys.executable,'-u','-m','utils.flow_iql_research.evaluate_energy_onestep_v1',
                '--run',str(output / 'full'),'--step','8','--mode',mode,'--episodes','10',
                '--implementation-only','--output',str(target)])
            value = json.loads(target.read_text())
            assert value['protocol'] == 'implementation_smoke_only_cpu10' and value['no_performance_claim']
            assert value['no_optimizer_updates'] and value['checkpoint_sha256'] == sha(output / 'full/checkpoint_8.pkl')
            evaluations.append(dict(mode=mode,path=str(target),sha256=sha(target),no_performance_claim=True))
        result = dict(time=time.time(),status='pass',environment=ENVIRONMENTS[tag],Q_Adam=8,actor_Adam=8,V_Adam=0,
            actual_hidden_width=config['identity']['config']['actor_hidden_dims'][0],full8_resume4plus4_all_saved_state_bitwise_equal=True,
            replay_rng_after_resume_bitwise_equal=True,portable_actor_only_client_isolation_passed=True,
            actual_actor_and_teacher_evaluator_API_passed=True,evaluator_implementation_only=evaluations,
            source_plan_sha256=sha(args.source_plan),fixture_actor=str(artifact),fixture_actor_sha256=sha(artifact),
            no_GPU_or_registered_learner_updates=True,no_performance_claim=True,
            scope='Real dataset/dimensions and default widths, CPU8update integration only. Not a trained1M result.')
        write(output / 'result.json',result)
        print(json.dumps(dict(case=tag,status='pass',output=str(output / 'result.json'))),flush=True)


if __name__ == '__main__':
    main()
