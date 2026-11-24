"""Exercise the real controller with in-process workers before GPU launch."""
import argparse
from contextlib import redirect_stdout
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    real=args.root.resolve()
    module=importlib.import_module('utils.codex.ukjo1.error_propagation.flow_iql_research.run_default_ema_n9_v1')
    plan=json.loads((real/'registration_v1.json').read_text())
    with tempfile.TemporaryDirectory(prefix='nondefault_controller_') as temporary:
        root=Path(temporary)
        (root/'validation').mkdir()
        (root/'validation/numerical_v1.json').write_text('{"status":"PASS"}')
        (root/'validation/controller_v1.json').write_text('{"status":"PASS"}')
        plan['source_sha256']={}
        for job in plan['jobs']:
            job['output']=str(root/'runs'/job['id'])
            job['evaluation_directory']=str(root/'evaluations'/job['id'])
        (root/'registration_v1.json').write_text(json.dumps(plan))
        clock=[1000000.]
        workers,attempts={},{}
        counts={'train':0,'eval':0}
        peaks=dict(counts)

        class Worker:
            def __init__(self,command,**kwargs):
                self.pid=10000+len(workers)
                self.returncode=None
                self.training='--env' in command
                self.role='train' if self.training else 'eval'
                self.output=Path(command[command.index('--output')+1])
                key=str(self.output)
                attempts[key]=attempts.get(key,0)+1
                self.fail=((self.training and self.output.name==plan['jobs'][0]['id']) or
                           (not self.training and self.output.name=='800000_lean.json')) and attempts[key]==1
                workers[self.pid]=self
                counts[self.role]+=1
                peaks[self.role]=max(peaks[self.role],counts[self.role])

            def poll(self):
                if self.returncode is not None:
                    return self.returncode
                self.returncode=1 if self.fail else 0
                counts[self.role]-=1
                if not self.fail:
                    self.output.parent.mkdir(parents=True,exist_ok=True)
                    if self.training:
                        self.output.mkdir(exist_ok=True)
                        (self.output/'completed.json').write_text('{}')
                        for step in plan['evaluation_steps']:
                            for stem in ('checkpoint','actor'):
                                (self.output/f'{stem}_{step}.pkl').write_bytes(b'stub')
                    else:
                        self.output.write_text('{}')
                return self.returncode

        def now():
            clock[0]+=1
            return clock[0]

        module.subprocess=SimpleNamespace(Popen=Worker,STDOUT=-2,
                                         run=lambda *a,**kw:SimpleNamespace(returncode=0))
        module.time=SimpleNamespace(time=now,sleep=lambda n:clock.__setitem__(0,clock[0]+n))
        module.birth=lambda pid:str(pid) if pid in workers and workers[pid].returncode is None else None
        sys.argv=['run_default_ema_n9_v1','--root',str(root)]
        with (real/'validation/controller_simulation_stdout.log').open('w') as stream:
            with redirect_stdout(stream):
                module.main()
        state=json.loads((root/'controller_status.json').read_text())
        assert state['phase']=='complete' and state['training_complete']==len(plan['jobs']) and state['evaluations_complete']==6*len(plan['jobs'])
        assert peaks['train']==len(plan['resources']['train_slots'])
        assert peaks['eval']==len(plan['resources']['eval_cpus'])
        assert any(e['kind']=='training' for e in state['retry_history'])
        assert any(e['kind']=='evaluation' for e in state['retry_history'])
        result=dict(status='PASS',training_completed=len(plan['jobs']),evaluations_completed=6*len(plan['jobs']),
                    maximum_concurrency=peaks,automatic_refill=True,
                    training_and_evaluation_retry=True,actual_GPU_work=False)
        (real/'validation/controller_v1.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
