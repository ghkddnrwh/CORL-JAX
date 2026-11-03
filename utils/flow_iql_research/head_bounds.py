"""Output-range geometry of the official post-GELU LayerNorm value networks.

For last hidden normalization u, sum(u)=0 and ||u||<=sqrt(d). For head
q=w' (gamma*u+beta)+b, define m=w'beta+b and c=w*gamma-mean(w*gamma).
Then m-sqrt(d)||c|| <= q <= m+sqrt(d)||c|| for every real input to the
LayerNorm layer. This is a parameter-dependent range, not a training contraction
proof or an accuracy guarantee. Floating point overflow outside finite inputs is
not covered by this real-arithmetic statement.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np


def bounds_for(module):
    net=module['value_net']
    last=max(int(k.split('_')[-1]) for k in net if k.startswith('Dense_'))
    layer=net[f'LayerNorm_{last-1}'];head=net[f'Dense_{last}']
    w=np.asarray(head['kernel'],np.float64)[...,0]
    beta=np.asarray(layer['bias'],np.float64);gamma=np.asarray(layer['scale'],np.float64)
    b=np.asarray(head['bias'],np.float64)[...,0]
    location=(w*beta).sum(-1)+b
    centered=w*gamma-(w*gamma).mean(-1,keepdims=True)
    radius=np.sqrt(w.shape[-1])*np.linalg.norm(centered,axis=-1)
    return dict(location=np.atleast_1d(location).tolist(),radius=np.atleast_1d(radius).tolist(),
                lower=np.atleast_1d(location-radius).tolist(),upper=np.atleast_1d(location+radius).tolist(),
                width=w.shape[-1])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root.resolve()
    paths=list((root/'runs').glob('*/checkpoint_*.pkl'))+list((root/'analysis/checkpoint_archive').glob('*/checkpoint_*.pkl'))
    seen=set();rows=[];model_records=[]
    for path in sorted(paths):
        stat=path.stat();inode=(stat.st_dev,stat.st_ino)
        if inode in seen:continue
        seen.add(inode)
        with path.open('rb') as f:saved=pickle.load(f)
        runname=path.parent.name;run=root/'runs'/runname
        conf=json.loads((run/'config.json').read_text());ident=conf['identity']
        params=saved['state']['agent']['network']['params']
        dataset=json.loads((run/'dataset.json').read_text());scale=max(abs(dataset['return_lower']),abs(dataset['return_upper']),1.)
        record=dict(run=runname,step=saved['step'],checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bounds={})
        for name in ('critic','target_critic','value'):
            key='modules_'+name
            if key not in params:continue
            result=bounds_for(params[key]);record['bounds'][name]=result
            for head,(m,r,lower,upper) in enumerate(zip(result['location'],result['radius'],result['lower'],result['upper'])):
                rows.append(dict(run=runname,environment=ident['environment'],kind=ident['kind'],profile=ident['profile'],
                    discount=ident['config']['discount'],expectile=ident['config']['expectile'],step=saved['step'],
                    module=name,head=head,location=m,radius=r,lower=lower,upper=upper,
                    location_over_return_scale=m/scale,radius_over_return_scale=r/scale,
                    width=result['width']))
        model_records.append(record)
    result=dict(time=time.time(),kind='parameter_dependent_global_output_range_in_real_arithmetic',
        formula='m=w dot beta+b; c=w*gamma-mean(w*gamma); radius=sqrt(width)*norm(c); range=[m-radius,m+radius]',
        proof='LayerNorm normalized channels have zero sum and squared norm <=width. Center the last-head coefficient because its constant component is orthogonal to the normalized channels; apply Cauchy-Schwarz.',
        caveats=['This is an instantaneous range determined by trained parameters, not a bound uniform over future optimizer updates.',
                 'The interval may include impossible positive returns even when sampled dataset Q values are negative.',
                 'The normalization epsilon only tightens the real-arithmetic norm bound. Finite precision and earlier arithmetic overflow are separate concerns.',
                 'A finite output range does not certify value accuracy, a small conditional residual, Bellman contraction or good policy performance.'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models=model_records)
    out=root/'analysis';(out/'layernorm_output_bounds.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    if rows:
        with (out/'layernorm_output_bounds.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(models=len(model_records),head_records=len(rows))))
    for record in model_records:
        if record['step']==300000 and 'scene' in record['run']:
            print(record['run'],record['bounds']['critic'],record['bounds'].get('value'))


if __name__=='__main__':main()
