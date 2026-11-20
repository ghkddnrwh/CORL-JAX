"""Recompute the old selected benchmark from all three requested evaluations."""
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    old=Path('logs/diagnosis/propagation_research_20261025')
    out=Path('logs/diagnosis/flow_iql_research_20261026/analysis')
    ap=old/'analysis/selected_benchmark_audit.json';audit=json.loads(ap.read_text());rows=[]
    for r in audit['rows']:
        if r['family'] not in ('candidate','original_baseline'):continue
        p=old/'runs'/r['run']/'evaluations.jsonl'
        es=[json.loads(s) for s in p.read_text().splitlines()];scores=[]
        for step in (800000,900000,1000000):
            matches=[e for e in es if e['step']==step and e['n_episodes']==50 and e['eval_device']=='cpu']
            assert len(matches)==1,(r['run'],step)
            e=matches[0];assert len(e['episode_successes'])==50
            assert set(e['episode_successes'])<={0.,1.}
            assert abs(statistics.mean(e['episode_successes'])-e['success_rate'])<1e-12
            scores.append(e['success_rate'])
        rows.append(dict(family=r['family'],environment=r['environment'],seed=r['seed'],run=r['run'],
            success_800k=scores[0],success_900k=scores[1],success_1m=scores[2],
            last3_mean=statistics.mean(scores),evaluation_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    macros={}
    for family in ('candidate','original_baseline'):
        fr=[r for r in rows if r['family']==family];seeds=sorted({r['seed'] for r in fr});sm={}
        for seed in seeds:
            sr=[r for r in fr if r['seed']==seed];assert len(sr)==9
            sm[str(seed)]=statistics.mean(r['last3_mean'] for r in sr)
        macros[family]=dict(per_seed=sm,last3_macro=statistics.mean(sm.values()),
            final1m_macro=statistics.mean(r['success_1m'] for r in fr),seeds=seeds,
            sample_SD=statistics.stdev(sm.values()) if len(sm)>1 else None,
            actual_successes=int(round(sum(r['last3_mean']*150 for r in fr))),
            total_episodes=len(fr)*150)
    result=dict(protocol='Mean(800k,900k,1M) equally, then equal task/seed weights; existing primary CPU50 logs; historical scores preserved',
        rows=rows,macros=macros,input_audit_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveats=['Old primary evaluation protocol differs from the new fixed independent episode seed protocol. No scores are replaced by re-evaluations.',
                 'Candidate has three training seeds; fresh original native reference has one.',
                 'Three repeated checkpoints of one training run are not three independent training seeds.'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'previous_results_last3_correction.json').write_text(json.dumps(result,indent=2))
    with (out/'previous_results_last3_correction.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(macros,indent=2))


if __name__=='__main__':main()
