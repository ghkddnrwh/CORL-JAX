"""Approximate evaluation uncertainty in all matched pre-existing seed pairs.

Resample reset IDs as clusters, keeping their three checkpoint outcomes paired.
These exploratory intervals condition on the trained policies; they do not
estimate uncertainty over new training seeds or identify a causal mechanism.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from utils.flow_iql_research.analyze_discount_0995_n9_v2 import csv_file, sha, atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();plan=json.loads((root/'registration_v1.json').read_text())
    rows=[]; evidence={}; rng=np.random.default_rng(13072026)
    for env in plan['environments']:
        jobs=sorted([j for j in plan['reused_jobs'] if j['environment']==env],key=lambda j:j['seed'])
        for mode in plan['deployment_modes']:
            values={}
            for job in jobs:
                scores=[]
                for step in plan['primary_evaluation_steps']:
                    path=Path(job['evaluation_directory'])/f'{step}_{mode}.json'
                    v=json.loads(path.read_text())
                    assert v['first_episode_seed']==42 and len(v['episode_successes'])==50
                    assert v['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
                    evidence[str(path)]=sha(path);scores.append(v['episode_successes'])
                values[job['seed']]=np.asarray(scores,float).mean(0)*100
            for a,b in itertools.combinations(values,2):
                difference=values[b]-values[a]
                boot=difference[rng.integers(0,50,size=(20000,50))].mean(1)
                lo,hi=np.quantile(boot,[.025,.975])
                rows.append(dict(environment=env,deployment=mode,seed_A=a,seed_B=b,
                    A_mean_pct=float(values[a].mean()),B_mean_pct=float(values[b].mean()),
                    difference_B_minus_A_pp=float(difference.mean()),
                    conditional_reset_bootstrap_95_low=float(lo),conditional_reset_bootstrap_95_high=float(hi),
                    reset_clusters=50,checkpoints_per_cluster=3,
                    interpretation='Exploratory evaluation-only uncertainty; fixed policies, not training-seed uncertainty; no multiplicity adjustment.'))
    csv_file(root/'analysis/prior_seed_pairs_evaluation_uncertainty.csv',rows)
    atomic(root/'analysis/prior_seed_pairs_evaluation_uncertainty_audit.json',dict(
        status='PASS',source_sha256=sha(__file__),all_available_matched_seed_pairs=True,
        pairs=len(rows),bootstrap_replicates=20000,evidence_sha256=evidence))
    for row in rows:
        if row['deployment']=='lean': print(json.dumps(row),flush=True)


if __name__=='__main__': main()
