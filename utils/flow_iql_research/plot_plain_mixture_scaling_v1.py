"""Standalone scientific figure from the audited frozen scaling diagnostic."""
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/plain_mixture_scaling_CPU_v1'
    inp=area/'SCALING_AUDIT_v1.json';audit=json.loads(inp.read_text());assert audit['status']=='pass'
    png=area/'scaling_diagnosis_v1.png';pdf=area/'scaling_diagnosis_v1.pdf';assert not png.exists()and not pdf.exists()
    fig,axes=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[1.4,1]})
    for row,color in zip(audit['rows'],['#386cb0','#ef8a23']):
        points=[x for x in row['snapshots']if x['actual_Adam']>0]
        axes[0].plot([x['actual_Adam']for x in points],[x['heldout_NLL_per_dimension']for x in points],
            '-o',color=color,label=f"{row['training_states']:,} replay states",linewidth=2,markersize=4)
    axes[0].set_xscale('log');axes[0].set_xticks([1000,5000,20000,50000,100000],['1K','5K','20K','50K','100K'])
    axes[0].set_xlabel('Actor Adam updates');axes[0].set_ylabel('Held-out NLL / action dimension')
    axes[0].set_title('More replay coverage improves generalization');axes[0].grid(alpha=.2);axes[0].legend(fontsize=9)
    successes=[0,2,50];n=50;rates=np.asarray(successes)/n;low=[];high=[]
    for rate in rates:
        z=1.96;den=1+z*z/n;center=(rate+z*z/(2*n))/den
        half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den;low.append(center-half);high.append(center+half)
    axes[1].bar(range(3),rates,color=['#386cb0','#ef8a23','#7f7f7f'],width=.6,
        yerr=np.stack([rates-np.asarray(low),np.asarray(high)-rates]).clip(0),capsize=4)
    axes[1].set_xticks(range(3),['MDN 65K\n100K updates','MDN 524K\n100K updates','Best9 teacher\n1M checkpoint'])
    axes[1].set_ylabel('Puzzle success rate');axes[1].set_ylim(0,1.12);axes[1].set_title('Policy extraction remains unresolved')
    for i,(s,p,h)in enumerate(zip(successes,rates,high)):axes[1].text(i,h+.025,f'{s}/50',ha='center',fontsize=9)
    axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
    fig.suptitle('Frozen-teacher plain MDN: data coverage and long-horizon success',fontsize=12)
    fig.text(.5,.015,'CPU diagnostics on one task; not joint 1M or nine-task results. Bars: fixed-policy episode-only 95% Wilson intervals.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (area/'scaling_figure_provenance_v1.json').write_text(json.dumps(dict(input=str(inp),input_sha256=sha(inp),source_sha256=sha(__file__),png_sha256=sha(png),pdf_sha256=sha(pdf),no_training_seed_uncertainty_claim=True),indent=2))
    print(png)


if __name__=='__main__':main()
