"""Plot the oracle diagnostic counterexample separately from OGBench results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);args=parser.parse_args()
    result=json.loads(args.result.read_text());assert result['status']=='complete'
    rows=[r for r in result['rows']if r['discount']==.999 and r['expectile']==.9]
    fig,(left,right)=plt.subplots(1,2,figsize=(10,4.5))
    colors=['#247ba0','#e07a35'];names=['Common successor offset','Heterogeneous successor bias']
    geometries=['common_successor_offset','heterogeneous_successor_bias'];offset=.17
    for i,geometry in enumerate(geometries):
        selected=[r for r in rows if r['geometry']==geometry]
        base=next(r for r in selected if r['horizon']==1)
        left.bar(np.arange(3)+(i-.5)*2*offset,
            [base[k]for k in ('value_residual_sup','conditional_score_sup','propagated_Q_bias_sup')],
            width=.31,color=colors[i],label=names[i])
        right.bar(np.arange(3)+(i-.5)*2*offset,[next(r['true_start_regret']for r in selected if r['horizon']==h)for h in (1,5,20)],
            width=.31,color=colors[i])
    left.set_yscale('log');left.set_ylim(5e-5,2)
    left.set_xticks(range(3),['V minus expectile','Conditional score','Propagated Q bias'])
    left.set_ylabel('Sup norm (log scale)');left.set_title('Equal norms, different spatial errors')
    left.legend(fontsize=8,loc='upper left')
    right.set_xticks(range(3),['H=1','H=5 trace','H=20 trace']);right.set_ylim(0,1.13)
    right.set_ylabel('True discounted regret at start');right.set_title('Trace preconditioning preserves the biased root')
    for axis in (left,right):
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    for i in range(3):right.text(i+.17,1.02,'0.999',ha='center',fontsize=9)
    fig.suptitle('Oracle counterexample: finite values and small residual norms do not certify policy quality',fontsize=11)
    fig.text(.5,.015,'Known three-state/two-action MDP; gamma=.999, tau=.9; fixed injected V errors; exact population expectations.\nBoth geometries stay within known return bounds. This is not an OGBench result or a diagnosis of its actual error mechanism.\nTraces can change fitting speed while retaining the systematic residual bias; neural/stochastic effects may differ.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.15,1,.93))
    for extension in ('png','pdf'):
        path=args.result.with_name(f'trace_residual_fixedpoint_oracle_v1.{extension}');assert not path.exists()
        fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
