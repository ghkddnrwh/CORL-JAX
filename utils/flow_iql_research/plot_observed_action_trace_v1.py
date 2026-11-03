"""Plot the frozen two-by-two backup trace; never interpret learned Q as truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    folder = root/'analysis/observed_action_backup_trace_v1'
    names = [('cubed', 'Cube-double'), ('ant', 'Antmaze-large'), ('pz3', 'Puzzle-3x3')]
    colors = dict(V2='#2368ad', finite='#df6a24')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    inputs, all_rows = {}, []
    for column, (short, label) in enumerate(names):
        path = folder/(short+'.json'); raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest(); records = json.loads(raw)['rows']
        all_rows.extend(dict(environment=short, **row) for row in records)
        for source in ('V2', 'finite'):
            rows = sorted([row for row in records if row['critic_source'] == source and row['proposal_source'] == 'V2'], key=lambda row:row['step'])
            x = [row['step']/1e6 for row in rows]
            axes[0, column].plot(x, [row['generated_Q_mean'] for row in rows], '-o', color=colors[source], lw=1.8,
                                 label=source+' critic, generated actions', markersize=4)
            axes[0, column].plot(x, [row['observed_Q_mean'] for row in rows], '--x', color=colors[source], lw=1.2,
                                 label=source+' critic, observed next action', markersize=5)
            axes[1, column].plot(x, [row['generated_Q_within_state_std_mean'] for row in rows], '-o', color=colors[source], lw=1.8, markersize=4)
        if short == 'cubed':
            axes[0, column].axhline(-2000, color='#666666', ls=':', lw=1)
            axes[0, column].text(.98, .07, 'Known lower return bound: -2000', transform=axes[0,column].transAxes,
                                 ha='right', fontsize=8, color='#555555')
        axes[0, column].set_title(label, fontsize=12)
        axes[0, column].set_ylabel('Mean target min-Q')
        axes[1, column].set_ylabel('Within-state Q std over 9 actions')
        axes[1, column].set_yscale('log'); axes[1, column].set_ylim(.01, 30)
        for row in range(2):
            axes[row, column].set_xticks([.1, .3, .8, 1.])
            axes[row, column].set_xlabel('Q updates (millions)')
            axes[row, column].grid(alpha=.18)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .055), frameon=False, fontsize=9)
    fig.suptitle('Observed and generated action values fall together; action contrast collapses in Cube-double', fontsize=13, y=.99)
    fig.text(.5, .012, '128 heldout next states. The same actual V2 BC proposals are held fixed across critic sources at each checkpoint.\nThese are learned Q predictions, not true Q errors or oracle conditional residuals; dataset states are not policy-visited states.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .145, 1, .955))
    for extension in ('png', 'pdf'):
        fig.savefig(folder/f'Q_scale_and_action_contrast_v1.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    with (folder/'all_conditions_v1.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    (folder/'figure_provenance_v1.json').write_text(json.dumps(dict(time=time.time(), inputs_sha256=inputs,
        figure_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_benchmark_evaluations_modified=True), indent=2))


if __name__ == '__main__':
    main()
