"""Plot frozen learned-value and finite-backup critic diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 7.3), sharex=True)
    metrics = [('expected_root9_MC_mean', 'Generated-action expectile level'),
               ('target_generated_action_std_mean', 'Q contrast across generated actions'),
               ('heldout_transition_backup_residual_MC9_RMS', 'Observed-transition backup residual RMS')]
    for row, (short, name) in enumerate([('hmmed', 'Humanoid medium'), ('hmlarge', 'Humanoid large')]):
        data = json.loads((args.directory/(short+'.json')).read_text())
        arrays = np.load(args.directory/(short+'.npz'))
        for source, color, label in [('V2', '#286b9b', 'Learned-V backup'), ('finite', '#d67d28', 'Finite-9 backup')]:
            points = sorted([entry for entry in data['rows'] if entry['critic_source'] == entry['proposal_source'] == source],
                            key=lambda entry:entry['step'])
            x = [point['step']/1000 for point in points]
            for column, (metric, title) in enumerate(metrics):
                axes[row, column].plot(x, [point[metric] for point in points], marker='o', color=color, label=label)
                axes[row, column].grid(alpha=.18)
                if row == 0: axes[row, column].set_title(title, fontsize=10)
            td = arrays['500000_'+source+'_'+source+'_transition_backup_residual']
            share = np.max(td**2)/np.sum(td**2)
            if source == 'finite':
                axes[row, 2].text(.03, .95 if row == 0 else .58, f'Largest transition: {100*share:.1f}%\nof residual squared sum at 500K',
                                  transform=axes[row, 2].transAxes, va='top', fontsize=8, color=color)
        axes[row, 0].axhline(-1000, color='#777777', linestyle=':', linewidth=1)
        axes[row, 0].set_ylabel(name+'\nValue (reward units)')
        axes[row, 1].set_yscale('log')
        axes[row, 1].set_ylabel('Conditional standard deviation')
        axes[row, 2].set_yscale('log')
        axes[row, 2].set_ylabel('RMS (reward units)')
        for ax in axes[row]: ax.set_xticks([100, 300, 500])
    for ax in axes[1]: ax.set_xlabel('Training step (K)')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('Frozen Humanoid diagnostics: value collapse and conditional errors', fontsize=13)
    fig.text(.5, .015, 'Same 64 validation queries at 100K / 300K / 500K; each curve uses its own actual critic and BC generator.\n'
             'Learned-Q diagnostics, not oracle interior value errors. Separate fresh 128-query train/validation replications are recorded in JSON.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .065, 1, .95])
    for extension in ('png', 'pdf'): fig.savefig(args.directory/('diagnostic_v1.'+extension), dpi=180)


if __name__ == '__main__': main()
