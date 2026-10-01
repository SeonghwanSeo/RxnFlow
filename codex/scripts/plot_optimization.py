"""Plot raw and rolling-mean optimization metrics from local JSONL logs."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=100)
    args = parser.parse_args()
    panels = [
        ('mean_qed_valid', 'QED among valid molecules'),
        ('mean_reward', 'QED per rollout attempt'),
        ('valid_fraction', 'Valid rollout fraction'),
        ('unique_fraction', 'Unique fraction within valid batch'),
        ('lipinski_fraction', 'Final MW / HBA / HBD pass fraction'),
        ('loss', 'Trajectory-balance loss'),
        ('mean_reactions', 'Reaction count per attempt'),
        ('step_seconds', 'Seconds per update (excludes checkpoint I/O)'),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    colors = ['#2364aa', '#e07a21', '#2b9348']
    summary = {}
    for run_index, path in enumerate(args.runs):
        rows = [json.loads(line) for line in (path / 'training.jsonl').read_text().splitlines()]
        steps = np.array([r['step'] for r in rows])
        label = path.name.replace('_', ' ')
        color = colors[run_index % len(colors)]
        summary[path.name] = {'logged_steps': len(rows), 'last_step': int(steps[-1])}
        for axis, (key, title) in zip(axes.flat, panels, strict=True):
            values = np.array([r[key] for r in rows])
            axis.plot(steps, values, color=color, alpha=.12, linewidth=.6)
            width = min(args.window, len(values))
            rolling = np.convolve(values, np.ones(width) / width, mode='valid')
            axis.plot(steps[width - 1:], rolling, color=color, label=label, linewidth=1.6)
            axis.set_title(title, fontsize=10)
            axis.grid(alpha=.2)
            summary[path.name][key] = {
                'first_500_mean': float(values[:500].mean()),
                'last_500_mean': float(values[-500:].mean()),
            }
            if key == 'loss':
                axis.set_yscale('log')
            if key.endswith('fraction'):
                axis.set_ylim(-.03, 1.03)
    for axis in axes[-1]:
        axis.set_xlabel('Training step')
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f'QED optimization — raw batches + {args.window}-step mean', fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, .975))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf', '.svg'):
        figure.savefig(args.output.with_suffix(suffix), dpi=160)
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
