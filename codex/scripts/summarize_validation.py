"""Summarize completed QED runs and structural diversity from saved raw samples."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def sample_statistics(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    valid = [row for row in rows if row['valid']]
    molecules = [Chem.MolFromSmiles(row['smiles']) for row in valid]
    assert all(mol is not None for mol in molecules)
    fingerprints = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True,
    ).GetFingerprints(molecules)
    # Include duplicate samples: repeated structures should reduce measured
    # diversity. No quadratic matrix is retained in memory.
    similarity_sum = 0.0
    pair_count = 0
    for index, fingerprint in enumerate(fingerprints):
        values = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[:index])
        similarity_sum += sum(values)
        pair_count += len(values)
    scaffolds = Counter(
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol) for mol in molecules
    )
    acyclic = scaffolds.pop('', 0)
    count = len(valid)
    result = {
        'attempts': len(rows), 'valid': count,
        'valid_fraction': count / len(rows),
        'unique_molecules': len({row['smiles'] for row in valid}),
        'unique_fraction': len({row['smiles'] for row in valid}) / max(count, 1),
        'nonempty_murcko_scaffolds': len(scaffolds),
        'acyclic_fraction': acyclic / max(count, 1),
        'most_common_nonempty_scaffolds': scaffolds.most_common(5),
        'internal_diversity': 1 - similarity_sum / pair_count if pair_count else None,
        'mean_qed_valid': statistics.mean(row['qed'] for row in valid) if valid else None,
        'mean_qed_all': sum(row['qed'] for row in valid) / len(rows),
        'reaction_counts': dict(Counter(row['reactions'] for row in valid)),
        'qed_by_reaction_count': {
            reactions: statistics.mean(row['qed'] for row in valid if row['reactions'] == reactions)
            for reactions in sorted({row['reactions'] for row in valid})
        },
        'invalid_reasons': dict(Counter(row['reason'] for row in rows if not row['valid'])),
        'property_violation_counts': {
            name: sum(row[name] > bound for row in valid)
            for name, bound in [('mw', 500), ('hba', 10), ('hbd', 5)]
        },
        'all_three_pass_fraction': sum(
            row['mw'] <= 500 and row['hba'] <= 10 and row['hbd'] <= 5 for row in valid
        ) / max(count, 1),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    results = {}
    for run in args.runs:
        completed = json.loads((run / 'completed.json').read_text())
        assert completed['completed_steps'] == 5000
        logs = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
        assert [row['step'] for row in logs] == list(range(1, 5001))
        checkpoint = torch.load(completed['checkpoint'], map_location='cpu', weights_only=False)
        assert checkpoint['step'] == 5000
        # The final replay contains recent training trajectories with exploration.
        # Keep its reaction mix separate from the independent evaluation samples.
        replay = [trajectory for trajectory in checkpoint['replay']['items'] if trajectory['valid']]
        results[run.name] = {
            'training_seconds': completed['training_seconds'],
            'peak_allocated_mib': completed['peak_allocated_mib'],
            'step_seconds_median': statistics.median(row['step_seconds'] for row in logs),
            'rollout_seconds_median': statistics.median(row['rollout_seconds'] for row in logs),
            'before': sample_statistics(run / 'evaluation_before.jsonl'),
            'after': sample_statistics(run / 'evaluation_after.jsonl'),
            'final_replay': {
                'valid_trajectories': len(replay),
                'first_library_counts': dict(Counter(t['steps'][0]['action']['block_type'] for t in replay)),
                'reaction_counts': dict(Counter(len(t['steps']) - 1 for t in replay)),
                'reaction_use_counts': dict(Counter(
                    step['action']['reaction'] for trajectory in replay for step in trajectory['steps'][1:]
                )),
                'common_reaction_sequences': Counter(
                    tuple(step['action']['reaction'] for step in trajectory['steps'][1:])
                    for trajectory in replay
                ).most_common(10),
            },
        }
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
