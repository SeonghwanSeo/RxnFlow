"""QED optimization with per-step metrics and unbiased raw-rollout evaluations."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch
from rdkit.Chem import QED, Descriptors, Lipinski

from rxnflow.config import Config
from rxnflow.gflownet.policy import SynthesisPolicy
from rxnflow.reward import QEDReward
from rxnflow.trainer import RxnFlowTrainer


class ValidationQEDReward(QEDReward):
    """Reward stays pure QED; final-property statistics are diagnostics only."""

    def score(self, molecules):
        scores = super().score(molecules)
        properties = [
            (Descriptors.MolWt(mol), Lipinski.NumHAcceptors(mol), Lipinski.NumHDonors(mol))
            for mol in molecules
        ]
        count = len(molecules)
        self.last_metrics = {
            "mean_qed_valid": float(scores.sum()) / max(1, count),
            "lipinski_fraction": sum(mw <= 500 and hba <= 10 and hbd <= 5 for mw, hba, hbd in properties) / max(1, count),
        }
        for index, name in enumerate(("mw", "hba", "hbd")):
            self.last_metrics[f"mean_{name}"] = sum(p[index] for p in properties) / max(1, count)
        return scores

    def metrics(self):
        return self.last_metrics


def evaluate(trainer, label, count, beta):
    # Isolate both CPU library draws and device-side Gumbel draws from training.
    policy = SynthesisPolicy(
        trainer.env, trainer.sampling_model, trainer.config, trainer.device,
        torch.Generator().manual_seed(1729),
    )
    rows = []
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        for start in range(0, count, trainer.config.training.batch_size):
            batch = policy.rollouts(
                min(trainer.config.training.batch_size, count - start),
                1.0, 0.0, analyze_backward=False,
                beta=torch.full((min(trainer.config.training.batch_size, count - start),), float(beta)),
                preferences=torch.ones(min(trainer.config.training.batch_size, count - start), 1),
            )
            for trajectory in batch:
                row = {
                    "smiles": trajectory.final_smiles, "valid": trajectory.valid,
                    "reason": trajectory.invalid_reason,
                    "reactions": max(0, len(trajectory.steps) - 1),
                }
                if trajectory.valid:
                    # Replay the full forward path to check that its recorded states
                    # and terminal molecule match actual reaction execution.
                    state = trainer.env.initial_state()
                    for step in trajectory.steps:
                        assert state.smiles == step.state.smiles
                        state = trainer.env.step(state, step.action)
                        assert state.smiles == step.product_smiles
                    assert state.terminated and state.smiles == trajectory.final_smiles
                    mol = state.mol
                    row.update(qed=QED.qed(mol), mw=Descriptors.MolWt(mol),
                               hba=Lipinski.NumHAcceptors(mol), hbd=Lipinski.NumHDonors(mol))
                rows.append(row)
    valid = [r for r in rows if r['valid']]
    summary = {
        'attempts': len(rows), 'valid': len(valid),
        'valid_fraction': len(valid) / len(rows),
        'unique': len({r['smiles'] for r in valid}),
        'mean_qed_valid': sum(r['qed'] for r in valid) / max(1, len(valid)),
        'mean_qed_all': sum(r['qed'] for r in valid) / len(rows),
        'lipinski_fraction': sum(r['mw'] <= 500 and r['hba'] <= 10 and r['hbd'] <= 5 for r in valid) / max(1, len(valid)),
        'reaction_counts': dict(Counter(r['reactions'] for r in valid)),
        'invalid_reasons': dict(Counter(r['reason'] for r in rows if not r['valid'])),
        'forward_paths_verified': len(valid),
    }
    path = trainer.output_dir / f'evaluation_{label}'
    path.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    path.with_suffix('.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--evaluation-samples', type=int, default=1024)
    parser.add_argument('--evaluation-beta', type=float, default=32.0)
    args = parser.parse_args()
    config = Config.from_file(args.config)
    trainer = RxnFlowTrainer(config, ValidationQEDReward())
    try:
        before = evaluate(trainer, 'before', args.evaluation_samples, args.evaluation_beta)
        print(json.dumps({'before': before}), flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        checkpoint = trainer.run()
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        after = evaluate(trainer, 'after', args.evaluation_samples, args.evaluation_beta)
        result = {
            'checkpoint': str(checkpoint), 'completed_steps': trainer.step,
            'training_seconds': elapsed, 'gpu': torch.cuda.get_device_name(),
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'before': before, 'after': after,
        }
        (trainer.output_dir / 'completed.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        if 'retro_analyzer' in trainer.env.__dict__:
            trainer.env.retro_analyzer.close()


if __name__ == '__main__':
    main()
