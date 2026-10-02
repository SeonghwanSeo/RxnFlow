"""Small local QED pilot; reports raw rollout validity without retry filtering."""

import argparse
import hashlib
import json
import platform
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdFingerprintGenerator

from rxnflow.config import (
    Config,
    DataConfig,
    ModelConfig,
    RewardConfig,
    SubsamplingConfig,
    TrainingConfig,
)
from rxnflow.envs.chemistry.features import PROPERTY_NAMES, molecular_properties
from rxnflow.gflownet.types import State
from examples.qed import QEDReward
from rxnflow.trainer import RxnFlowTrainer


def evaluate(trainer: RxnFlowTrainer, count: int, seed: int, name: str) -> dict:
    # Evaluation must not advance the training action/subsampling generator.
    previous_rng = trainer.generator.get_state()
    trainer.generator.manual_seed(seed)
    started = time.perf_counter()
    try:
        trajectories = trainer.sampling_policy.rollouts(count, analyze_backward=False, beta=torch.full((count,), trainer.config.reward.beta[1][0]), preferences=torch.ones(count, 1))
    finally:
        trainer.generator.set_state(previous_rng)
    elapsed = time.perf_counter() - started
    valid = [trajectory for trajectory in trajectories if trajectory.valid]
    molecules = [Chem.MolFromSmiles(trajectory.final_smiles) for trajectory in valid]
    rewards = [QED.qed(mol) for mol in molecules]
    # Fingerprints here are only an offline diversity metric, never policy input.
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(mol) for mol in molecules]
    similarities = [
        similarity
        for i, fingerprint in enumerate(fingerprints)
        for similarity in DataStructs.BulkTanimotoSimilarity(
            fingerprint, fingerprints[:i]
        )
    ]
    records = [
        {
            "valid": trajectory.valid,
            "smiles": trajectory.final_smiles,
            "invalid_reason": trajectory.invalid_reason,
            "trajectory": [
                {
                    **trainer.env.action_to_dict(step.action),
                    "product_smiles": step.product_smiles,
                }
                for step in trajectory.steps
            ],
        }
        for trajectory in trajectories
    ]
    (trainer.output_dir / f"{name}_trajectories.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    return {
        "attempts": count,
        "valid": len(valid),
        "valid_fraction": len(valid) / count,
        "unique_valid": len({trajectory.final_smiles for trajectory in valid}),
        "unique_fraction_among_valid": len(
            {trajectory.final_smiles for trajectory in valid}
        )
        / len(valid)
        if valid
        else None,
        "mean_qed_valid": float(np.mean(rewards)) if rewards else None,
        "mean_qed_all_attempts": sum(rewards) / count,
        "mean_pairwise_tanimoto_distance": 1 - float(np.mean(similarities))
        if similarities
        else None,
        "selected_action_kinds": dict(
            Counter(
                step.action.action_type.name
                for trajectory in trajectories
                for step in trajectory.steps
            )
        ),
        "selected_reactions": dict(
            Counter(
                step.action.reaction
                for trajectory in trajectories
                for step in trajectory.steps
                if step.action.reaction is not None
            )
        ),
        # Additive budgets do not guarantee exact terminal bounds. Measure this
        # approximation explicitly rather than silently filtering the outputs.
        "terminal_property_exceedances": {
            name: sum(
                int(molecular_properties(mol)[PROPERTY_NAMES.index(name)] > limit)
                for mol in molecules
            )
            for name, limit in trainer.config.property_penalty.items()
        },
        "reaction_counts_valid": dict(
            Counter(len(trajectory.steps) - 1 for trajectory in valid)
        ),
        "invalid_reasons": dict(
            Counter(
                trajectory.invalid_reason
                for trajectory in trajectories
                if not trajectory.valid
            )
        ),
        "seconds": elapsed,
    }


@torch.no_grad()
def action_diagnostics(trainer: RxnFlowTrainer) -> dict:
    """Measure competing Uni/Bi logit scales on fixed synthon states."""
    previous_rng = trainer.generator.get_state()
    trainer.generator.manual_seed(17)
    report = {}
    try:
        for smiles in ("[11*]CC", "[33*]NCC", "[1*]NCC", "[3*]C"):
            candidates = trainer.sampling_policy.forward(
                [State.from_smiles(smiles)],
                torch.tensor([trainer.config.reward.beta[1][0]], device=trainer.device),
                torch.ones(1, 1, device=trainer.device),
            )
            groups = candidates.action_subspaces
            weighted = [g.logits[0] + trainer.config.subsampling.importance_temp * g.log_importance for g in groups]
            log_partition = torch.logsumexp(torch.cat(weighted), 0)
            by_kind = {}
            for action_type in {g.action_type for g in groups}:
                values = torch.cat([g.logits[0] for g in groups if g.action_type == action_type])
                weights = torch.cat([w for g, w in zip(groups, weighted, strict=True) if g.action_type == action_type])
                valid = torch.isfinite(values)
                by_kind[action_type.name] = {
                    "sampled_actions": int(valid.sum()),
                    "logit_min": float(values[valid].min().cpu()) if valid.any() else None,
                    "logit_max": float(values[valid].max().cpu()) if valid.any() else None,
                    "probability_mass": float((weights - log_partition).exp().sum().cpu()),
                }
            report[smiles] = by_kind
    finally:
        trainer.generator.set_state(previous_rng)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("output-dir must be a new path")
    config = Config(
        data=DataConfig(env_dir=str(args.env_dir.resolve()), max_atoms=50),
        reward=RewardConfig(beta=("fixed", [8.0])),
        property_penalty={"mw": 500.0},
        subsampling=SubsamplingConfig(sampling_ratio=0.002, min_sampling=10),
        model=ModelConfig(num_emb=64, num_heads=4, num_layers=2),
        training=TrainingConfig(
            steps=args.steps,
            batch_size=4,
            replay_batch_size=4,
            replay_capacity=256,
            learning_rate=1e-3,
            ema_decay=0.9,
            checkpoint_every=args.steps,
            log_every=5,
            retrosynthesis_workers=4,
        ),
        output_dir=str(args.output_dir.resolve()),
        seed=args.seed,
        device="auto",
    )
    started = time.perf_counter()
    trainer = RxnFlowTrainer(config, QEDReward())
    report = {
        "host": platform.node(),
        "device": str(trainer.device),
        "environment_load_and_model_seconds": time.perf_counter() - started,
        "environment": trainer.env.signature,
        "config": config.to_file_dict(),
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(
                (Path(__file__).resolve().parents[2] / "src/rxnflow").rglob("*.py")
            )
        },
    }
    try:
        libraries = trainer.env.blocks.values()
        report["library_tensor_bytes"] = sum(
            tensor.numel() * tensor.element_size()
            for library in libraries
            for tensor in (library.properties, library.fingerprints, library.heavy_atoms)
        )
        # Equal FP+property rows within one library receive identical block
        # embeddings. Quantify this finite-representation limit explicitly.
        collisions = {}
        for name, library in trainer.env.blocks.items():
            features = [
                fingerprint.numpy().tobytes() + properties.numpy().tobytes()
                for fingerprint, properties in zip(
                    library.fingerprints, library.properties, strict=True
                )
            ]
            duplicate_rows = len(features) - len(set(features))
            if duplicate_rows:
                collisions[name] = duplicate_rows
        report["duplicate_block_feature_rows_by_library"] = collisions
        report["before"] = evaluate(trainer, args.count, 11, "before")
        report["action_diagnostics_before"] = action_diagnostics(trainer)
        print(json.dumps({"before": report["before"]}), flush=True)
        started = time.perf_counter()
        report["checkpoint"] = str(trainer.run())
        report["training_seconds"] = time.perf_counter() - started
        report["after"] = evaluate(trainer, args.count, 11, "after")
        report["action_diagnostics_after"] = action_diagnostics(trainer)
        report["learned_logit_scale"] = float(trainer.model.logit_scale(trainer.model.encode_cond(torch.tensor([trainer.config.reward.beta[1][0]], device=trainer.device), torch.ones(1, 1, device=trainer.device)))[0].detach().cpu())
        report["peak_rss_mib_main_process"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        if trainer.device.type == "cuda":
            report["peak_cuda_allocated_mib"] = (
                torch.cuda.max_memory_allocated(trainer.device) / 1024**2
            )
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {"after": report["after"], "training_seconds": report["training_seconds"]}
            ),
            flush=True,
        )
    finally:
        trainer.env.retro_analyzer.close()


if __name__ == "__main__":
    main()
