"""Checkpoint-backed local RxnFlow sampling."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from rxnflow import __version__
from rxnflow.config import Config
from rxnflow.core.types import SamplingResult, Trajectory
from rxnflow.envs.env import SynthesisEnv
from rxnflow.gflownet.conditioning import ConditionSampler
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.models import RxnFlowModel


class RxnFlowSampler:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        env_dir: str | Path | None = None,
    ) -> None:
        """Load EMA weights from a full checkpoint or extracted model.

        device selects the sampling device; env_dir overrides the saved catalog path.
        """
        # Load once on CPU: sampling needs only EMA weights, not the optimizer
        # and replay tensors copied to the GPU with the entire checkpoint.
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ckpt["rxnflow_version"] != __version__:
            raise ValueError(
                f"checkpoint was created by RxnFlow {ckpt['rxnflow_version']!r}"
            )
        config = Config.from_dict(ckpt["config"])

        if env_dir is not None:
            config.data.env_dir = str(env_dir)
        self.config = config
        self.objectives = tuple(ckpt["objectives"])
        self.device = torch.device(device)
        self.env = SynthesisEnv(
            config.data.env_dir,
            config.data.max_atoms,
            config.generation.max_reactions,
            0,
            config.property_penalty,
            min_synthons=config.generation.min_synthons,
            max_synthons=config.generation.max_synthons,
            min_reactions=config.generation.min_reactions,
        )
        if ckpt["templates"] != self.env.templates:
            raise ValueError(
                "reaction, synthon or exclusion definitions differ from the checkpoint"
            )
        self.model = (
            RxnFlowModel(
                self.env,
                config.model,
                len(self.objectives),
                preference_conditioning=config.reward.moo_preferences[0] != "none",
            )
            .to(self.device)
            .eval()
        )
        self.model.load_state_dict(ckpt["sampling_model"])
        del ckpt  # Release optimizer/replay data when loading a full checkpoint.
        self.rng = np.random.default_rng(0)
        self.policy = RxnFlowPolicy(self.env, self.model, config, self.device, self.rng)

    def _result(self, trajectory: Trajectory) -> SamplingResult:
        actions = [
            {
                **self.env.action_to_dict(step.action),
                "product_smiles": step.product_smiles,
            }
            for step in trajectory.steps
        ]
        return SamplingResult(
            smiles=trajectory.final_smiles,
            traj=actions,
            metadata={
                "beta": trajectory.beta,
                "preferences": trajectory.preferences,
            },
        )

    def sample(
        self,
        num_samples: int,
        *,
        beta: tuple[str, list[float]] | None = None,
        preferences: tuple[str, list[float]] | None = None,
        batch_size: int = 64,
        softmax_temperature: float = 1.0,
        seed: int | None = None,
    ) -> list[SamplingResult]:
        """Generate num_samples valid trajectories, retaining duplicate molecules.

        beta and preferences are (distribution, parameters) tuples. Omitted
        conditions use their training settings. batch_size is independent of the
        training batch size; softmax_temperature scales the policy softmax.
        A supplied seed resets sampling RNGs; None continues their current state.
        """
        # 1. Resolve requested conditions and reset sampling RNGs when seeded.
        if beta is None:
            beta = self.config.reward.beta
        if preferences is None:
            preferences = self.config.reward.moo_preferences
        if (preferences[0] == "none") != (
            self.config.reward.moo_preferences[0] == "none"
        ):
            raise ValueError("preferences must match the checkpoint's conditioning mode")
        conditions = ConditionSampler(
            beta,
            preferences,
            len(self.objectives),
            self.config.reward.moo_scalarization,
        )
        # Fixed training conditions support only the same value at inference.
        # Variable beta supports fixed values or subranges inside its training range.
        trained_beta = self.config.reward.beta
        if trained_beta[0] == "fixed":
            if beta != trained_beta:
                raise ValueError("fixed-beta training requires the same fixed beta")
        elif min(beta[1]) < trained_beta[1][0] or max(beta[1]) > trained_beta[1][1]:
            raise ValueError("sampling beta must stay within the training range")
        trained_preferences = self.config.reward.moo_preferences
        if trained_preferences[0] == "fixed":
            if preferences[0] != "fixed":
                raise ValueError("fixed-preference training requires fixed preferences")
            trained = np.asarray(trained_preferences[1])
            requested = np.asarray(preferences[1])
            if not np.allclose(trained / trained.sum(), requested / requested.sum()):
                raise ValueError(
                    "sampling preferences differ from fixed training weights"
                )
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if num_samples <= 0:
            raise ValueError("sample count must be positive")
        if seed is not None:
            self.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
            torch.manual_seed(seed)  # CPU conditions and device-side categorical draws.
        if softmax_temperature <= 0:
            raise ValueError("softmax temperature must be positive")
        print(
            "Training reward config: "
            + json.dumps(self.config.to_file_dict()["reward"], sort_keys=True),
            flush=True,
        )
        print(
            "Sampling settings: "
            + json.dumps(
                {
                    "num_samples": num_samples,
                    "batch_size": batch_size,
                    "beta": beta,
                    "preferences": preferences,
                    "moo_scalarization": self.config.reward.moo_scalarization,
                    "softmax_temperature": softmax_temperature,
                    "seed": seed,
                    "device": str(self.device),
                    "env_dir": str(self.env.env_dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        # 2. Generate until enough valid terminal trajectories or the attempt limit.
        trajectories: list[Trajectory] = []
        attempts = 0
        maximum_attempts = max(100, num_samples * 100)
        while len(trajectories) < num_samples and attempts < maximum_attempts:
            batch_count = min(
                batch_size,
                num_samples - len(trajectories),
                maximum_attempts - attempts,
            )
            sampled_beta, weights = conditions.sample(batch_count)
            batch = self.policy.sample_from_model(
                batch_count,
                softmax_temperature,
                0.0,
                analyze_backward=False,
                beta=sampled_beta,
                preferences=weights,
            )
            attempts += batch_count
            trajectories.extend(trajectory for trajectory in batch if trajectory.valid)
        if len(trajectories) != num_samples:
            raise RuntimeError(
                f"generated only {len(trajectories)} valid samples in {maximum_attempts} attempts"
            )
        return [self._result(trajectory) for trajectory in trajectories]

    @staticmethod
    def write(
        results: list[SamplingResult],
        path: str | Path,
        *,
        output_format: str | None = None,
    ) -> None:
        destination = Path(path)
        format_name = (output_format or destination.suffix.lstrip(".")).lower()
        if format_name == "smi":
            with destination.open("w", encoding="utf-8") as handle:
                for result in results:
                    handle.write(result.smiles + "\n")
        elif format_name == "csv":
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "smiles",
                        "traj",
                        "beta",
                        "preferences",
                    ],
                )
                writer.writeheader()
                for result in results:
                    writer.writerow(
                        {
                            "smiles": result.smiles,
                            "traj": json.dumps(result.traj, separators=(",", ":")),
                            "beta": result.metadata["beta"],
                            "preferences": json.dumps(
                                result.metadata["preferences"], separators=(",", ":")
                            ),
                        }
                    )
        elif format_name == "json":
            with destination.open("w", encoding="utf-8") as handle:
                json.dump([result.to_dict() for result in results], handle, indent=2)
                handle.write("\n")
        else:
            raise ValueError("output format must be smi, csv, or json")
