"""Checkpoint-backed local RxnFlow sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rxnflow.config import Config
from rxnflow.core.compatibility import (
    check_library_compatibility,
    check_model_compatibility,
)
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
        check_model_compatibility(ckpt["rxnflow_version"])
        check_library_compatibility(ckpt["environment"]["rxnflow_version"])
        config = Config.from_dict(ckpt["config"])

        if env_dir is not None:
            config.env_dir = str(env_dir)
        self.config = config
        self.objectives = tuple(ckpt["objectives"])
        self.device = torch.device(device)
        self.env = SynthesisEnv(
            config.env_dir,
            config.generation.max_atoms,
            config.generation.max_reactions,
            config.generation.min_synthons,
            config.generation.max_synthons,
            config.generation.min_reactions,
            config.property_penalty,
            0,
        )
        if ckpt["environment"]["synthesis"] != self.env.signature["synthesis"]:
            raise ValueError(
                "reaction, synthon or exclusion definitions differ from the checkpoint"
            )
        self.model = (
            RxnFlowModel(
                self.env,
                config.model,
                len(self.objectives),
            )
            .to(self.device)
            .eval()
        )
        self.model.load_state_dict(ckpt["sampling_model"])
        del ckpt  # Release optimizer/replay data when loading a full checkpoint.
        self.rng = np.random.default_rng(0)
        self.policy = RxnFlowPolicy(self.env, self.model, config, self.device, self.rng)

    def _result(self, trajectory: Trajectory) -> SamplingResult:
        actions = []
        for step in trajectory.steps:
            actions.append(
                {
                    **self.env.action_to_dict(step.action),
                    "product_smiles": step.product_smiles,
                }
            )
        return SamplingResult(
            smiles=trajectory.final_smiles,
            traj=actions,
            metadata={
                "beta": trajectory.beta,
                "preference": trajectory.preference,
            },
        )

    def sample(
        self,
        num_samples: int,
        *,
        beta: tuple[str, list[float]] | None = None,
        preference: tuple[str, list[float]] | None = None,
        batch_size: int = 64,
        softmax_temperature: float = 1.0,
        seed: int | None = None,
    ) -> list[SamplingResult]:
        """Attempt num_samples trajectories and return the valid results.

        beta and preference are (distribution, parameters) tuples. Omitted
        conditions use their training settings. batch_size is independent of the
        training batch size; softmax_temperature scales the policy softmax.
        A supplied seed resets sampling RNGs; None continues their current state.
        """
        # 1. Resolve requested conditions and reset sampling RNGs when seeded.
        if beta is None:
            beta = self.config.reward.beta
        if preference is None:
            preference = self.config.reward.moo_preference
        if (preference[0] == "none") != (
            self.config.reward.moo_preference[0] == "none"
        ):
            raise ValueError("preference 'none' must match the checkpoint setting")
        conditions = ConditionSampler(
            beta,
            preference,
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
        trained_preference = self.config.reward.moo_preference
        if trained_preference[0] == "fixed":
            if preference[0] != "fixed":
                raise ValueError("fixed-preference training requires fixed preference")
            trained = np.asarray(trained_preference[1])
            requested = np.asarray(preference[1])
            if not np.allclose(trained / trained.sum(), requested / requested.sum()):
                raise ValueError(
                    "sampling preference differ from fixed training weights"
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
                    "preference": preference,
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
        # 2. Sample a fixed number of trajectories without retrying invalid ones.
        trajectories: list[Trajectory] = []
        for start in range(0, num_samples, batch_size):
            batch_count = min(batch_size, num_samples - start)
            sampled_beta, weights = conditions.sample(batch_count)
            batch = self.policy.sample_from_model(
                batch_count,
                softmax_temperature,
                0.0,
                analyze_backward=False,
                beta=sampled_beta,
                preference=weights,
            )
            trajectories.extend(trajectory for trajectory in batch if trajectory.valid)
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
                for index, result in enumerate(results):
                    handle.write(f"{result.smiles}\tsample_{index}\n")
        elif format_name == "json":
            with destination.open("w", encoding="utf-8") as handle:
                json.dump([result.to_dict() for result in results], handle, indent=2)
                handle.write("\n")
        elif format_name == "jsonl":
            with destination.open("w", encoding="utf-8") as handle:
                for result in results:
                    handle.write(json.dumps(result.to_dict()) + "\n")
        else:
            raise ValueError("output format must be smi, json, or jsonl")
