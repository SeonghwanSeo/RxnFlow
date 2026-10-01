"""Checkpoint-backed local RxnFlow sampling."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from rxnflow._version import __version__
from rxnflow.config import Config
from rxnflow.envs import SynthesisEnv
from rxnflow.gflownet.runtime import PolicyRuntime, resolve_device, trajectory_sample
from rxnflow.models import RxnFlowModel
from rxnflow.reward import RewardFunction, SampleFilter, evaluate_rewards
from rxnflow.types import SamplingResult, Trajectory


class RxnFlowSampler:
    def __init__(
        self,
        config: Config,
        checkpoint: str | Path,
        reward: RewardFunction | None = None,
        sample_filter: SampleFilter | None = None,
    ):
        config.validate()
        self.config = config
        self.reward = reward
        self.sample_filter = sample_filter
        self.device = resolve_device(config.device)
        self.env = SynthesisEnv(
            config.data.env_dir,
            config.data.max_atoms,
            config.generation.min_reactions,
            config.generation.max_reactions,
            0,
            config.property_penalty,
        )
        self.model = RxnFlowModel(self.env, config.model).to(self.device).eval()
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if payload.get("rxnflow_version") != __version__:
            raise ValueError(
                f"checkpoint was created by RxnFlow {payload.get('rxnflow_version')!r}; "
                f"this installation is RxnFlow {__version__}"
            )
        checkpoint_config = dict(payload["config"])
        runtime_config = config.to_dict()
        checkpoint_config.pop("device", None)
        runtime_config.pop("device", None)
        if checkpoint_config != runtime_config:
            raise ValueError(
                "sampling configuration differs from the resolved checkpoint configuration"
            )
        if payload.get("environment") != self.env.signature:
            raise ValueError(
                "prepared environment differs from the checkpoint environment"
            )
        self.model.load_state_dict(payload["sampling_model"])
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.runtime = PolicyRuntime(
            self.env, self.model, config, self.device, self.generator
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        reward: RewardFunction | None = None,
        device: str | None = None,
        sample_filter: SampleFilter | None = None,
    ) -> RxnFlowSampler:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("rxnflow_version") != __version__:
            raise ValueError(
                f"checkpoint was created by RxnFlow {payload.get('rxnflow_version')!r}; "
                f"this installation is RxnFlow {__version__}"
            )
        config = Config.from_dict(payload["config"])
        if device is not None:
            config.device = device
        return cls(config, checkpoint, reward, sample_filter)

    def _result(self, trajectory: Trajectory) -> SamplingResult:
        actions = [self.env.action_to_dict(step.action) for step in trajectory.steps]
        intermediates = [step.product_smiles for step in trajectory.steps]
        return SamplingResult(
            smiles=trajectory.final_smiles,
            trajectory=actions,
            intermediates=intermediates,
            metadata={"valid": trajectory.valid},
        )

    def sample(
        self,
        count: int,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> list[SamplingResult]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        if seed is not None:
            self.generator.manual_seed(seed)
        sampling_temperature = (
            self.config.training.sampling_temperature
            if temperature is None
            else temperature
        )
        if sampling_temperature <= 0:
            raise ValueError("temperature must be positive")
        trajectories: list[Trajectory] = []
        attempts = 0
        maximum_attempts = max(100, count * 100)
        while len(trajectories) < count and attempts < maximum_attempts:
            trajectory = self.runtime.rollout(
                sampling_temperature, 0.0, analyze_backward=False
            )
            attempts += 1
            if trajectory.valid:
                trajectories.append(trajectory)
        if len(trajectories) != count:
            raise RuntimeError(
                f"generated only {len(trajectories)} valid samples in {maximum_attempts} attempts"
            )
        results = [self._result(trajectory) for trajectory in trajectories]
        if self.reward is not None and results:
            values, metrics = evaluate_rewards(
                self.reward,
                [trajectory_sample(value) for value in trajectories],
                self.sample_filter,
            )
            for result, value in zip(results, values, strict=True):
                result.reward = value
                result.metadata.update(metrics)
        return results

    @staticmethod
    def write(
        results: list[SamplingResult], path: str | Path, output_format: str | None = None
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
                        "trajectory",
                        "intermediates",
                    ],
                )
                writer.writeheader()
                for result in results:
                    writer.writerow(
                        {
                            "smiles": result.smiles,
                            "trajectory": json.dumps(
                                result.trajectory, separators=(",", ":")
                            ),
                            "intermediates": json.dumps(
                                result.intermediates, separators=(",", ":")
                            ),
                        }
                    )
        elif format_name == "json":
            with destination.open("w", encoding="utf-8") as handle:
                json.dump([result.to_dict() for result in results], handle, indent=2)
                handle.write("\n")
        else:
            raise ValueError("output format must be smi, csv, or json")
