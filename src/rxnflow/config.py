"""Configuration for local RxnFlow preparation, training, and sampling."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

RUN_FIELDS = ("output_dir", "seed", "device")


def parse_distribution(value: str) -> tuple[str, list[float]]:
    """Parse a CLI/YAML condition. Bare uniform means simplex-uniform weights."""
    if not isinstance(value, str):
        raise ValueError("external condition specifications must be strings")
    value = value.strip()
    if value == "none":
        return "none", []
    if value == "uniform":
        return "dirichlet", [1.0]
    match = re.fullmatch(r"(fixed|uniform|dirichlet)\(([^)]+)\)", value)
    if match:
        return match[1], [float(x) for x in match[2].split(",")]
    return "fixed", [float(value)]


@dataclass
class DataConfig:
    """Prepared Enamine synthon environment and graph capacity settings."""

    env_dir: str = ""
    max_atoms: int = 50

    def validate(self) -> None:
        if self.max_atoms <= 0:
            raise ValueError("data.max_atoms must be positive")


@dataclass
class SubsamplingConfig:
    """Uniform synthon action-space sampling."""

    sampling_ratio: float = 0.05
    min_sampling: int = 50
    importance_temp: float = 1.0

    def validate(self) -> None:
        if not 0 < self.sampling_ratio <= 1:
            raise ValueError("subsampling.sampling_ratio must be in (0, 1]")
        if self.min_sampling <= 0:
            raise ValueError("subsampling.min_sampling must be positive")
        if self.importance_temp < 0:
            raise ValueError("subsampling.importance_temp must be non-negative")


@dataclass
class RewardConfig:
    """Reward transformation and constructor settings."""

    # Reward exponent and preference sampling specifications.
    beta: tuple[str, list[float]] = field(default_factory=lambda: ("fixed", [32.0]))
    preferences: tuple[str, list[float]] = field(default_factory=lambda: ("none", []))
    scalarization: Literal["sum", "mul"] = "mul"
    floor: float = 1e-4
    settings: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for spec in (self.beta, self.preferences):
            if (
                not isinstance(spec, tuple)
                or len(spec) != 2
                or not isinstance(spec[0], str)
                or not isinstance(spec[1], list)
            ):
                raise ValueError(
                    "internal conditions must be (distribution, parameters) tuples"
                )
        if self.scalarization not in ("sum", "mul"):
            raise ValueError("reward.scalarization must be sum or mul")
        if self.floor <= 0:
            raise ValueError("reward.floor must be positive")
        if not isinstance(self.settings, dict):
            raise ValueError("reward.settings must be a mapping")


@dataclass
class ModelConfig:
    # Residual GINE width/depth; readout concatenates mean and virtual-node features.
    num_emb: int = 128
    num_layers: int = 4
    num_synthon_emb: int = 128
    # Total Linear layers, including the output layer.
    num_mlp_layers: int = 2
    num_mlp_layers_synthon: int = 2
    dropout: float = 0.0

    def validate(self) -> None:
        if self.num_emb <= 0 or self.num_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if (
            self.num_synthon_emb <= 0
            or self.num_mlp_layers < 1
            or self.num_mlp_layers_synthon < 1
        ):
            raise ValueError("invalid synthon dimension or MLP depth")
        if not 0 <= self.dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")


@dataclass
class GenerationConfig:
    """Dynamic synthesis trajectory limits."""

    min_synthons: int = 2
    max_synthons: int = 3
    min_reactions: int = 1
    max_reactions: int = 3

    def validate(self) -> None:
        if not 1 <= self.min_synthons <= self.max_synthons:
            raise ValueError("generation requires 1 <= min_synthons <= max_synthons")
        if not 1 <= self.min_reactions <= self.max_reactions:
            raise ValueError("generation requires 1 <= min_reactions <= max_reactions")
        if self.min_synthons > self.max_reactions + 1:
            raise ValueError("min_synthons cannot be reached within max_reactions")


@dataclass
class TrainingConfig:
    steps: int = 1_000
    batch_size: int = 64
    replay_batch_size: int = 64
    replay_capacity: int = 10_000
    learning_rate: float = 1e-4
    log_z_learning_rate: float = 1e-3
    lr_decay_steps: float = 20_000
    weight_decay: float = 1e-8
    # Additional softmax temperature, separate from reward exponent beta.
    sampling_temperature: float = 1.0
    random_action_prob: float = 0.05
    ema_decay: float = 0.99
    checkpoint_every: int = 500
    log_every: int = 10
    retrosynthesis_workers: int = 4
    # Route mass is divided by this factor for each additional synthon.
    backward_synthon_penalty: float = 100.0

    def validate(self) -> None:
        positive_ints = {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "checkpoint_every": self.checkpoint_every,
            "log_every": self.log_every,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.replay_batch_size < 0:
            raise ValueError("training.replay_batch_size must be non-negative")
        if self.replay_capacity < 0:
            raise ValueError("training.replay_capacity must be non-negative")
        if self.retrosynthesis_workers < 0:
            raise ValueError("training.retrosynthesis_workers must be non-negative")
        if (
            not math.isfinite(self.backward_synthon_penalty)
            or self.backward_synthon_penalty <= 0
        ):
            raise ValueError(
                "training.backward_synthon_penalty must be finite and positive"
            )
        if any(
            value <= 0
            for value in (
                self.learning_rate,
                self.log_z_learning_rate,
                self.lr_decay_steps,
                self.sampling_temperature,
            )
        ):
            raise ValueError(
                "learning rates, decay steps and sampling temperature must be positive"
            )
        if not 0 <= self.random_action_prob <= 1:
            raise ValueError("training.random_action_prob must be in [0, 1]")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("training.ema_decay must be in [0, 1)")


@dataclass
class Config:
    """Top-level runtime configuration.

    The resolved dictionary is embedded in every checkpoint, including the
    heavy-atom capacity and action-space subsampling parameters.
    """

    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    property_penalty: dict[str, float] = field(default_factory=dict)
    subsampling: SubsamplingConfig = field(default_factory=SubsamplingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "runs/rxnflow"
    seed: int = 0
    device: str = "auto"

    def validate(self) -> None:
        self.data.validate()
        self.reward.validate()
        from rxnflow.envs.features import PROPERTY_NAMES

        if not isinstance(self.property_penalty, dict):
            raise ValueError("property_penalty must be a mapping")
        unknown = set(self.property_penalty) - set(PROPERTY_NAMES)
        if unknown:
            raise ValueError(f"unknown property_penalty properties: {sorted(unknown)}")
        # Zero is useful for counts (e.g. no rings/HBD); logP may be negative.
        # These are upper bounds, so positivity is not a general requirement.
        if any(not math.isfinite(value) for value in self.property_penalty.values()):
            raise ValueError("property_penalty values must be finite")
        self.subsampling.validate()
        self.generation.validate()
        self.model.validate()
        self.training.validate()
        if not self.data.env_dir:
            raise ValueError(
                "data.env_dir must point to a prepared Enamine synthon environment"
            )
        if not self.output_dir:
            raise ValueError("run.output_dir must not be empty")
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("run.device must be 'auto', 'cpu', or a CUDA device")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_file_dict(self) -> dict[str, Any]:
        """Return the shallow user-facing YAML representation."""

        reward = asdict(self.reward)
        for name in ("beta", "preferences"):
            distribution, params = reward[name]
            reward[name] = (
                "none"
                if distribution == "none"
                else f"{distribution}({','.join(str(x) for x in params)})"
            )
        return {
            "data": asdict(self.data),
            "run": {
                "output_dir": self.output_dir,
                "seed": self.seed,
                "device": self.device,
            },
            "reward": reward,
            "property_penalty": dict(self.property_penalty),
            "subsampling": asdict(self.subsampling),
            "generation": asdict(self.generation),
            "model": asdict(self.model),
            "training": asdict(self.training),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(self.to_file_dict()), path)

    @staticmethod
    def _flatten_run_section(raw: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(raw)
        root_run_fields = set(normalized) & set(RUN_FIELDS)
        if root_run_fields:
            raise ValueError(
                f"run fields must be configured under run: {sorted(root_run_fields)}"
            )
        run = normalized.pop("run", {})
        if not isinstance(run, dict):
            raise ValueError("run must be a mapping")
        unknown = set(run) - set(RUN_FIELDS)
        if unknown:
            raise ValueError(f"unknown run configuration fields: {sorted(unknown)}")
        for name, value in run.items():
            normalized[name] = value
        return normalized

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        reward = dict(raw["reward"])
        # JSON/OmegaConf serialize tuples as sequences; restore the typed config.
        for name in ("beta", "preferences"):
            dist, params = reward[name]
            reward[name] = (dist, list(params))
        cfg = cls(
            data=DataConfig(**raw["data"]),
            reward=RewardConfig(**reward),
            property_penalty=dict(raw["property_penalty"]),
            subsampling=SubsamplingConfig(**raw["subsampling"]),
            generation=GenerationConfig(**raw["generation"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            output_dir=str(raw["output_dir"]),
            seed=int(raw["seed"]),
            device=str(raw["device"]),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        # 1. Load YAML and map the run section to the internal flat fields.
        base = OmegaConf.create(cls().to_dict())
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(loaded, dict):
            raise ValueError("configuration file must contain a mapping")
        normalized = cls._flatten_run_section(loaded)
        # 2. Parse condition strings once, before constructing numeric config.
        reward = normalized.get("reward")
        if reward is not None:
            if not isinstance(reward, dict):
                raise ValueError("reward must be a mapping")
            for name in ("beta", "preferences"):
                if name in reward:
                    reward[name] = parse_distribution(reward[name])
            if "settings" in reward and not isinstance(reward["settings"], dict):
                raise ValueError("reward.settings must be a mapping")
        property_penalty = normalized.get("property_penalty")
        if property_penalty is not None and not isinstance(property_penalty, dict):
            raise ValueError("property_penalty must be a mapping")
        # 3. Merge omitted defaults and validate the resolved configuration.
        merged = OmegaConf.merge(base, normalized)
        raw = OmegaConf.to_container(merged, resolve=True)
        assert isinstance(raw, dict)
        return cls.from_dict(raw)
