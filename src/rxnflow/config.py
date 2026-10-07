"""Configuration for local RxnFlow preparation, training, and sampling."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf


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
class SubsamplingConfig:
    """Uniform synthon action-space sampling."""

    sampling_ratio: float = 0.1
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
    moo_preferences: tuple[str, list[float]] = field(default_factory=lambda: ("none", []))
    moo_scalarization: Literal["sum", "mul"] = "mul"
    property_penalty_ratio: float = 0.2
    settings: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for spec in (self.beta, self.moo_preferences):
            if (
                not isinstance(spec, tuple)
                or len(spec) != 2
                or not isinstance(spec[0], str)
                or not isinstance(spec[1], list)
            ):
                raise ValueError(
                    "internal conditions must be (distribution, parameters) tuples"
                )
        if self.moo_scalarization not in ("sum", "mul"):
            raise ValueError("reward.moo_scalarization must be sum or mul")
        if (
            not math.isfinite(self.property_penalty_ratio)
            or self.property_penalty_ratio < 0
        ):
            raise ValueError(
                "reward.property_penalty_ratio must be finite and non-negative"
            )
        if not isinstance(self.settings, dict):
            raise ValueError("reward.settings must be a mapping")


@dataclass
class ModelConfig:
    """GFN model hyperparameters."""

    state_dim: int = 256
    num_state_layers: int = 4
    synthon_dim: int = 256
    num_synthon_layers: int = 3
    hidden_dim: int = 256
    num_action_layers: int = 3

    def validate(self) -> None:
        for name, value in (
            ("state_dim", self.state_dim),
            ("num_state_layers", self.num_state_layers),
            ("synthon_dim", self.synthon_dim),
            ("num_synthon_layers", self.num_synthon_layers),
            ("hidden_dim", self.hidden_dim),
            ("num_action_layers", self.num_action_layers),
        ):
            if value <= 0:
                raise ValueError(f"model.{name} must be positive")


@dataclass
class GenerationConfig:
    """Dynamic synthesis trajectory limits."""

    max_atoms: int = 50
    min_synthons: int = 2
    max_synthons: int = 3
    min_reactions: int = 1
    max_reactions: int = 3

    def validate(self) -> None:
        if self.max_atoms <= 0:
            raise ValueError("generation.max_atoms must be positive")
        if not 1 <= self.min_synthons <= self.max_synthons:
            raise ValueError("generation requires 1 <= min_synthons <= max_synthons")
        if not 1 <= self.min_reactions <= self.max_reactions:
            raise ValueError("generation requires 1 <= min_reactions <= max_reactions")
        if self.min_synthons > self.max_reactions + 1:
            raise ValueError("min_synthons cannot be reached within max_reactions")


@dataclass
class TrainingConfig:
    log_every: int = 10
    checkpoint_every: int = 1000
    num_online: int = 64
    num_replay: int = 64
    ema_decay: float = 0.99
    replay_capacity: int = 100_000
    num_replay_insert: int | None = None
    replay_insert_priority: Literal["uniform", "reward"] = "uniform"
    learning_rate: float = 1e-4
    learning_rate_logZ: float = 1e-2
    lr_decay_steps: float = 10_000
    weight_decay: float = 1e-8
    reward_floor: float = 1e-5
    loss_fn: Literal["mse", "mae", "huber"] = "mse"
    random_action_prob: float = 0.1
    backward_synthon_penalty: float = 100.0
    retrosynthesis_workers: int = 4

    def validate(self) -> None:
        positive_ints = {
            "num_online": self.num_online,
            "checkpoint_every": self.checkpoint_every,
            "log_every": self.log_every,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.num_replay < 0:
            raise ValueError("training.num_replay must be non-negative")
        if self.replay_capacity < 0:
            raise ValueError("training.replay_capacity must be non-negative")
        if self.num_replay_insert is not None and self.num_replay_insert < 0:
            raise ValueError("training.num_replay_insert must be non-negative or None")
        if self.replay_insert_priority not in ("uniform", "reward"):
            raise ValueError("training.replay_insert_priority must be uniform or reward")
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
                self.learning_rate_logZ,
                self.lr_decay_steps,
            )
        ):
            raise ValueError("learning rates and decay steps must be positive")
        if not math.isfinite(self.reward_floor) or self.reward_floor <= 0:
            raise ValueError("training.reward_floor must be finite and positive")
        if self.loss_fn not in ("mse", "mae", "huber"):
            raise ValueError("training.loss_fn must be mse, mae, or huber")
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

    env_dir: str = ""
    reward: RewardConfig = field(default_factory=RewardConfig)
    property_penalty: dict[str, float] = field(default_factory=dict)
    subsampling: SubsamplingConfig = field(default_factory=SubsamplingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.reward.validate()
        from rxnflow.envs.features import PROPERTY_PENALTY_INDICES

        if not isinstance(self.property_penalty, dict):
            raise ValueError("property_penalty must be a mapping")
        unknown = set(self.property_penalty) - set(PROPERTY_PENALTY_INDICES)
        if unknown:
            raise ValueError(f"unknown property_penalty properties: {sorted(unknown)}")
        # Zero is useful for counts (e.g. no rings/HBD).
        # These are upper bounds, so positivity is not a general requirement.
        if any(not math.isfinite(value) for value in self.property_penalty.values()):
            raise ValueError("property_penalty values must be finite")
        self.subsampling.validate()
        self.generation.validate()
        self.model.validate()
        self.training.validate()
        if not self.env_dir:
            raise ValueError("env_dir must point to a prepared synthon environment")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_file_dict(self) -> dict[str, Any]:
        """Return the shallow user-facing YAML representation."""

        reward = asdict(self.reward)
        for name in ("beta", "moo_preferences"):
            distribution, params = reward[name]
            reward[name] = (
                "none"
                if distribution == "none"
                else f"{distribution}({','.join(str(x) for x in params)})"
            )
        return {
            "env_dir": self.env_dir,
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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        reward = dict(raw["reward"])
        # JSON/OmegaConf serialize tuples as sequences; restore the typed config.
        for name in ("beta", "moo_preferences"):
            dist, params = reward[name]
            reward[name] = (dist, list(params))
        cfg = cls(
            env_dir=raw["env_dir"],
            reward=RewardConfig(**reward),
            property_penalty=dict(raw["property_penalty"]),
            subsampling=SubsamplingConfig(**raw["subsampling"]),
            generation=GenerationConfig(**raw["generation"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        # 1. Load the user-facing YAML settings.
        base = OmegaConf.create(cls().to_dict())
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(loaded, dict):
            raise ValueError("configuration file must contain a mapping")
        normalized = loaded
        # 2. Parse condition strings once, before constructing numeric config.
        reward = normalized.get("reward")
        if reward is not None:
            if not isinstance(reward, dict):
                raise ValueError("reward must be a mapping")
            for name in ("beta", "moo_preferences"):
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
