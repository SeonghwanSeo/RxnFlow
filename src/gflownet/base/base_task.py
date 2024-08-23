import numpy as np
import torch
import torch.nn as nn

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.config import Config
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils import metrics
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward
from gflownet.utils.conditioning import FocusRegionConditional, MultiObjectiveWeightedPreferences


class BaseTask(GFNTask):
    """Sets up a common structure of task"""

    def __init__(self, cfg: Config, rng: np.random.Generator, wrap_model: Callable[[nn.Module], nn.Module]):
        self._wrap_model: Callable[[nn.Module], nn.Module] = wrap_model
        self.rng: np.random.Generator = rng
        self.cfg: Config = cfg
        self.models: dict[str, nn.Module] = self._load_task_models()
        self.temperature_conditional: TemperatureConditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim: int = self.temperature_conditional.encoding_size()

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        raise NotImplementedError

    def cond_info_to_logreward(self, cond_info: dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return self._to_reward_scalar(cond_info, flat_reward)

    def _load_task_models(self) -> dict[str, nn.Module]:
        return {}

    def _to_reward_scalar(self, cond_info: dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def flat_reward_transform(self, y: float | Tensor | list[float]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp


class BaseMOOTask(BaseTask):
    """Sets up a common structure of task"""

    def __init__(self, cfg: Config, rng: np.random.Generator, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, rng, wrap_model)
        self.setup_moo()

    def setup_moo(self):
        mcfg = self.cfg.task.moo
        self.objectives: list[str] = mcfg.objectives
        self.num_objectives = len(mcfg.objectives)

        if self.cfg.cond.focus_region.focus_type is not None:
            self.focus_cond = FocusRegionConditional(self.cfg, mcfg.n_valid, self.rng)
        else:
            self.focus_cond = None
        self.pref_cond = MultiObjectiveWeightedPreferences(self.cfg)
        self.temperature_sample_dist = self.cfg.cond.temperature.sample_dist
        self.temperature_dist_params = self.cfg.cond.temperature.dist_params
        self.num_thermometer_dim = self.cfg.cond.temperature.num_thermometer_dim
        self.num_cond_dim = (
            self.temperature_conditional.encoding_size()
            + self.pref_cond.encoding_size()
            + (self.focus_cond.encoding_size() if self.focus_cond is not None else 0)
        )

    def sample_conditional_information(self, n: int, train_it: int, final: bool = True) -> dict[str, Tensor]:
        cond_info = super().sample_conditional_information(n, train_it, final)
        pref_ci = self.pref_cond.sample(n)
        focus_ci = (
            self.focus_cond.sample(n, train_it) if self.focus_cond is not None else {"encoding": torch.zeros(n, 0)}
        )
        cond_info = {
            **cond_info,
            **pref_ci,
            **focus_ci,
            "encoding": torch.cat([cond_info["encoding"], pref_ci["encoding"], focus_ci["encoding"]], dim=1),
        }
        return cond_info

    def encode_conditional_information(self, steer_info: Tensor) -> dict[str, Tensor]:
        """
        Encode conditional information at validation-time
        We use the maximum temperature beta for inference
        Args:
            steer_info: Tensor of shape (Batch, 2 * n_objectives) containing the preferences and focus_dirs
            in that order
        Returns:
            dict[str, Tensor]: dictionary containing the encoded conditional information
        """
        n = len(steer_info)
        if self.temperature_sample_dist == "constant":
            beta = torch.ones(n) * self.temperature_dist_params[0]
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            beta = torch.ones(n) * self.temperature_dist_params[-1]
            beta_enc = torch.ones((n, self.num_thermometer_dim))

        assert len(beta.shape) == 1, f"beta should be of shape (Batch,), got: {beta.shape}"

        # TODO: positional assumption here, should have something cleaner
        preferences = steer_info[:, : len(self.objectives)].float()
        focus_dir = steer_info[:, len(self.objectives) :].float()

        preferences_enc = self.pref_cond.encode(preferences)
        if self.focus_cond is not None:
            focus_enc = self.focus_cond.encode(focus_dir)
            encoding = torch.cat([beta_enc, preferences_enc, focus_enc], 1).float()
        else:
            encoding = torch.cat([beta_enc, preferences_enc], 1).float()
        return {
            "beta": beta,
            "encoding": encoding,
            "preferences": preferences,
            "focus_dir": focus_dir,
        }

    def relabel_condinfo_and_logrewards(
        self, cond_info: dict[str, Tensor], log_rewards: Tensor, flat_rewards: FlatRewards, hindsight_idxs: Tensor
    ):
        # TODO: we seem to be relabeling tensors in place, could that cause a problem?
        if self.focus_cond is None:
            raise NotImplementedError("Hindsight relabeling only implemented for focus conditioning")
        if self.focus_cond.cfg.focus_type is None:
            return cond_info, log_rewards
        # only keep hindsight_idxs that actually correspond to a violated constraint
        _, in_focus_mask = metrics.compute_focus_coef(
            flat_rewards, cond_info["focus_dir"], self.focus_cond.cfg.focus_cosim
        )
        out_focus_mask = torch.logical_not(in_focus_mask)
        hindsight_idxs = hindsight_idxs[out_focus_mask[hindsight_idxs]]

        # relabels the focus_dirs and log_rewards
        cond_info["focus_dir"][hindsight_idxs] = nn.functional.normalize(flat_rewards[hindsight_idxs], dim=1)

        preferences_enc = self.pref_cond.encode(cond_info["preferences"])
        focus_enc = self.focus_cond.encode(cond_info["focus_dir"])
        cond_info["encoding"] = torch.cat(
            [cond_info["encoding"][:, : self.num_thermometer_dim], preferences_enc, focus_enc], 1
        )

        log_rewards = self.cond_info_to_logreward(cond_info, flat_rewards)
        return cond_info, log_rewards

    def cond_info_to_logreward(self, cond_info: dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        """
        Compute the logreward from the flat_reward and the conditional information
        """
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)

        scalarized_rewards = self.pref_cond.transform(cond_info, flat_reward)
        scalarized_logrewards = to_logreward(scalarized_rewards)
        focused_logreward = (
            self.focus_cond.transform(cond_info, flat_reward, scalarized_logrewards)
            if self.focus_cond is not None
            else scalarized_logrewards
        )
        tempered_logreward = self.temperature_conditional.transform(cond_info, focused_logreward)
        clamped_logreward = tempered_logreward.clamp(min=self.cfg.algo.illegal_action_logreward)

        return RewardScalar(clamped_logreward)
