import torch
import torch.nn as nn
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.utils import metrics
from gflownet.utils.conditioning import (
    FocusRegionConditional,
    MultiObjectiveWeightedPreferences,
    TemperatureConditional,
)
from gflownet.utils.transforms import to_logreward
from rxnflow.config import Config


class BaseTask(GFNTask):
    """Sets up a common structure of task"""

    is_moo: bool = False

    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.temperature_conditional: TemperatureConditional = TemperatureConditional(cfg)
        self.num_cond_dim: int = self.temperature_conditional.encoding_size()
        if self.is_moo:
            self.setup_moo()

    def compute_obj_properties(self, mols: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        """we only generate rdkit molecules -> rename objs to mols"""
        raise NotImplementedError

    def setup_moo(self):
        mcfg = self.cfg.task.moo
        self.objectives: list[str] = mcfg.objectives
        self.num_objectives = len(mcfg.objectives)

        if self.cfg.cond.focus_region.focus_type is not None:
            self.focus_cond = FocusRegionConditional(self.cfg, mcfg.n_valid)
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

    def cond_info_to_logreward(self, cond_info: dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        """
        Compute the logreward from the obj_props and the conditional information
        """
        if self.is_moo:
            if isinstance(obj_props, list):
                if isinstance(obj_props[0], Tensor):
                    obj_props = torch.stack(obj_props)
                else:
                    obj_props = torch.tensor(obj_props)
            scalarized_rewards = self.pref_cond.transform(cond_info, obj_props)
            scalarized_logrewards = to_logreward(scalarized_rewards)
            focused_logreward = (
                self.focus_cond.transform(cond_info, (obj_props, scalarized_logrewards))
                if self.focus_cond is not None
                else scalarized_logrewards
            )
            logreward = focused_logreward
        else:
            logreward = to_logreward(obj_props)
        tempered_logreward = self.temperature_conditional.transform(cond_info, logreward)
        clamped_logreward = tempered_logreward.clamp(min=self.cfg.algo.illegal_action_logreward)
        return LogScalar(clamped_logreward)

    def sample_conditional_information(self, n: int, train_it: int) -> dict[str, Tensor]:
        cond_info = self.temperature_conditional.sample(n)
        if self.is_moo:
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
        self, cond_info: dict[str, Tensor], log_rewards: Tensor, obj_props: ObjectProperties, hindsight_idxs: Tensor
    ):
        # TODO: we seem to be relabeling tensors in place, could that cause a problem?
        if self.focus_cond is None:
            raise NotImplementedError("Hindsight relabeling only implemented for focus conditioning")
        if self.focus_cond.cfg.focus_type is None:
            return cond_info, log_rewards
        # only keep hindsight_idxs that actually correspond to a violated constraint
        _, in_focus_mask = metrics.compute_focus_coef(
            obj_props, cond_info["focus_dir"], self.focus_cond.cfg.focus_cosim
        )
        out_focus_mask = torch.logical_not(in_focus_mask)
        hindsight_idxs = hindsight_idxs[out_focus_mask[hindsight_idxs]]

        # relabels the focus_dirs and log_rewards
        cond_info["focus_dir"][hindsight_idxs] = nn.functional.normalize(obj_props[hindsight_idxs], dim=1)

        preferences_enc = self.pref_cond.encode(cond_info["preferences"])
        focus_enc = self.focus_cond.encode(cond_info["focus_dir"])
        cond_info["encoding"] = torch.cat(
            [cond_info["encoding"][:, : self.num_thermometer_dim], preferences_enc, focus_enc], 1
        )

        log_rewards = self.cond_info_to_logreward(cond_info, obj_props)
        return cond_info, log_rewards
