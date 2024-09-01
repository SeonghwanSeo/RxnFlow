import numpy as np
import torch
from gflownet.envs.synthesis import SynthesisEnv
from gflownet.envs.synthesis.env_context import (
    SynthesisEnvContext,
    DEFAULT_ATOMS,
    DEFAULT_CHIRAL_TYPES,
    DEFAULT_CHARGES,
    DEFAULT_EXPL_H_RANGE,
)


class RGFN_EnvContext(SynthesisEnvContext):
    """This context specifies how to create molecules by applying reaction templates."""

    def __init__(
        self,
        env: SynthesisEnv,
        num_cond_dim: int = 0,
        fp_radius_building_block: int = 2,
        fp_nbits_building_block: int = 1024,
        *args,
        atoms: list[str] = DEFAULT_ATOMS,
        chiral_types: list = DEFAULT_CHIRAL_TYPES,
        charges: list[int] = DEFAULT_CHARGES,
        expl_H_range: list[int] = DEFAULT_EXPL_H_RANGE,
        allow_explicitly_aromatic: bool = False,
    ):
        super().__init__(
            env,
            num_cond_dim,
            fp_radius_building_block,
            fp_nbits_building_block,
            atoms=atoms,
            chiral_types=chiral_types,
            charges=charges,
            expl_H_range=expl_H_range,
            allow_explicitly_aromatic=allow_explicitly_aromatic,
        )
        # NOTE: Use only MACCSFingerprint
        self.building_block_features = (self.building_block_features[0][:, :166], self.building_block_features[1])
        self.num_block_features = 166

    def get_block_data(
        self, block_indices: torch.Tensor | list[int] | np.ndarray, device: torch.device
    ) -> torch.Tensor:
        if len(block_indices) >= self.num_building_blocks:
            fp = self.building_block_features[0]
        else:
            fp = self.building_block_features[0][block_indices]
        return torch.as_tensor(fp, dtype=torch.float32, device=device)
