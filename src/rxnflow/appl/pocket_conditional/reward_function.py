from __future__ import annotations

import torch
from pmnet_appl.base import BaseProxy
from pmnet_appl.tacogfn_reward import TacoGFN_Proxy
from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.tasks.utils import chem_metrics


def get_reward_function(proxy: BaseProxy, objectives: list[str]) -> RewardFunction:
    if isinstance(proxy, TacoGFN_Proxy):
        return TacoGFNRewardFunction(proxy, objectives)
    else:
        raise ValueError(proxy)


class RewardFunction:
    def __init__(self, proxy: BaseProxy, objectives: list[str]):
        self.proxy: BaseProxy = proxy
        self.objectives: list[str] = objectives

    def __call__(self, mols: list[RDMol], pocket_key: str | list[str]) -> tuple[Tensor, dict[str, Tensor]]:
        return self.run(mols, pocket_key)

    @torch.no_grad()
    def run(self, mols: list[RDMol], pocket_key: str | list[str]) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(pocket_key, str):
            assert len(pocket_key) == len(mols)
        return self._run(mols, pocket_key)

    def _run(self, mols: list[RDMol], pocket_key: str | list[str]) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError


class TacoGFNRewardFunction(RewardFunction):
    proxy: TacoGFN_Proxy

    def __init__(self, proxy: BaseProxy, objectives: list[str]):
        super().__init__(proxy, objectives)
        assert set(self.objectives) < {"vina", "qed", "sa"}

    def _run(self, mols: list[RDMol], pocket_key: str | list[str]) -> tuple[Tensor, dict[str, Tensor]]:
        """TacoGFN Reward Function
        r_aff:
            - 0                             (0 <= affinity)
            - -affinity * 0.04              (-8 <= affinity <= 0)
            - (-affinity - 8) * 0.2 + 0.32  (-13 <= affinity <= -8)
            - 1.32                          (affinity <= -13)
        r_qed:
            - qed * 0.7                     (0 <= qed <= 0.7)
            - 1.0                           (0.7 <= qed)
        r_sa:
            - sa * 0.8                      (0 <= sa <= 0.8)
            - 1.0                           (0.8 <= sa)
        HAC:
            - num_heavy_atoms
        r = 3 * r_aff * r_qed * r_sa / HAC^(1/3)
        """

        info = {}
        affinity = info["vina"] = self.mol2proxy(mols, pocket_key)
        r_aff = -1 * (0.2 * (affinity + 8).clip(-5, 0) + 0.04 * affinity.clip(min=-8))
        if "qed" in self.objectives:
            info["qed"] = qed = chem_metrics.mol2qed(mols)
            r_qed = (qed / 0.7).clip(0, 1)
        else:
            r_qed = 1
        if "sa" in self.objectives:
            info["sa"] = sa = chem_metrics.mol2sascore(mols)
            r_sa = (sa / 0.8).clip(0, 1)
        else:
            r_sa = 1
        num_heavy_atoms = torch.tensor([mol.GetNumHeavyAtoms() for mol in mols], dtype=torch.float32)
        reward = 3 * r_aff * r_qed * r_sa / (num_heavy_atoms ** (1 / 3))
        return reward, info

    def mol2proxy(self, mols: list[RDMol], pocket_key: str | list[str]) -> Tensor:
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        if isinstance(pocket_key, str):
            scores = self.proxy.scoring_list(pocket_key, smiles).cpu()
        else:
            scores = torch.tensor(
                [float(self.proxy.scoring(target, smi)) for target, smi in zip(pocket_key, smiles, strict=True)],
                dtype=torch.float32,
            )
        return scores
