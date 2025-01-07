from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor

from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from rdkit.Chem import rdMolDescriptors, Lipinski, Crippen, QED

from collections import OrderedDict
from collections.abc import Callable

from rxnflow.config import Config
from rxnflow.base import BaseTask
from rxnflow.utils.chem_metrics import mol2qed, mol2sascore


aux_tasks = {"qed": mol2qed, "sa": mol2sascore}


class VinaTask(BaseTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)

        # binding affinity estimation
        self.protein_path: Path = Path(cfg.task.docking.protein_path)
        self.center: tuple[float, float, float] = cfg.task.docking.center
        self.size: tuple[float, float, float] = cfg.task.docking.size

        self.filter: str | None = cfg.task.constraint.rule
        assert self.filter in [None, "lipinski", "veber"]

        self.save_dir: Path = Path(cfg.log_dir) / "docking"
        self.save_dir.mkdir()

        self.topn_vina: OrderedDict[str, float] = OrderedDict()
        self.batch_vina: list[float] = []

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        fr = self.calc_vina_reward(objs)
        return fr.reshape(-1, 1)

    def _calc_vina_score(self, obj: RDMol) -> float:
        raise NotImplementedError

    def calc_vina_reward(self, objs: list[RDMol]) -> Tensor:
        # NOTE: save pose
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        with Chem.SDWriter(str(out_path)) as w:
            for obj in objs:
                w.write(obj)

        # NOTE: calculate score (vina local opt)
        vina_scores = [self._calc_vina_score(obj) for obj in objs]
        self.update_storage(objs, vina_scores)
        fr = torch.tensor(vina_scores, dtype=torch.float32) * -0.1
        return fr.clip(min=1e-5)

    def filter_object(self, obj: RDMol) -> bool:
        if self.filter is None:
            pass
        elif self.filter in ("lipinski", "veber"):
            if rdMolDescriptors.CalcExactMolWt(obj) > 500:
                return False
            if Lipinski.NumHDonors(obj) > 5:
                return False
            if Lipinski.NumHAcceptors(obj) > 10:
                return False
            if Crippen.MolLogP(obj) > 5:
                return False
            if self.filter == "veber":
                if rdMolDescriptors.CalcTPSA(obj) > 140:
                    return False
                if Lipinski.NumRotatableBonds(obj) > 10:
                    return False
        else:
            raise ValueError(self.filter)
        return True

    def update_storage(self, objs: list[RDMol], scores: list[float]):
        self.batch_vina = scores
        smiles_list = [Chem.MolToSmiles(obj) for obj in objs]
        self.topn_vina.update(zip(smiles_list, scores, strict=True))
        if len(self.topn_vina) > 2000:
            topn = sorted(list(self.topn_vina.items()), key=lambda v: v[1])[:2000]
            self.topn_vina = OrderedDict(topn)


class VinaMOOTask(VinaTask):
    """Sets up a task where the reward is computed using a Vina, QED."""

    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"vina", "qed", "sa"}

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        flat_r: list[Tensor] = []
        self.avg_reward_info = OrderedDict()
        for prop in self.objectives:
            if prop == "vina":
                fr = self.calc_vina_reward(objs)
            else:
                fr = aux_tasks[prop](objs)
            flat_r.append(fr)
            self.avg_reward_info[prop] = fr.mean().item()
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(objs)
        return flat_rewards

    def update_storage(self, objs: list[RDMol], scores: list[float]):
        def _filter(obj: RDMol) -> bool:
            """Check the object passes a property filter"""
            return QED.qed(obj) > 0.5

        pass_idcs = [i for i, obj in enumerate(objs) if _filter(obj)]
        pass_objs = [objs[i] for i in pass_idcs]
        pass_scores = [scores[i] for i in pass_idcs]
        super().update_storage(pass_objs, pass_scores)
