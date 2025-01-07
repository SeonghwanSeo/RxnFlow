from pathlib import Path
from rdkit import Chem

from .env import SynthesisEnv3D
from .pose_prediction import unidock


class SynthesisEnv3D_unidock(SynthesisEnv3D):
    def __init__(
        self,
        env_dir: str | Path,
        protein_path: str | Path,
        pocket_center: tuple[float, float, float],
    ):
        super().__init__(env_dir, protein_path, pocket_center)
        # NOTE: to reduce overhead
        self.__protein_pdbqt_path: Path = self.protein_path.parent / (self.protein_path.name + "qt")

    def set_binding_pose(self, obj: Chem.Mol):
        protein_pdbqt = self.__protein_pdbqt_path
        center = self.pocket_center
        if not protein_pdbqt.exists():
            unidock.pdb2pdbqt(self.protein_path, protein_pdbqt)

        docked_obj, docking_score = unidock.run(obj, protein_pdbqt, center)
        assert docked_obj is not None
        assert obj.GetNumAtoms() == docked_obj.GetNumAtoms()
        for a1, a2 in zip(obj.GetAtoms(), docked_obj.GetAtoms(), strict=True):
            if a1.GetSymbol() == "*":
                assert a2.GetSymbol() == "C"
            else:
                assert a1.GetSymbol() == a2.GetSymbol()
        conf = docked_obj.GetConformer()
        obj.AddConformer(conf)
        obj.SetDoubleProp("docking_score", docking_score)

    def set_binding_pose_batch(self, objs: list[Chem.Mol]) -> None:
        protein_pdbqt = self.__protein_pdbqt_path
        center = self.pocket_center
        if not protein_pdbqt.exists():
            unidock.pdb2pdbqt(self.protein_path, protein_pdbqt)

        docking_results = unidock.run_batch(objs, protein_pdbqt, center)
        for obj, (docked_obj, docking_score) in zip(objs, docking_results, strict=True):
            if docked_obj is None:
                continue
            if obj.GetNumAtoms() != docked_obj.GetNumAtoms():
                continue
            flag = True
            for a1, a2 in zip(obj.GetAtoms(), docked_obj.GetAtoms(), strict=True):
                if a1.GetSymbol() == "*":
                    if a2.GetSymbol() != "C":
                        flag = False
                        break
                elif a1.GetSymbol() != a2.GetSymbol():
                    flag = False
                    break
            if not flag:
                continue
            conf = docked_obj.GetConformer()
            obj.AddConformer(conf)
            obj.SetDoubleProp("docking_score", docking_score)
