from pathlib import Path
from .env import SynthesisEnv3D


class SynthesisEnv3D_semla(SynthesisEnv3D):
    def __init__(
        self,
        env_dir: str | Path,
        protein_path: str | Path,
        pocket_center: tuple[float, float, float],
    ):
        super().__init__(env_dir, protein_path, pocket_center)
        raise NotImplementedError
