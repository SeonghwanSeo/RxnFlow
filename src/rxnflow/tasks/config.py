from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class MOOTaskConfig:
    """Common Config for the MOOTasks

    Attributes
    ----------
    n_valid : int
        The number of valid cond_info tensors to sample.
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors.
    objectives : list[str]
        The objectives to use for the multi-objective optimization..
    online_pareto_front : bool
        Whether to calculate the pareto front online.
    """

    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: list[str] = field(default_factory=lambda: ["vina", "qed"])
    log_topk: bool = False
    online_pareto_front: bool = True


@dataclass
class PocketConditionalConfig:
    """Config for PocketConditional Training

    Attributes
    ----------
    proxy: tuple[str, str, str] (proxy_name, docking_program, train_dataset)
        Proxy Key from PharmacoNet
    pocket_db: str (path)
        Index file including pocket key-filepath pairs (e.g. 10gs,./data/protein/10gs.pdb)
    pocket_dim: int
        Pocket embedding dimension
    """

    proxy: tuple[str, str, str] = MISSING
    pocket_db: str = MISSING
    pocket_dim: int = 128


@dataclass
class DockingTaskConfig:
    """Config for Docking
    protein_path: required
    center | ref_ligand_path: required
    size: 22.5 (default)

    Attributes
    ----------
    protein_path: str (path)
        Protein Path
    center: tuple[float, float, float] (optional)
        Pocket Center
    ref_ligand_path: str (path; optional)
        Reference ligand to identy pocket center
    size: tuple[float, float, float]
        Pocket Box Size
    """

    protein_path: str = MISSING
    center: tuple[float, float, float] | None = None
    ref_ligand_path: str | None = None
    size: tuple[float, float, float] = (22.5, 22.5, 22.5)  # unidock default


@dataclass
class DrugFilter:
    """Config for SBDDConfig

    Attributes
    ----------
    rule: str (path)
        DrugFilter Rule
            - None
            - lipinski
            - veber
    """

    rule: str | None = None


@dataclass
class TasksConfig:
    moo: MOOTaskConfig = field(default_factory=MOOTaskConfig)
    pocket_conditional: PocketConditionalConfig = field(default_factory=PocketConditionalConfig)
    constraint: DrugFilter = field(default_factory=DrugFilter)
    docking: DockingTaskConfig = field(default_factory=DockingTaskConfig)
