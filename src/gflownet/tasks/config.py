from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING

from gflownet.utils.misc import StrictDataClass


@dataclass
class SEHTaskConfig(StrictDataClass):
    reduced_frag: bool = False


@dataclass
class SEHMOOTaskConfig(StrictDataClass):
    """Config for the SEHMOOTask

    Attributes
    ----------
    n_valid : int
        The number of valid cond_info tensors to sample.
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors.
    objectives : List[str]
        The objectives to use for the multi-objective optimization. Should be a subset of ["seh", "qed", "sa", "mw"].
    online_pareto_front : bool
        Whether to calculate the pareto front online.
    """

    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: List[str] = field(default_factory=lambda: ["seh", "qed", "sa", "mw"])
    log_topk: bool = False
    online_pareto_front: bool = True


@dataclass
class QM9TaskConfig(StrictDataClass):
    h5_path: str = "./data/qm9/qm9.h5"  # see src/gflownet/data/qm9.py
    model_path: str = "./data/qm9/qm9_model.pt"


@dataclass
class QM9MOOTaskConfig(StrictDataClass):
    """
    Config for the QM9MooTask

    Attributes
    ----------
    n_valid : int
        The number of valid cond_info tensors to sample.
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors.
    objectives : List[str]
        The objectives to use for the multi-objective optimization. Should be a subset of ["gap", "qed", "sa", "mw"].
        While "mw" can be used, it is not recommended as the molecules are already small.
    online_pareto_front : bool
        Whether to calculate the pareto front online.
    """

    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: List[str] = field(default_factory=lambda: ["gap", "qed", "sa"])
    online_pareto_front: bool = True


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
class TasksConfig(StrictDataClass):
    qm9: QM9TaskConfig = field(default_factory=QM9TaskConfig)
    qm9_moo: QM9MOOTaskConfig = field(default_factory=QM9MOOTaskConfig)
    seh: SEHTaskConfig = field(default_factory=SEHTaskConfig)
    seh_moo: SEHMOOTaskConfig = field(default_factory=SEHMOOTaskConfig)
    moo: MOOTaskConfig = field(default_factory=MOOTaskConfig)
    docking: DockingTaskConfig = field(default_factory=DockingTaskConfig)
