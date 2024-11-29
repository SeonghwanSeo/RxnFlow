from dataclasses import field, dataclass

from omegaconf import MISSING

from gflownet import config
from gflownet.data.config import ReplayConfig
from gflownet.utils.config import ConditionalsConfig

from rxnflow.algo.config import AlgoConfig
from rxnflow.models.config import ModelConfig
from rxnflow.tasks.config import TasksConfig


@dataclass
class Config(config.Config):
    """Base configuration for training

    Attributes
    ----------
    desc : str
        A description of the experiment
    log_dir : str
        The directory where to store logs, checkpoints, and samples.
    device : str
        The device to use for training (either "cpu" or "cuda[:<device_id>]")
    seed : int
        The random seed
    validate_every : int
        The number of training steps after which to validate the model
    checkpoint_every : Optional[int]
        The number of training steps after which to checkpoint the model
    store_all_checkpoints : bool
        Whether to store all checkpoints or only the last one
    print_every : int
        The number of training steps after which to print the training loss
    start_at_step : int
        The training step to start at (default: 0)
    num_final_gen_steps : Optional[int]
        After training, the number of steps to generate graphs for
    num_training_steps : int
        The number of training steps
    num_workers : int
        The number of workers to use for creating minibatches (0 = no multiprocessing)
    hostname : Optional[str]
        The hostname of the machine on which the experiment is run
    pickle_mp_messages : bool
        Whether to pickle messages sent between processes (only relevant if num_workers > 0)
    git_hash : Optional[str]
        The git hash of the current commit
    overwrite_existing_exp : bool
        Whether to overwrite the contents of the log_dir if it already exists
    """

    desc: str = "noDesc"
    log_dir: str = MISSING
    device: str = "cuda"
    seed: int = 0
    validate_every: int = 1000
    checkpoint_every: int | None = None
    store_all_checkpoints: bool = False
    print_every: int = 100
    start_at_step: int = 0
    num_final_gen_steps: int | None = None
    num_validation_gen_steps: int | None = None
    num_training_steps: int = 10_000
    num_workers: int = 0
    hostname: str | None = None
    pickle_mp_messages: bool = False
    git_hash: str | None = None
    overwrite_existing_exp: bool = False
    env_dir: str = MISSING
    num_workers_retrosynthesis: int = 4
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: config.OptimizerConfig = field(default_factory=config.OptimizerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    task: TasksConfig = field(default_factory=TasksConfig)
    cond: ConditionalsConfig = field(default_factory=ConditionalsConfig)


def init_empty(cfg: Config):
    """
    Initialize a dataclass instance with all fields set to MISSING,
    including nested dataclasses.

    This is meant to be used on the user side (tasks) to provide
    some configuration using the Config class while overwritting
    only the fields that have been set by the user.
    """
    return config.init_empty(cfg)
