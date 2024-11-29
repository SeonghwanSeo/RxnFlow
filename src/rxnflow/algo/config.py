from dataclasses import dataclass, field
from omegaconf import MISSING

from gflownet.utils.misc import StrictDataClass
from gflownet.algo.config import TBConfig


@dataclass
class SubsamplingConfig(StrictDataClass):
    """Configuration for action space subsampling

    Attributes
    ----------
    sampling_ratio : float
        global subsampling ratio, [0, 1]
    num_sampling_first_block : int
        the number of actions for AddFirstReactant (override sampling_ratio)
    sampling_ratio_bi_rxn : float
        subsampling ratio for ReactBi (override sampling_ratio)
    min_sampling_reatbi : int
        the minimum number of actions for each reaction template; ReactBi

    ** Advanced Parameters **
    num_mc_sampling : int
        Theoretically, set num_mc_sampling to 1 and set larger sampling ratio is better.
    onpolicy_temp : float
        Recommended to set 1.0. We do not investigate this option in the paper.
    """

    sampling_ratio: float = 1.0
    num_sampling_first_block: int = MISSING
    sampling_ratio_bi_rxn: float = MISSING
    min_sampling_bi_rxn: int = 100

    # Advanced parameters
    onpolicy_temp: float = 1.0


@dataclass
class AlgoConfig(StrictDataClass):
    """Generic configuration for algorithms

    Attributes
    ----------
    method : str
        The name of the algorithm to use (e.g. "TB")
    num_from_policy : int
        The number of on-policy samples for a training batch.
        If using a replay buffer, see `replay.num_from_replay` for the number of samples from the replay buffer, and
        `replay.num_new_samples` for the number of new samples to add to the replay buffer (e.g. `num_from_policy=0`,
        and `num_new_samples=N` inserts `N` new samples in the replay buffer at each step, but does not make that data
        part of the training batch).
    num_from_dataset : int
        The number of samples from the dataset for a training batch
    valid_num_from_policy : int
        The number of on-policy samples for a validation batch
    valid_num_from_dataset : int
        The number of samples from the dataset for a validation batch
    max_len : int
        The maximum length of a trajectory
    max_nodes : int
        The maximum number of nodes in a generated graph
    max_edges : int
        The maximum number of edges in a generated graph
    illegal_action_logreward : float
        The log reward an agent gets for illegal actions
    train_random_action_prob : float
        The probability of taking a random action during training
    train_det_after: Optional[int]
        Do not take random actions after this number of steps
    valid_random_action_prob : float
        The probability of taking a random action during validation
    sampling_tau : float
        The EMA factor for the sampling model (theta_sampler = tau * theta_sampler + (1-tau) * theta)
    """

    method: str = "TB"
    num_from_policy: int = 64
    num_from_dataset: int = 0
    valid_num_from_policy: int = 64
    valid_num_from_dataset: int = 0
    min_len: int = 2
    max_len: int = 3
    illegal_action_logreward: float = -100
    train_random_action_prob: float = 0.05
    train_det_after: int | None = None
    valid_random_action_prob: float = 0.0
    sampling_tau: float = 0.0
    tb: TBConfig = field(default_factory=TBConfig)
    action_subsampling: SubsamplingConfig = SubsamplingConfig()
