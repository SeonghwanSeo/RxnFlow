import sys
import os
import time
import logging
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf


class RewardModule:
    """Reward Module for communicating with trainer running on the other process.
    To avoid the dependency issues, gflownet sources are not imported.
    If you want to run reward function on your own dependencies or environments, copy this class.
    """

    def __init__(self, log_dir: str | Path, load_config: bool = True, verbose: bool = True):
        # config
        self.num_objectives: int = len(self.objectives)
        self.logger: logging.Logger = self.create_logger(
            "reward", loglevel=logging.INFO if verbose else logging.WARNING
        )
        self.oracle_idx: int = 0  # reward oracle index
        self._tick: float = 0.1  # tick for wait
        self._max_timeout: float = 300  # 5 min

        # gflownet communicate structure
        self.log_dir = log_dir = Path(log_dir)
        self.config_path = self.log_dir / "config.yaml"
        self._load_sample_dir = log_dir / "_samples/"
        self._save_reward_dir = log_dir / "_rewards/"
        self._lock_file: Path = log_dir / "_wait.lock"
        _communication_lock = log_dir / "_process.lock"

        # Wait while the gflownet start
        while not self.config_path.exists():
            time.sleep(self._tick)
        if load_config:
            self.cfg = OmegaConf.load(self.config_path)
        else:
            self.cfg = None

        # Only one reward calculation process for a single gflownet process
        assert not _communication_lock.exists()
        with open(_communication_lock, "w"):
            pass

    """Running API"""

    def run(self):
        while True:
            oracle_idx = self._wait()
            if oracle_idx is None:
                break
            self.oracle_idx = oracle_idx
            self._run_oracle()

    """Implement following methods!"""

    @property
    def objectives(self) -> list[str]:
        """Return the objective names"""
        raise NotImplementedError

    def str_repr_to_object(self, obj_repr: str) -> Any:
        """Convert a string representation to an object
        e.g., return Chem.MolFromSmiles(obj_repr)

        Parameters
        ----------
        obj_repr : str
            A string representation of the object

        Returns
        -------
        obj: Any
            An object
        """

    def filter_object(self, obj: Any) -> bool:
        """Implement the constraint here if required

        Parameters
        ----------
        obj : Any
            A valid object(graph, seq, mol)

        Returns
        -------
        is_valid: bool
            return whether the object is valid or not
        """
        return True

    def compute_reward_single(self, obj: Any) -> list[float]:
        """Implement the reward function here

        Parameters
        ----------
        obj : Any
            A valid object(graph, seq, mol)

        Returns
        -------
        rewards: list[float]
            It shoule be list of the property for each objective
            The negative value would be clipped to 0 because GFlowNets require a non-negative reward.
            assert len(reward) == self.num_objectives
        """
        raise NotImplementedError

    def compute_rewards(self, objs: list[Any]) -> list[list[float]]:
        """Modify here if parallel computation is required

        Parameters
        ----------
        objs : list[Any]
            A list of valid objects(graphs, seqs, mols, ...)

        Returns
        -------
        rewards_list: list[list[float]]
            Each item of list should be list of reward for each objective
            assert len(rewards_list) == len(objs)
        """
        return [self.compute_reward_single(obj) for obj in objs]

    def log(
        self,
        obj_reprs: list[str],
        rewards: list[list[float]],
        is_valid: list[bool],
    ):
        """Log Hook

        Parameters
        ----------
        obj_reprs : list[str]
            A list of string representation of objects
        rewards : list[list[float]]
            A list of reward for each object
        is_valid : list[bool]
            A list of valid flag of each objects
        """
        # oracle_idx = self.oracle_idx
        pass

    """Inner functions"""

    def _run_oracle(self):
        """Calculate Reward"""
        # load the sample objects
        obj_reprs = self._get()
        objs = [self.str_repr_to_object(repr) for repr in obj_reprs]

        # filter the valid molecules
        is_valid = [self.filter_object(obj) for obj in objs]
        valid_objs = [obj for flag, obj in zip(is_valid, objs, strict=True) if flag]

        # reward calculation
        st = time.time()
        self.logger.info(f"reward calculation for {len(valid_objs)} valid molecules...")
        if len(valid_objs) > 0:
            rewards = self._compute_rewards(valid_objs)
        else:
            rewards = []
        tick = time.time() - st
        self.logger.info(f"reward calculation finish! ({tick:.3f} sec)")

        # save rewards
        self._put(obj_reprs, rewards, is_valid)
        self._unlock()

        # logging
        self.log(obj_reprs, rewards, is_valid)

    def _wait(self) -> int | None:
        self.logger.info("wait request...")
        st = time.time()
        while not self._lock_file.exists():
            time.sleep(self._tick)
            if time.time() - st > self._max_timeout:
                self.logger.warning("Timeout!")
                return None
        tick = time.time() - st
        with open(self._lock_file) as f:
            oracle_idx = int(f.readline().strip())
        self.logger.info(f"receive objects! ({tick:.3f} sec)")
        return oracle_idx

    def _get(self):
        sample_file = self._load_sample_dir / f"{self.oracle_idx}.csv"
        return self._load_samples(sample_file)

    def _put(
        self,
        obj_reprs: list[str],
        rewards: list[list[float]],
        is_valid: list[bool],
    ):
        save_file = self._save_reward_dir / f"{self.oracle_idx}.csv"
        return self._save_reward(obj_reprs, rewards, is_valid, save_file)

    def _unlock(self):
        os.remove(self._lock_file)

    def _compute_rewards(self, objs: list[Any]) -> list[list[float]]:
        """To prevent unsafe actions"""
        rs = self.compute_rewards(objs)
        for r in rs:
            assert (
                len(r) == self.num_objectives
            ), f"The number of reward ({len(r)}) should be same to the number of objectives ({self.num_objectives})"
        assert len(rs) == len(
            objs
        ), f"The number of outputs {len(rs)} should be same to the number of samples ({len(objs)})"
        return rs

    def _load_samples(self, sample_file: Path) -> list[str]:
        with sample_file.open() as f:
            lines = f.readlines()[1:]
        return [ln.split(",")[1].strip() for ln in lines]

    def _save_reward(
        self,
        obj_reprs: list[str],
        rewards: list[list[float]],
        is_valid: list[bool],
        save_file: Path,
    ):
        """save rewards to csv

        Parameters
        ----------
        obj_reprs : list[str]
            A string representations of n objects
        rewards : list[list[float]]
            Rewards for m valid objects where m <= n
        is_valid : list[bool]
            Valid flags of n objects
        save_file : Path
            The path for csv file
        """
        num_samples = len(obj_reprs)
        assert len(rewards) == sum(
            is_valid
        ), f"The length of rewards ({len(rewards)}) should be same to the number of valid molecules ({sum(is_valid)})"
        t = 0
        with open(save_file, "w") as w:
            w.write(",object,is_valid," + ",".join(self.objectives) + "\n")
            for i in range(num_samples):
                obj = obj_reprs[i]
                valid = is_valid[i]
                if valid:
                    rs = [str(r) for r in rewards[t]]
                    t += 1
                else:
                    rs = ["0.0"] * len(self.objectives)
                w.write(f"{i},{obj},{valid}," + ",".join(rs) + "\n")

    def create_logger(self, name="logger", loglevel=logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(loglevel)
        while len([logger.removeHandler(i) for i in logger.handlers]):
            pass  # Remove all handlers (only useful when debugging)
        formatter = logging.Formatter(
            fmt=f"%(asctime)s - %(levelname)s - {name} - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
