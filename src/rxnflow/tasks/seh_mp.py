import torch
import torch_geometric.data as gd
from pathlib import Path
from rdkit import Chem

from rxnflow.communication.reward import RewardModule
from gflownet.models import bengio2021flow


class SEHReward(RewardModule):
    def __init__(self, log_dir: str | Path, device: str | torch.device = "cuda", verbose: bool = True):
        super().__init__(log_dir, load_config=False, verbose=True)
        self.device: torch.device = torch.device(device)
        self.model = bengio2021flow.load_original_model()
        self.model.to(self.device)

    @property
    def objectives(self) -> list[str]:
        return ["seh"]

    def str_repr_to_object(self, obj_repr: str) -> Chem.Mol:
        return Chem.MolFromSmiles(obj_repr)

    def compute_rewards(self, objs: list[Chem.Mol]) -> list[list[float]]:
        graphs = [bengio2021flow.mol2graph(i) for i in objs]
        preds = self.compute_reward_from_graph(graphs)
        assert preds.shape == (len(graphs), 1)
        return preds.tolist()

    def compute_reward_from_graph(self, graphs: list[gd.Data]) -> torch.Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.model(batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1, 1))


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    import sys

    flag = sys.argv[1]
    assert flag in ["train", "reward"]

    log_dir = "./logs/debug-seh-communicate/"

    if flag == "train":
        from rxnflow.communication.trainer import MolCommunicationTrainer
        from rxnflow.config import Config, init_empty

        config = init_empty(Config())
        config.log_dir = log_dir
        config.env_dir = "./data/envs/real"
        config.overwrite_existing_exp = True
        config.print_every = 10

        config.num_training_steps = 1000
        config.algo.action_subsampling.sampling_ratio = 0.1

        trial = MolCommunicationTrainer(config)
        trial.run()
    else:
        SEHReward(log_dir)
