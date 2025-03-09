import torch
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties
from gflownet.data.data_source import DataSource
from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.utils.misc import get_worker_rng
from rxnflow.appl.pocket_conditional.model import RxnFlow_MultiPocket, RxnFlow_PocketConditional, RxnFlow_SinglePocket
from rxnflow.appl.pocket_conditional.utils import PocketDB
from rxnflow.base import BaseTask, RxnFlowTrainer
from rxnflow.config import Config


class PocketConditionalTask(BaseTask):
    """Sets up a task where the reward is computed using a Proxy, QED."""

    pocket_db: PocketDB

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.objectives = cfg.task.moo.objectives
        self.setup_pocket_db()

    def compute_obj_properties(self, mols: list[RDMol], sample_idcs: list[int]) -> tuple[ObjectProperties, Tensor]:
        raise NotImplementedError

    @property
    def num_pockets(self) -> int:
        return len(self.pocket_db)

    def setup_pocket_db(self):
        """set protein db for training"""
        raise NotImplementedError

    def sample_conditional_information(self, n: int, train_it: int) -> dict[str, Tensor]:
        cond_info = super().sample_conditional_information(n, train_it)
        # sample pocket for training
        rng = get_worker_rng()
        pocket_indices = rng.choice(len(self.pocket_db), n).tolist()
        self.pocket_db.set_batch_idcs(pocket_indices)
        cond_info["pocket_idx"] = torch.LongTensor(pocket_indices)
        return cond_info


class PocketConditionalTrainer(RxnFlowTrainer):
    task: PocketConditionalTask
    model: RxnFlow_PocketConditional

    def step(self, loss: Tensor):
        self.model.clear_cache()
        self.sampling_model.clear_cache()
        return super().step(loss)

    def create_data_source(self, replay_buffer: ReplayBuffer | None = None, is_algo_eval: bool = False):
        return _DataSource(self.cfg, self.ctx, self.algo, self.task, replay_buffer, is_algo_eval)


class PocketConditionalTrainer_SinglePocket(PocketConditionalTrainer):
    task: PocketConditionalTask

    def setup_model(self):
        self.model = RxnFlow_SinglePocket(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
            freeze_pocket_embedding=True,
            freeze_action_embedding=True,
        )


class PocketConditionalTrainer_MultiPocket(PocketConditionalTrainer):
    task: PocketConditionalTask

    def setup_model(self):
        self.model = RxnFlow_MultiPocket(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )


class _DataSource(DataSource):
    task: PocketConditionalTask

    def compute_properties(self, trajs, mark_as_online=False):
        """Sets trajs' obj_props and is_valid keys by querying the task."""
        valid_idcs = torch.tensor([i for i in range(len(trajs)) if trajs[i].get("is_valid", True)]).long()
        mols = [self.ctx.graph_to_obj(trajs[i]["result"]) for i in valid_idcs]

        # NOTE: This is only the different part
        obj_props, m_is_valid = self.task.compute_obj_properties(mols, valid_idcs.tolist())
        # obj_props, m_is_valid = self.task.compute_obj_properties(mols) # previous

        assert obj_props.ndim == 2, "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
        # The task may decide some of the mols are invalid, we have to again filter those
        valid_idcs = valid_idcs[m_is_valid]
        all_fr = torch.zeros((len(trajs), obj_props.shape[1]))
        all_fr[valid_idcs] = obj_props
        for i in range(len(trajs)):
            trajs[i]["obj_props"] = all_fr[i]
            trajs[i]["is_online"] = mark_as_online
        # Override the is_valid key in case the task made some mols invalid
        for i in valid_idcs:
            trajs[i]["is_valid"] = True
