from rxnflow.appl.pocket_conditional.model import RxnFlow_SinglePocket
from rxnflow.appl.pocket_conditional.pocket.data import generate_protein_data
from rxnflow.appl.pocket_conditional.utils import PocketDB
from rxnflow.config import Config, init_empty
from rxnflow.tasks.unidock_vina import VinaTask, VinaTrainer
from rxnflow.tasks.unidock_vina_moo import VinaMOOTask, VinaMOOTrainer
from rxnflow.tasks.utils.chem_metrics import mol2qed, mol2sascore

aux_tasks = {"qed": mol2qed, "sa": mol2sascore}

"""Multi-objective optimization but not MO-GFN (production-based)"""


class VinaTask_Fewshot(VinaTask):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.pocket_db = PocketDB({"protein": generate_protein_data(self.vina.protein_pdb_path, self.vina.center)})
        self.pocket_db.set_batch_idcs([0])


class VinaMOOTask_Fewshot(VinaMOOTask):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.pocket_db = PocketDB({"protein": generate_protein_data(self.vina.protein_pdb_path, self.vina.center)})
        self.pocket_db.set_batch_idcs([0])


class VinaTrainer_Fewshot(VinaTrainer):
    task: VinaTask_Fewshot

    def setup_model(self):
        self.model = RxnFlow_SinglePocket(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
            freeze_pocket_embedding=True,
            freeze_action_embedding=True,
        )

    def setup_task(self):
        self.task = VinaTask_Fewshot(self.cfg)


class VinaMOOTrainer_Fewshot(VinaMOOTrainer):
    task: VinaMOOTask_Fewshot

    def setup_model(self):
        self.model = RxnFlow_SinglePocket(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
            freeze_pocket_embedding=True,
            freeze_action_embedding=True,
        )

    def setup_task(self):
        self.task = VinaMOOTask_Fewshot(self.cfg)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-vina-moo-fewshot/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True
    config.pretrained_model_path = "./logs/pretrain/sbdd-uniform-0-64/model_state.pt"
    config.algo.action_subsampling.sampling_ratio = 0.1
    config.algo.sampling_tau = 0.98

    config.task.docking.protein_path = "./data/examples/6oim_protein.pdb"
    config.task.docking.center = (1.872, -8.260, -1.361)

    config.replay.use = False

    trial = VinaMOOTrainer_Fewshot(config)
    trial.run()
