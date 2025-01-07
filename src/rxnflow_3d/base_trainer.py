from rxnflow.base.trainer import RxnFlowTrainer
from rxnflow_3d.envs.env import SynthesisEnvContext3D, SynthesisEnv3D
from rxnflow_3d.envs.unidock_env import SynthesisEnv3D_unidock
from rxnflow_3d.algo import SynthesisTB3D


class RxnFlow3DTrainer(RxnFlowTrainer):
    env: SynthesisEnv3D

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB3D(self.env, self.ctx, self.cfg)

    def setup_env_context(self):
        self.ctx = SynthesisEnvContext3D(self.env, num_cond_dim=self.task.num_cond_dim)


class RxnFlow3DTrainer_unidock(RxnFlow3DTrainer):
    def setup_env(self):
        env_dir = self.cfg.env_dir
        protein_path = self.cfg.task.docking.protein_path
        pocket_center = self.cfg.task.docking.center
        self.env = SynthesisEnv3D_unidock(env_dir, protein_path, pocket_center)


# TODO: implement semla version
class RxnFlow3DTrainer_semla(RxnFlow3DTrainer):
    def setup_env(self):
        raise NotImplementedError
