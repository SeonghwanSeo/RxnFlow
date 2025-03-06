import os
from pathlib import Path
import time
from typing import Any
from tqdm import tqdm
import numpy as np
import random

import torch

from rxnflow.config import Config, init_empty
from rxnflow.base.generator import RxnFlowSampler
from rxnflow.appl.pocket_conditional.model import RxnFlow_SinglePocket
from rxnflow.appl.pocket_conditional.trainer import PocketConditionalTask


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestTask(PocketConditionalTask):
    def setup_pocket_db(self):
        pass


class TestSampler(RxnFlowSampler):
    task: TestTask

    def setup_model(self):
        self.model = RxnFlow_SinglePocket(self.ctx, self.cfg, num_graph_out=self.cfg.algo.tb.do_predict_n + 1)

    def setup_task(self):
        self.task = TestTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    @torch.no_grad()
    def set_pocket(self, protein_path: str | Path, center: tuple[float, float, float]):
        self.sampling_model.clear_cache()
        self.model.clear_cache()
        self.task.set_protein(str(protein_path), center)

    @torch.no_grad()
    def sample_against_pocket(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float],
        n: int,
        calc_reward: bool = False,
    ) -> list[dict[str, Any]]:
        """
        samples = sampler.sample_against_pocket(<pocket_file>, <center>, <n>, calc_reward = False)
        samples[0] = {'smiles': <smiles>, 'traj': <traj>, 'info': <info>}
        samples[0]['traj'] = [
            (('StartingBlock',), smiles1),        # None    -> smiles1
            (('UniMolecularReaction', template), smiles2),  # smiles1 -> smiles2
            ...                                 # smiles2 -> ...
        ]
        samples[0]['info'] = {'beta': <beta>, ...}


        samples = sampler.sample_against_pocket(..., calc_reward = True)
        samples[0]['info'] = {
            'beta': <beta>,
            'reward': <reward>,
            'reward_qed': <qed>,
            'reward_docking': <proxy>,
        }
        """
        self.set_pocket(protein_path, center)
        return self.sample(n, calc_reward)


if __name__ == "__main__":
    MODEL_PATH = "./logs/pocket_conditional_qvina_crossdocked2020/model_state.pt"
    ROOT_DIR = "./data/experiments/CrossDocked2020/"
    TEST_POCKET_DIR = os.path.join(ROOT_DIR, "protein/test/")
    TEST_POCKET_CENTER_INFO: dict[str, tuple[float, float, float]] = {}
    with open(os.path.join(ROOT_DIR, "center_info/test.csv")) as f:
        for line in f.readlines():
            pocket_name, x, y, z = line.split(",")
            TEST_POCKET_CENTER_INFO[pocket_name] = (float(x), float(y), float(z))

    # NOTE: Create sampler
    config = init_empty(Config())
    config.algo.num_from_policy = 100
    device = "cuda"
    ckpt_path = MODEL_PATH
    sampler = TestSampler(config, ckpt_path, device)
    sampler.update_temperature("uniform", [16, 64])

    # NOTE: Run
    save_path = Path("./result/crossdocked/")
    save_path.mkdir(exist_ok=True)
    runtime = []

    for pocket_file in tqdm(sorted(list(Path(TEST_POCKET_DIR).iterdir()))):
        set_seed(1)
        key = pocket_file.stem
        center = TEST_POCKET_CENTER_INFO[key]

        st = time.time()
        res = sampler.sample_against_pocket(pocket_file, center, 100, calc_reward=False)
        runtime.append(time.time() - st)

        with open(save_path / f"{key}.csv", "w") as w:
            w.write(",SMILES\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                w.write(f"{idx},{smiles}\n")
    print("avg time", np.mean(runtime))
