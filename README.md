[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2410.04542)
[![Python versions](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# RxnFlow: Generative Flows on Synthetic Pathway for Drug Design

<img src="image/overview.png" width=600>

Official implementation of **_Generative Flows on Synthetic Pathway for Drug Design_** by Seonghwan Seo, Minsu Kim, Tony Shen, Martin Ester, Jinkyu Park, Sungsoo Ahn, and Woo Youn Kim. [[arXiv](https://arxiv.org/abs/2410.04542)]

RxnFlow are a synthesis-oriented generative framework that aims to discover diverse drug candidates through GFlowNet objective and a large action space.

- RxnFlow can operate on large synthetic action spaces comprising 1M building blocks and 100 reaction templates without compute overhead
- RxnFlow can explore broader chemical space within less reaction steps, resulting in higher diversity, higher potency, and lower synthetic complexity of generated molecules.
- RxnFlow can generate molecules with expanded or modified building block libaries without retraining.

This project is based on gflownet, and `src/gflownet/` is a clone of [recursionpharma/gflownet@v0.2.0](https://github@v0.2.0.com/recursionpharma/gflownet/tree/v0@v0.2.0.2@v0.2.0.0).
Since we have continued to improve it, we do not guarantee that current version will reproduce the same results as the paper.
You can access the reproducing codes and scripts from [tag: paper-archive](https://github.com/SeonghwanSeo/RxnFlow/tree/paper-archive).

This repository was developed for research.
The repository for real-world drug discovery will be released later.

## Installation

```bash
# python>=3.10,<3.13, torch>=2.3.1
pip install -e . --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html

# For UniDock
conda install unidock
pip install -e '.[unidock]' --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html

# For Pocket Conditional Generation
pip install -e '.[pmnet]' --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

## Data Preparation

To construct datas, please follow the process in [data/README.md](data/README.md).

### Reaction Template

We provide the two reaction template sets:

- **Real**: We provide the 109-size reaction template set [templates/real.txt](templates/real.txt) from Enamine REAL synthesis protocol ([Gao et al.](https://github.com/wenhao-gao/synformer)).
- **HB**: The reaction template used in this paper contains 13 uni-molecular reactions and 58 bi-molecular reactions, which is constructed by [Cretu et al](https://github.com/mirunacrt/synflownet). The template set is available under [templates/hb_edited.txt](template/hb_edited.txt).

### Building Block Library

We support two building block libraries.

- **ZINCFrag:** For reproducible benchmark study, we propose a new public building block library, which is a subset of ZINC22 fragment set. All fragments are also included in AiZynthFinder's built-in ZINC stock.
- **Enamine:** We support the Enamine building block library, which is available upon request at [https://enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog).

## Experiments

<details>
<summary><h3 style="display:inline-block"> Docking optimization with GPU-accelerated UniDock</h3></summary>

You can optimize the docking score with GPU-accelerated [UniDock](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145).

```bash
python script/opt_unidock.py -h
python script/opt_unidock.py \
  -p <Protein PDB path> \
  -c <Center X> <Center Y> <Center Z> \
  -l <Reference ligand, required if center is empty. > \
  -s <Size X> <Size Y> <Size Z> \
  -o <Output directory> \
  -n <Num Oracles (default: 1000)> \
  --filter <drugfilter; choice=(lipinski, veber, null); default: lipinski> \
  --batch_size <Num generations per oracle; default: 64> \
  --env_dir <Environment directory> \
  --subsampling_ratio <Subsample ratio; memory-variance trade-off; default: 0.01>
```

You can also perform multi-objective optimization ([Multi-objective GFlowNet](https://arxiv.org/abs/2210.12765)) for docking score and QED.

```bash
python script/opt_unidock_moo.py -h
python script/opt_unidock_moo.py \
  -p <Protein PDB path> \
  -c <Center X> <Center Y> <Center Z> \
  -l <Reference ligand, required if center is empty. > \
  -s <Size X> <Size Y> <Size Z> \
  -o <Output directory> \
  -n <Num Oracles (default: 1000)> \
  --batch_size <Num generations per oracle; default: 64> \
  --env_dir <Environment directory> \
  --subsampling_ratio <Subsample ratio; memory-variance trade-off; default: 0.01>
```

**Example (KRAS G12C mutation)**

- Use center coordinates

  ```bash
  python script/opt_unidock.py -p ./data/examples/6oim_protein.pdb -c 1.872 -8.260 -1.361 -o ./log/kras --filter veber
  ```

- Use center of the reference ligand

  ```bash
  python script/opt_unidock_moo.py -p ./data/examples/6oim_protein.pdb -l ./data/examples/6oim_ligand.pdb -o ./log/kras
  ```

</details>

<details>
<summary><h3 style="display:inline-block"> Zero-shot sampling with Pharmacophore-based QuickVina Proxy</h3></summary>

Sample high-affinity molecules. The QuickVina docking score is estimated by Proxy Model [[github](https://github.com/SeonghwanSeo/PharmacoNet/tree/main/src/pmnet_appl)].
To create dataset, please refer [data/](./data/)

The trained model will be updated soon.

- Training

  ```bash
  python script/train_pocket_conditional.py -h
  python script/train_pocket_conditional.py \
    --env_dir <Environment directory> \
    --subsampling_ratio <Subsample ratio; memory-variance trade-off; default: 0.01> \
  ```

- Sampling
  ```bash
  python script/sampling_zeroshot.py -h
  python script/sampling_zeroshot.py \
    -p <Protein PDB path> \
    -c <Center X> <Center Y> <Center Z> \
    -l <Reference ligand, required if center is empty. > \
    -o <Output path: `smi|csv`> \
    -n <Num samples (default: 100)> \
    --env_dir <Environment directory> \
    --model_path <Checkpoint path; default: None (auto-downloaded)> \
    --subsampling_ratio <Subsample ratio; memory-variance trade-off; default: 0.01> \
    --cuda
  ```

**Example (KRAS G12C mutation)**

- `csv` format: save molecules with their rewards (GPU is recommended)

  ```bash
  python script/sampling_zeroshot.py -o out.csv -p ./data/examples/6oim_protein.pdb -l ./data/examples/6oim_ligand.pdb --cuda
  ```

- `smi` format: save molecules only (CPU: 0.06s/mol, GPU: 0.04s/mol)

  ```bash
  python script/sampling_zeroshot.py -o out.smi -p ./data/examples/6oim_protein.pdb -c 1.872 -8.260 -1.361
  ```

</details>

<details>
<summary><h3 style="display:inline-block">Custom optimization</h3></summary>

If you want to train RxnFlow with your custom reward function, you can use the base classes from `rxnflow.base`. The reward should be **Non-negative**.

Example codes are provided in `./examples/`.

- Example (QED)

  ```python
  import torch
  from rdkit.Chem import Mol, QED
  from gflownet import ObjectProperties
  from rxnflow.base import RxnFlowTrainer, RxnFlowSampler, BaseTask

  class QEDTask(BaseTask):
      def compute_obj_properties(self, objs: list[Chem.Mol]) -> tuple[ObjectProperties, torch.Tensor]:
          fr = torch.tensor([QED.qed(mol) for mol in mols], dtype=torch.float)
          fr = fr.reshape(-1, 1) # reward dimension should be [Nobj, Nprop]
          is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
          return ObjectProperties(fr), is_valid_t

  class QEDTrainer(RxnFlowTrainer):  # For online training
      def setup_task(self):
          self.task = QEDTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

  class QEDSampler(RxnFlowSampler):  # Sampling with pre-trained GFlowNet
      def setup_task(self):
          self.task = QEDTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)
  ```

- Example (Multi-objective optimization)
  The example scripts will be provided soon!

  ```python
  import torch
  from rdkit.Chem import Mol as RDMol
  from gflownet import ObjectProperties
  from rxnflow.base import RxnFlowTrainer, RxnFlowSampler, BaseTask

  class MOOTask(BaseTask):
      is_moo=True
      def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, torch.Tensor]:
          fr1 = torch.tensor([reward1(mol) for mol in mols], dtype=torch.float)
          fr2 = torch.tensor([reward2(mol) for mol in mols], dtype=torch.float)
          fr = torch.stack([fr1, fr2], dim=-1)
          is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
          return ObjectProperties(fr), is_valid_t

  class MOOTrainer(RxnFlowTrainer):  # For online training
      def set_default_hps(self, base: Config):
          super().set_default_hps(base)
          base.task.moo.objectives = ["obj1", "obj2"] # set the objective names

      def setup_task(self):
          self.task = MOOTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

  class MOOSampler(RxnFlowSampler):  # Sampling with pre-trained GFlowNet
      def setup_task(self):
          self.task = MOOTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)
  ```

</details>

<details>
<summary><h3 style="display:inline-block">Reproducing experimental results</h3></summary>

The training/sampling scripts are provided in `experiments/`.

**_NOTE_**: Current version do not fully reproduce the paper result. Please switch to [tag: paper-archive](https://github.com/SeonghwanSeo/RxnFlow/tree/paper-archive).

</details>

## Citation

If you use our code in your research, we kindly ask that you consider citing our work in papers:

```bibtex
@article{seo2024generative,
  title={Generative Flows on Synthetic Pathway for Drug Design},
  author={Seo, Seonghwan and Kim, Minsu and Shen, Tony and Ester, Martin and Park, Jinkyoo and Ahn, Sungsoo and Kim, Woo Youn},
  journal={arXiv preprint arXiv:2410.04542},
  year={2024}
}

```

## Related Works

- [GFlowNet](https://arxiv.org/abs/2106.04399) (github: [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet))
- [TacoGFN](https://arxiv.org/abs/2310.03223) [github: [tsa87/TacoGFN-SBDD](https://github.com/tsa87/TacoGFN-SBDD)]
- [PharmacoNet](https://arxiv.org/abs/2310.00681) [github: [SeonghwanSeo/PharmacoNet](https://github.com/SeonghwanSeo/PharmacoNet)]
- [UniDock](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145) [github: [dptech-corp/Uni-Dock](https://github.com/dptech-corp/Uni-Dock)]
