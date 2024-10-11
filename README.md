[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2410.04542)
[![Python versions](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# RxnFlow: Generative Flows on Synthetic Pathway for Drug Design

Official implementation of ***Generative Flows on Synthetic Pathway for Drug Design*** by Seonghwan Seo, Minsu Kim, Tony Shen, Martin Ester, Jinkyu Park, Sungsoo Ahn, and Woo Youn Kim. [[arXiv](https://arxiv.org/abs/2410.04542)]

RxnFlow are a synthesis-oriented generative framework that aims to discover diverse drug candidates through GFlowNet objective and a large action space.

- RxnFlow can operate on large synthetic action spaces comprising 1.2M building blocks and 71 reaction templates without compute overhead
- RxnFlow can explore broader chemical space within less reaction steps, resulting in higher diversity, higher potency, and lower synthetic complexity of generated molecules.
- RxnFlow can generate molecules with expanded or modified building block libaries without retraining.

The implementation of this project builds upon the [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) with MIT license. This repository was developed for research, and the code for real-world drug discovery will be released later.

## Setup

### Install

```bash
# python: 3.10
conda install openbabel # For PharmacoNet
pip install -e . --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html

# For UniDock
conda install openbabel unidock
pip install -e '.[unidock]' --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

### Data

To construct the synthetic action space, RxnFlow requires the reaction template set and the building block library.

The reaction template used in this paper contains 13 uni-molecular reactions and 58 bi-molecular reactions, which is constructed by [Cretu et al](https://github.com/mirunacrt/synflownet). The template set is available under [data/template/hb_edited.txt](data/template/hb_edited.txt).

The Enamine building block library is available upon request at [https://enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog). We used the "Comprehensive Catalog" released at 2024.06.10.

- Use Comprehensive Catalog

  ```bash
  cd data
  # case1: single-step
  python scripts/a_sdf_to_env.py -b <CATALOG_SDF> -d envs/enamine_all --cpu <CPU>
  
  # case2: two-step
  python scripts/b1_sdf_to_smi.py -b <CATALOG_SDF> -o building_blocks/blocks.smi --cpu <CPU>
  python scripts/b2_smi_to_env.py -b building_blocks/blocks.smi -d envs/enamine_all --cpu <CPU> --skip_sanitize
  ```

- Use custom SMILES file (`.smi`)

  ```bash
  python scripts/b2_smi_to_env.py -b <SMILES-FILE> -d ./envs/<ENV> --cpu <CPU>
  ```

## Experiments

### Docking-QED multi-objective optimization with GPU-accelerated UniDock

Multi-objective optimization ([Multi-objective GFlowNet](https://arxiv.org/abs/2210.12765)) for docking score and QED. This uses GPU-accelerated [UniDock](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145).

```bash
python script/opt_unidock.py -h
python script/opt_unidock.py \
  -p <Protein PDB path> \
  -c <Center X> <Center Y> <Center Z> \
  -l <Reference ligand, required if center is empty. > \
  -o <Output directory> \
  -n <Num Oracles (default: 1000)> \
  --batch_size <Num generations per oracle; default: 64> \
  --env_dir <Environment directory> \
  --subsample_ratio <Subsample ratio; memory-variance trade-off; default: 0.01>
```

**Example (KRAS G12C mutation)**

- Use center coordinates
  ```bash
  python script/opt_unidock.py -p ./data/examples/6oim_protein.pdb -c 1.872 -8.260 -1.361 -o ./log/kras
  ```
- Use center of the reference ligand
  ```bash
  python script/opt_unidock.py -p ./data/examples/6oim_protein.pdb -l ./data/examples/6oim_ligand.pdb -o ./log/kras
  ```

### Zero-shot sampling with Pharmacophore-based QuickVina Proxy

Sample high-affinity molecules. The QuickVina docking score is estimated by Proxy Model [[github](https://github.com/SeonghwanSeo/PharmacoNet/tree/main/src/pmnet_appl)].

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
  --subsample_ratio <Subsample ratio; memory-variance trade-off; default: 0.01> \
  --cuda
```

**Example (KRAS G12C mutation)**

- csv file: Save molecules with their rewards (GPU is recommended for reward calculation)
  ```bash
  python script/sampling_zeroshot.py -o out.csv -p ./data/examples/6oim_protein.pdb -l ./data/examples/6oim_ligand.pdb --cuda
  ```
- smi file: Save molecules only (CPU: 0.06s/mol, GPU: 0.04s/mol)
  ```bash
  python script/sampling_zeroshot.py -o out.smi -p ./data/examples/6oim_protein.pdb -c 1.872 -8.260 -1.361
  ```

### Custom optimization

If you want to train RxnFlow with your custom reward function, you can use the base classes from  `gflownet.base`. The reward should be **Non-negative**.

- Example (QED)

  ```python
  import torch
  from gflownet.base import SynthesisTrainer, SynthesisGFNSampler, BaseTask
  from gflownet.trainer import FlatRewards
  from rdkit.Chem import Mol as RDMol, QED

  class QEDTask(BaseTask):
      def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, torch.Tensor]:
          fr = torch.tensor([QED.qed(mol) for mol in mols], dtype=torch.float).reshape(-1, 1)
          is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
          return FlatRewards(fr), is_valid_t

  class QEDSynthesisTrainer(SynthesisTrainer): # For online training
      def setup_task(self):
          self.task: QEDTask = QEDTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

  class QEDSynthesisSampler(SynthesisGFNSampler): # Sampling with pre-trained GFlowNet
      def setup_task(self):
          self.task: QEDTask = QEDTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)
  ```

### Reproducing experimental results

All scripts to reproduce the results of paper are in `./experiments/`.

The dataset is available at [Google Drive](https://drive.google.com/drive/folders/1ZngDj3-b8ZLcR9J4ekIrGpxTklMXNIn-). Please decompress them at `./data/experiments/`.

## Citation

If you use this code in your research, please cite:

```
@article{seo2024rxnflow,
      title={Generative Flows on Synthetic Pathway for Drug Design}, 
      author={Seonghwan Seo and Minsu Kim and Tony Shen and Martin Ester and Jinkyoo Park and Sungsoo Ahn and Woo Youn Kim},
      journal={arXiv preprint arXiv:2410.04542},
      year={2024},
}
```

## Related Works

- [GFlowNet](https://arxiv.org/abs/2106.04399) (github: [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet))
- [TacoGFN](https://arxiv.org/abs/2310.03223) [github: [tsa87/TacoGFN-SBDD](https://github.com/tsa87/TacoGFN-SBDD)]
- [PharmacoNet](https://arxiv.org/abs/2310.00681) [github: [SeonghwanSeo/PharmacoNet](https://github.com/SeonghwanSeo/PharmacoNet)]
- [UniDock](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145) [github: [dptech-corp/Uni-Dock](https://github.com/dptech-corp/Uni-Dock)]
