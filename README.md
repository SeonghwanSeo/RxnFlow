[![arXiv](https://img.shields.io/badge/arXiv-2410.04542-b31b1b.svg)](https://arxiv.org/abs/2410.04542)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# RxnFlow: Generative Flows on Synthetic Pathway for Drug Design

RxnFlow uses generative flow networks to design molecules through synthesis pathways. Train it with a molecular reward function, then sample molecules together with their reaction paths and building-block sources.

This repository accompanies **Generative Flows on Synthetic Pathway for Drug Design** by Seonghwan Seo, Minsu Kim, Tony Shen, Martin Ester, Jinkyu Park, Sungsoo Ahn, and Woo Youn Kim. [[Paper](https://arxiv.org/abs/2410.04542)]

The current implementation is under development for v1.0.0. It supports an Enamine-derived synthon library, linear synthesis with unary and binary reactions, and single- or multi-objective rewards.

## Installation

Requires Python 3.10 or later. From the repository root:

```bash
pip install -e .
```

RxnFlow uses PyTorch and RDKit. Training and sampling support CPU and CUDA; GPU execution requires a CUDA-enabled PyTorch installation.

## Prepare synthon libraries

Provide an Enamine building-block file with one `SMILES<TAB>ID` record per line:

```text
CCN	BB001
CC(=O)O	BB002
```

Prepare the environment with the supplied [synthon](data/templates/synthon.yaml) and [reaction](data/templates/reaction.yaml) templates:

```bash
python scripts/prepare.py \
  --building-blocks /path/to/enamine_stock.smi \
  --template-dir data/templates \
  --env-dir /path/to/prepared/enamine \
  --num-workers 16 \
  --min-library-size 10
```

Use a new output directory. Preparation removes salts, excludes source building blocks above 50 heavy atoms, and creates synthon libraries in `synthons/*.smi` with precomputed features in `synthon_features.npz`. `--min-library-size` keeps libraries with at least that many distinct synthons; use `1` for small trial datasets. Building-block source IDs are retained for tracing generated synthesis paths. Preparation also writes `action_space.json` with action-space pairs and `signature.json` with the environment identity and the RxnFlow version used to prepare it. Keep the prepared files together and regenerate them after changing templates or building blocks; environments without these files must be regenerated.

## Train

Start with [configs/qed.yaml](configs/qed.yaml). Set `data.env_dir` to your prepared environment and `run.output_dir` to the desired output directory:

```yaml
data:
  env_dir: /path/to/prepared/enamine
  max_atoms: 50

run:
  output_dir: runs/qed
  device: auto
  seed: 0

reward:
  beta: "32"

property_penalty:
  mw: 500

generation:
  min_synthons: 2
  max_synthons: 3
  min_reactions: 1
  max_reactions: 3

training:
  steps: 3000
  batch_size: 64
  replay_batch_size: 64
```

Run the [QED example](examples/qed.py):

```bash
python examples/qed.py --config configs/qed.yaml
```

`device: auto` selects CUDA when available. `min_synthons`/`max_synthons` bound the number of selected synthons, including FirstSynthon. `min_reactions`/`max_reactions` count chemical reactions after FirstSynthon; deprotection consumes a reaction but no synthon. State tracks `num_synthons` and `num_reactions`. Action spaces retain only type-level paths that can terminate within both bounds, then apply the atom-capacity action mask and property-level penalty Ω. `property_penalty` limits additive state-plus-synthon property estimates during action selection; it does not guarantee exact final-product property bounds. Add `hba: 10` and `hbd: 5` alongside `mw: 500` for those additional constraints.

The policy distinguishes `FIRST_SYNTHON`, `UNIRXN_TRANSFORM`, `UNIRXN_TERMINAL`, `BIRXN_BRICK`, and `BIRXN_LINKER`. Each type has its own output head; graph and synthon encoders are shared, and all eligible actions compete in one softmax.

Training estimates backward probabilities from catalog-supported reverse paths. Reverse search preserves generated paths and prunes paths requiring more synthons than the smallest count found; unary reactions do not increase this count. Reaction depth is bounded separately. This is an approximation over chemical paths: alternatives may use different synthon or reaction counts from the generated history.

The output directory contains `checkpoints/`, per-update metrics in `training.jsonl`, and fresh trajectories in `samples/step_XXXXXX.jsonl` (one file per update and one trajectory per line, including invalid attempts). Each sample contains final SMILES, rewards, conditions, validity, and a compact `traj`: each entry records the pre-action state SMILES, reaction template name, and oriented synthon SMILES (`null` for unary reactions). FirstSynthon is omitted; the first reaction's state contains the initial brick. Resume with the same configuration and prepared environment:

```bash
python examples/qed.py \
  --config configs/qed.yaml \
  --restart runs/qed/checkpoints/latest.ckpt \
  --steps 1000
```

`--steps` specifies additional updates. See the [configuration guide](configs/README.md) and [complete template](configs/template.yaml) for available settings.

## Custom rewards

Implement `RewardFunction.score()` and pass the reward to `RxnFlowTrainer`. Scores must be finite, non-negative float32 NumPy arrays with shape `[number of molecules, number of objectives]`, including empty batches. Higher scores should indicate better molecules.

```python
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import QED

from rxnflow.config import Config
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class MyReward(RewardFunction):
    objectives = ("qed",)

    def score(self, molecules: list[Chem.Mol]) -> NDArray[np.float32]:
        return np.array([QED.qed(mol) for mol in molecules], dtype=np.float32).reshape(-1, 1)


config = Config.from_file("configs/qed.yaml")
trainer = RxnFlowTrainer(config, MyReward())
trainer.run()
```

Override `filter_object(mol)` to assign zero reward to molecules that should not be scored. See [examples/custom_reward.py](examples/custom_reward.py) for an example.

For multiple objectives, return one column per name in `objectives`. The trainer combines them using sampled preference weights. Scale each objective in your reward function, then configure the reward exponent and preferences:

```yaml
reward:
  beta: "uniform(8,64)"
  preferences: "uniform"
  floor: 0.0001
```

`beta` controls reward sharpening; it can be fixed, such as `"32"`, or sampled uniformly. `preferences: "uniform"` draws weights uniformly on the simplex. Use `"fixed(0.3,0.7)"` for fixed weights in a two-objective task, or `"dirichlet(0.5)"` for a different preference distribution.

## Sample molecules

Generate molecules from a trained checkpoint:

```bash
python scripts/sample.py \
  --checkpoint runs/qed/checkpoints/latest.ckpt \
  --num-samples 100 \
  --beta 32 \
  --seed 0 \
  --output samples.json
```

Use `.smi` for molecule SMILES, `.csv` for molecules and paths, or `.json` for structured trajectories and metadata. JSON output includes intermediate structures, reaction choices, and the original building-block IDs and structures.

For a multi-objective checkpoint, add `--preferences "fixed(0.3,0.7)"` to choose a trade-off. Omitted preferences are sampled uniformly on the simplex. Choose beta and preferences within the trained conditions when comparing generated samples. `--sampling-temperature` controls an additional softmax temperature and defaults to 1.

Python sampling is available through `RxnFlowSampler`. Pass a reward implementation to score the generated molecules:

```python
from examples.qed import QEDReward
from rxnflow.sampler import RxnFlowSampler

sampler = RxnFlowSampler("runs/qed/checkpoints/latest.ckpt", reward=QEDReward())
results = sampler.sample(100, beta=("fixed", [32.0]), seed=0)
sampler.write(results, "scored_samples.json")
```

## Citation

```bibtex
@article{seo2024generative,
  title={Generative Flows on Synthetic Pathway for Drug Design},
  author={Seo, Seonghwan and Kim, Minsu and Shen, Tony and Ester, Martin and Park, Jinkyoo and Ahn, Sungsoo and Kim, Woo Youn},
  journal={arXiv preprint arXiv:2410.04542},
  year={2024}
}
```

## License

[MIT](LICENSE).
