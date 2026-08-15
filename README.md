[![Python versions](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# RxnFlow

RxnFlow is a generative flow network for designing molecules through the eMolecules eXplore–Synple reaction space. It provides local data preparation, QED optimization, custom reward injection, checkpoint restart, and molecule sampling using PyTorch and RDKit.

The research project is described in [Generative Flows on Synthetic Pathway for Drug Design](https://arxiv.org/abs/2410.04542). This repository is developed from work with HITS-AI and eMolecules.

## Requirements and installation

RxnFlow requires Python 3.10 or newer.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

For development tools, use `uv sync --extra dev`.

## Prepare eMolecules data

RxnFlow supports the eMolecules eXplore–Synple library. Access and licensing information is available from the [official eMolecules virtual-compounds page](https://www.emolecules.com/virtual-compounds). Source records are user-supplied and are not included in this repository.

Prepare an environment with the included Synple and eXplore templates:

```bash
rxnflow-prepare all \
  --raw-data-dir /path/to/emolecules/records \
  --env-dir /path/to/prepared/emolecules \
  --template-dir data/templates/emolecules/synple \
  --template-dir data/templates/emolecules/explore
```

Preparation is resumable. Use `--force` to rebuild completed stages. The expected input columns and template format are documented in [data/README.md](data/README.md).

## Configure training

Copy the minimal [QED configuration](configs/qed.yaml) and set `data.env_dir` and `run.output_dir`:

```yaml
data:
  env_dir: /path/to/prepared/emolecules
  max_atoms: 50

run:
  output_dir: runs/qed
  device: auto
  seed: 0

reward:
  exponent: 32.0
  settings: {}

property_penalty:
  mw: 500

subsampling:
  sampling_ratio: 0.1

training:
  steps: 1000
  batch_size: 128
  replay_batch_size: 128
```

`max_atoms` is the maximum number of heavy atoms in generated molecules. `property_penalty` contains optional hard upper bounds such as `mw`, `tpsa`, and `logp`; use `{}` for no property constraints. Reduce `batch_size`, `replay_batch_size`, or `sampling_ratio` when compute or memory is limited.

The complete set of options is shown in [configs/template.yaml](configs/template.yaml) and explained in the [configuration guide](configs/README.md).

## Train with QED

```bash
rxnflow-train --config configs/qed.yaml
```

The equivalent Python API is:

```python
from rxnflow import Config, QEDReward, RxnFlowTrainer

config = Config.from_file("configs/qed.yaml")
reward = QEDReward(**config.reward.settings)
checkpoint = RxnFlowTrainer(config, reward).run()
```

Every run writes its resolved configuration and checkpoints under `run.output_dir`.

## Use a custom reward

Implement `RewardFunction.score` and construct the reward explicitly in Python. Each `Sample` provides canonical `smiles` and an RDKit `mol`.

```python
from rxnflow import Config, RewardFunction, RxnFlowTrainer, Sample


class HeavyAtomReward(RewardFunction):
    def __init__(self, scale: float = 40.0):
        self.scale = scale

    def score(self, samples: list[Sample]) -> list[float]:
        return [sample.mol.GetNumHeavyAtoms() / self.scale for sample in samples]


config = Config.from_file("config.yaml")
reward = HeavyAtomReward(**config.reward.settings)
RxnFlowTrainer(config, reward).run()
```

Place constructor keyword arguments under `reward.settings`. YAML does not select or import the reward class. Rewards must be finite and non-negative; invalid molecules receive zero.

## Restart training

Restart with the resolved configuration written by the original run:

```bash
rxnflow-train \
  --config runs/qed/config.yaml \
  --restart runs/qed/checkpoint_latest.pt \
  --steps 500
```

Restart requires the same RxnFlow version, reward implementation, prepared environment, and resolved configuration.

## Sample molecules

```bash
rxnflow-sample \
  --checkpoint runs/qed/checkpoint_latest.pt \
  --num-samples 100 \
  --output samples.json \
  --format json \
  --qed
```

Supported output formats are SMILES (`smi`), CSV, and structured JSON.

## Development checks

```bash
./test.sh quick
./test.sh heavy
```

The heavy suite uses representative eMolecules data supplied through `RXNFLOW_RAW_DATA` or an existing environment supplied through `RXNFLOW_ENV_DIR`.

## Acknowledgements

We thank HITS-AI and eMolecules for their collaboration on the eXplore–Synple environment and RxnFlow development.
