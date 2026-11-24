# RxnFlow: Synthesis-aware drug design with GFlowNets

<img src="image/overview.png" alt="RxnFlow synthesis pathway overview" width="600">

Official implementation of RxnFlow, a generative model for synthesis-aware drug design. RxnFlow uses a generative flow network (GFlowNet) to generate diverse, high-reward molecules.

**Updates:**

- **2026.10.11 — RxnFlow v2 released.** 
  - This update incorporates our in-house model **Hyper Screening X (HSX)**, developed in collaboration with [HITS](https://hits.ai/index_en.html) and [eMolecules](https://www.emolecules.com/). Together with wet-lab collaborators at Yonsei University, we identified two first-in-class lead compounds targeting an SLC1A5 variant.

Papers: [RxnFlow (ICLR 2025)](https://arxiv.org/abs/2410.04542) · [HSX (preprint)](https://www.biorxiv.org/content/10.64898/2026.10.01.755933)

## Installation

RxnFlow requires Python 3.10 or later.

```bash
git clone https://github.com/SeonghwanSeo/RxnFlow
cd RxnFlow/
pip install -e .
```

To install a specific version, use the following command:

```bash
git checkout v2.0.0
pip install -e .
```

## Quick start

Start with the supplied [example SMILES file](data/building_blocks/example_5k.smi), which contains 5,000 ZINC building blocks.

### Prepare a synthesis environment

To prepare a synthesis environment, you need a building-block library and reaction templates. Three template sets are available in `data/templates/`:

- `explore` (default): 26 curated bimolecular reaction templates, 4 unimolecular transformations, and reaction-reactant compatibility constraints to improve synthesizability.
- `explore-expanded`: the same reactions without these compatibility constraints, allowing a broader chemical space.
- `cgflow`: the set of 38 bimolecular reaction templates used in [CGFlow](https://github.com/tsa87/cgflow).

Save your building-block library, for example from [eMolecules](https://www.emolecules.com/data-downloads) or [Enamine](https://enamine.net/building-blocks/building-blocks-catalog), as a `.smi` file with one `SMILES<TAB>ID` record per line.

Prepare the supplied example library using the default templates:

```bash
python scripts/prepare.py \
  --block-smi ./data/building_blocks/example_5k.smi \
  --env-dir ./data/envs/example_5k/ \
  --config ./data/templates/explore/config.yaml \
  --num-workers 16
```

Set `env_dir` in your training configuration to the prepared directory. See [environment preparation](docs/environment.md) for more details.

### Train a GFlowNet

Start with an [example configuration](configs/examples/README.md) for your objective, or [define a custom reward](docs/rewards.md).
Each example below pairs a Python script that defines the reward with a YAML file that controls training.

**Single-objective optimization**

The [QED example](examples/qed.py) trains a model to generate molecules with high QED.
A higher `beta` makes the model favor high-QED molecules more strongly.
Training with [beta conditioning](configs/examples/qed_beta.yaml) lets you adjust that preference when generating molecules.

```bash
# Fixed beta
python examples/qed.py --config configs/examples/qed.yaml --steps 2000 --output-dir runs/qed

# Beta conditioning
python examples/qed.py --config configs/examples/qed_beta.yaml --steps 2000 --output-dir runs/qed_beta
```

**Multi-objective optimization**

The [MW/logP example](examples/mw_logp.py) favors molecules with lower MW and higher logP.
The preference weights determine how much each objective contributes to the reward.
Training with [preference conditioning](configs/examples/mw_logp_pref.yaml) lets you choose whether to emphasize lower MW or higher logP when generating molecules.

```bash
# Equal objective weights: R = 0.5 * R_mw + 0.5 * R_logp
python examples/mw_logp.py --config configs/examples/mw_logp.yaml --steps 2000 --output-dir runs/mw_logp

# Preference conditioning: R = w_1 * R_mw + w_2 * R_logp, where w_1 + w_2 = 1
python examples/mw_logp.py --config configs/examples/mw_logp_pref.yaml --steps 2000 --output-dir runs/mw_logp_pref
```

**Conditional generation**

The [TPSA script](examples/tpsa_cond.py) trains the model to generate molecules near a target TPSA.
The [external conditions guide](docs/conditioning.md#external-conditions) shows how to specify a target when sampling.

```bash
python examples/tpsa_cond.py --config configs/examples/tpsa_cond.yaml --steps 2000 --output-dir runs/tpsa_cond
```

### Sampling

Generate molecules and their synthesis routes from a trained checkpoint:

```bash
python scripts/sample.py \
  --checkpoint runs/mw_logp/checkpoints/latest.ckpt \
  --num-samples 100 \
  --output samples.jsonl
```

See [sampling](docs/training.md#sampling) for available options and output formats.

## Citation

If you use this code in your research, please cite the following papers:

```bibtex
@article{seo2026hsx,
  title={Synthesis-aware generative design in trillion-scale chemical spaces for automated drug discovery},
  author={Seonghwan Seo and Yulseung Sung and Sang-Yeon Hwang and Mincheol Kang and Joonseong Lee and Samuele Bordi and Luka Raguz and Jihye Choi and Daniil Melnichenko and Wan Namkung and Sehan Lee and Jaechang Lim and Benedikt M. Wanner and Jung Min Han and Woo Youn Kim},
  year={2026},
  doi={10.64898/2026.10.01.755933},
  url={https://www.biorxiv.org/content/10.64898/2026.10.01.755933},
  journal={bioRxiv}
}

@inproceedings{seo2025rxnflow,
  title={Generative Flows on Synthetic Pathway for Drug Design},
  author={Seonghwan Seo and Minsu Kim and Tony Shen and Martin Ester and Jinkyoo Park and Sungsoo Ahn and Woo Youn Kim},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=pB1XSj2y4X}
}
```
