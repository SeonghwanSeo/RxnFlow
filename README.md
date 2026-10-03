# RxnFlow: Generative Flows on Synthetic Pathway for Drug Design

[![arXiv](https://img.shields.io/badge/arXiv-2410.04542-b31b1b.svg)](https://arxiv.org/abs/2410.04542)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

RxnFlow uses generative flow networks to design molecules through synthesis pathways. Optimize molecular properties with your own reward function and generate molecules with their synthesis routes.

This repository accompanies the ICLR paper **Generative Flows on Synthetic Pathway for Drug Design** by Seonghwan Seo, Minsu Kim, Tony Shen, Martin Ester, Jinkyu Park, Sungsoo Ahn, and Woo Youn Kim. [[Paper](https://arxiv.org/abs/2410.04542)]

This is a preview of the core framework behind Hyper Screening X (HSX), developed with eMolecules and HITS. The preprint is coming soon.

## Installation

RxnFlow requires Python 3.10 or later. Install from the repository root:

```bash
pip install -e .
```

## Prepare a synthesis environment

Prepare an eMolecules or Enamine building-block catalog with the supplied reaction templates:

```bash
python scripts/prepare.py \
  --building-blocks /path/to/building_blocks.smi \
  --env-dir /path/to/prepared/environment \
  --num-workers 16
```

Set `data.env_dir` in your training configuration to the prepared directory. See [environment preparation](docs/environment.md) for catalog format, filters and custom reaction templates.

## Single-objective optimization

Train with QED as the reward:

```bash
python examples/qed.py --config configs/qed.yaml --steps 1000 --output-dir runs/qed
```

See [training settings and resuming a run](docs/training.md) or [define your own reward](docs/rewards.md).

## Multi-objective optimization

Train on QED and synthetic accessibility, with SA reward `(10 - SA score) / 9`:

```bash
python examples/qed_sa.py --config configs/qed_sa.yaml --steps 3000 --output-dir runs/qed_sa
```

This example uses the product of the two rewards and samples the reward exponent beta uniformly from 1 to 64. To learn different objective trade-offs, set `reward.moo_preferences: "uniform"`. See [multi-objective rewards and conditioning](docs/rewards.md#combine-multiple-objectives).

## Sampling

Generate molecules from a trained checkpoint and save them with their routes to CSV:

```bash
python scripts/sample.py \
  --checkpoint runs/qed_sa/checkpoints/latest.ckpt \
  --num-samples 100 \
  --output samples.csv
```

Omitted beta and preferences use their training settings. Use `--beta 32` to select a fixed exponent supported by the trained model. Both full checkpoints and [extracted sampling models](docs/training.md#extract-a-sampling-model) are accepted. See [sampling](docs/training.md#sampling) for batch size, objective preferences and output formats.

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
