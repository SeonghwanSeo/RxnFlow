# RxnFlow: Generative Flows on Synthetic Pathway for Drug Design

[![arXiv](https://img.shields.io/badge/arXiv-2410.04542-b31b1b.svg)](https://arxiv.org/abs/2410.04542)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

RxnFlow uses generative flow networks to design molecules through synthesis pathways. Train it with a molecular reward function, then sample molecules together with their reaction paths and building-block sources.

This repository accompanies the ICLR paper **Generative Flows on Synthetic Pathway for Drug Design** by Seonghwan Seo, Minsu Kim, Tony Shen, Martin Ester, Jinkyu Park, Sungsoo Ahn, and Woo Youn Kim. [[Paper](https://arxiv.org/abs/2410.04542)]

**v0.9.0 is a preview release of the core architecture behind Hyper Screening X (HSX), developed with eMolecules and HITS. It does not include the full implementation described in the HSX preprint.**

## Installation

RxnFlow requires Python 3.10 or later. Install from the repository root:

```bash
pip install -e .
```

### Data preparation

Prepare an eMolecules or Enamine building-block catalog with the supplied reaction templates:

```bash
python scripts/prepare.py \
  --building-blocks /path/to/building_blocks.smi \
  --config data/templates/basic/config.yaml \
  --env-dir /path/to/prepared/environment \
  --num-workers 16 \
  --max-atoms 30 \
  --min-library-size 10
```

Set `data.env_dir` and `run.output_dir` in your chosen configuration. See the [environment guide](docs/environment.md) for catalog format and preparation details.

## Single-objective optimization

Train with QED as the reward:

```bash
python examples/qed.py --config configs/qed.yaml
```

See [training and restart](docs/training.md) and [custom rewards](docs/rewards.md) for other objectives.

## Multi-objective optimization

Train on QED and synthetic accessibility, with SA reward `(10 - SA score) / 9`:

```bash
python examples/qed_sa.py --config configs/qed_sa.yaml
```

The default reward is the product of QED and SA rewards, with beta sampled uniformly from 1 to 64. Preference conditioning is optional: set `reward.moo_preferences: "uniform"` to train across trade-offs. See [MOO rewards and conditioning](docs/rewards.md).

## Sampling

Replace the checkpoint path with your trained model and choose a beta value from its training range:

```bash
python scripts/sample.py \
  --checkpoint runs/qed_sa/checkpoints/latest.ckpt \
  --num-samples 100 \
  --beta 32 \
  --output samples.json
```

For preference-conditioned checkpoints, use `--preferences "fixed(0.3,0.7)"` to choose a trade-off. See the [sampling guide](docs/training.md#sampling) for formats and reward evaluation.

## Documentation

[Environment preparation](docs/environment.md) · [Configuration, training and sampling](docs/training.md) · [Custom rewards and MOO](docs/rewards.md)

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
