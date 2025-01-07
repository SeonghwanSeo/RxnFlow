[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2410.04542)
[![Python versions](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# RxnFlow: Generative Flows on Synthetic Pathway for Drug Design

This project is based on gflownet, and `src/gflownet/` is a clone of [recursionpharma/gflownet@v0.2.0](https://github@v0.2.0.com/recursionpharma/gflownet/tree/v0@v0.2.0.2@v0.2.0.0).
Since we have updated the gflownet version and performed modularization after submission, we do not guarantee that current version will reproduce the same results as the paper.

## Setup

### Install

```bash
# python: 3.12
pip install -e . --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

### Data

To construct the synthetic action space, RxnFlow requires the reaction template set and the building block library.
We provide two reaction template set:

- We provide the reaction template set [data/template/real/] from Enamine REAL synthesis protocol (ref: [Gao et al.](https://github.com/wenhao-gao/synformer)).

The Enamine building block library is available upon request at [https://enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog). We used the "Comprehensive Catalog" released at 2024.06.10.

- Use Comprehensive Catalog

  ```bash
  cd data
  python scripts/a_sdf_to_smi.py -b <CATALOG-SDF> -o ./building_blocks/enamine_blocks.smi --cpu <CPU>
  python scripts/b_create_env.py -b ./building_blocks/enamine_blocks.smi --env_dir ./envs/real/ --cpu <CPU>
  ```

- Use custom SMILES file (`.smi`)

  ```bash
  python scripts/a_sdf_to_smi.py -b <BLOCK-SMI> -o ./building_blocks/<CLEAN-SMI> --cpu <CPU>
  python scripts/b_create_env.py -b ./building_blocks/<CLEAN-SMI> --env_dir ./envs/custom/ --cpu <CPU>
  ```

## Experiments

## Citation

If you use our code in your research, we kindly ask that you consider citing our work in papers:

```bibtex
@article{shen2025???,
  title={???},
  author={???},
  journal={arXiv preprint arXiv:????.?????},
  year={2025}
}
@article{seo2024generative,
  title={Generative Flows on Synthetic Pathway for Drug Design},
  author={Seo, Seonghwan and Kim, Minsu and Shen, Tony and Ester, Martin and Park, Jinkyoo and Ahn, Sungsoo and Kim, Woo Youn},
  journal={arXiv preprint arXiv:2410.04542},
  year={2024}
}
@article{shen2024tacogfn,
  title={TacoGFN: Target Conditioned GFlowNet for Structure-Based Drug Design},
  author={Shen, Tony and Seo, Seonghwan and Lee, Grayson and Pandey, Mohit and Smith, Jason R and Cherkasov, Artem and Kim, Woo Youn and Ester, Martin},
  journal={arXiv preprint arXiv:2310.03223},
  year={2024},
  note={Published in Transactions on Machine Learning Research(TMLR)}
}
@article{seo2023molecular,
  title={Molecular generative model via retrosynthetically prepared chemical building block assembly},
  author={Seo, Seonghwan and Lim, Jaechang and Kim, Woo Youn},
  journal={Advanced Science},
  volume={10},
  number={8},
  pages={2206674},
  year={2023},
  publisher={Wiley Online Library}
}

```

## Related Works

- [GFlowNet](https://arxiv.org/abs/2106.04399) [github: [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet)]
- [TacoGFN](https://arxiv.org/abs/2310.03223) [github: [tsa87/TacoGFN-SBDD](https://github.com/tsa87/TacoGFN-SBDD)]
- [PharmacoNet](https://doi.org/10.1039/D4SC04854G) [github: [SeonghwanSeo/PharmacoNet](https://github.com/SeonghwanSeo/PharmacoNet)]
- [SemlaFlow](https://arxiv.org/abs/2406.07266) [github: [rssrwn/semla-flow](https://github.com/rssrwn/semla-flow)]
