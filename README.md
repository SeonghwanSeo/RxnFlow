[![Python versions](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# RxnFlow: Generative Flows on Synthetic Pathways for Drug Design

Official implementation of ***Generative Flows on Synthetic Pathways for Drug Design*** by Seonghwan Seo, Minsu Kim, Tony Shen, Martin Ester, Jinkyu Park, Sungsoo Ahn, and Woo Youn Kim.

[paper]

RxnFlow are a synthesis-oriented generative framework that aims to discover diverse drug candidates through GFlowNet objective and a large action space.

- RxnFlow can operate on large synthetic action spaces comprising 1.2M building blocks and 71 reaction templates without memory overhead.
- RxnFlow can explore broader chemical space within less reaction steps, resulting in higher diversity, higher potency, and lower synthetic complexity of generated molecules.
- RxnFlow can generate molecules with expanded or modified building block libaries without retraining.

The implementation of this project builds upon the [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) with MIT license.

## Setup

### Install

```bash
# python: 3.10
conda install openbabel
pip install -e . --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

### Data

To construct the synthetic action space, RxnFlow requires the reaction teamplate set and the building block library.

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
