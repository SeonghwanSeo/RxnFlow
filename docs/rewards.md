# Custom rewards

Implement `RewardFunction.score(smiles_list, cond_list=None)` to return one score per objective for each molecule. `cond_list` contains raw external condition dictionaries (`dict[str, Any]`) in the same order as `smiles_list`; condition-independent rewards can ignore it:

```python
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rxnflow import RewardFunction


class MyReward(RewardFunction):
    objectives = ("qed",)

    def score(
        self, smiles_list: list[str], cond_list: list[dict[str, Any]] | None = None
    ) -> np.ndarray:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        qeds = [QED.qed(m) for m in mols]
        rewards = np.array(qeds, dtype=np.float32).reshape(-1, 1)
        return rewards  # [batch, 1]
```

- Return a float32 array of shape `[batch, num_objectives]` with finite, non-negative scores; larger values should be better.
- Preserve input order and the column order in `objectives`. RxnFlow applies preference weights and [terminal property penalties](conditioning.md#property-rewards).
- Call `reward(smiles_list, cond_list=conditions)` or `reward.run(...)`. Failed molecules (`None`) receive zero scores.
- Without conditions, duplicate and cached SMILES are scored once. With conditions, every valid pair is scored without caching or deduplication.

## Pass reward settings

Use `reward.settings` to configure your reward's constructor. For example, reward molecules whose molecular weight is close to a target:

```python
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rxnflow import RewardFunction


class MolecularWeightReward(RewardFunction):
    objectives = ("mw_target",)

    def __init__(self, target_mw: float):
        self.target_mw = target_mw

    def score(
        self, smiles_list: list[str], cond_list: list[dict[str, Any]] | None = None
    ) -> np.ndarray:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        mws = [Descriptors.ExactMolWt(m) for m in mols]
        mws = np.array(mws)
        rewards = 1 / (1 + np.abs(mws - self.target_mw))
        return rewards.astype(np.float32).reshape(-1, 1)  # [batch, 1]
```

Set the target molecular weight in your training YAML, as in [mw.yaml](../configs/mw.yaml):

```yaml
reward:
  settings:
    target_mw: 300.0
```

Pass these settings when constructing the reward in your training script:

```python
from rxnflow import Config

config = Config.from_file("configs/mw.yaml")
reward = MolecularWeightReward(**config.reward.settings)
```

Setting names must match the constructor arguments. Initialize any scoring model or docking setup in the constructor.

## Use a target for each molecule

Read each target from `cond_list` when it varies between molecules. For example, the [TPSA reward](../examples/tpsa_cond.py) compares each molecule with its supplied target:

```python
from examples.tpsa_cond import TPSAReward

reward = TPSAReward()
scores = reward(
    ["CCO", "CCO", None],
    cond_list=[{"tpsa": 40.0}, {"tpsa": 80.0}, {"tpsa": 40.0}],
)
```

The two ethanol entries use different targets; the failed entry receives zero. See [external conditions](conditioning.md#external-conditions) to train and sample with these targets.

## Define multiple objectives

For multi-objective optimization, return one column per objective instead of combining the scores inside `score()`. For example, minimize molecular weight and maximize logP:

```python
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rxnflow import RewardFunction


class MWLogPReward(RewardFunction):
    objectives = ("mw", "logp")

    def __init__(self, mw_scale: float = 300.0, logp_scale: float = 1.0):
        self.mw_scale = mw_scale
        self.logp_scale = logp_scale

    def score(
        self, smiles_list: list[str], cond_list: list[dict[str, Any]] | None = None
    ) -> np.ndarray:
        # Convert both properties to positive rewards: lower MW and higher logP.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        mws = np.array([Descriptors.ExactMolWt(m) for m in mols])
        logps = np.array([Descriptors.MolLogP(m) for m in mols])
        mw_rewards = 1 / (1 + mws / self.mw_scale)
        logp_rewards = 1 / (1 + np.exp(-logps / self.logp_scale))
        rewards = np.stack([mw_rewards, logp_rewards], axis=1)
        return rewards.astype(np.float32)  # [batch, 2]
```

The result has shape `[batch, 2]`, with MW rewards first and logP rewards second, matching `objectives`. Positive `mw_scale` and `logp_scale` control each score's sensitivity.

```yaml
reward:
  beta: "uniform(1,64)"
  moo_scalarization: mul
  moo_preference: "dirichlet(1.5)"
```

Construct `MWLogPReward()` in your training script. The [mw_logp.yaml](../configs/mw_logp.yaml) configuration varies beta and objective preferences. This is a descriptor optimization example; the extent of the trade-off depends on the available molecules.

## Training and conditioning

- [Reward conditioning](conditioning.md): combine objectives, choose preference distributions and set the reward exponent.
- [Training guide](training.md): train, resume and sample with your reward.
