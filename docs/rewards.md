# Custom rewards

Implement `RewardFunction.score()` to return one score per objective for each molecule:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rxnflow import RewardFunction


class MyReward(RewardFunction):
    objectives = ("qed",)

    def score(self, mols: list[Chem.Mol]) -> np.ndarray:
        qeds = [QED.qed(mol) for mol in mols]
        return np.array(qeds, dtype=np.float32).reshape(-1, 1)
```

- Rewards must be non-negative. The scores will be automatically clipped `1e-5` for numerical stability.
- Return a float32 NumPy array of shape `[batch, num_objectives]`, including for an empty batch. Columns follow the order in `objectives`.
- Return the individual objective scores; RxnFlow applies preference weights.
- Assign zero scores to unwanted molecules inside `score()`, preserving the input order and batch size. This lowers their reward; it does not remove them from sample outputs.

For direct evaluation, use `reward.run(mols)` or `reward(mols)`. See [MW/logP](../examples/mw_logp.py) for a runnable two-objective example.

## Pass reward settings

Use `reward.settings` to configure your reward's constructor. For example, reward molecules whose molecular weight is close to a target. The distance `abs(mw - target_mw)` is converted to `1 / (1 + distance)` so closer molecules receive higher rewards:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rxnflow import RewardFunction


class MolecularWeightReward(RewardFunction):
    objectives = ("mw_target",)

    def __init__(self, target_mw: float):
        self.target_mw = target_mw

    def score(self, mols: list[Chem.Mol]) -> np.ndarray:
        rewards = [
            1 / (1 + abs(Descriptors.ExactMolWt(mol) - self.target_mw))
            for mol in mols
        ]
        return np.array(rewards, dtype=np.float32).reshape(-1, 1)
```

Set the target molecular weight in your training YAML, as in [mw_cond.yaml](../configs/mw_cond.yaml):

```yaml
reward:
  settings:
    target_mw: 300.0
```

Pass these settings when constructing the reward in your training script:

```python
from rxnflow import Config

config = Config.from_file("configs/mw_cond.yaml")
reward = MolecularWeightReward(**config.reward.settings)
```

Setting names must match the constructor arguments. The same approach can pass a target protein structure path (`target_path`) and docking options (`docking_settings`) to a binding-affinity reward. Initialize the scoring model or docking setup in the constructor, then evaluate molecules in `score()`.

## Define multiple objectives

For multi-objective optimization, return one column per objective instead of combining the scores inside `score()`. For example, minimize molecular weight and maximize logP:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rxnflow import RewardFunction


class MWLogPReward(RewardFunction):
    objectives = ("mw", "logp")

    def __init__(self, mw_scale: float = 300.0, logp_scale: float = 1.0):
        self.mw_scale = mw_scale
        self.logp_scale = logp_scale

    def score(self, mols: list[Chem.Mol]) -> np.ndarray:
        # Convert both properties to positive rewards: lower MW and higher logP.
        rewards = [
            [
                1 / (1 + Descriptors.ExactMolWt(mol) / self.mw_scale),
                1 / (1 + np.exp(-Descriptors.MolLogP(mol) / self.logp_scale)),
            ]
            for mol in mols
        ]
        return np.array(rewards, dtype=np.float32).reshape(-1, 2)
```

The result has shape `[batch, 2]`, with MW rewards first and logP rewards second, matching `objectives`. The inverse transform rewards lower MW; the sigmoid rewards higher logP, including when raw logP is negative. Positive `mw_scale` and `logp_scale` control each score's sensitivity.

```yaml
reward:
  beta: "uniform(1,64)"
  moo_scalarization: mul
  moo_preferences: "dirichlet(1.5)"
```

Construct `MWLogPReward()` in your training script. The [mw_logp.yaml](../configs/mw_logp.yaml) configuration varies beta and objective preferences. This is a descriptor optimization example; the extent of the trade-off depends on the available molecules.

## Training and conditioning

- [Reward conditioning](conditioning.md): combine objectives, choose preference distributions and set the reward exponent.
- [Training guide](training.md): train, resume and sample with your reward.
