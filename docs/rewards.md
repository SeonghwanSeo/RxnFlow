# Custom rewards

Implement `RewardFunction.score()` to accept a list of SMILES strings and return one score per objective for each molecule:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rxnflow import RewardFunction


class MyReward(RewardFunction):
    objectives = ("qed",)

    def score(self, smiles_list: list[str]) -> np.ndarray:
        """
        inputs: list of unique, uncached SMILES strings.
        outputs: NumPy array of shape [batch, num_objectives]
        """
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        qeds = [QED.qed(m) for m in mols]
        rewards = np.array(qeds, dtype=np.float32).reshape(-1, 1)
        return rewards  # [batch, 1]
```

- Return finite, non-negative scores.
- Return a NumPy array of shape `[batch, num_objectives]`, preserving input molecule order and objectives column order.
- Return the individual objective scores; RxnFlow applies preference weights and the configured [terminal property penalty](conditioning.md#property-rewards).
- Scores are cached by SMILES in the `_cache` dictionary.
- `run()` and `__call__()` accept `list[str | None]`, including duplicates and `None` for molecules that failed to generate. `score()` receives only unique, uncached SMILES strings.

## Pass reward settings

Use `reward.settings` to configure your reward's constructor. For example, reward molecules whose molecular weight is close to a target:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rxnflow import RewardFunction


class MolecularWeightReward(RewardFunction):
    objectives = ("mw_target",)

    def __init__(self, target_mw: float):
        self.target_mw = target_mw

    def score(self, smiles_list: list[str]) -> np.ndarray:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        mws = [Descriptors.ExactMolWt(m) for m in mols]
        mws = np.array(mws)
        rewards = 1 / (1 + np.abs(mws - self.target_mw))
        return rewards.reshape(-1, 1)  # [batch, 1]
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

    def score(self, smiles_list: list[str]) -> np.ndarray:
        # Convert both properties to positive rewards: lower MW and higher logP.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        mws = np.array([Descriptors.ExactMolWt(m) for m in mols])
        logps = np.array([Descriptors.MolLogP(m) for m in mols])
        mw_rewards = 1 / (1 + mws / self.mw_scale)
        logp_rewards = 1 / (1 + np.exp(-logps / self.logp_scale))
        rewards = np.stack([mw_rewards, logp_rewards], axis=1)
        return rewards  # [batch, 2]
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
