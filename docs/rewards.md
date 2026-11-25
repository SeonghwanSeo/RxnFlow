# Reward functions

Implement either method on `RewardFunction`:

| Method | When to use it | Return value |
| --- | --- | --- |
| `score(smiles, cond=None)` | Compute a descriptor or other score one molecule at a time. The default `score_list` calls this method for each molecule and its condition. | One finite `float`. |
| `score_list(smiles_list, cond_list=None)` | Evaluate several molecules together, for example in a model inference batch. | A `list[float]` with one finite score per input, in input order. |

Call the reward as `reward(smiles_list, cond_list=conditions)` with either implementation. A condition is a dictionary for one molecule. In a batch, `cond_list[i]` belongs to `smiles_list[i]`, including when the same SMILES appears with different targets.

Higher scores are more favorable. With one objective, no prior and unit property penalties, the target is `p(x) ∝ exp(beta * R(x))`. The [QED example](../examples/qed.py) implements single-molecule scoring:

```python
from rdkit import Chem
from rdkit.Chem import QED
from rxnflow import RewardFunction


class QEDReward(RewardFunction):
    name = "qed"

    # example of `score()`
    def score(self, smiles: str, cond=None) -> float:
        return QED.qed(Chem.MolFromSmiles(smiles))

    # example of `score_list()`
    def score_list(self, smiles_list: list[str], cond_list=None) -> list[float]:
        return [QED.qed(Chem.MolFromSmiles(s)) for s in smiles_list]
```

## Pass reward settings

Pass constructor arguments when creating the reward in Python.
For example, reward molecules whose molecular weight is close to a target:

```python
from rdkit import Chem
from rdkit.Chem import Descriptors
from rxnflow import RewardFunction


class MWReward(RewardFunction):
    name = "mw"

    def __init__(self, target: float, scale: float = 100.0):
        self.target = target
        self.scale = scale

    def score(self, smiles: str, cond=None) -> float:
        mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
        return -abs(mw - self.target) / self.scale
```

For a target of 300 Da, pass this instance as the trainer’s `reward` argument:

```python
reward = MWReward(target=300.0, scale=100.0)
```

Initialize any scoring model or docking setup in the constructor.

## Use a target for each molecule

Pass one condition dictionary per SMILES. The reward receives each molecule's target in `cond` and scores its absolute TPSA error in units of 100 Å².

```python
from rdkit import Chem
from rdkit.Chem import Descriptors
from rxnflow import RewardFunction


class CondTPSAReward(RewardFunction):
    name = "tpsa"

    def score(self, smiles: str, cond=None) -> float:
        assert cond is not None
        tpsa = Descriptors.TPSA(Chem.MolFromSmiles(smiles))
        return -abs(tpsa - cond["tpsa"]) / 100.0


reward = CondTPSAReward()
scores = reward(
    ["CCO", "CCO"],
    cond_list=[{"tpsa": 40.0}, {"tpsa": 80.0}],
)
```

The result contains two scores, one for each ethanol–target pair.
See [external conditions](conditioning.md#external-conditions) to train and sample with these targets.

## Define multiple objectives

Give each reward class a unique, nonempty string `name` and pass one instance or a list of instances as `reward`. For multiple rewards, list order determines the order of preference weights and logged objective scores.

```python
from examples.mw_logp import MWReward, LogPReward

rewards = [MWReward(scale=100.0), LogPReward()]
```

With equal weights and arithmetic scalarization, a reduction of 100 Da contributes as much as an increase of one logP unit. See the [MW/logP config](../configs/examples/mw_logp.yaml).

## Training and conditioning

- [Reward conditioning](conditioning.md): combine objectives, choose preference distributions and set task beta.
- [Training guide](training.md): train, resume and sample with your reward.
- [Chemical prior guidance](prior_guidance.md)
