# Custom rewards and MOO

Implement `RewardFunction.score()` to return one score per objective for each molecule, then pass your reward to the trainer:

```python
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import QED

from rxnflow.config import Config
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class MyReward(RewardFunction):
    objectives = ("qed",)

    def score(self, mols: list[Chem.Mol]) -> NDArray[np.float32]:
        return np.array([QED.qed(mol) for mol in mols], dtype=np.float32).reshape(-1, 1)


if __name__ == "__main__":
    config = Config.from_file("configs/qed.yaml")
    reward = MyReward()
    trainer = RxnFlowTrainer(config, reward)
    trainer.run()
```

- Return a float32 NumPy array of shape `[batch, num_objectives]`, including for an empty batch. Columns follow the order in `objectives`.
- Scores must be finite, non-negative and larger-is-better. Transform or scale each objective to reflect your optimization goal.
- Return the individual objective scores; RxnFlow applies preference weights and the reward exponent separately.
- Assign zero scores to unwanted molecules inside `score()`, preserving the input order and batch size. This lowers their reward; it does not remove them from sample outputs.

For direct evaluation, use `reward.run(mols)` or `reward(mols)`. See [QED/SA](../examples/qed_sa.py) for a two-objective example returning `[QED, (10 - SA score) / 9]`.

## Combine multiple objectives

Choose how objective scores contribute to the combined reward:

```yaml
reward:
  moo_scalarization: mul
  moo_preferences: "none"
  beta: "32"
```

- **`mul` (default):** weighted product, `R = product(r_i ** w_i)`. With equal weights, this is the simple product of the objective scores. A low score in one objective reduces the combined reward even when another is high.
- **`sum`:** weighted sum, `R = sum(w_i * r_i)`. With equal weights, this is the arithmetic mean. A high score in one objective can compensate for a low score in another.

RxnFlow normalizes weights to sum to the number of objectives for `mul`, or to one for `sum`. For example, `moo_preferences: "fixed(0.3,0.7)"` gives weights `[0.6, 1.4]` for `mul` and `[0.3, 0.7]` for `sum`.

## Set the reward exponent

`beta` controls how strongly sampling favors high rewards through `R ** beta`. Larger values put more emphasis on high-reward molecules.

- `beta: "32"`: train with a fixed exponent.
- `beta: "uniform(1,64)"`: train across exponents sampled uniformly between 1 and 64. You can then choose a fixed beta within that range when sampling.

## Choose objective trade-offs

Use `moo_preferences` to set the relative importance of the objectives, in `objectives` order:

- **`"none"` (default):** use equal weights without preference conditioning.
- **`"fixed(0.3,0.7)"`:** use one fixed trade-off throughout training.
- **`"uniform"`:** sample uniformly over non-negative weight vectors that sum to one, covering different trade-offs.
- **`"dirichlet(0.5)"`:** favor weights closer to the extremes, emphasizing individual objectives more often than `"uniform"`.

To choose different trade-offs at sampling time, train with varying preferences such as `"uniform"` or `"dirichlet(0.5)"`. A model trained with `"none"` cannot enable preference conditioning only at sampling time. See the [sampling guide](training.md#sampling) for how to select beta and preferences.
