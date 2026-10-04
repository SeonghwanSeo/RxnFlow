# Custom rewards and MOO

Implement `RewardFunction.score()` to return one score per objective for each molecule, then pass your reward to the trainer:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rxnflow import Config, RewardFunction, RxnFlowTrainer


class MyReward(RewardFunction):
    objectives = ("qed",)

    def score(self, mols: list[Chem.Mol]) -> np.ndarray:
        qeds = [QED.qed(mol) for mol in mols]
        return np.array(qeds, dtype=np.float32).reshape(-1, 1)


if __name__ == "__main__":
    config = Config.from_file("configs/qed.yaml")
    reward = MyReward()
    trainer = RxnFlowTrainer(config, reward, output_dir="runs/custom_reward")
    trainer.run(1000)
```

- Rewards must be non-negative. The scores will be automatically clipped `1e-5` for numerical stability.
- Return a float32 NumPy array of shape `[batch, num_objectives]`, including for an empty batch. Columns follow the order in `objectives`.
- Return the individual objective scores; RxnFlow applies preference weights.
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

RxnFlow normalizes weights to sum to the number of objectives for `mul`, or to one for `sum`.
For example, `moo_preferences: "fixed(0.3,0.7)"` gives weights `[0.6, 1.4]` for `mul` and `[0.3, 0.7]` for `sum`.

## Set the reward exponent

`beta` controls how strongly sampling favors high rewards through `R ** beta`. Larger values put more emphasis on high-reward molecules.

- `beta: "32"`: train with a fixed exponent.
- `beta: "uniform(1,64)"`: train across exponents sampled uniformly between 1 and 64. You can then choose a fixed beta within that range when sampling.

## Choose objective trade-offs

Use `moo_preferences` to set the relative importance of the objectives, in `objectives` order:

- **`"none"` (default):** use fixed equal weights for all objectives.
- **`"dirichlet(1.5)"`:** sample from a Dirichlet distribution.
- **`"uniform"`:** sample uniformly over non-negative weight vectors that sum to one, covering different trade-offs.
- **`"fixed(0.3,0.7)"`:** use fixed weights.

To choose different trade-offs at sampling time, train with varying preferences such as `"uniform"` or `"dirichlet(1.5)"`.
A model trained with `"none"` keeps the same equal weights at sampling time.
See the [sampling guide](training.md#sampling) for how to select beta and preferences.
