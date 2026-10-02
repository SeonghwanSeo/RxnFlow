# Custom rewards and MOO

Implement `RewardFunction.score()` and pass the reward to `RxnFlowTrainer`:

```python
import numpy as np
from rdkit.Chem import QED

from rxnflow.config import Config
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class MyReward(RewardFunction):
    objectives = ("qed",)

    def score(self, molecules):
        return np.array([QED.qed(mol) for mol in molecules], dtype=np.float32).reshape(-1, 1)


trainer = RxnFlowTrainer(Config.from_file("configs/qed.yaml"), MyReward())
trainer.run()
```

- Return `[batch, num_objectives]`, including empty batches, as a float32 NumPy array.
- Scores must be finite, non-negative and larger-is-better; normalize each objective yourself.
- Columns follow `objectives` order.
- Optional `filter_object(mol)` skips evaluation and assigns zero objective values; the reward floor still applies.

See [QED/SA](../examples/qed_sa.py) for a two-objective example: `[QED, (10 - SA score) / 9]`.

## Combining objectives

```yaml
reward:
  beta: "uniform(1,64)"
  preferences: "none"
  scalarization: mul
  floor: 0.0001
```

| Mode | Effective weight sum | Log reward | Without preference conditioning |
| --- | --- | --- | --- |
| `mul` (default) | Number of objectives N | `sum(w_i * log(max(r_i, floor)))` | Simple product. |
| `sum` | 1 | `log(max(sum(w_i * r_i), floor))` | Arithmetic mean. |

Training uses `beta * log R`; logged scalar rewards are before beta.

## Conditions

| Setting | Behavior |
| --- | --- |
| `beta: "32"` | Fixed reward exponent. |
| `beta: "uniform(1,64)"` | Sample the exponent for each trajectory. |
| `preferences: "none"` (default) | Equal weights, no preference encoder. Beta conditioning remains active. |
| `preferences: "uniform"` | Sample a simplex-uniform direction, then normalize for the selected scalarization. |
| `preferences: "dirichlet(0.5)"` | Change the preference distribution's concentration. |
| `preferences: "fixed(0.3,0.7)"` | Fixed relative importance: `[0.6, 1.4]` for `mul`, `[0.3, 0.7]` for `sum`. |

Replay retains the original beta and effective weights. To select trade-offs during sampling, train with preference conditioning enabled; it cannot be enabled only at sampling time.

YAML/CLI use the strings above. Python uses tuples such as `beta=("fixed", [32.0])` and `preferences=("fixed", [0.3, 0.7])`. Omitted sampling preferences use the checkpoint setting.
