# Reward conditioning

Set `reward.beta`, `reward.moo_scalarization` and `reward.moo_preference` in your training configuration. Return individual objective scores from your [reward class](rewards.md); RxnFlow combines them and applies the reward exponent.

## Reward exponent

`beta` controls how strongly the model favors high rewards through `R ** beta`. Larger values increase this preference.

| Setting | Use |
| --- | --- |
| `beta: "32"` | Train with a fixed exponent. |
| `beta: "uniform(1,64)"` | Train across uniformly sampled exponents. |
| `beta: "log_uniform(1,64)"` | Sample exponents uniformly in log space, giving equal probability to intervals with the same ratio, such as `[1,2]` and `[2,4]`. |

Beta must be positive. Both range distributions require `0 < lower < upper`; bounds are beta values, not their logarithms.

## Combine objectives

Choose how scores `r_i` and weights `w_i` form the reward `R`:

| `moo_scalarization` | Reward | Equal weights |
| --- | --- | --- |
| `mul` | `R = product(r_i ** w_i)` | Product of scores. |
| `sum` | `R = sum(w_i * r_i)` | Arithmetic mean of scores. |

With `mul`, a low score in one objective reduces the reward even when another is high. With `sum`, higher scores can compensate for lower scores. Choose comparable score scales so one objective does not dominate unintentionally.

Weights are normalized to sum to the number of objectives for `mul`, or to one for `sum`. For two objectives, `fixed(0.3,0.7)` gives weights `[0.6, 1.4]` for `mul` and `[0.3, 0.7]` for `sum`.

## Property rewards

Molecules exceeding `property_penalty` limits or `generation.max_atoms` receive lower rewards, independently of MOO preferences.

- `reward.property_penalty_ratio` controls how gradually rewards decrease beyond a limit. The default `0.2` uses 20% of the limit as the smoothing width; larger values give gentler penalties.
- If the ratio or a limit is zero, violating that limit gives zero reward.
- Molecules and trajectories are retained even when they exceed a limit.

## Objective preferences

Preferences follow the column order in your reward's `objectives` tuple.

| `moo_preference` | Use |
| --- | --- |
| `"none"` | Equal weights throughout training and sampling. |
| `"fixed(0.3,0.7)"` | Fixed relative weights for two objectives. |
| `"dirichlet(1.5)"` | Sample preferences with more emphasis near the center of the simplex. |
| `"dirichlet(1)"` | Sample uniformly on the preference simplex. |
| `"dirichlet(0.5)"` | Sample preferences with more emphasis near the simplex boundaries. |

Fixed weights require one non-negative value per objective and a positive total.

`dirichlet` without parameters uses `dirichlet(1.5)`. Dirichlet concentrations must be positive. One value is shared across objectives: with three objectives, `dirichlet(0.5)` is equivalent to `dirichlet(0.5,0.5,0.5)`. You can also supply one concentration per objective. Equal concentrations of `1` sample uniformly on the preference simplex; values below `1` favor the boundaries, while values above `1` favor balanced preferences.

For example, to train across both exponents and trade-offs:

```yaml
reward:
  beta: "uniform(1,64)"
  moo_scalarization: mul
  moo_preference: "dirichlet"
```

## Choose conditions when sampling

Omitted beta and preferences reuse their training settings.

- Fixed-beta training requires the same beta at sampling. Uniform or log-uniform training allows a fixed value, uniform range or log-uniform range inside the training range.
- `none` keeps equal weights. Fixed-preference training requires the same relative weights.
- Training with `dirichlet` or `dirichlet(...)` allows you to choose a fixed trade-off or another preference distribution at sampling.

Use `--beta` and `--preference` in the [sampling command](training.md#sampling), or the corresponding arguments to `RxnFlowSampler.sample()`.

## External conditions

External conditions describe the generation target, such as a pocket key or target TPSA. Each trajectory retains one `dict[str, Any]`, for example `{"tpsa": 80.0}`. Beta and objective preferences remain separate settings.

Implement a `ConditioningModule` with:

- `encoding_dim`: the fixed number of model input features.
- `sample(num_samples, rng)`: return one raw condition dictionary per trajectory using the supplied NumPy generator.
- `encode(cond_list)`: return a deterministic float32 NumPy array of shape `[len(cond_list), encoding_dim]`, preserving order.

The supplied RNG is seeded and saved in training checkpoints. Use JSON-compatible condition values for sample logs and JSON exports.

The conditioning module selects and encodes targets; the [reward function](rewards.md) evaluates molecules against the raw targets.

### Train with target TPSA

The [TPSA example](../examples/tpsa_cond.py) samples target TPSA values, divides them by 100 for model input, and rewards molecules with `1 / (1 + abs(TPSA - target))`. Its [configuration](../configs/tpsa_cond.yaml) supplies constructor settings:

```yaml
conditioning:
  min_tpsa: 0
  max_tpsa: 200
```

Set `env_dir` in the YAML to your compatible prepared environment, then run from the repository root:

```bash
python examples/tpsa_cond.py --config configs/tpsa_cond.yaml --steps 1000
```

In custom scripts, construct `TPSAConditioningModule(**config.conditioning)` and pass it as `conditioning=` to `RxnFlowTrainer`. Use the same module and settings when [resuming](training.md#resume-training).

### Sample with explicit targets

Provide `cond_list` to choose the targets yourself. The module's `sample()` method is then skipped; only `encode()` is used. Supply one condition per generation attempt, including attempts that may fail:

```python
from examples.tpsa_cond import TPSAConditioningModule
from rxnflow import Config, RxnFlowSampler

config = Config.from_file("runs/tpsa_cond/config.yaml")
sampler = RxnFlowSampler(
    "runs/tpsa_cond/checkpoints/latest.ckpt",
    conditioning=TPSAConditioningModule(**config.conditioning),
    device="cuda",
)
results = sampler.sample(
    num_samples=100,
    cond_list=[{"tpsa": 80.0} for _ in range(100)],
    seed=1,
)
sampler.write(results, "samples_tpsa80.jsonl")
```

Omit `cond_list` to sample targets through the module. Keep the training feature mapping unchanged. External conditioning uses this Python interface; the generic sampling CLI does not construct custom modules.

Each raw condition field appears as `cond_<key>` in sampling metadata and training `sample.jsonl` records, such as `cond_tpsa: 80.0`.
