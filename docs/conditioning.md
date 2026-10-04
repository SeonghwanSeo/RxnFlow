# Reward conditioning

Set `reward.beta`, `reward.moo_scalarization` and `reward.moo_preferences` in your training configuration. Return individual objective scores from your [reward class](rewards.md); RxnFlow combines them and applies the reward exponent.

## Reward exponent

`beta` controls how strongly the model favors high rewards through `R ** beta`. Larger values increase this preference.

| Setting | Use |
| --- | --- |
| `beta: "32"` | Train with a fixed exponent. |
| `beta: "uniform(1,64)"` | Train across uniformly sampled exponents. |

Beta must be positive. A uniform range requires `lower < upper`.

## Combine objectives

Choose how scores `r_i` and weights `w_i` form the reward `R`:

| `moo_scalarization` | Reward | Equal weights |
| --- | --- | --- |
| `mul` | `R = product(r_i ** w_i)` | Product of scores. |
| `sum` | `R = sum(w_i * r_i)` | Arithmetic mean of scores. |

With `mul`, a low score in one objective reduces the reward even when another is high. With `sum`, higher scores can compensate for lower scores. Choose comparable score scales so one objective does not dominate unintentionally.

Weights are normalized to sum to the number of objectives for `mul`, or to one for `sum`. For two objectives, `fixed(0.3,0.7)` gives weights `[0.6, 1.4]` for `mul` and `[0.3, 0.7]` for `sum`.

`training.reward_floor` is applied before taking logarithms: to each objective score for `mul`, or to the combined reward for `sum`.

## Objective preferences

Preferences follow the column order in your reward's `objectives` tuple.

| `moo_preferences` | Use |
| --- | --- |
| `"none"` | Equal weights throughout training and sampling. |
| `"fixed(0.3,0.7)"` | Fixed relative weights for two objectives. |
| `"uniform"` | Sample uniformly on the preference simplex. |
| `"dirichlet(0.5)"` | Sample preferences with more emphasis near the simplex boundaries. |

Fixed weights require one non-negative value per objective and a positive total.

Dirichlet concentrations must be positive. One value is shared across objectives: with three objectives, `dirichlet(0.5)` is equivalent to `dirichlet(0.5,0.5,0.5)`. You can also supply one concentration per objective. Equal concentrations of `1` give the same distribution as `uniform`; values below `1` favor the boundaries, while values above `1` favor balanced preferences.

For example, to train across both exponents and trade-offs:

```yaml
reward:
  beta: "uniform(1,64)"
  moo_scalarization: mul
  moo_preferences: "uniform"
```

## Choose conditions when sampling

Omitted beta and preferences reuse their training settings.

- Fixed-beta training requires the same beta at sampling. Uniform-beta training allows a fixed value or uniform subrange inside the training range.
- `none` keeps equal weights. Fixed-preference training requires the same relative weights.
- Training with `uniform` or `dirichlet(...)` allows you to choose a fixed trade-off or another preference distribution at sampling.

Use `--beta` and `--preferences` in the [sampling command](training.md#sampling), or the corresponding arguments to `RxnFlowSampler.sample()`.
