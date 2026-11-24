# Reward conditioning

Set `gflownet.beta` and `gflownet.moo_preference` in your training configuration.
Return individual objective scores from your [reward class](rewards.md); RxnFlow combines them and multiplies the result by beta.

## Task beta

Larger `gflownet.beta` values favor higher task scores. With the same prior and property factors, a one-unit increase in the task score multiplies relative probability by `exp(beta)`.
See [reward composition](prior_guidance.md#enable-prior-guidance).

| Setting | Use |
| --- | --- |
| `beta: 32` | Fixed task beta. |
| `beta: "uniform(1,64)"` | Sample uniformly between the bounds. |
| `beta: "log_uniform(1,64)"` | Sample uniformly in log space. |

The QED examples use the same reward and MW penalty. Choose [fixed beta](../configs/examples/qed.yaml) for one reward strength or [uniform beta](../configs/examples/qed_beta.yaml) to vary it when sampling.

Fixed beta must be non-negative.
Use `0 <= lower < upper` for `uniform` and `0 < lower < upper` for `log_uniform`.

## Combine objectives

Set `gflownet.moo_scalarization` to choose how raw component scores form the task reward `R(x)`. Both modes normalize preference weights to sum to one. Equal preferences for two objectives give weights `[0.5, 0.5]`, and `fixed(3,7)` gives `[0.3, 0.7]`.

| Mode | Task reward for two equally weighted objectives | Use |
| --- | --- | --- |
| `sum` (default) | `(r1 + r2) / 2` | Weighted arithmetic mean. Supports signed scores such as negative molecular weight. |
| `mul` | `sqrt(max(r1, 1e-8) * max(r2, 1e-8))` | Weighted geometric mean. Use positive scores when a weak component should reduce the combined reward. |

For example, component scores of 0.2 and 0.8 give `R(x) = 0.5` with `sum` and `R(x) = 0.4` with `mul`. In both cases the task factor is `exp(beta * R(x))`. These means follow REINVENT4's `custom_sum` and `custom_product` for finite component scores, including the component floor in `mul`. [REINVENT4 aggregation code](https://github.com/MolecularAI/REINVENT4/blob/main/reinvent/scoring/aggregators/means.py)

## Property rewards

Molecules exceeding `property_penalty` limits or `generation.max_atoms` receive lower rewards, independently of MOO preferences.

- `gflownet.property_penalty_ratio` controls how gradually rewards decrease beyond a limit.
  The default `0.2` uses 20% of the limit as the smoothing width; larger values give gentler penalties.
- If the ratio or a limit is zero, violating that limit gives zero reward.
- Molecules and trajectories are retained even when they exceed a limit.

## Objective preferences

Preferences follow the order of the named rewards passed to `RxnFlowTrainer`.

| `moo_preference` | Use |
| --- | --- |
| `"none"` | Equal weights throughout training and sampling. |
| `"fixed(0.3,0.7)"` | Fixed relative weights for two objectives. |
| `"dirichlet(1.5)"` | Sample preferences with more emphasis near the center of the simplex. |
| `"dirichlet(1)"` | Sample uniformly on the preference simplex. |
| `"dirichlet(0.5)"` | Sample preferences with more emphasis near the simplex boundaries. |

Fixed weights require one non-negative value per objective and a positive total.

`dirichlet` without parameters uses `dirichlet(1.5)`.
Dirichlet concentrations must be positive.
One value is shared across objectives: with three objectives, `dirichlet(0.5)` is equivalent to `dirichlet(0.5,0.5,0.5)`.
You can also supply one concentration per objective.
Equal concentrations of `1` sample uniformly on the preference simplex; values below `1` favor the boundaries, while values above `1` favor balanced preferences.

The [fixed MW/logP example](../configs/examples/mw_logp.yaml) weights both objectives equally at beta 32. The [conditioned MW/logP example](../configs/examples/mw_logp_pref.yaml) keeps beta at 32 and samples objective preferences during training.

```yaml
gflownet:
  beta: 32
  moo_preference: "dirichlet"
```

## Choose conditions when sampling

Sampling defaults to checkpoint settings.

- For either beta, use the trained fixed value or choose a value or range within the training bounds.
- `none` keeps equal weights.
  Fixed-preference training requires the same relative weights.
- Training with `dirichlet` or `dirichlet(...)` allows you to choose a fixed trade-off or another preference distribution at sampling.

In `RxnFlowSampler.sample()`, `beta` sets the task beta, `prior_beta` sets the prior guidance strength and `preference` sets the objective weights. The corresponding CLI options are `--task-beta`, `--prior-beta` and `--preference`. See the [sampling guide](training.md#sampling).

## External conditions

External conditions describe the generation target, such as a pocket key or target TPSA.
Each trajectory retains one `dict[str, Any]`, for example `{"tpsa": 80.0}`.
Beta and objective preferences remain separate settings.

Implement a `ConditioningModule` with:

- `encoding_dim`: the fixed number of model input features.
- `sample(num_samples, rng)`: return one raw condition dictionary per trajectory using the supplied NumPy generator.
- `encode(cond_list)`: return a deterministic float32 NumPy array of shape `[len(cond_list), encoding_dim]`, preserving order.

The supplied RNG is seeded and saved in training checkpoints.
Use JSON-compatible condition values for sample logs and JSON exports.

The conditioning module selects and encodes targets; the [reward function](rewards.md) evaluates molecules against the raw targets.

### Train with target TPSA

The [TPSA example](../examples/tpsa_cond.py) samples target TPSA values, divides them by 100 for model input, and returns the negative absolute error scaled by `scale` (100 Å² by default), giving `-abs(TPSA - target) / scale`.
Its [configuration](../configs/examples/tpsa_cond.yaml) supplies constructor settings:

```yaml
conditioning:
  min_tpsa: 0
  max_tpsa: 200
```

Set `env_dir` in the YAML to your compatible prepared environment, then run from the repository root:

```bash
python examples/tpsa_cond.py --config configs/examples/tpsa_cond.yaml --steps 1000
```

In custom scripts, construct `TPSAConditioningModule(**config.conditioning)` and pass it as `conditioning=` to `RxnFlowTrainer`.
Use the same module and settings when [resuming](training.md#resume-training).

### Sample with explicit targets

Provide `cond_list` to choose the targets yourself.
The module's `sample()` method is then skipped; only `encode()` is used.
Supply one condition per generation attempt, including attempts that may fail:

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

Omit `cond_list` to sample targets through the module.
Keep the training feature mapping unchanged.
External conditioning uses this Python interface; the generic sampling CLI does not construct custom modules.

Each raw condition field appears as `cond_<key>` in sampling metadata and training `sample.jsonl` records, such as `cond_tpsa: 80.0`.
