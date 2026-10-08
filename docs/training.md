# Training and sampling

## Define your reward

Implement a `RewardFunction` subclass with an `objectives` tuple and a `score()` method returning a float32 array of shape `[batch, num_objectives]`. Each score must be finite, non-negative and larger-is-better. See [custom rewards](rewards.md) for the interface and examples. Save your implementation in `my_reward.py` as `MyReward`.

## Train your model

Write your own configuration file based on the [configuration template](../configs/template.yaml), and set `env_dir` to your [prepared synthesis environment](environment.md). Configure reward conditioning and training settings using the [configuration reference](../configs/README.md). `reward.settings` contains constructor arguments for your reward class.

```python
from my_reward import MyReward
from rxnflow import Config, RxnFlowTrainer


if __name__ == "__main__":
    config = Config.from_file("configs/custom_rewards.yaml")
    reward = MyReward(**config.reward.settings)
    trainer = RxnFlowTrainer(
        config,
        reward,
        output_dir="runs/custom_reward",
        device="cuda",
        seed=1,
    )
    checkpoint = trainer.run(num_steps=1000)
```

`trainer.run(steps)` performs the requested number of additional updates. Output directory, device, seed and update count are execution arguments, separate from YAML settings.

## Resume training

To resume an interrupted run, pass the previous run’s checkpoint path to `resume_from_checkpoint`.

```python
config = Config.from_file("configs/custom_rewards.yaml")
reward = MyReward(**config.reward.settings)
trainer = RxnFlowTrainer(
    config,
    reward,
    output_dir="runs/custom_reward_resumed",
    device="cuda",
    seed=1,
)
checkpoint = trainer.run(
    num_steps=2000,
    resume_from_checkpoint="runs/custom_reward/checkpoints/latest.ckpt",
)
```

This performs 2,000 additional updates.

## Outputs

Files are saved under the trainer’s `output_dir`, which must not already exist. Resumed runs write only their new updates into the new directory. Invalid attempts, such as trajectories with no feasible continuation, are also recorded in the sample files.

| Path | Contents |
| --- | --- |
| `config.yaml` | Resolved configuration. |
| `checkpoints/latest.ckpt` | Model, optimizer, replay and RNG state for restart. |
| `training.log` | Training log, also printed to terminal. |
| `training.jsonl` | Metrics for every update. |
| `samples/step_XXXXXX.jsonl` | All online attempts for each update, including invalid ones. |

### Sample records

`samples/step_XXXXXX.jsonl` stores individual online attempts. Each record contains `step`, `sample`, `final_smiles`, `reward`, `objective_rewards`, `property_reward`, `property_violation`, `beta`, `preference`, `valid`, `invalid_reason` and `traj`.

- `objective_rewards` follows the reward class's `objectives` order.
- `preference` stores effective weights: their sum is the number of objectives for `mul`, or one for `sum`.
- `traj` records the synthesis trajectory.

### Training log metrics

`training.jsonl` contains one record per update. Generation metrics describe all new online attempts, including invalid ones; loss and flow metrics describe the combined online + replay batch before the parameter update.

**Generation and rewards**

| Metric | Meaning |
| --- | --- |
| `step` | Completed training updates, continuing from the checkpoint when resumed. |
| `num_online`, `num_replay` | Actual numbers of new and replay trajectories used in the update. |
| `num_valid`, `num_invalid` | Valid and invalid online trajectory counts. |
| `num_unique` | Number of distinct valid online SMILES. |
| `reward` | Mean online reward after combining objectives and applying the property penalty, before the reward exponent. |
| `objective_rewards` | Mean online scores by objective, keyed as `r_qed`, etc. |
| `property_reward` | Mean online property penalty multiplier. |
| `num_property_violations` | Number of online trajectories violating at least one property limit. |
| `traj_lens` | Mean online trajectory length, including initial synthon selection. |

The terminal summary reports valid percentage out of online attempts and unique percentage out of valid attempts. Its `replay` value is the current buffer size, rather than the number used in the update.

**Loss and flow diagnostics**

| Metric | Meaning |
| --- | --- |
| `loss` | Mean trajectory-balance loss using the configured `loss_fn`. |
| `online_loss`, `replay_loss` | Mean loss within each source. |
| `valid_losses`, `invalid_losses` | Mean loss within valid and invalid trajectories. |
| `logZ` | Mean predicted log partition function for the batch conditions. |
| `beta` | Mean reward exponent. |
| `logit_scale` | Mean learned logit scale for the batch conditions. |
| `traj_log_p_F`, `traj_log_p_B` | Mean forward and backward log probabilities, summed over each trajectory. |
| `scaled_log_R` | Mean log reward used by TB after applying the reward exponent. |
| `tb_error` | Mean signed TB residual: `logZ + traj_log_p_F - traj_log_p_B - scaled_log_R`. Positive and negative errors can cancel. |
| `batch_entropy` | Negative mean trajectory log forward probability; not categorical action entropy. |
| `invalid_logprob` | Mean trajectory log forward probability among invalid trajectories. |
| `invalid_trajectories` | Fraction of invalid trajectories in the online + replay batch. |

Subset metrics are zero when their subset is empty.

**Optimization and timing**

| Metric | Meaning |
| --- | --- |
| `learning_rate` | Policy learning rate after the scheduler update. |
| `policy_grad_norm` | Policy gradient L2 norm before clipping. |
| `grad_norm` | Total gradient L2 norm, including the logZ head, before clipping. |
| `sampling_time` | Online trajectory generation time in seconds, excluding reward evaluation. |
| `time` | Online sampling and training time in seconds. |


## Sampling

Use your trained checkpoint; omitted beta and preferences reuse their training settings:

```bash
# Sample from the extracted GFN model (EMA weights and config)
python scripts/extract_model.py \
  --checkpoint runs/custom_reward/checkpoints/latest.ckpt \
  --output runs/custom_reward/model.pt

python scripts/sample.py \
  --checkpoint runs/custom_reward/model.pt \
  --num-samples 100 \
  --output samples.jsonl

# Directly sample from the checkpoint
python scripts/sample.py \
  --checkpoint runs/custom_reward/checkpoints/latest.ckpt \
  --num-samples 100 \
  --output samples.jsonl
```

Use `--beta 32` to select a fixed exponent, or `--preference "fixed(0.3,0.7)"` to select a trade-off for two objectives. See [sampling conditions](conditioning.md#choose-conditions-when-sampling) for the options supported by your training settings.

- `--num-samples` sets the number of trajectory attempts. Invalid results are excluded without retrying, so fewer molecules may be returned.
- Sampling prints the training reward configuration and the requested sampling settings before generating molecules.
- `--env-dir` selects another compatible prepared environment. Its reaction, synthon and exclusion definitions must match training; BB identities and library sizes may differ. Types defined in the templates but unused during training may have untrained embeddings.
- `--batch-size` controls trajectories generated per batch (default 64), independently of the training batch size.
- `--softmax-temperature` is an additional softmax temperature (default 1), separate from beta.

Supported output formats:

- `.smi`: one `SMILES<TAB>sample_i` record per line, numbered from zero in saved-result order.
- `.jsonl`: one record per line containing `smiles`, `traj` and `metadata`. Metadata holds `beta` and `preference`; trajectories include initial synthon selection and each action's product.
- `.json`: the same records as a single array, formatted with two-space indentation.

### JSON and JSONL records

Both formats store the same sample records. JSON contains an array of records with two-space indentation; JSONL contains one record per line. Only valid sampling results are saved.

| Field | Contents |
| --- | --- |
| `smiles` | Final generated molecule's SMILES. |
| `traj` | Ordered list of actions, starting with initial synthon selection and ending with the terminal product. |
| `metadata.beta` | Reward exponent sampled for this trajectory, including when fixed during training. |
| `metadata.preference` | Effective objective weights in the reward class's `objectives` order. They sum to the number of objectives for `mul`, or to one for `sum`. |

For example, two objectives with equal weights under `mul` have:

```json
{
  "beta": 32.0,
  "preference": [1.0, 1.0]
}
```

This is the `metadata` object within each record. Sampling does not evaluate rewards, so the records contain neither objective scores nor combined rewards. No sample name or likelihood is added.

**Trajectory actions**

| Field | Contents |
| --- | --- |
| `action_type` | `FIRST_SYNTHON`, `UNIRXN_TRANSFORM`, `UNIRXN_TERMINAL`, `BIRXN_BRICK` or `BIRXN_LINKER`. |
| `reaction` | Reaction template name; `first_synthon` for initial selection. |
| `reaction_smarts` | Executable forward reaction SMARTS, including incoming-site orientation; `null` for initial selection. |
| `product_smiles` | State after this action. Intermediate states can contain typed dummy atoms; the last action's product matches the top-level `smiles`. |
| `synthon_id` | Selected synthon's ID in the prepared environment's `synthons/*.tsv` files. Present only when selecting a synthon. |
| `synthon_smiles` | Selected synthon's SMILES in the orientation used by the action. Present only when selecting a synthon. |
| `building_blocks` | Original building-block candidates corresponding to the selected synthon. Each entry contains `id` and `smiles`. Present only when selecting a synthon. |

Initial selection and bimolecular reactions include the three synthon fields. Unimolecular reactions act on the current state without adding a synthon, so those fields are omitted rather than set to `null`.
