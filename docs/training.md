# Training and sampling

## Define your reward

Implement a `RewardFunction` subclass with an `objectives` tuple and a `score()` method returning a float32 array of shape `[batch, num_objectives]`. Each score must be finite, non-negative and larger-is-better. See [custom rewards](rewards.md) for the interface and examples. Save your implementation in `my_reward.py` as `MyReward`.

## Train your model

Copy the [configuration template](../configs/template.yaml) to `config.yaml` and set `env_dir` to your [prepared synthesis environment](environment.md). Configure reward conditioning and training settings using the [configuration reference](../configs/README.md). `reward.settings` contains constructor arguments for your reward class.

Save the following as `train.py` alongside `my_reward.py` and run `python train.py`:

```python
import torch

from my_reward import MyReward
from rxnflow import Config, RxnFlowTrainer


if __name__ == "__main__":
    config = Config.from_file("config.yaml")
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

`trainer.run(steps)` performs the requested number of additional updates. Output directory, device, seed and update count are execution arguments, separate from YAML settings. The output directory must not already exist.

## Resume training

To resume an interrupted run, use its saved configuration and checkpoint with the same reward class, constructor settings and objective order. Choose a new output directory and keep the prepared environment and package version unchanged. The checkpoint restores RNG state; the supplied seed only initializes new runs.

In your training script, replace the training block with:

```python
config = Config.from_file("runs/custom_reward/config.yaml")
reward = MyReward(**config.reward.settings)
trainer = RxnFlowTrainer(
    config,
    reward,
    output_dir="runs/custom_reward_resumed",
    device="cuda",
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

- `reward` is before beta; `objective_rewards` contains the individual scores.
- `num_valid`, `num_invalid` and `num_unique` describe online samples only; `num_unique` counts distinct valid SMILES. Terminal summaries calculate valid and unique percentages from these counts.
- In sample records, `preferences` contains effective weights: they sum to the number of objectives for `mul`, or to one for `sum`.
- `traj_lens` includes initial synthon selection. The compact `traj` omits that selection and records pre-reaction state, reaction name and added synthon.
- `batch_entropy` is mean negative trajectory log forward probability over online + replay data, not categorical action entropy.

## Sampling

Use your trained checkpoint; omitted beta and preferences reuse their training settings:

```bash
python scripts/sample.py \
  --checkpoint runs/custom_reward/checkpoints/latest.ckpt \
  --num-samples 100 \
  --output samples.csv
```

Use `--beta 32` to select a fixed exponent, or `--preferences "fixed(0.3,0.7)"` to select a trade-off for two objectives. See [sampling conditions](conditioning.md#choose-conditions-when-sampling) for the options supported by your training settings.

- Sampling prints the training reward configuration and the requested sampling settings before generating molecules.
- `--env-dir` selects another prepared building-block catalog. Its reaction, synthon and exclusion definitions must match training; BB identities and library sizes may differ. Types defined in the templates but unused during training may have untrained embeddings.
- `--batch-size` controls trajectories generated per batch (default 64), independently of the training batch size. Duplicate molecules are retained.
- Formats: `.smi` for SMILES, `.csv` for molecules and paths, `.json` for structured trajectories and provenance.
- `--softmax-temperature` is an additional softmax temperature (default 1), separate from beta.

JSON records contain `smiles`, `traj` and `metadata`; metadata holds `beta` and `preferences`. CSV columns are `smiles`, `traj`, `beta` and `preferences`; `traj` and `preferences` are JSON strings. Sampling trajectories include the initial synthon selection and each action’s product. SMILES output contains only SMILES.

Sampling does not evaluate rewards. Evaluate generated molecules separately when needed:

```python
from rdkit import Chem

from my_reward import MyReward
from rxnflow.sampler import RxnFlowSampler

sampler = RxnFlowSampler("runs/custom_reward/checkpoints/latest.ckpt")
results = sampler.sample(100, seed=0)
mols: list[Chem.Mol] = [Chem.MolFromSmiles(result.smiles) for result in results]
reward = MyReward(**sampler.config.reward.settings)
objective_rewards = reward(mols)
```

## Extract a sampling model

Extraction is optional: `--checkpoint` accepts either a full training checkpoint or an extracted model. Extract once to avoid loading optimizer and replay data on each sampling run:

```bash
python scripts/extract_model.py \
  --checkpoint runs/custom_reward/checkpoints/latest.ckpt \
  --output runs/custom_reward/model.pt
```

Use `--checkpoint runs/custom_reward/model.pt` in the sampling command. The file contains the configuration, EMA model weights and metadata needed for sampling. Use the configured environment path or select a compatible catalog with `--env-dir`. Extracted models cannot resume training.
