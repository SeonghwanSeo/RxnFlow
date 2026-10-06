# Configuration reference

Defaults apply when settings are omitted. Start from the [template](template.yaml); see the [training guide](../docs/training.md) for output directory, device, seed and update count.

## Synthetic Environment

| Setting | Default | Meaning |
| --- | --- | --- |
| `env_dir` | Required | Prepared synthesis environment directory. |

## Reward

See [custom rewards](../docs/rewards.md) for implementation and [conditioning](../docs/conditioning.md) for distributions, objective weights and sampling options.

| Setting | Default | Meaning |
| --- | --- | --- |
| `reward.beta` | `"32"` | Reward exponent: a fixed value or `uniform(lower,upper)`. |
| `reward.moo_scalarization` | `mul` | Combine objectives by weighted product (`mul`) or weighted sum (`sum`). |
| `reward.moo_preferences` | `"none"` | Objective weights: `none`, `fixed(...)`, `uniform` or `dirichlet(...)`. |
| `reward.settings` | `{}` | Constructor arguments for your reward class. |

## Property penalty

Upper bounds used to mask synthon actions, disabled by default (`{}`). Omitted bounds are unset. These use additive reactant estimates and do not guarantee exact final-product bounds.

| Setting | Default | Meaning |
| --- | --- | --- |
| `property_penalty.mw` | — | Exact molecular weight limit (Da). |
| `property_penalty.tpsa` | — | Topological polar surface area limit (Å²). |
| `property_penalty.hbd` | — | Hydrogen-bond donor limit. |
| `property_penalty.hba` | — | Hydrogen-bond acceptor limit. |
| `property_penalty.rotatable_bonds` | — | Rotatable-bond limit. |
| `property_penalty.rings` | — | Ring count limit. |
| `property_penalty.aromatic_rings` | — | Aromatic-ring count limit. |

## Action Subsampling

| Setting | Default | Meaning |
| --- | --- | --- |
| `subsampling.sampling_ratio` | `0.1` | Fraction of each synthon library evaluated per action, in `(0, 1]`. |
| `subsampling.min_sampling` | `50` | Minimum sampled candidates per library, capped by library size. |
| `subsampling.importance_temp` | `1.0` | Subsampling correction strength: `1` applies the full correction, `0` disables it during action selection. |

## Generation Constraints

Synthon counts include the initial selection; reaction counts include unary and binary reactions.

| Setting | Default | Meaning |
| --- | --- | --- |
| `generation.max_atoms` | `50` | Maximum heavy atoms per generated state. |
| `generation.min_synthons` | `2` | Minimum selected synthons in a completed route. |
| `generation.max_synthons` | `3` | Maximum selected synthons in a route. |
| `generation.min_reactions` | `1` | Minimum reaction steps in a completed route. |
| `generation.max_reactions` | `3` | Maximum reaction steps in a route. |

## Model Configuration

MLP layer counts include the output linear layer.

| Setting | Default | Meaning |
| --- | --- | --- |
| `model.state_dim` | `256` | State graph encoder width. |
| `model.num_state_layers` | `4` | State graph message-passing layers. |
| `model.synthon_dim` | `256` | Synthon embedding width. |
| `model.num_synthon_layers` | `3` | Synthon encoder MLP layers. |
| `model.hidden_dim` | `256` | Condition/reaction embedding and action-head hidden width. |
| `model.num_action_layers` | `3` | Action-head MLP layers. |

## Training settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `training.log_every` | `10` | Summary interval in updates; metrics are saved every update. |
| `training.checkpoint_every` | `1000` | Checkpoint interval in updates; the final state is also saved. |
| `training.num_online` | `64` | New trajectories per update. |
| `training.num_replay` | `64` | Stored trajectories reused per update; `0` disables reuse. |
| `training.ema_decay` | `0.99` | EMA coefficient for the sampling model; `0` copies current weights. |
| `training.replay_capacity` | `100000` | FIFO replay capacity; `0` disables storage. |
| `training.num_replay_insert` | `null` | Maximum insertions per update; `null` keeps all, `0` keeps none. |
| `training.replay_insert_priority` | `uniform` | Limited insertion selection: random (`uniform`) or highest reward (`reward`). |
| `training.learning_rate` | `1e-4` | Adam learning rate for policy parameters. |
| `training.learning_rate_logZ` | `1e-2` | Adam learning rate for the logZ head. |
| `training.lr_decay_steps` | `10000` | Updates over which both learning rates halve. |
| `training.weight_decay` | `1e-8` | Adam weight decay. |
| `training.reward_floor` | `1e-5` | Reward lower bound before taking logarithms. |
| `training.random_action_prob` | `0.1` | Exploration-action probability during online training. |
| `training.backward_synthon_penalty` | `100.0` | Backward-route penalty per additional synthon; values above `1` favor fewer synthons. |
| `training.retrosynthesis_workers` | `4` | Reverse-search workers; `0` runs synchronously. |
