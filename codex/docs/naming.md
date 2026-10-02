# GFlowNet naming alignment

This feedback pass changes identifiers and their callers, config names, state-dict module paths and serialized action/transition field names. It does not change tensor operations, dimensions, initialization, probability corrections, sampling distributions or the TB objective. There are no compatibility aliases. Training-log keys are included in this naming pass; none are preserved solely because they appeared in earlier discussion.

## Architecture

Reference: `source/rxnflow_hits/src/rxnflow/models/gfn.py` and `source/CGFlow/src/rxnflow/models/gfn.py`. Use the reference naming vocabulary while retaining the agreed architecture.

| Previous | Current | Meaning |
| --- | --- | --- |
| `RxnFlowModel` | `RxnFlowModel` | Retained by explicit user preference |
| `hidden_dim`, `block_dim` | `num_emb`, `num_block_emb` | Model/config widths |
| `mlp_layers`, `block_mlp_layers` | `num_mlp_layers`, `num_mlp_layers_block` | Model/config MLP depths |
| `first_block_head`, `bi_reaction_head` | `mlp_firstblock`, `mlp_birxn` | HSX policy-head names |
| `uni_reaction_head` | `mlp_unirxn` | Same naming pattern for current unary actions |
| `fingerprint_encoder`, `property_encoder` | `lin_fp`, `lin_prop` | Reference block-feature projections |
| `block_type_embedding` | `emb_type` | Reference block-type embedding |
| `block_encoder` | `mlp_block` | Fusion MLP; not the complete reference BlockEmbedding module |
| `action_embedding` | `emb_rxn` | Reaction identity; not a workflow/protocol identity |
| `beta_encoder`, `preference_encoder` | `emb_beta`, `emb_preferences` | Condition encoders |
| `graph_condition` | `cond2h` | Condition projection into virtual-node initialization |
| `graph_encoder` | `mpnn` | Current GINE backbone; deliberately not reference `transf` |
| `mean_norm`, `virtual_norm` | `norm_mean`, `norm_virtual` | Existing separate readout normalizations |
| `encode_condition`, `encode_graphs` | `encode_cond`, `graph_embedding` | Encoding operations |
| `encode_block_features`, `_encode_blocks` | `block_embedding`, `get_block_emb` | Block encoding from features or indexed library rows |
| `action_query` | `forward_mdp` | State/reaction projection used for scalar/dot-product logits |
| `score_scalar`, `score_blocks` | `get_unirxn_logits`, `get_block_logits` | Explicit action-logit calculations |

`_logZ`/`logZ()` and `_logit_scale`/`logit_scale()` already follow HSX. Local embeddings use `graph_emb`, `state_emb`, `block_emb`, `rxn_emb` and `cond_info`. GINE uses `node_emb`, `edge_emb`, `msg`, `aggr`, `src_idx` and `dst_idx`; its update network is `mlp`. No transformer/workflow name is introduced for an object with different semantics.

## Trajectory balance

Reference: `source/rxnflow_hits/src/rxnflow/base/gflownet/trajectory_balance.py`, also shared with the RxnFlow/CGFlow GFlowNet implementation.

| Previous | Current | Meaning |
| --- | --- | --- |
| `_loss` | `compute_batch_losses` | Batch TB calculation |
| `condition` | `cond_info` | Encoded trajectory conditions |
| `log_z` | `log_Z` | Per-trajectory log partition prediction |
| `values` (forward probabilities) | `log_p_F` | Per-transition forward log probabilities |
| `forward_flow` | `traj_log_p_F` | Sum of forward log probabilities per trajectory |
| `backward_flows` | `traj_log_p_B` | Sum of backward log probabilities per trajectory |
| `indices` (trajectory reduction) | `batch_idx` | Transition-to-trajectory index |
| `log_reward` | `clip_log_R` | Floored, beta-scaled log reward |
| `residual`, `trajectory_losses` | `tb_residual`, `traj_losses` | TB residual and per-trajectory losses |
| `Transition.log_pb` | `Transition.log_p_B` | Stored per-transition backward log probability |

The residual remains `log_Z + traj_log_p_F - traj_log_p_B - clip_log_R`; operand order and numerical operations are unchanged. JSON TB metrics follow the same notation (`logZ`, `traj_log_p_F`, `traj_log_p_B`, `clip_log_R`).

## Policy and action metadata

| Previous | Current | Meaning |
| --- | --- | --- |
| `candidate_batch` | `forward` | Construct the masked forward categorical distribution |
| `observed_logits` | `get_action_logits` | Evaluate selected/observed actions |
| `action_log_probabilities` | `log_prob` | Forward action log probabilities |
| `action_log_probability` | `log_prob_single` | Single-state convenience call |
| `choose_actions`, `choose_action` | `sample_actions`, `sample_action` | Draw actions |
| `categorical` | `fwd_cat` | Forward policy categorical |
| `embeddings` | `graph_emb` | Graph embeddings carried with the categorical |
| `ActionKind`, `kind` | `ActionType`, `action_type` | Action enum and metadata field |
| `temperature` (sampling argument) | `sampling_temperature` | Additional softmax temperature |

`ActionGroup` is removed. `ActionSpace` is a list of `(ActionType, reaction_name, block_type)` tuples, precomputed for init and each synthon type. `ActionSubspace` carries sampled library indices and scored/masked columns for one reaction; its selected column converts to `Action`. These do not become workflow objects or individual candidate-action objects. `rollout`/`rollouts`, state, transition and trajectory terms retain their standard RL meanings.

## Logging

| Previous | Current |
| --- | --- |
| `log_z` | `logZ` |
| `log_pf` | `traj_log_p_F` |
| `log_pb` | `traj_log_p_B` |
| `log_reward` | `clip_log_R` |
| `sample_loss` | `online_loss` |
| `valid_loss` | `valid_losses` |
| `invalid_loss` | `invalid_losses` |
| `invalid_fraction` | `invalid_trajectories` |
| `sample_count` | `num_online` |
| `replay_count` | `num_replay` |
| `valid_count` | `num_valid` |
| `invalid_count` | `num_invalid` |
| `sample_valid_fraction` | `online_valid_fraction` |
| `sample_unique_fraction` | `online_unique_fraction` |
| `num_steps` | `traj_lens` |
| `sample_time` | `sampling_time` |
| `log_time` | `logging_time` |
| `time` | `iteration_time` |

`loss`, `batch_entropy`, `replay_loss`, reward summaries and gradient diagnostics already describe their quantities directly. They remain without aliases. Timing names reflect the measured scope; reference `train_time` is narrower than the current full update timer. See [training-logging.md](training-logging.md) for metric populations.
