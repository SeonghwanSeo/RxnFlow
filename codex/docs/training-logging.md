# Training logging and references

The trainer writes `training.jsonl` and `samples.jsonl` under `run.output_dir` on every update. `training.log_every` controls console output only. Restarting from the latest checkpoint appends subsequent updates. Logs are not rolled back when an older checkpoint is loaded; use a separate output directory for a new experiment. No fingerprint, scaffold, or pairwise diversity computations are performed during logging.

## Reference implementations inspected

| Reference | Revision | Relevant files |
| --- | --- | --- |
| RxnFlow `master` | `a39c7aebd45fddfdf72109ffa0f8e4b92f9c4458` | `src/rxnflow/base/gflownet/trajectory_balance.py`, `src/rxnflow/base/gflownet/sqlite_log.py` |
| CGFlow | `89fe021304164764f4c027da2ee0c274de3065fa` | `src/gflownet/algo/trajectory_balance.py`, `src/rxnflow/base/gflownet/sqlite_log.py` |
| HSX `explore_250509` | `2d472443806ac107b9cdc8a65d03866302394d84` | `src/rxnflow/base/gflownet/trajectory_balance.py`, `src/rxnflow/base/gflownet/sqlite_log.py` |
| HSX `main` | `e999c3d1b2f911d1b33fb8245c0a66a2946f14b0` | `src/rxnflow/gflownet/algo/trajectory_balance.py`, `src/rxnflow/gflownet/online_trainer.py`, `src/rxnflow/gflownet/utils/sqlite_log.py` |

CGFlow's `SynthesisTB` inherits `gflownet.algo.trajectory_balance.TrajectoryBalance`. Its active info dictionary includes `loss`, `logZ`, `batch_entropy`, `traj_lens`, and invalid fraction; several more detailed loss fields are commented out. HSX main's TB dictionary is also deliberately small. The richer invalid and online/offline diagnostics come from RxnFlow master and HSX-250509, not from assuming every reference logs everything.

## Scalar diagnostics

TB metrics describe the optimization batch (fresh trajectories followed by sampled replay) before the optimizer update. `log_z` is captured at the same point as the residual. `learning_rate` retains the existing post-scheduler convention. Existing rollout summaries (`mean_reward`, `valid_fraction`, `unique_fraction`, `mean_reactions`) describe only fresh attempts, including failures where applicable.

| Logged fields | Definition | Reference / adaptation |
| --- | --- | --- |
| `loss`, `log_z` | Mean squared TB residual and mean pre-update conditional logZ | All references; existing MSE objective unchanged |
| `batch_entropy` | `-mean(sum_transition log P_F)` over fresh + replay | All four references. This is observed trajectory surprisal, not categorical entropy or molecular diversity. Off-policy/replay data and subsampled probabilities prevent treating it as an unbiased policy entropy estimate. |
| `mean_log_pf`, `mean_log_pb`, `mean_log_reward`, `mean_tb_residual` | Batch means of the existing TB terms; `mean_log_reward` includes reward floor and exponent | Lightweight diagnostics exposing the terms already computed by reference TB objectives; not claimed as identical reference log keys. `mean_log_pf = -batch_entropy`. |
| `invalid_loss`, `invalid_logprob` | Mean TB loss and summed forward log probability for invalid batch trajectories | RxnFlow master / HSX-250509 `invalid_losses`, `invalid_logprob`; local naming and exact nonempty denominator |
| `valid_loss` | Mean TB loss for valid batch trajectories | Complement to the reference invalid-loss diagnostic |
| `fresh_loss`, `replay_loss` | Mean loss for each source of the optimization batch | Adaptation of RxnFlow master / HSX-250509 online/offline loss separation. Replay is not an external offline dataset, so the names are explicit. |
| `batch_invalid_fraction`, `batch_valid_count`, `batch_invalid_count`, `fresh_count`, `replay_count` | Batch population and source sizes | Reference invalid-fraction diagnostic plus explicit denominators. Distinct from fresh-only `valid_fraction`. |
| `grad_norm` | Pre-clipping global L2 norm including logZ | HSX main |
| `policy_grad_norm`, `policy_grad_clipped` | Pre-clipping norm of the policy-only group and 0/1 indicator for exceeding 100 | Local clarification of HSX main's policy-only clipping. Reuse the return value of the existing clip call; no extra parameter traversal. |

Empty valid, invalid, or replay groups have group mean 0, following the reference convention for absent groups. Consult the corresponding counts to distinguish an absent group from a true zero loss. A chemically valid trajectory filtered by the reward API is still valid with zero reward; invalid metrics describe environment failure, not reward filtering.

Tensor diagnostics are detached and transferred to CPU together. Logging does not recompute policy scores, reactions, rewards, or descriptors and does not consume sampling RNG. `step_seconds` covers rollout through optimizer/EMA and scalar aggregation, excluding sample/training-log writes and checkpoints. `sample_log_seconds` separately measures sample serialization and file writing; it does not measure durable disk flush latency.

## Fresh rollout counts

`reaction_counts` records the distribution of attempted reaction counts, excluding FirstBlock and including failed trajectories. `action_counts` records selected transitions under `first_block`, `brick`, `linker`, `uni_terminal`, and `uni_continue`. A selected reaction that fails execution is counted; an empty candidate space adds no selected transition. `invalid_reasons` counts environment failure messages. Absent categories are omitted. These are inexpensive current-environment diagnostics, not fields claimed to be copied directly from a reference implementation.

## Persistent generated samples

The reference SQLite hooks retain generated structures, rewards and synthesis paths. The local single-process trainer preserves that information in append-only `samples.jsonl`, reusing `Trajectory.to_dict()` rather than adding a database or logging framework. Each row contains `step`, a zero-based within-update `sample` index, `final_smiles`, raw `reward`, `valid`, `invalid_reason`, and `steps`. Each transition contains its parent state SMILES/metadata, action identifiers, product SMILES, and stored `log_pb`. Failed selected actions retain their empty product SMILES and invalid reason.

Only fresh attempts are written, including invalid ones. Replayed trajectories are not written again. Every record includes `beta`, `preferences` and `objective_rewards` in checkpoint objective order. Raw scalar reward is the preference-weighted sum before TB flooring/exponentiation; custom `RewardFunction.metrics()` remain per-update summaries. No additional property calculations are introduced for individual samples. The serialized trajectories can be reconstructed with `Trajectory.from_dict()` after removing the two log-index fields. Storage grows with the number of fresh attempts; FIFO replay eviction and checkpoint replacement do not remove sample history.

## Deferred

Exact categorical entropy, scaffold/fingerprint diversity, reward quantiles, and periodic exploration-free evaluation are outside this change. Existing evaluation scripts can still perform separate before/after evaluations. There is no new automatic evaluation run or training job.

Conditional training additionally records `mean_beta` and `mean_logit_temperature` over the fresh+replay optimization batch. `mean_objective_rewards` and `mean_preferences` are dictionaries keyed by objective name for fresh attempts, including zero rewards for invalid/filtered samples. The beta/preference encoder is shared by policy and logZ; its parameters belong to the policy optimizer group, while only the logZ head uses `log_z_learning_rate`.
