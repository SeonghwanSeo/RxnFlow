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

TB metrics describe the optimization batch (fresh trajectories followed by sampled replay) before the optimizer update. `logZ` is captured at the same point as the residual. `learning_rate` retains the existing post-scheduler convention. Existing rollout summaries (`reward`, `online_valid_fraction`, `online_unique_fraction`, `traj_lens`) describe only fresh attempts, including failures where applicable.

| Logged fields | Definition | Reference / adaptation |
| --- | --- | --- |
| `loss`, `logZ` | Mean squared TB residual and mean pre-update conditional logZ | All references; existing MSE objective unchanged |
| `batch_entropy` | `-mean(sum_transition log P_F)` over fresh + replay | All four references. This is observed trajectory surprisal, not categorical entropy or molecular diversity. Off-policy/replay data and subsampled probabilities prevent treating it as an unbiased policy entropy estimate. |
| `traj_log_p_F`, `traj_log_p_B`, `clip_log_R`, `tb_residual` | Batch means of the existing TB terms; `clip_log_R` includes reward floor and exponent | Lightweight diagnostics exposing the terms already computed by reference TB objectives; not claimed as identical reference log keys. `traj_log_p_F = -batch_entropy`. |
| `invalid_losses`, `invalid_logprob` | Mean TB loss and summed forward log probability for invalid batch trajectories | RxnFlow master / HSX-250509 `invalid_losses`, `invalid_logprob`; local naming and exact nonempty denominator |
| `valid_losses` | Mean TB loss for valid batch trajectories | Complement to the reference invalid-loss diagnostic |
| `online_loss`, `replay_loss` | Mean loss for each source of the optimization batch | Adaptation of RxnFlow master / HSX-250509 online/offline loss separation. Replay is not an external offline dataset, so the names are explicit. |
| `invalid_trajectories`, `num_valid`, `num_invalid`, `num_online`, `num_replay` | Batch population and source sizes | Reference invalid-fraction diagnostic plus explicit denominators. Distinct from fresh-only `online_valid_fraction`. |
| `grad_norm` | Pre-clipping global L2 norm including logZ | HSX main |
| `policy_grad_norm`, `policy_grad_clipped` | Pre-clipping norm of the policy-only group and 0/1 indicator for exceeding 100 | Local clarification of HSX main's policy-only clipping. Reuse the return value of the existing clip call; no extra parameter traversal. |

Empty valid, invalid, or replay groups have group mean 0, following the reference convention for absent groups. Consult the corresponding counts to distinguish an absent group from a true zero loss. A chemically valid trajectory filtered by the reward API is still valid with zero reward; invalid metrics describe environment failure, not reward filtering.

Tensor diagnostics are detached and transferred to CPU together. Logging does not recompute policy scores, reactions, rewards, or descriptors and does not consume sampling RNG. `iteration_time` covers rollout through optimizer/EMA and scalar aggregation, excluding sample/training-log writes and checkpoints. `logging_time` separately measures sample serialization and file writing; it does not measure durable disk flush latency.

## Fresh rollout counts

`traj_lens` is the mean number of selected actions per fresh trajectory, including FirstBlock and failed selected actions. An empty candidate space adds no step. Reaction/action histograms and aggregate failure-reason counts are omitted; individual paths and failure reasons remain in `samples.jsonl`. `sampling_time` measures rollout generation in seconds; `iteration_time` measures update computation and `logging_time` measures sample serialization. The reference GFlowNet uses `traj_lens` for trajectory lengths and `train_time` for training-batch time; these are not identical timing scopes.

## Persistent generated samples

The reference SQLite hooks retain generated structures, rewards and synthesis paths. The local single-process trainer preserves that information in append-only `samples.jsonl`, reusing `Trajectory.to_dict()` rather than adding a database or logging framework. Each row contains `step`, a zero-based within-update `sample` index, `final_smiles`, raw `reward`, `valid`, `invalid_reason`, and `steps`. Each transition contains its parent state SMILES/metadata, action identifiers, product SMILES, and stored `log_p_B`. Failed selected actions retain their empty product SMILES and invalid reason.

Only fresh attempts are written, including invalid ones. Replayed trajectories are not written again. Every record includes `beta`, `preferences` and `objective_rewards` in checkpoint objective order. Raw scalar reward is the preference-weighted sum before TB flooring/exponentiation; custom `RewardFunction.metrics()` remain per-update summaries. No additional property calculations are introduced for individual samples. The serialized trajectories can be reconstructed with `Trajectory.from_dict()` after removing the two log-index fields. Storage grows with the number of fresh attempts; FIFO replay eviction and checkpoint replacement do not remove sample history.

## Deferred

Exact categorical entropy, scaffold/fingerprint diversity, reward quantiles, and periodic exploration-free evaluation are outside this change. Existing evaluation scripts can still perform separate before/after evaluations. There is no new automatic evaluation run or training job.

Conditional training additionally records `beta` and `logit_scale` over the fresh+replay optimization batch. `objective_rewards` and `preferences` are dictionaries keyed by objective name for fresh attempts, including zero rewards for invalid/filtered samples. The beta/preference encoder is shared by policy and logZ; its parameters belong to the policy optimizer group, while only the logZ head uses `log_z_learning_rate`.

The naming pass now aligns `logZ`, `traj_lens`, `online_loss`, `invalid_losses` and `invalid_trajectories` with reference keys. `replay_loss`/`num_replay` stay explicit because replay is not an offline dataset. `iteration_time` includes sampling and optimization, unlike reference `train_time` which measures only the training batch; `sampling_time` and `logging_time` expose their actual scopes. Fresh-only summaries use `online_*`; optimization-batch valid/invalid counts use `num_valid`/`num_invalid`. TB-term means use the same names as their mathematical variables.
