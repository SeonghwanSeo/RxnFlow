# Beta and preference conditioning

The model takes beta and preferences from its caller. The beta encoder always uses `u=(beta-1)/63` and `[u, sin(2*pi*u*f), cos(2*pi*u*f)]` for fixed frequencies `(1, 2, 4, 8)` (nine inputs). These are encoder coordinates, not allowed beta bounds; no clamp is applied. A two-Linear MLP encodes these features, another encodes the preference vector, and their outputs are added. Separate branches inject this condition into the initial virtual node, predict bounded reaction temperatures `T_r(beta, w)`, and predict `log Z(beta, w)`. There is no per-layer FiLM and no generic unused condition argument.

`RewardFunction.objectives` defines ordered objective names. `score(molecules)` returns a detached float32 tensor `[B, K]`, including `[0, K]` for an empty batch. Objective transforms and scale are the reward implementation's responsibility; larger scores must be better and all scores must be finite and non-negative. The trainer computes `R(x,w)=sum(w*r(x))` and `log_reward=beta*log(max(R, floor))`. Invalid/filtered samples receive a zero objective vector. A single objective follows exactly the same path with K=1 and w=[1]. The environment, action validity and backward analysis do not depend on beta or preferences.

## External settings

```yaml
reward:
  exponent: [8.0, 64.0]  # Or a fixed scalar, e.g. 32.0 (the default).
  preferences: null      # Uniform Dirichlet(1); or fixed weights, e.g. [0.8, 0.2].
  floor: 0.0001
  settings: {}
```

These ranges are examples, not new default training settings. The trainer draws beta and preferences once per fresh trajectory and retains them throughout generation and optimization. For Dirichlet(1), normalized independent Exp(1) values use the existing checkpointed CPU generator. Sampling uses explicit `sampler.sample(count, beta=32.0, preferences=[0.8, 0.2])`; CLI equivalents are `--beta 32 --preferences 0.8 0.2`. The existing `temperature` argument is an additional sampling softmax temperature, not reward exponent beta. Same-architecture checkpoints store objective names/order, both model copies, the configuration, replay, optimizer and RNG states. Older scalar-logZ checkpoints are unsupported.

## Replay: current implementation

Uniform FIFO replay stores the original trajectory, beta, preferences, objective reward vector and untempered weighted reward. Replay sampling preserves the conditions. TB uses the stored objective vector and weights, recomputes forward probabilities and conditional logZ with the current model, and uses the stored backward probabilities (the current backward procedure is independent of reward conditions). No property/reward recomputation or condition relabeling occurs during replay. Checkpoint serialization and replay sampling copy mutable score/weight lists so changing a sampled item does not change the buffer.

The shared condition encoder is trained with the policy parameter group. Only the logZ head uses the separate logZ learning rate and is excluded from policy clipping. Temperature output weights initialize at zero and biases produce 0.2 for every condition; logZ output initializes at zero. Both heads become condition-dependent as their output weights train.

## TODO: replay research and experiments

These are deferred, not implemented switches. Test them independently before changing the default.

- Beta relabeling: sample a new training beta independently of replay trajectories, following [Logit-GFN, Kim et al., ICML 2024, Eq. 5](https://arxiv.org/html/2310.02823v3#S4.S2). Compare with original-condition replay at equal update and reward-evaluation budgets. Exploration and training beta distributions can differ.
- Preference relabeling: compare original preferences with partial relabeling and target-preference selection. [Zhu et al., NeurIPS 2023, §4.3](https://proceedings.neurips.cc/paper_files/paper/2023/file/fbc9981dd6316378aee7fd5975250f21-Paper-Conference.pdf) uses target-specific high-reward buffers, not unconditional random weight replacement. Recompute scalarized reward and all conditional model terms together.
- Prioritized replay: compare uniform sampling with reward-rank/quantile sampling, using [Shen et al., ICML 2023](https://proceedings.mlr.press/v202/shen23a/shen23a.pdf) and [Vemgal et al., 2023](https://arxiv.org/abs/2307.07674). Distinguish eviction policy from sampling policy. Across different beta/preferences, raw tempered rewards are not directly comparable priorities.
- Terminal-object replay/backward trajectory regeneration and [Local Search GFlowNets, ICLR 2024](https://arxiv.org/abs/2310.02710): assess reverse-search cost and route coverage before replacing stored-trajectory replay.
- Replay capacity and update-to-collection ratio: measure condition coverage, conditional reward distributions and Pareto/diversity evaluation in separate evaluation jobs; avoid adding per-step fingerprint/scaffold calculations.
