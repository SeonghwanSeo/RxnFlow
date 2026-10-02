# Beta and preference conditioning

The model takes numeric beta and preferences from its caller. The beta encoder always uses `u=(beta-1)/63` and `[u, sin(2*pi*u*f), cos(2*pi*u*f)]` for fixed frequencies `(1, 2, 4, 8)` (nine inputs). These are encoder coordinates, not allowed beta bounds; no clamp is applied. A two-Linear MLP encodes these features, another encodes the preference vector, and their outputs are added. Separate branches inject this condition into the initial virtual node, predict a shared positive `logit_scale(beta, w) = ELU(head) + 1`, and predict `log Z(beta, w)`. There is no per-layer FiLM and no generic unused condition argument.

`RewardFunction.objectives` defines ordered objective names. `score(molecules)` returns a detached float32 tensor `[B, K]`, including `[0, K]` for an empty batch. Objective transforms and scale are the reward implementation's responsibility; larger scores must be better and all scores must be finite and non-negative. The trainer computes `R(x,w)=sum(w*r(x))` and `log_reward=beta*log(max(R, floor))`. Invalid/filtered samples receive a zero objective vector. A single objective follows exactly the same path with K=1 and w=[1]. The environment, action validity and backward analysis do not depend on beta or preferences.

## External settings

```yaml
reward:
  beta: "uniform(8,64)"  # Default: "32".
  preferences: "uniform" # Or "dirichlet(0.5)" / "fixed(0.8,0.2)".
  floor: 0.0001
  settings: {}
```

The YAML `reward.beta` is a string specifying the reward exponent: `"32"` or `"uniform(1,64)"`. `reward.preferences` defaults to `"uniform"` (uniform on the simplex, exactly Dirichlet(1)); alternatives are `"dirichlet(0.5)"` for symmetric concentration or `"fixed(0.3,0.7)"` in objective order. CLI/YAML boundaries parse strings into `tuple[str, list[float]]`: beta `("fixed", [32.0])` or `("uniform", [1.0, 64.0])`, preferences `("dirichlet", [1.0])` or `("fixed", [0.3, 0.7])`. Python Config/Sampler/ConditionSampler consume these tuples, not strings. `sample_distribution()` supplies shared fixed/uniform/Dirichlet draws. Trajectories/replay store sampled numeric beta and weights. `reward.floor` applies before exponentiation. CLI sampling accepts the same strings: `--beta "uniform(1,64)" --preferences "fixed(0.3,0.7)"`. A single objective always has weight `[1]`. These are external settings, independent of the fixed model encoder coordinates. `reward.settings` contains constructor kwargs; reward selection stays explicit in Python. The trainer and sampler draw conditions once per trajectory using the checkpointed CPU global Torch RNG, independently of the library subsampling generator. `sampling_temperature` is an additional softmax temperature (default 1), separate from beta. Checkpoints preserve objective order, both models, configuration, replay, optimizers and RNG states.

Uniform preferences do not imply uniformly spaced points along a Pareto frontier. For reproducible frontier comparisons, evaluate repeated samples at a fixed grid of preferences including endpoints; `fixed(...)` supports these calls. No automatic grid evaluation is implemented. Normalizing independent Uniform(0,1) coordinates is not uniform on the simplex.

`RewardFunction.filter_object(mol)` owns optional pre-scoring rejection. Rejected molecules get zero objective rewards and remain in sample outputs. There is no separate trainer/sampler filter callback.

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


## Reference alignment and naming TODO

The scalar head and multiplicative `logit_scale()` convention follow `source/rxnflow_hits/src/rxnflow/models/gfn.py` and `policy/action_categorical.py`. The [Logit-GFN paper, Eq. 4](https://arxiv.org/html/2310.02823v3#S4.S1) writes logits divided by a learned scalar temperature; HSX parameterizes the multiplicative inverse scale. We retain the agreed virtual-node conditioning and preference conditioning. The removed 0.01–10 bounds and per-reaction temperatures were local additions, not requirements of either reference. Initial scale is now 1 (previous effective scale was 5 from temperature 0.2), and head dimensions changed; this is an architecture change, not a numerical-equivalence refactor. The MLP width/activation retains this repository's implementation; only the scalar scale formula/naming follows HSX here.

`State` replaces `MoleculeState`. `ActionSpace` is a list of static reaction/library tuples stored by the environment; `ActionSubspace` carries sampled indices, learned logits and subsampling weights; categorical sampling computes random-policy library balancing when needed. Custom exceptions are defined in `rxnflow/errors.py`.

Architecture, TB and policy naming alignment is recorded in [naming.md](naming.md). Computational behavior is unchanged by that naming pass. No intermediate commits during the current feedback process.
