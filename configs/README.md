# Configuration guide

RxnFlow merges a YAML file with built-in defaults. `qed.yaml` is the minimal example and `template.yaml` lists every supported option.

`data.env_dir` points to a prepared Enamine synthon environment. `data.max_atoms` is the fixed RDKit heavy-atom capacity; graph tensors reserve one additional slot for the single dummy handle. Molecules are never truncated.

`generation.max_reactions` counts only UniReaction and BiReaction actions. FirstBlock does not contribute. The default allows up to three reactions. A zero-site product terminates immediately, including after the first reaction. On the last allowed reaction, only brick closures or terminal UniReactions are allowed. There is no Stop action.

`subsampling.sampling_ratio` and `subsampling.min_sampling` control uniform candidate sampling within each compatible block library. `importance_temp` scales the inclusion-probability correction used during online action selection. All distinct site outcomes of a sampled block share its inclusion probability. A policy batch shares one draw per library. TB/replay forces inclusion of the union of observed rows in that library and uses conditional inclusion probabilities for the remaining sampled population. A subsample with no feasible continuation produces an invalid trajectory; there is no retry or full-library fallback.

The shallow top-level `property_penalty` mapping accepts `mw`, `tpsa`, `hbd`, `hba`, `logp`, `rotatable_bonds`, `rings`, `aromatic_rings`, and `heavy_atoms`. These are hard upper bounds checked on each actual candidate product before scoring, for FirstBlock, UniReaction, and BiReaction. Intermediate descriptors describe the synthon, with dummy isotope labels excluded from mass. Terminal descriptors describe the final molecule. They are not additive reactant estimates or reward penalties.

The YAML `reward.beta` is a string specifying the reward exponent: `"32"` or `"uniform(1,64)"`. `reward.preferences` defaults to `"uniform"` (uniform on the simplex, exactly Dirichlet(1)); alternatives are `"dirichlet(0.5)"` for symmetric concentration or `"fixed(0.3,0.7)"` in objective order. CLI/YAML boundaries parse strings into `tuple[str, list[float]]`: beta `("fixed", [32.0])` or `("uniform", [1.0, 64.0])`, preferences `("dirichlet", [1.0])` or `("fixed", [0.3, 0.7])`. Python Config/Sampler/ConditionSampler consume these tuples, not strings. `sample_distribution()` supplies shared fixed/uniform/Dirichlet draws. Trajectories/replay store sampled numeric beta and weights. `reward.floor` applies before exponentiation. CLI sampling accepts the same strings: `--beta "uniform(1,64)" --preferences "fixed(0.3,0.7)"`. A single objective always has weight `[1]`. These are external settings, independent of the fixed model encoder coordinates. `reward.settings` contains constructor kwargs; reward selection stays explicit in Python.

`training.retrosynthesis_workers` controls the local CPU process pool used to calculate backward probabilities. Set it to `0` for synchronous execution.

At startup RxnFlow writes the complete resolved config into `run.output_dir/config.yaml`. Restart requires the exact current config and prepared environment signature.
