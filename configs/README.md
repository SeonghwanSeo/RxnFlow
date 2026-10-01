# Configuration guide

RxnFlow merges a YAML file with built-in defaults. `qed.yaml` is the minimal example and `template.yaml` lists every supported option.

`data.env_dir` points to a prepared Enamine synthon environment. `data.max_atoms` is the fixed RDKit heavy-atom capacity; graph tensors reserve one additional slot for the single dummy handle. Molecules are never truncated.

`generation.min_reactions` and `generation.max_reactions` count only UniReaction and BiReaction actions. FirstBlock does not contribute. The defaults allow one to three reactions. A zero-site product terminates immediately; before the minimum, terminating actions are masked. On the last allowed reaction, only brick closures or terminal UniReactions are allowed. There is no Stop action.

`subsampling.sampling_ratio` and `subsampling.min_sampling` control uniform candidate sampling within each compatible block library. `importance_temp` scales the inclusion-probability correction used during online action selection. All distinct site outcomes of a sampled block share its inclusion probability. TB/replay forces inclusion of the observed block and uses conditional inclusion probabilities for the other sampled blocks. A subsample with no feasible continuation produces an invalid trajectory; there is no retry or full-library fallback.

The shallow top-level `property_penalty` mapping accepts `mw`, `tpsa`, `hbd`, `hba`, `logp`, `rotatable_bonds`, `rings`, `aromatic_rings`, and `heavy_atoms`. These are hard upper bounds checked on each actual candidate product before scoring, for FirstBlock, UniReaction, and BiReaction. Intermediate descriptors describe the synthon, with dummy isotope labels excluded from mass. Terminal descriptors describe the final molecule. They are not additive reactant estimates or reward penalties.

`reward.exponent` and `reward.floor` control trajectory-balance reward scaling. `reward.settings` is expanded into the explicitly selected local `RewardFunction` constructor; YAML never imports a reward class.

`training.retrosynthesis_workers` controls the local CPU process pool used to calculate backward probabilities. Set it to `0` for synchronous execution.

At startup RxnFlow writes the complete resolved config into `run.output_dir/config.yaml`. Restart requires the exact current config and prepared environment signature.
