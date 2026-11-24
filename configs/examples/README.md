# Example training configurations

These configurations use the [5,000-building-block environment](../../README.md#prepare-a-synthesis-environment) with a subsampling ratio of 0.2. Run the scripts from the repository root with `--config configs/examples/<file>.yaml` and `--steps`. The Python script defines the reward, and the YAML selects the training settings.

| Configuration | Purpose | Training script |
| --- | --- | --- |
| [qed.yaml](qed.yaml) | Maximize QED at a fixed beta of 32, with a penalty above 500 Da. | [qed.py](../../examples/qed.py) |
| [qed_beta.yaml](qed_beta.yaml) | Train the same QED objective and MW penalty with beta sampled uniformly from 1 to 64. Choose beta within this range when sampling. | [qed.py](../../examples/qed.py) |
| [qed_prior.yaml](qed_prior.yaml) | Add a ChEMBL chemical prior to the fixed-beta QED example. Train with prior strength sampled uniformly from 0 to 1. | [qed.py](../../examples/qed.py) |
| [mw_logp.yaml](mw_logp.yaml) | Minimize MW and maximize logP with equal weights on `-MW / 100` and logP at a fixed beta of 32. | [mw_logp.py](../../examples/mw_logp.py) |
| [mw_logp_pref.yaml](mw_logp_pref.yaml) | Train the same two objectives across trade-offs by sampling Dirichlet preference weights at a fixed beta of 32. | [mw_logp.py](../../examples/mw_logp.py) |
| [tpsa_cond.yaml](tpsa_cond.yaml) | Generate molecules near a requested topological polar surface area (TPSA). Training samples target values from 0 to 200 Å² and beta from 1 to 64. | [tpsa_cond.py](../../examples/tpsa_cond.py) |
