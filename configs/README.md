# Configuration examples

- [Complete template](template.yaml): all supported model and training settings. Explicit values override the defaults.
- [QED](qed.yaml): single-objective optimization.
- [QED/SA](qed_sa.yaml): product-reward optimization with beta conditioning.

Execution options such as output directory, device and seed are passed to the training script, not YAML. See the [training guide](../docs/training.md) for key settings and [reward guide](../docs/rewards.md) for conditioning.
