# Chemical prior guidance

Prior guidance shifts the sampling distribution toward the chemical space learned by the prior model.

## Enable prior guidance

Set `prior_guidance` in your training configuration.

```yaml
gflownet:
  beta: 32

prior_guidance:
  model_name_or_path: chembl
  strength: "uniform(0,1)" # or 0.5 for fixed guidance strength
```

See the [QED example configuration](../configs/examples/qed_prior.yaml).

Increasing `gflownet.beta` favors molecules with higher task rewards. Increasing `prior_guidance.strength` favors molecules with higher sequence probability under the prior. Setting either value to zero removes that factor from the target. The property penalty applies independently.

```text
log p(x) = beta * R_task(x) + prior_strength * log p_prior(x) + log(property_penalty) - log Z
```

The optional normalized score is a human-readable representation:

```text
s_prior = clip(1 + log p_prior(x) / 100, 0, 1)
```

## Adjust prior guidance strength

If you train with a range of guidance strengths, use `--prior-beta` to select a value within that range when sampling.

```bash
python scripts/sample.py \
  --checkpoint runs/qed_prior/checkpoints/latest.ckpt \
  --prior-beta 0.5 \
  --num-samples 100 \
  --output samples.jsonl
```

## Fine-tune a prior model

Use `pretrain` for the large model pretrained on ZINC and PubChem, or `chembl` for the ChEMBL-adapted model.
You can fine-tune either model on your own molecular dataset.

### Prepare training data

Provide one SMILES per line, optionally followed by a whitespace-separated ID.

```bash
python scripts/chemprior/preprocess.py \
  --input data/raw.smi \
  --output data/clean.smi \
  --max-length 255
```

### Fine-tune

```bash
python scripts/chemprior/train.py \
  --model chembl \
  --input data/clean.smi \
  --save-dir runs/prior_finetune
```

`--val-ratio` sets the validation fraction and defaults to `0.1`.
Set it to `0` to skip validation.
Run `python scripts/chemprior/train.py --help` for all options.

The script saves the model as `config.json` and `model.pth` in `runs/prior_finetune`.
It saves the weights from the epoch with the lowest validation loss, or from the final epoch if validation is disabled.
Set `prior_guidance.model_name_or_path` to `runs/prior_finetune` to use the fine-tuned model.

## Teacher scoring API

`ChemPrior.screening()` and `ChemPrior.evaluate_batch()` return sequence `log p(x)` by default (`normalize=False`).
Use `normalize=True` for the human-readable normalized score `clip(1 + log p(x) / 100, 0, 1)`.
Guidance uses log probability; normalized scores are for reporting and visualization.
Invalid screening inputs return `-inf` by default and `0` with `normalize=True`.
`PriorGuidance` converts nonfinite outputs to its finite invalid-trajectory penalty.
