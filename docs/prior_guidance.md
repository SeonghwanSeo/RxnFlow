# Chemical prior guidance

Prior guidance shifts the sampling distribution toward the chemical space learned by the prior model.

## Enable prior guidance

Set `prior_guidance` in your training configuration.

```yaml
prior_guidance:
  model_name_or_path: chembl
  beta: "uniform(0,50)" # or "25" for a fixed value
```

See the [QED example configuration](../configs/qed_prior.yaml).

With prior guidance enabled, the GFlowNet reward combines the task reward, the prior score, and the property penalty.

```text
R_gfn = R_task^beta * R_prior^prior_beta * property_penalty
```

`R_prior` is derived from the prior model's sequence log-likelihood.

```text
R_prior = clip(1 + log_likelihood / 100, 0, 1)
```

## Adjust prior guidance strength

If you train with a range of prior beta values, use `--prior-beta` to select a value within that range when sampling.

```bash
python scripts/sample.py \
  --checkpoint runs/qed_prior/checkpoints/latest.ckpt \
  --prior-beta 25 \
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
Training and validation both drop incomplete batches.
Run `python scripts/chemprior/train.py --help` for all options.

The script saves the model as `config.json` and `model.pth` in `runs/prior_finetune`.
It saves the weights from the epoch with the lowest validation loss, or from the final epoch if validation is disabled.
Set `prior_guidance.model_name_or_path` to `runs/prior_finetune` to use the fine-tuned model.
