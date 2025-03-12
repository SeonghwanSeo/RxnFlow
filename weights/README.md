# Pretrained Model

All model files are uploaded at [Google Drive](https://drive.google.com/drive/folders/1OaJ9vaaykDhkyfl-JE5A5DzTeOMrUeqy)

## Download

You can download the pretrained model from Google Drive or using `gdown` package in terminal:

```bash
gdown <id> # if it fails, use gdown --id <id>

# example
gdown 1romQcMrJ7UM6Ps7sFrrvGp4Rmj_I2OBi
```

In addition, the pretrained models are auto-downloaded by declaring name.

```bash
# non-pocket-conditional gflownet (non-MOGFN)
python scripts/opt_unidock.py ... --pretrained_model qed-unif-0-64
python scripts/opt_unidock_moo.py ... --pretrained_model qed-unif-0-64

# few-shot training using pocket-conditional gflownet
python scripts/few_shot_unidock_moo.py ... --pretrained_model qvina-unif-0-64
```

## Model List

| Name            | Target-conditional | Temperature | Google drive Id                   | Updated    | Note                                          |
| --------------- | ------------------ | ----------- | --------------------------------- | ---------- | --------------------------------------------- |
| qed-unif-0-64   | X                  | $U(0, 64)$  | 1romQcMrJ7UM6Ps7sFrrvGp4Rmj_I2OBi | 2025-05-12 | [script](scripts/pretrain_qed.py)             |
| qvina-unif-0-64 | O                  | $U(0, 64)$  | 13ATBmJIgIhU3Kclwut0yssjaciPCieEV | 2025-05-12 | [script](scripts/train_pocket_conditional.py) |

\* All Model dimensions follow the default configuration.

## Prepare own pretrained model

You can train the model
