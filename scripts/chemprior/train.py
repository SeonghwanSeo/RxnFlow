"""Fine-tune ChemPrior on prepared .smi files using native PyTorch."""

import argparse
import gc
import json
import math
import shutil
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chemprior import ChemPrior, SmilesTokenizer


def load_smiles(path: str | Path, max_length: int = 255) -> list[str]:
    """Read prepared SMILES and filter unsupported or overlong sequences before splitting."""
    tokenizer = SmilesTokenizer()
    accepted = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        smiles = line.split()[0]
        try:
            tokens = tokenizer.encode(smiles, add_special_tokens=False)
        except (KeyError, ValueError, AttributeError):
            continue
        if len(tokens) <= max_length:
            accepted.append(smiles)
    if not accepted:
        raise ValueError(f"No valid SMILES in {path}.")
    return accepted


class SmilesDataset(Dataset):
    """Tokenize an already filtered train or validation SMILES list."""

    def __init__(self, smiles: list[str], max_length: int = 255):
        self.tokenizer = SmilesTokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.max_length = max_length + 1
        self.data = [
            np.array(
                [*self.tokenizer.encode(s, add_special_tokens=False), self.eos_token_id],
                dtype=np.int64,
            )
            for s in smiles
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.data[index])


def validate_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if (
        not all(math.isfinite(x) and 0 <= x < 1 for x in (val_ratio, test_ratio))
        or val_ratio <= 0
        or val_ratio + test_ratio >= 1
    ):
        raise ValueError(
            "Validation must be positive; split fractions must sum to less than 1"
        )


def split_smiles(
    smiles: list[str],
    val_ratio: float,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic row split of a preprocessed, deduplicated dataset."""
    validate_split_ratios(val_ratio, test_ratio)
    order = np.random.default_rng(seed).permutation(len(smiles))
    # Preserve the original train suffix; partition only its held-out prefix.
    heldout_ratio = val_ratio + test_ratio
    n_holdout = round(len(smiles) * heldout_ratio)
    n_val = int(n_holdout * val_ratio / heldout_ratio)
    n_test = n_holdout - n_val
    if (val_ratio and not n_val) or (test_ratio and not n_test):
        raise ValueError("Requested validation/test split is empty")
    train_data = [smiles[i] for i in order[n_holdout:]]
    val_data = [smiles[i] for i in order[:n_val]]
    test_data = [smiles[i] for i in order[n_val:n_holdout]]

    print(
        f"Split: Train: {len(train_data)}, "
        f"Val: {len(val_data)}, "
        f"Test: {len(test_data)}"
    )

    return train_data, val_data, test_data


def _collate_smiles(
    batch: list[torch.Tensor],
    pad_token_id: int,
    max_length: int,
) -> torch.Tensor:
    """Pad target sequences to max_length."""
    padded = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    for index, sequence in enumerate(batch):
        padded[index, : len(sequence)] = sequence
    return padded


def make_loader(
    dataset: SmilesDataset,
    batch_size: int,
    *,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int = 42,
    drop_last: bool = True,
) -> DataLoader:
    """Build a padded loader; training shuffles, validation keeps input order."""
    if not len(dataset) or (drop_last and len(dataset) < batch_size):
        raise ValueError(
            "Dataset must contain at least one full batch with drop_last=True"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=partial(
            _collate_smiles,
            pad_token_id=dataset.pad_token_id,
            max_length=dataset.max_length,
        ),
    )


def train_epoch(
    model: ChemPrior,
    dataset: SmilesDataset,
    optimizer: torch.optim.Optimizer,
    accumulation: int,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    precision: str = "float32",
) -> tuple[float, int]:
    """Return token-averaged training loss and the number of optimizer updates."""
    loader = make_loader(dataset, batch_size, num_workers=num_workers, shuffle=True, seed=seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_nll = 0.0
    total_tokens = 0
    group_tokens = 0
    updates = 0

    for batch_index, seq in enumerate(loader, start=1):
        seq = seq.to(model.device)
        num_tokens = int((seq != model.pad_token_id).sum())

        with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            logits = model(seq).float()

        mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        mask[model.pad_token_id] = True
        mask[model.bos_token_id] = True
        logits = logits.masked_fill(mask, -torch.inf)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            seq.flatten(),
            ignore_index=model.pad_token_id,
            reduction="sum",
        )

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        total_nll += loss.detach().item()
        total_tokens += num_tokens
        group_tokens += num_tokens

        group_complete = batch_index % accumulation == 0
        last_batch = batch_index == len(loader)
        if not (group_complete or last_batch):
            continue

        # Average over actual tokens, including an incomplete final group.
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(group_tokens)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        group_tokens = 0

    return total_nll / total_tokens, updates


@torch.no_grad()
def val_epoch(
    model: ChemPrior, dataset: SmilesDataset, *, batch_size: int,
    num_workers: int = 0, precision: str = "float32",
) -> dict[str, float]:
    """Evaluate every validation row, weighting loss by non-padding tokens."""
    loader = make_loader(dataset, batch_size, num_workers=num_workers, drop_last=False)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_score = 0.0
    total_rows = 0

    for seq in loader:
        seq = seq.to(model.device)
        with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            logits = model(seq).float()
        log_p = model.calc_p_x(seq, logits)
        total_nll -= log_p.sum().item()
        total_tokens += int((seq != model.pad_token_id).sum())
        total_score += (1 + log_p / 100).clamp(0, 1).sum().item()
        total_rows += len(seq)

    loss = total_nll / total_tokens
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite validation loss")
    return {"val_loss": loss, "val_score": total_score / total_rows}


def restore_checkpoint(
    checkpoint: dict,
    training_config: dict,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, float, dict[str, torch.Tensor] | None]:
    """Restore optimizer and progress for an epoch-boundary restart."""
    for key in (
        "input",
        "val_ratio",
        "batch_size",
        "accumulate_grad",
        "lr",
        "max_length",
        "precision",
        "seed",
    ):
        if checkpoint["args"][key] != training_config[key]:
            raise ValueError(f"Resume requires unchanged {key}")
    if checkpoint["epoch"] >= training_config["epochs"]:
        raise ValueError("epochs must exceed the completed checkpoint epoch")
    if checkpoint["args"].get("val_input"):
        raise ValueError(
            "Cannot resume an external-validation checkpoint with internal splits"
        )
    if checkpoint["args"].get("test_ratio", 0.0) != training_config["test_ratio"]:
        raise ValueError("Resume requires unchanged test_ratio")
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        checkpoint["epoch"],
        checkpoint["step"],
        checkpoint["best_loss"],
        checkpoint["best_state_dict"],
    )


def save_checkpoint(
    path: Path,
    model: ChemPrior,
    optimizer: torch.optim.Optimizer,
    training_config: dict,
    *,
    epoch: int,
    step: int,
    best_loss: float,
    best_state_dict: dict[str, torch.Tensor] | None,
) -> None:
    """Preserve an epoch checkpoint and atomically update the latest copy."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "best_state_dict": best_state_dict,
        "args": training_config,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    latest = path.parent / "last.ckpt"
    shutil.copyfile(path, latest)


def train(
    train_dataset: SmilesDataset,
    val_dataset: SmilesDataset,
    destination: str | Path,
    *,
    model_path: str,
    device: str | torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    num_workers: int = 0,
    accumulate_grad: int = 1,
    precision: str = "float32",
    compile_model: bool = False,
    resume: str | None = None,
    training_config: dict,
) -> None:
    """Train on prepared datasets and export the best-validation checkpoint."""
    device = torch.device(device)
    destination = Path(destination)
    torch.manual_seed(seed)
    checkpoint = None
    if resume:
        checkpoint = torch.load(Path(resume), map_location="cpu", weights_only=True)
        model = ChemPrior(ChemPrior.Config(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
    else:
        model = ChemPrior.from_pretrained(model_path, device=device)

    if compile_model:
        model.compile(dynamic=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    start_epoch = 0
    step = 0
    best_loss = math.inf
    best_state_dict = None
    if checkpoint is not None:
        start_epoch, step, best_loss, best_state_dict = restore_checkpoint(
            checkpoint, training_config, optimizer
        )
    del checkpoint

    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir()
    (destination / "training_args.json").write_text(
        json.dumps(training_config, indent=2) + "\n"
    )
    with (destination / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for epoch in range(start_epoch, epochs):
            epoch_started = time.monotonic()

            # Training
            loss, updates = train_epoch(
                model, train_dataset, optimizer, accumulate_grad,
                batch_size=batch_size, seed=seed + epoch,
                num_workers=num_workers, precision=precision,
            )
            step += updates
            record = {"epoch": epoch + 1, "step": step, "train_loss": loss}

            # Validation
            record.update(val_epoch(model, val_dataset, batch_size=batch_size,
                                    num_workers=num_workers, precision=precision))
            if record["val_loss"] < best_loss:
                best_loss = record["val_loss"]
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            record["train_validation_seconds"] = time.monotonic() - epoch_started
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.ckpt",
                model,
                optimizer,
                training_config,
                epoch=epoch + 1,
                step=step,
                best_loss=best_loss,
                best_state_dict=best_state_dict,
            )

    torch.save(best_state_dict, destination / "model.pth")
    (destination / "config.json").write_text(
        json.dumps(asdict(model.config), indent=2) + "\n"
    )


@torch.no_grad()
def evaluate_test(
    model_dir: str | Path,
    smiles: list[str],
    *,
    batch_size: int,
    device: str | torch.device,
    precision: str = "float32",
) -> ChemPrior:
    """Reload best weights and score held-out SMILES through ChemPrior's public API."""
    destination = Path(model_dir)
    model = ChemPrior.from_pretrained(destination, device=device)
    with torch.autocast(
        torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=precision == "bfloat16",
    ):
        scores = model.screening(smiles, batch_size=batch_size)
    if smiles:
        (destination / "test_metrics.json").write_text(
            json.dumps(
                {
                    "n": len(smiles),
                    "test_score": sum(scores) / len(scores),
                },
                indent=2,
            )
            + "\n"
        )
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, help="Preprocessed deduplicated .smi file"
    )
    parser.add_argument("--save-dir", required=True, help="Empty output directory")
    parser.add_argument(
        "--model",
        default="chembl",
        help="Model name, HF repo or local path",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Validation fraction"
    )
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test fraction")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Total epochs, including resumed epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Molecules per batch; incomplete batches are dropped",
    )
    parser.add_argument(
        "--accumulate-grad",
        type=int,
        default=1,
        help="Microbatches per optimizer update",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile(dynamic=False)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Constant AdamW learning rate"
    )
    parser.add_argument(
        "--max-length", type=int, default=255, help="Maximum SMILES tokens, excluding EOS"
    )
    parser.add_argument(
        "--device", default="cuda", choices=("cpu", "cuda"), help="Training device"
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Training precision",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 loads in the main process",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for splitting and training"
    )
    parser.add_argument(
        "--resume",
        help="Path to an epoch checkpoint or last.ckpt",
    )
    args = parser.parse_args(argv)
    try:
        validate_split_ratios(args.val_ratio, args.test_ratio)
    except ValueError as error:
        parser.error(str(error))
    if min(args.batch_size, args.accumulate_grad, args.epochs) < 1:
        parser.error("batch-size, accumulate-grad and epochs must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if not 1 <= args.max_length <= 2047:
        parser.error("max-length must be between 1 and 2047 (excluding EOS)")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    return args


if __name__ == "__main__":
    args = parse_args()

    destination = Path(args.save_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {destination}")

    # Load and split the dataset, then train and evaluate.
    smiles = load_smiles(args.input, args.max_length)
    train_smiles, val_smiles, test_smiles = split_smiles(
        smiles, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    destination.mkdir(parents=True, exist_ok=True)
    for name, sequences in (
        ("train", train_smiles),
        ("val", val_smiles),
        ("test", test_smiles),
    ):
        (destination / f"{name}.smi").write_text(
            "".join(smiles + "\n" for smiles in sequences)
        )

    train_dataset = SmilesDataset(train_smiles, args.max_length)
    val_dataset = SmilesDataset(val_smiles, args.max_length)

    # Train the model
    training_config = vars(args).copy()
    train(
        train_dataset,
        val_dataset,
        destination,
        model_path=args.model,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        accumulate_grad=args.accumulate_grad,
        precision=args.precision,
        compile_model=args.compile,
        resume=args.resume,
        training_config=training_config,
    )
    gc.collect()
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    if test_smiles:
        # Evaluate the best checkpoint on the held-out test set.
        evaluate_test(
            destination,
            test_smiles,
            batch_size=args.batch_size,
            device=args.device,
            precision=args.precision,
        )
