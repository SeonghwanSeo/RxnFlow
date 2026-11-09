"""Fine-tune ChemPrior on prepared .smi files using native PyTorch."""

import argparse
import copy
import json
import math
import shutil
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chemprior import ChemPrior, SmilesTokenizer


class SmilesDataset(Dataset):
    """Tokenize prepared SMILES, skipping unsupported and overlong sequences."""

    def __init__(self, path: str | Path, max_length: int = 255) -> None:
        tokenizer = SmilesTokenizer()
        self.eos_token_id: int = tokenizer.eos_token_id
        self.pad_token_id: int = tokenizer.pad_token_id
        self.sequence_length = max_length + 1  # Include EOS.

        with Path(path).expanduser().open(encoding="utf-8") as source:
            smiles_list = [
                line.split()[0]
                for line in source
                if line.strip() and not line.lstrip().startswith("#")
            ]

        self.sequences: list[np.ndarray] = []
        n_total = len(smiles_list)
        n_accepted = 0
        n_skipped = 0
        n_invalid = 0
        for smi in smiles_list:
            try:
                tokens = tokenizer.encode(smi, add_special_tokens=False)
            except (KeyError, ValueError, AttributeError):
                n_invalid += 1
                continue  # Skip invalid SMILES
            if len(tokens) > max_length:
                n_skipped += 1
                continue  # Skip sequences that exceed max_length
            n_accepted += 1
            self.sequences.append(np.array([*tokens, self.eos_token_id], dtype=np.int64))
        print(
            f"Loaded {n_accepted} SMILES from {path}. "
            f"Total: {n_total}, Accepted: {n_accepted}, "
            f"Skipped: {n_skipped}, Invalid: {n_invalid}"
        )
        if n_accepted == 0:
            raise ValueError(f"No valid SMILES in {path}.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.sequences[index])


def split_dataset(dataset, val_ratio, seed):
    """Deterministic row split of a preprocessed, deduplicated dataset."""
    order = np.random.default_rng(seed).permutation(len(dataset))
    n_val = round(len(dataset) * val_ratio)
    if val_ratio > 0 and n_val == 0:
        raise ValueError("Validation split is empty")
    train = copy.copy(dataset)
    train.sequences = [dataset.sequences[i] for i in order[n_val:]]
    val = None
    if n_val:
        val = copy.copy(dataset)
        val.sequences = [dataset.sequences[i] for i in order[:n_val]]
    print(f"Split: train={len(train)}, val={n_val}, seed={seed}", flush=True)
    return train, val


def _collate_smiles(
    batch: list[torch.Tensor], pad_token_id: int, sequence_length: int
) -> torch.Tensor:
    """Pad target sequences to max_length + 1, including EOS."""
    padded = batch[0].new_full((len(batch), sequence_length), pad_token_id)
    for index, sequence in enumerate(batch):
        padded[index, : sequence.numel()] = sequence
    return padded


def make_loader(
    dataset: SmilesDataset,
    batch_size: int,
    *,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Build a padded loader; training shuffles, validation keeps input order."""
    if len(dataset) < batch_size:
        raise ValueError(
            "Dataset must contain at least one full batch with drop_last=True"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=partial(
            _collate_smiles,
            pad_token_id=dataset.pad_token_id,
            sequence_length=dataset.sequence_length,
        ),
    )


def train_epoch(
    model: ChemPrior,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accumulation: int,
    bf16: bool,
) -> tuple[float, int]:
    """Return token-averaged training loss and the number of optimizer updates."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_nll = 0.0
    total_tokens = 0
    group_tokens = 0
    updates = 0

    for batch_index, seq in enumerate(loader, start=1):
        seq = seq.to(model.device)
        num_tokens = int((seq != model.pad_token_id).sum())

        with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(seq)
            logits = logits.float()
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
def evaluate(model: ChemPrior, loader: DataLoader, bf16: bool) -> dict[str, float]:
    """Evaluate full validation batches, weighting loss by non-padding tokens."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_score = 0.0
    total_rows = 0

    for seq in loader:
        seq = seq.to(model.device)
        with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(seq)
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
    args: argparse.Namespace,
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
        if checkpoint["args"][key] != getattr(args, key):
            raise ValueError(f"Resume requires unchanged {key}")
    if checkpoint["epoch"] >= args.epochs:
        raise ValueError("epochs must exceed the completed checkpoint epoch")
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
    args: argparse.Namespace,
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
        "args": vars(args),
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    temporary = path.with_suffix(".ckpt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    latest = path.parent / "last.ckpt"
    latest_temporary = latest.with_suffix(".ckpt.tmp")
    shutil.copyfile(path, latest_temporary)
    latest_temporary.replace(latest)


def train(args: argparse.Namespace) -> ChemPrior:
    device = torch.device(args.device)
    destination = Path(args.save_dir).expanduser()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {destination}")

    torch.manual_seed(args.seed)
    dataset = SmilesDataset(args.input, args.max_length)
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.seed)
    val_loader = None
    if val_dataset is not None:
        val_loader = make_loader(
            val_dataset, args.batch_size, num_workers=args.num_workers
        )
    if len(train_dataset) < args.batch_size:
        raise ValueError("Training split must contain at least one full batch")

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(
            Path(args.resume).expanduser(), map_location="cpu", weights_only=True
        )
        model = ChemPrior(ChemPrior.Config(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
    else:
        model = ChemPrior.from_pretrained(args.model, device=device)

    if args.compile:
        model.compile(dynamic=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    start_epoch = 0
    step = 0
    best_loss = math.inf
    best_state_dict = None
    if checkpoint is not None:
        start_epoch, step, best_loss, best_state_dict = restore_checkpoint(
            checkpoint, args, optimizer
        )
    del checkpoint

    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir()
    (destination / "training_args.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )
    bf16 = args.precision == "bfloat16"
    with (destination / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for epoch in range(start_epoch, args.epochs):
            train_loader = make_loader(
                train_dataset,
                args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                seed=args.seed + epoch,
            )
            loss, updates = train_epoch(
                model, train_loader, optimizer, args.accumulate_grad, bf16
            )
            step += updates
            record = {"epoch": epoch + 1, "step": step, "train_loss": loss}

            if val_loader is not None:
                record.update(evaluate(model, val_loader, bf16))
                if record["val_loss"] < best_loss:
                    best_loss = record["val_loss"]
                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }

            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.ckpt",
                model,
                optimizer,
                args,
                epoch=epoch + 1,
                step=step,
                best_loss=best_loss,
                best_state_dict=best_state_dict,
            )

    state_dict = model.state_dict() if best_state_dict is None else best_state_dict
    torch.save(state_dict, destination / "model.pth")
    (destination / "config.json").write_text(
        json.dumps(asdict(model.config), indent=2) + "\n"
    )
    return model.eval()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, help="Preprocessed deduplicated .smi file"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation fraction; 0 disables validation",
    )
    parser.add_argument("--save-dir", required=True, help="Empty output directory")
    parser.add_argument(
        "--model",
        default="chembl",
        help="Model name, HF repo or local path",
    )
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
    if not 0 <= args.val_ratio < 1:
        parser.error("val-ratio must be in [0, 1)")
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
    train(args)
