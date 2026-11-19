"""Train a proxy model"""

import argparse
import copy
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rxnflow.models.proxy_model import MolProxy, collate, encode_mol


class ScoreDataset(Dataset):
    """Source SMILES with optional compressed graph caching."""

    def __init__(self, path: str | Path) -> None:
        self.graph_cache = []
        self.ids: list[str] = []
        self.smiles: list[str] = []
        self.scores: list[float] = []
        with open(path) as f:
            for row in csv.DictReader(f):
                self.ids.append(row["id"])
                self.smiles.append(row["smiles"])
                self.scores.append(float(row["score"]))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], float]:
        if self.graph_cache:
            arrays, offsets = self.graph_cache[index // 4096]
            local = index % 4096
            start, end = offsets[local : local + 2]
            graph = {
                key: torch.from_numpy(value[start:end]) for key, value in arrays.items()
            }
        else:
            graph = encode_mol(self.smiles[index])
        return graph, self.scores[index]

    def cache_graphs(self) -> None:
        """Pack graphs into CPU arrays; collate restores the training dtypes."""
        self.graph_cache = []
        dtypes = {
            "h": np.float16,
            "e": np.bool_,
            "edge_index": np.uint8,
            "edge_mask": np.bool_,
        }
        for start in range(0, len(self), 4096):
            graphs = [encode_mol(s) for s in self.smiles[start : start + 4096]]
            if any(g["h"].shape[0] > 256 for g in graphs):
                raise ValueError(
                    "uint8 graph cache supports at most 256 atoms per molecule"
                )
            offsets = np.cumsum([0] + [g["h"].shape[0] for g in graphs], dtype=np.int64)
            arrays = {
                key: np.concatenate([g[key].numpy().astype(dtype) for g in graphs])
                for key, dtype in dtypes.items()
            }
            self.graph_cache.append((arrays, offsets))

    def select(self, indices: list[int]) -> "ScoreDataset":
        dataset = copy.copy(self)
        dataset.graph_cache = []
        dataset.ids = [self.ids[i] for i in indices]
        dataset.smiles = [self.smiles[i] for i in indices]
        dataset.scores = [self.scores[i] for i in indices]
        return dataset

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "smiles", "score"])
            writer.writerows(zip(self.ids, self.smiles, self.scores, strict=True))


def random_split(dataset, seed: int, split: tuple[float, float, float]):
    count = len(dataset)
    n_train = int(count * split[0])
    n_val = int(count * split[1])
    if split[2] == 0:
        n_val = count - n_train

    order = np.random.default_rng(seed).permutation(count)
    val_end = n_train + n_val
    train_dataset = dataset.select(order[:n_train])
    val_dataset = dataset.select(order[n_train:val_end])
    test_dataset = None
    if split[2] > 0 and val_end < count:
        test_dataset = dataset.select(order[val_end:])
    return train_dataset, val_dataset, test_dataset


def load_data(
    data_paths: list[str | Path],
    seed: int,
    split_sizes: tuple[float, float, float] | None,
):
    if len(data_paths) == 1:
        # train+val+test
        dataset = ScoreDataset(data_paths[0])
        if split_sizes is None:
            split_sizes = (0.8, 0.1, 0.1) if len(dataset) < 10000 else (0.9, 0.05, 0.05)
        train_ds, val_ds, test_ds = random_split(dataset, seed, split_sizes)

    elif len(data_paths) == 2:
        # train+val / test
        dataset = ScoreDataset(data_paths[0])
        if split_sizes is None:
            split_sizes = (0.9, 0.1, 0.0)
        train_ds, val_ds, _ = random_split(dataset, seed, split_sizes)
        test_ds = ScoreDataset(data_paths[1])

    else:
        # train / val / test
        train_ds = ScoreDataset(data_paths[0])
        val_ds = ScoreDataset(data_paths[1])
        test_ds = ScoreDataset(data_paths[2])

    return train_ds, val_ds, test_ds


def _collate_scores(rows):
    graphs, scores = zip(*rows, strict=True)
    return collate(list(graphs)), torch.tensor(scores, dtype=torch.float32)


def make_loader(dataset, args, *, shuffle=False, seed=None):
    if seed is None:
        seed = args.seed
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate_scores,
    )


def train_epoch(model, dataset, optimizer, scheduler, args, epoch):
    loader = make_loader(dataset, args, shuffle=True, seed=args.seed + epoch)
    model.train()
    loss_sum = 0.0
    for graph, score in loader:
        graph = {key: value.to(model.device) for key, value in graph.items()}
        target = (score.to(model.device) - model.score_mean) / model.score_std
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(graph, normalized=True), target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += loss.item() * len(score)
    return loss_sum / len(dataset)


def make_scheduler(optimizer, args, steps_per_epoch):
    """Linear warmup followed by exponential decay, updated per step."""
    init_lr = args.lr * 0.1
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_epochs * steps_per_epoch, total_steps)

    def lr_factor(step):
        if warmup_steps == total_steps:
            lr = init_lr + (args.lr - init_lr) * min(step / warmup_steps, 1.0)
        elif step < warmup_steps:
            lr = init_lr + (args.lr - init_lr) * step / warmup_steps
        else:
            fraction = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
            lr = args.lr * (args.final_lr / args.lr) ** fraction
        return lr / init_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def regression_metrics(prediction, target):
    error = prediction - target
    squared_error = np.square(error)
    target_variance = float(np.square(target - target.mean()).sum())
    metrics = {
        "rmse": float(np.sqrt(squared_error.mean())),
        "mae": float(np.abs(error).mean()),
        "r2": 1.0 - float(squared_error.sum()) / target_variance
        if target_variance > 0
        else None,
    }
    return metrics


@torch.no_grad()
def val_epoch(model, dataset, args):
    loader = make_loader(dataset, args)
    model.eval()
    predictions, targets = [], []
    for graph, score in loader:
        graph = {
            key: value.to(model.device, non_blocking=True) for key, value in graph.items()
        }
        prediction = model(graph)
        predictions.append(prediction.cpu().numpy())
        targets.append(score.numpy())
    prediction = np.concatenate(predictions).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    return regression_metrics(prediction, target)


@torch.no_grad()
def test(model, dataset, args, prediction_path: Path):
    model.eval()
    prediction = np.asarray(
        model.screening(dataset.smiles, batch_size=args.batch_size), dtype=np.float64
    )
    target = np.asarray(dataset.scores, dtype=np.float32).astype(np.float64)
    metrics = regression_metrics(prediction, target)
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "smiles", "score", "prediction"])
        writer.writerows(
            zip(dataset.ids, dataset.smiles, target, prediction, strict=True)
        )
    return metrics


def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    scheduler,
    args,
    *,
    epoch,
    step,
    best_epoch,
    best_rmse,
    best_validation,
):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "step": step,
        "best_epoch": best_epoch,
        "best_rmse": best_rmse,
        "best_validation": best_validation,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(model.device)
        if model.device.type == "cuda"
        else None,
    }
    paths = [checkpoint_dir / f"epoch_{epoch:03d}.ckpt", checkpoint_dir / "last.ckpt"]
    if epoch == best_epoch:
        paths.append(checkpoint_dir / "best.ckpt")
    for path in paths:
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["rng_state"])
    if model.device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], model.device)
    return checkpoint


def train(args: argparse.Namespace) -> MolProxy:
    torch.manual_seed(args.seed)
    destination = Path(args.save_dir).expanduser()
    if args.resume:
        train_dataset = ScoreDataset(destination / "train.csv")
        val_dataset = ScoreDataset(destination / "val.csv")
        test_path = destination / "test.csv"
        test_dataset = ScoreDataset(test_path) if test_path.exists() else None
    else:
        train_dataset, val_dataset, test_dataset = load_data(
            args.data_path, args.seed, args.split_sizes
        )
    if args.cache_graphs:
        train_dataset.cache_graphs()
        val_dataset.cache_graphs()
    model = (
        MolProxy(model_dim=args.model_dim, depth=3, p_dropout=0.0).float().to(args.device)
    )
    model.score_mean.fill_(float(np.mean(train_dataset.scores)))
    model.score_std.fill_(float(np.std(train_dataset.scores)) or 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    scheduler = make_scheduler(optimizer, args, steps_per_epoch)

    checkpoint_dir = destination / "checkpoints"
    start_epoch, step, best_epoch, best_rmse = 1, 0, 0, math.inf
    best_validation = None
    if args.resume:
        checkpoint = restore_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        step = checkpoint["step"]
        best_epoch = checkpoint["best_epoch"]
        best_rmse = checkpoint["best_rmse"]
        best_validation = checkpoint["best_validation"]
        best_path = checkpoint_dir / "best.ckpt"
        temporary = best_path.with_suffix(".tmp")
        shutil.copyfile(checkpoint_dir / f"epoch_{best_epoch:03d}.ckpt", temporary)
        temporary.replace(best_path)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        train_dataset.to_csv(destination / "train.csv")
        val_dataset.to_csv(destination / "val.csv")
        if test_dataset is not None:
            test_dataset.to_csv(destination / "test.csv")
        checkpoint_dir.mkdir()
        (destination / "training_args.json").write_text(
            json.dumps(vars(args), indent=2) + "\n"
        )
    print(
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset) if test_dataset is not None else 0}",
        flush=True,
    )
    log_path = destination / (
        f"metrics_resume_{start_epoch:03d}.jsonl" if args.resume else "metrics.jsonl"
    )
    with log_path.open("a", encoding="utf-8") as metrics:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_epoch(
                model, train_dataset, optimizer, scheduler, args, epoch
            )
            step += steps_per_epoch
            validation = val_epoch(model, val_dataset, args)
            if validation["rmse"] < best_rmse:
                best_rmse, best_epoch = validation["rmse"], epoch
                best_validation = validation
            record = {
                "epoch": epoch,
                "step": step,
                "train_mse": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "val_rmse": validation["rmse"],
                "val_mae": validation["mae"],
                "val_r2": validation["r2"],
            }
            log_line = json.dumps(record, allow_nan=False)
            save_checkpoint(
                checkpoint_dir,
                model,
                optimizer,
                scheduler,
                args,
                epoch=epoch,
                step=step,
                best_epoch=best_epoch,
                best_rmse=best_rmse,
                best_validation=best_validation,
            )
            metrics.write(log_line + "\n")
            metrics.flush()
            print(log_line, flush=True)

    checkpoint = torch.load(
        checkpoint_dir / f"epoch_{best_epoch:03d}.ckpt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    final = {
        "best_epoch": best_epoch,
        "step": step,
        "validation": best_validation,
    }
    if test_dataset is not None:
        final["test"] = test(
            model,
            test_dataset,
            args,
            destination / "test_predictions.csv",
        )
    (destination / "evaluation.json").write_text(
        json.dumps(final, indent=2, allow_nan=False) + "\n"
    )
    torch.save(model.state_dict(), destination / "model.pth")
    print(json.dumps(final, allow_nan=False), flush=True)
    return model.eval()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        help="CSV files: full dataset; train/val pool + test; or train + val + test",
    )
    parser.add_argument("--save-dir", help="Empty output directory for a new run")
    parser.add_argument("--resume", help="Latest checkpoint of an interrupted run")
    parser.add_argument(
        "--split-sizes",
        type=float,
        nargs=3,
        help="Train/validation/test fractions for automatic splitting",
    )
    parser.add_argument(
        "--cache-graphs",
        action="store_true",
        help="Cache compressed train/validation graphs in CPU RAM",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda", help="cpu, cuda or cuda:<index>")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Graph encoding workers"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Initialization and loader shuffle seed"
    )
    args = parser.parse_args(argv)
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        epoch_paths = list(resume_path.parent.glob("epoch_*.ckpt"))
        if resume_path.name == "last.ckpt" and epoch_paths:
            resume_path = max(epoch_paths, key=lambda path: int(path.stem.split("_")[1]))
        if not resume_path.is_file():
            parser.error(f"Checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        saved_args = checkpoint["args"]
        saved_args.setdefault("model_dim", 256)
        run_dir = resume_path.parent.parent
        defaults = dict(saved_args)
        defaults.update(resume=str(resume_path), save_dir=str(run_dir))
        parser.set_defaults(**defaults)
        args = parser.parse_args(argv)
        args.resume = str(resume_path)
        if Path(args.save_dir).expanduser().resolve() != run_dir:
            parser.error("Resume must use the original run directory")
        for name in (
            "data_path",
            "split_sizes",
            "epochs",
            "batch_size",
            "model_dim",
            "lr",
            "final_lr",
            "warmup_epochs",
            "seed",
        ):
            if getattr(args, name) != saved_args[name]:
                parser.error(f"Cannot change --{name.replace('_', '-')} when resuming")
        latest_epoch = max(
            int(path.stem.split("_")[1])
            for path in resume_path.parent.glob("epoch_*.ckpt")
        )
        if checkpoint["epoch"] != latest_epoch:
            parser.error(
                "Resume from the latest epoch checkpoint to preserve later checkpoints"
            )
        args.save_dir = str(run_dir)
        return args

    if args.data_path is None or args.save_dir is None:
        parser.error("New runs require --data-path and --save-dir")
    if len(args.data_path) > 3:
        parser.error("data-path accepts one, two or three CSV files")
    if args.split_sizes is not None:
        if len(args.data_path) == 3:
            parser.error("Do not specify --split-sizes with three CSV files")
        train_ratio, val_ratio, test_ratio = args.split_sizes
        if not (
            0 < train_ratio < 1
            and 0 < val_ratio < 1
            and 0 <= test_ratio < 1
            and math.isclose(sum(args.split_sizes), 1.0, abs_tol=1e-8)
        ):
            parser.error(
                "Split ratios must sum to 1 with train > 0, val > 0 and test >= 0"
            )
        if len(args.data_path) == 2 and test_ratio != 0:
            parser.error("With two CSV files, use --split-sizes TRAIN VAL 0")

    destination = Path(args.save_dir).expanduser()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error(f"Use an empty output directory: {destination}")
    return args


if __name__ == "__main__":
    train(parse_args())
