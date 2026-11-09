"""Transformer-based drug-likeness scoring."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from rdkit import Chem
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing_extensions import Self

from .cleaning import ParentCleaner
from .tokenizer import SmilesTokenizer
from .transformer import TransformerStack

NAMED_MODEL = {
    "pretrain": "SeonghwanSeo/chemprior-large-pretrain",
    "chembl": "SeonghwanSeo/chemprior-large-chembl",
}


def resolve_model_path(
    name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local model or download a Hub model."""
    path = Path(name_or_path).expanduser()
    if isinstance(name_or_path, Path) or path.exists():
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=NAMED_MODEL.get(name_or_path, name_or_path),
            allow_patterns=["config.json", "model.pth"],
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            max_workers=1,
        )
    )


class ChemPrior(nn.Module):
    """Autoregressive SMILES model for drug-likeness scoring."""

    @dataclass
    class Config:
        input_size: int = field(default_factory=lambda: len(SmilesTokenizer()))
        hidden_size: int = 512
        n_layers: int = 12
        n_heads: int = 8

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer: SmilesTokenizer = SmilesTokenizer()
        self.vocab: dict[str, int] = self.tokenizer.token_to_id
        self.pad_token_id: int = self.tokenizer.pad_token_id
        self.bos_token_id: int = self.tokenizer.bos_token_id
        self.eos_token_id: int = self.tokenizer.eos_token_id
        c = self.config
        self.embedding: nn.Embedding = nn.Embedding(c.input_size, c.hidden_size)
        self.transformer: TransformerStack = TransformerStack(
            c.hidden_size, c.n_heads, c.n_layers
        )
        self.fc: nn.Linear = nn.Linear(c.hidden_size, c.input_size, bias=False)
        self.apply(self._init_weight)
        self.fc.weight = self.embedding.weight

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "chembl",
        device: torch.device | str = "cpu",
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> Self:
        """Load local or named HF weights and return a model ready for inference."""
        path = resolve_model_path(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        config = cls.Config(**json.loads((path / "config.json").read_text()))
        state = torch.load(path / "model.pth", map_location="cpu", weights_only=True)
        model = cls(config)
        model.load_state_dict(state)
        return model.to(device).eval()

    @torch.no_grad()
    def screening(
        self,
        smiles_list: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> list[float]:
        """Score canonical parents in input order, preserving stereochemistry.

        Return 0–1 scores by default or summed log probabilities with normalize=False.
        Invalid or empty inputs receive 0.0 or -inf, respectively.
        Unresolved mixtures raise ValueError.
        """
        scores = [0.0 if normalize else -math.inf] * len(smiles_list)

        cleaner = ParentCleaner()
        pending: list[tuple[int, str]] = []

        def flush() -> None:
            values = self.evaluate_batch([s for _, s in pending], normalize=normalize)
            for (index, _), value in zip(pending, values, strict=True):
                scores[index] = max(scores[index], value)
            pending.clear()

        order = sorted(
            range(len(smiles_list)),
            key=lambda i: (len(smiles_list[i]), smiles_list[i]),
            reverse=True,
        )
        for index in order:
            mol = cleaner.clean(smiles_list[index])
            if mol is None:
                scores[index] = 0.0 if normalize else -math.inf
                continue
            smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            pending.append((index, smi))
            if len(pending) == batch_size:
                flush()
        if pending:
            flush()
        return scores

    @staticmethod
    def _init_weight(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Predict [B, L, vocab_size] logits for [B, L] targets including EOS."""
        bos = seq.new_full((seq.size(0), 1), self.bos_token_id)
        x = self.embedding(torch.cat((bos, seq[:, :-1]), dim=1))
        return self.fc(self.transformer(x))

    def calc_p_x(self, seq: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Sum token log probabilities per sequence, including EOS and skipping PAD."""
        logits = logits.float()
        # Exclude PAD and BOS from the prediction distribution.
        mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        mask[self.pad_token_id] = mask[self.bos_token_id] = True
        logits = logits.masked_fill(mask, -torch.inf)

        log_p_t = logits.log_softmax(-1)

        log_p_t = log_p_t.gather(-1, seq.unsqueeze(-1)).squeeze(-1)

        log_p_t = log_p_t.masked_fill(seq == self.pad_token_id, 0.0)
        log_p_seq = log_p_t.sum(dim=-1)
        return log_p_seq

    def tokenize(self, smiles: str, add_special_tokens: bool = True) -> list[int]:
        """Return token IDs, appending EOS when add_special_tokens is True."""
        ids = self.tokenizer.encode(smiles, add_special_tokens=False)
        return [*ids, self.eos_token_id] if add_special_tokens else ids

    def encode(self, input: list[str]) -> torch.Tensor:
        """Encode SMILES as padded [B, L] target token IDs."""
        rows = [torch.tensor(self.tokenize(s), dtype=torch.long) for s in input]
        seq = pad_sequence(rows, batch_first=True, padding_value=self.pad_token_id)
        return seq.to(self.device)

    @torch.no_grad()
    def evaluate_batch(
        self,
        smiles_list: list[str],
        normalize: bool = True,
    ) -> list[float]:
        """Score prepared SMILES without cleaning or stereoisomer enumeration.

        normalize=True returns 0–1 scores; False returns summed log probabilities.
        Use screening() for unvalidated input.
        """
        if not smiles_list:
            return []
        training = self.training
        self.eval()
        try:
            seq = self.encode(smiles_list)
            likelihood = self.calc_p_x(seq, self(seq))
            scores = (1 + likelihood / 100).clamp(0, 1) if normalize else likelihood
            return scores.tolist()
        finally:
            self.train(training)
