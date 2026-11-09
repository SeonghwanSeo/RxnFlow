"""SMILES tokenization and vocabulary management."""

import re
from collections.abc import Iterable
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Any

from typing_extensions import Self

VOCAB_PATH: Path = Path(__file__).with_name("vocab.txt")

OUTER = re.compile(r"\[[^\[\]\s]+\]|Cl|Br|%\([0-9]+\)|%[0-9]{2}|->|<-|[\s\S]")
BRACKET = re.compile(r"\[([0-9]*)([A-Z][a-z]?|si|se|as|te|[bcnops]|\*)(.*)\]")
CHIRAL_CLASSES = {"TH", "AL", "SP", "TB", "OH"}


class SmilesTokenizer:
    """Split SMILES into element and syntax tokens."""

    def __init__(self, token_path: str | Path = VOCAB_PATH) -> None:
        self.tokens: tuple[str, ...] = tuple(Path(token_path).read_text().splitlines())
        self.token_to_id: dict[str, int] = {
            token: i for i, token in enumerate(self.tokens)
        }
        self.pad_token_id: int = self.token_to_id["<PAD>"]
        self.bos_token_id: int = self.token_to_id["<BOS>"]
        self.eos_token_id: int = self.token_to_id["<EOS>"]
        self._init_caches()

    def _init_caches(self) -> None:
        self._parts = lru_cache(maxsize=4096)(self._token_parts)
        self._part_ids = lru_cache(maxsize=4096)(self._token_ids)

    def __getstate__(self) -> dict[str, Any]:
        # Cached callables are recreated after unpickling.
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"_parts", "_part_ids"}
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_caches()

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Load a vocabulary from a text file."""
        return cls(path)

    def save(self, path: str | Path) -> None:
        """Save one vocabulary token per line."""
        Path(path).write_text("\n".join(self.tokens) + "\n")

    def _split_bracket(self, text: str) -> list[str]:
        mass, element, tail = BRACKET.fullmatch(text).groups()
        parts = ["[", *mass, element]
        i = 0
        while i < len(tail):
            pair = tail[i : i + 2]
            if pair == "@@" or (i > 0 and tail[i - 1] == "@" and pair in CHIRAL_CLASSES):
                parts.append(pair)
                i += 2
            else:
                parts.append(tail[i])
                i += 1
        return parts + ["]"]

    def _token_parts(self, text: str) -> tuple[str, ...]:
        if text[0] == "[":
            return tuple(self._split_bracket(text))
        if text[0] == "%":
            return tuple(text)
        return (text,)

    def _token_ids(self, text: str) -> tuple[int, ...]:
        return tuple(self.token_to_id[token] for token in self._parts(text))

    def tokenize(self, smiles: str) -> list[str]:
        """Return the tokens of a SMILES string."""
        return list(chain.from_iterable(map(self._parts, OUTER.findall(smiles))))

    def encode(self, smiles: str, *, add_special_tokens: bool = True) -> list[int]:
        """Return token IDs, optionally adding BOS and EOS."""
        ids = chain.from_iterable(map(self._part_ids, OUTER.findall(smiles)))
        return (
            [self.bos_token_id, *ids, self.eos_token_id]
            if add_special_tokens
            else list(ids)
        )

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Convert token IDs to SMILES, optionally omitting special tokens."""
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        return "".join(
            self.tokens[i] for i in ids if not skip_special_tokens or i not in special
        )
