"""Tokenizer for SMILES strings."""

import itertools
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Self


class SMILESTokenizer:
    """Tokenizer for SMILES strings."""

    _REGEXP = re.compile(r"(\[[^\]]+]|%\d{2}|Br|Cl|.)")

    def __init__(self, tok_iterable: Iterable[str]):
        """Initializer.

        Args:
            tok_iterable (Iterable[str]): An iterable of tokens in the
                vocabulary.
        """
        self.vocab = list(tok_iterable)

        self._tok_to_idx = {tok: idx for idx, tok in enumerate(tok_iterable)}
        self.pad_idx = self._tok_to_idx["<pad>"]
        self.bos_idx = self._tok_to_idx["<bos>"]
        self.eos_idx = self._tok_to_idx["<eos>"]

    def __len__(self) -> int:
        """Returns the total number of tokens in the vocabulary.

        Returns:
            int: The total number of tokens in the vocabulary.
        """
        return len(self.vocab)

    @classmethod
    def from_vocab_file(cls, vocab_file: str | Path) -> Self:
        """Creates a tokenizer from a vocabulary file.

        Args:
            vocab_file (str | Path): A path to a vocabulary file.

        Returns:
            SMILESTokenizer: A tokenizer.
        """
        with Path(vocab_file).open() as f:
            return cls([line.strip() for line in f])

    @classmethod
    def tokenize(cls, smiles: str) -> list[str]:
        """Tokenizes a SMILES string.

        Args:
            smiles (str): A SMILES string.

        Returns:
            list[str]: A list of tokens.
        """
        return cls._REGEXP.findall(smiles)

    def encode(self, smiles: str, with_begin_and_end: bool = True) -> list[int]:
        """Encodes a SMILES string.

        Args:
            smiles (str): A SMILES string.
            with_begin_and_end (bool, optional): Whether to include the
                '<bos>' and '<eos>' tokens. Defaults to True.

        Returns:
            list[int]: A list of token indices.
        """
        idx_list = [self._tok_to_idx[tok] for tok in self.tokenize(smiles)]
        if with_begin_and_end:
            idx_list = [self.bos_idx] + idx_list + [self.eos_idx]
        return idx_list

    def decode(self, idx_iterable: Iterable[int]) -> str:
        """Decodes an iterable of token indices.

        Args:
            idx_iterable (Iterable[int]): An iterable of token indices.

        Returns:
            str: A SMILES string.
        """
        idx_iter = iter(idx_iterable)
        idx_iter = itertools.dropwhile(lambda idx: idx == self.bos_idx, idx_iter)
        idx_iter = itertools.takewhile(lambda idx: idx != self.eos_idx, idx_iter)
        return "".join(self.vocab[idx] for idx in idx_iter)
