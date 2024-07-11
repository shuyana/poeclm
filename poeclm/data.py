"""Data module for SMILES strings."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .chem import randomize_smiles
from .tokenizer import SMILESTokenizer


class SMILESDataset(Dataset):
    """Dataset for SMILES strings."""

    def __init__(
        self,
        smiles_seq: Sequence[str],
        tokenizer: SMILESTokenizer,
        randomize_smiles: bool = True,
    ):
        """Initializer.

        Args:
            smiles_seq (Sequence[str]): A sequence of SMILES strings.
            tokenizer (SMILESTokenizer): A tokenizer.
            randomize_smiles (bool, optional): Whether to randomize
                SMILES strings. Defaults to True.
        """
        self.smiles_seq = smiles_seq
        self.tokenizer = tokenizer
        self.randomize_smiles = randomize_smiles

    def __len__(self) -> int:
        """Returns the number of SMILES strings in the dataset."""
        return len(self.smiles_seq)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Returns the token indices at the given index.

        Args:
            index (int): An index.

        Returns:
            torch.Tensor: The token indices of the SMILES string at
                the given index.
        """
        smiles = self.smiles_seq[index]

        if not self.randomize_smiles:
            return torch.tensor(self.tokenizer.encode(smiles), dtype=torch.long)

        for _ in range(10):
            random_smiles = randomize_smiles(smiles)
            try:
                return torch.tensor(self.tokenizer.encode(random_smiles), dtype=torch.long)
            except KeyError:
                pass

        raise ValueError(f"Failed to randomize SMILES string: {smiles}")

    def collate(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        """Collates a list of token indices into a batched tensor.

        Args:
            sequences (list[torch.Tensor]): A list of token indices.

        Returns:
            torch.Tensor: A batched tensor of token indices.
        """
        return pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_idx)


class SMILESDataModule:
    """Data module for a dataset of SMILES strings."""

    def __init__(
        self,
        root: str | Path,
        tokenizer: SMILESTokenizer,
        filter_expr: str | None = None,
        randomize_smiles: bool = True,
    ):
        """Initializer.

        Args:
            root (str | Path): A root directory containing the parquet
                files for each split.
            tokenizer (SMILESTokenizer): A tokenizer.
            filter_expr (str | None, optional): A filter expression for
                the dataset. Defaults to None.
            randomize_smiles (bool, optional): Whether to randomize
                SMILES strings. Defaults to True.
        """
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.tokenizer = tokenizer
        self.filter_expr = filter_expr
        self.randomize_smiles = randomize_smiles

        self.data = {}
        for split in ("train", "val", "test"):
            df = pd.read_parquet(
                root / f"{split}.parquet",
                engine="pyarrow",
                dtype_backend="pyarrow",
            )
            if filter_expr is not None:
                df = df.query(filter_expr)
            df = df.drop(columns=df.columns.difference(["standard_smiles"]))
            self.data[split] = df["standard_smiles"].reset_index(drop=True)

    def get_dataset(self, split: Literal["train", "val", "test"]) -> SMILESDataset:
        """Gets a dataset for a split.

        Args:
            split (Literal['train', 'val', 'test']): A split.

        Returns:
            SMILESDataset: A dataset.
        """
        return SMILESDataset(
            self.data[split],
            self.tokenizer,
            randomize_smiles=self.randomize_smiles,
        )

    def get_dataloader(
        self,
        split: Literal["train", "val", "test"],
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Gets a data loader for a split.

        Args:
            split (Literal['train', 'val', 'test']): A split.
            batch_size (int): A batch size.
            shuffle (bool): Whether to shuffle the dataset. Defaults
                to True.
            num_workers (int): The number of workers for data loading.
                Defaults to 0, which means that the data will be loaded
                in the main process.
            pin_memory (bool): Whether to pin memory. Defaults to True.

        Returns:
            DataLoader: A data loader.
        """
        dataset = self.get_dataset(split)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate,
            pin_memory=pin_memory,
        )
