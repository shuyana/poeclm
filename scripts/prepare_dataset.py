"""Script to prepare a dataset from a list of SMILES strings."""

import argparse
from fractions import Fraction
from pathlib import Path

import datamol as dm
import pandas as pd

from poeclm.chem import compute_descriptors, process_smiles
from poeclm.tokenizer import SMILESTokenizer


def random_split(
    df: pd.DataFrame,
    frac: tuple[Fraction, Fraction, Fraction],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly splits a dataset into train/validation/test sets.

    Args:
        df (DataFrame): A dataframe.
        frac (tuple[Fraction, Fraction, Fraction]): A tuple of fractions
            for the train, validation, and test sets.
        seed (int, optional): A random seed. Defaults to 42.

    Returns:
        tuple[DataFrame, DataFrame, DataFrame]: A tuple of train,
            validation, and test datasets.
    """
    if sum(frac) != 1.0:
        raise ValueError("The sum of `frac` must be 1.0.")

    df = df.sample(frac=1.0, random_state=seed, ignore_index=True)  # Shuffle the dataset

    train_size, val_size = int(frac[0] * len(df)), int(frac[1] * len(df))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test_df = df.iloc[train_size + val_size :].reset_index(drop=True)
    return train_df, val_df, test_df


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the input file in .csv or .smi format.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Path to the output directory to write the train, validation, and test datasets.",
    )
    parser.add_argument(
        "-f",
        "--frac",
        type=Fraction,
        nargs=3,
        required=True,
        help="Fraction of the dataset to use for train, validation, and test sets.",
    )
    parser.add_argument(
        "--vocab_file",
        type=Path,
        default=None,
        help="Path to the vocabulary file to filter the SMILES strings.",
    )
    parser.add_argument(
        "--filter_expr",
        type=str,
        default=None,
        help="Filter expression to apply to the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    if ".smi" in args.input.suffixes:
        df = pd.read_csv(args.input, sep=" ", names=["smiles", "id"])
    else:
        df = pd.read_csv(args.input)
    print(f"Input size: {len(df):,}")

    df["standard_smiles"] = dm.parallelized(
        process_smiles,
        df["smiles"],
        progress=True,
        tqdm_kwargs={"desc": "Processing SMILES"},
    )
    df = df.dropna(subset="standard_smiles")
    df = df.drop_duplicates(subset="standard_smiles", ignore_index=True)

    descriptors = dm.parallelized(
        compute_descriptors,
        df["standard_smiles"],
        progress=True,
        tqdm_kwargs={"desc": "Computing descriptors"},
    )
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame.from_records(descriptors)], axis=1)

    if args.vocab_file is not None:
        with args.vocab_file.open() as f:
            tok_set = {line.strip() for line in f}
        vocab_filter = lambda s: all(tok in tok_set for tok in SMILESTokenizer.tokenize(s))
        df = df[df["standard_smiles"].apply(vocab_filter)].reset_index(drop=True)

    if args.filter_expr is not None:
        df = df.query(args.filter_expr).reset_index(drop=True)

    train_df, val_df, test_df = random_split(df, args.frac, seed=args.seed)
    print(
        f"Output size: {len(df):,}",
        f"(train: {len(train_df):,}, val: {len(val_df):,}, test: {len(test_df):,})",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(
        args.out / "train.parquet",
        engine="pyarrow",
        index=False,
        row_group_size=2**20,
    )
    val_df.to_parquet(
        args.out / "val.parquet",
        engine="pyarrow",
        index=False,
        row_group_size=2**20,
    )
    test_df.to_parquet(
        args.out / "test.parquet",
        engine="pyarrow",
        index=False,
        row_group_size=2**20,
    )


if __name__ == "__main__":
    main()
