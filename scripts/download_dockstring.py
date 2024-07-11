"""Script to download the DOCKSTRING dataset."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://figshare.com/ndownloader/files/35948138",
        help="URL to the DOCKSTRING dataset.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.url, sep="\t")
    df = df.rename(columns={"inchikey": "id"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
