"""Script to run ligand docking with DOCKSTRING."""

import argparse
from pathlib import Path
from typing import Any

import datamol as dm
import dockstring
import pandas as pd


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the input file in .smi format.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Path to the output file to write the docked molecules.",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=True,
        help="Name of the target to dock against.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=" ", names=["smiles", "id"])

    target = dockstring.load_target(args.target)

    def _dock(smiles: str) -> dict[str, Any]:
        try:
            score, aux = target.dock(smiles, num_cpus=1)
        except dockstring.DockstringError as e:
            print(f"Failed to dock {smiles}: {e}")
            return {}

        mol = aux["ligand"]
        for i in range(1, mol.GetNumConformers()):
            mol.RemoveConformer(i)
        return {"score": score, "mol": mol}

    results = dm.parallelized(
        _dock,
        df["smiles"],
        progress=True,
        tqdm_kwargs={"desc": f"Docking against {args.target}"},
    )
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame.from_records(results)], axis=1)
    df = df.dropna(subset=["score", "mol"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dm.to_sdf(df, args.out, mol_column="mol")


if __name__ == "__main__":
    main()
