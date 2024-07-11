"""Script to generate config files for SpaceLight."""

import argparse
import json
from pathlib import Path

import datamol as dm
import numpy as np
import pandas as pd

from poeclm.chem import process_smiles


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bb",
        type=Path,
        required=True,
        help="Path to the input file containing building blocks in .csv or .smi format.",
    )
    parser.add_argument(
        "-r",
        "--rxn",
        type=Path,
        required=True,
        help="Path to the input file containing reactions in .csv format.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Path to the output directory to write the config files.",
    )
    args = parser.parse_args()

    # Prepare building blocks
    if ".smi" in args.bb.suffixes:
        bb_df = pd.read_csv(args.bb, sep=" ", names=["smiles", "id"])
    else:
        bb_df = pd.read_csv(args.bb)
    bb_df["standard_smiles"] = dm.parallelized(process_smiles, bb_df["smiles"])
    bb_df = bb_df.dropna(subset="standard_smiles")
    bb_df = bb_df.drop_duplicates(subset="standard_smiles", ignore_index=True)
    bb_list = dm.parallelized(dm.to_mol, bb_df["standard_smiles"])
    print(f"Number of building blocks: {len(bb_list):,}")

    # Prepare reactions
    rxn_df = pd.read_csv(args.rxn)
    rxn_list = []
    with dm.without_rdkit_log():
        for smarts in rxn_df["smarts"]:
            rxn = dm.reactions.rxn_from_smarts(smarts)
            if not dm.reactions.is_reaction_ok(rxn):
                raise ValueError(f"Invalid SMARTS: {smarts}")
            if rxn.GetNumProductTemplates() != 1:
                raise ValueError(f"Reaction has more than 1 product template: {smarts}")
            rxn_list.append(rxn)
    print(f"Number of reactions: {len(rxn_list):,}")

    # Prepare reagents
    total = 0
    reagents_list = []
    for n, rxn in enumerate(rxn_list, start=1):
        reagents = []
        for k, rct in enumerate(rxn.GetReactants(), start=1):
            match_list = [mol for mol in bb_list if mol.HasSubstructMatch(rct)]
            if len(match_list) == 0:
                raise ValueError(f"rxn{n} has no matching building blocks for reactant {k}.")
            reagents.append(match_list)
        total += int(np.prod([len(r) for r in reagents]))
        reagents_list.append(tuple(reagents))
    print(f"Total number of combinations: {total:,}")

    # Write config files
    args.out.mkdir(parents=True, exist_ok=True)
    topologies = []
    for n, (smarts, reagents) in enumerate(  # type: ignore
        zip(rxn_df["smarts"], reagents_list, strict=True),
        start=1,
    ):
        topologies.append(
            {
                "name": f"rxn{n}",
                "reactions": [
                    {
                        "components": list(range(1, len(reagents) + 1)),
                        "reaction": smarts,
                    }
                ],
                "reagentGroups": [
                    {
                        "groupId": i,
                        "reagents": f"rxn{n}block{i}.smi",
                    }
                    for i in range(1, len(reagents) + 1)
                ],
            }
        )
        for i, match_list in enumerate(reagents, start=1):
            path = args.out / f"rxn{n}block{i}.smi"
            with path.open("w") as f:
                for j, mol in enumerate(match_list, start=1):
                    f.write(f"{dm.to_smiles(mol)} block{i}frag{j}\n")
    config_path = args.out / "config.json"
    with config_path.open("w") as f:
        json.dump({"topologies": topologies}, f, indent=2)


if __name__ == "__main__":
    main()
