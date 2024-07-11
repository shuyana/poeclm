"""Script to enumerate molecules from building blocks and reactions."""

import argparse
import functools
import random
from collections.abc import Generator
from fractions import Fraction
from pathlib import Path

import datamol as dm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm

from poeclm.chem import ThresholdTanimotoSearch, evaluate_rd_filters, process_smiles


def enumerate_random_molecules(
    rxn_list: list[rdChemReactions.ChemicalReaction],
    reagents_list: list[tuple[list[Chem.Mol], ...]],
    seed: int = 42,
) -> Generator[str, None, None]:
    """Randomly enumerates molecules from a set of BBs and RXNs.

    Molecules are enumerated by randomly choosing a reaction and a
    set of reagents.

    Args:
        rxn_list (list[ChemicalReaction]): A list of chemical reactions.
        reagents_list (list[tuple[list[Mol], ...]]): A list of reagents
            for each reaction.
        seed (int, optional): A random seed. Defaults to 42.

    Yields:
        Generator[str, None, None]: A generator of SMILES strings.
    """
    rng = random.Random(seed)
    library_list = [
        rdChemReactions.EnumerateLibrary(
            rxn,
            [rng.sample(match_list, len(match_list)) for match_list in reagents],
            rdChemReactions.RandomSampleStrategy(),
        )
        for rxn, reagents in zip(rxn_list, reagents_list, strict=True)
    ]
    while True:
        library = rng.choice(library_list)
        product_set = {process_smiles(products[0]) for products in library.nextSmiles()}
        yield rng.choice(sorted(smiles for smiles in product_set if smiles is not None))


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
        help="Path to the output file to write the enumerated molecules.",
    )
    parser.add_argument(
        "-n",
        "--num_mols",
        type=int,
        required=True,
        help="Number of molecules to enumerate.",
    )
    parser.add_argument(
        "--prop_filters",
        type=str,
        nargs="+",
        default=[],
        help="Property filters to apply to the enumerated molecules.",
    )
    parser.add_argument(
        "--rd_filters",
        type=str,
        nargs="+",
        default=[],
        help="Structural alerts to apply to the enumerated molecules.",
    )
    parser.add_argument(
        "--threshold",
        type=Fraction,
        default=None,
        help="Tanimoto similarity threshold to apply to the enumerated molecules.",
    )
    parser.add_argument(
        "--fp_type",
        choices=["ecfp", "fcfp"],
        default="ecfp",
        help="Type of fingerprint to use for similarity thresholding.",
    )
    parser.add_argument(
        "--fp_radius",
        type=int,
        default=2,
        help="Radius of the fingerprint to use for similarity thresholding.",
    )
    parser.add_argument(
        "--fp_bits",
        choices=[512, 1024, 2048],
        default=512,
        help="Number of bits of the fingerprint to use for similarity thresholding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    # Prepare filters
    filters = []
    for prop_filter in args.prop_filters:
        if prop_filter == "Ro5":
            filters.append(
                lambda mol: dm.descriptors.mw(mol) <= 500.0
                and dm.descriptors.clogp(mol) <= 5.0
                and dm.descriptors.n_lipinski_hba(mol) <= 10
                and dm.descriptors.n_lipinski_hbd(mol) <= 5
            )
        elif prop_filter == "Veber":
            filters.append(
                lambda mol: dm.descriptors.n_rotatable_bonds(mol) <= 10
                and dm.descriptors.tpsa(mol) <= 140.0
            )
        else:
            raise ValueError(f"Unrecognized prop_filter: {prop_filter}")
    for rd_filter in args.rd_filters:
        if rd_filter in {
            "BMS",
            "Dundee",
            "Glaxo",
            "Inpharmatica",
            "LINT",
            "MLSMR",
            "PAINS",
            "SureChEMBL",
        }:
            filters.append(functools.partial(evaluate_rd_filters, rule_set_name=rd_filter))
        else:
            raise ValueError(f"Unrecognized rd_filter: {rd_filter}")

    threshold_search = None
    if args.threshold is not None:
        threshold_search = ThresholdTanimotoSearch(args.fp_bits)

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
    reagents_list = []
    for n, rxn in enumerate(rxn_list, start=1):
        reagents = []
        for k, rct in enumerate(rxn.GetReactants(), start=1):
            match_list = [mol for mol in bb_list if mol.HasSubstructMatch(rct)]
            if len(match_list) == 0:
                raise ValueError(f"rxn{n} has no matching building blocks for reactant {k}.")
            reagents.append(match_list)
        reagents_list.append(tuple(reagents))

    # Enumerate a library of molecules
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w") as f, tqdm(total=args.num_mols) as pbar:
        for smiles in enumerate_random_molecules(rxn_list, reagents_list, seed=args.seed):
            mol = dm.to_mol(smiles)
            if mol is None or not all(filter_fn(mol) for filter_fn in filters):
                continue

            if threshold_search is not None:
                fp = dm.to_fp(
                    mol,
                    as_array=False,
                    fp_type=args.fp_type,
                    radius=args.fp_radius,
                    nBits=args.fp_bits,
                )
                if threshold_search.hit(fp, args.threshold):
                    continue
                threshold_search.add_target(fp)

            count += 1
            f.write(f"{smiles} enum{count}\n")
            pbar.update(1)

            if count >= args.num_mols:
                break


if __name__ == "__main__":
    main()
