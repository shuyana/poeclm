"""Utilities for processing SMILES strings."""

import datamol as dm


def process_smiles(smiles: str) -> str | None:
    """Sanitizes and standardizes a SMILES string.

    Args:
        smiles (str): A SMILES string.

    Returns:
        str | None: A standardized SMILES string or None if the input
            SMILES string is invalid.
    """
    if smiles == "":
        return None

    with dm.without_rdkit_log():
        sane_smiles = dm.sanitize_smiles(smiles, isomeric=False)
        if sane_smiles is None:
            return None

        standard_smiles = dm.standardize_smiles(sane_smiles)
        if dm.to_mol(standard_smiles) is None:
            return None

        return standard_smiles


def randomize_smiles(smiles: str) -> str:
    """Randomizes a SMILES string by shuffling the order of the atoms.

    Args:
        smiles (str): A SMILES string.

    Returns:
        str: A randomized SMILES string.
    """
    mol = dm.to_mol(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    random_smiles = dm.to_smiles(mol, randomize=True)
    if random_smiles is None:
        raise ValueError(f"Could not randomize SMILES: {smiles}")

    return random_smiles
