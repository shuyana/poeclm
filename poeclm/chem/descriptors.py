"""Functions for computing molecular descriptors."""

from typing import Any

import datamol as dm

from .rd_filters import evaluate_rd_filters


def compute_descriptors(smiles: str) -> dict[str, Any]:
    """Computes molecular descriptors for a SMILES string.

    Args:
        smiles (str): A SMILES string.

    Returns:
        dict[str, Any]: A dictionary of molecular descriptors.
    """
    mol = dm.to_mol(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    descriptors = dm.descriptors.compute_many_descriptors(mol)
    descriptors["Ro5"] = (
        descriptors["mw"] <= 500.0
        and descriptors["clogp"] <= 5.0
        and descriptors["n_lipinski_hba"] <= 10
        and descriptors["n_lipinski_hbd"] <= 5
    )
    descriptors["Veber"] = descriptors["n_rotatable_bonds"] <= 10 and descriptors["tpsa"] <= 140.0
    for name in ("BMS", "Dundee", "Glaxo", "Inpharmatica", "LINT", "MLSMR", "PAINS", "SureChEMBL"):
        descriptors[name] = evaluate_rd_filters(mol, name)
    return descriptors
