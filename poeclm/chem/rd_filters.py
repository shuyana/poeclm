"""Wrapper for Patrick Walters' rd_filters package."""

import functools
import importlib.resources

import pandas as pd
from rdkit import Chem


@functools.cache
def _get_rule_df(rule_set_name: str) -> pd.DataFrame:
    traversable = importlib.resources.files("rd_filters") / "data/alert_collection.csv"
    with importlib.resources.as_file(traversable) as alert_file_name:
        rule_df = pd.read_csv(alert_file_name).dropna()
    rule_df = rule_df[rule_df["rule_set_name"] == rule_set_name]
    rule_df["smarts_mol"] = rule_df["smarts"].apply(Chem.MolFromSmarts)
    return rule_df


def evaluate_rd_filters(mol: Chem.Mol, rule_set_name: str) -> bool:
    """Evaluates a molecule against a rule set.

    Args:
        mol (Mol): A molecule.
        rule_set_name (str): The name of the rule set to evaluate
            against.

    Returns:
        bool: True if the molecule passes the rule set, False otherwise.
    """
    rule_df = _get_rule_df(rule_set_name)
    return all(
        len(mol.GetSubstructMatches(patt)) <= max_val
        for patt, max_val in rule_df[["smarts_mol", "max"]].itertuples(index=False, name=None)
    )
