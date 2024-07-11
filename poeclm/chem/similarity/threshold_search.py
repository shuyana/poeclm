"""Fast threshold similarity search using CFFI.

Based on the code from Andrew Dalke's blog post:
    http://www.dalkescientific.com/writings/diary/archive/2020/10/07/intersection_popcount.html
"""

import math
from fractions import Fraction
from typing import Literal

from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from ._popc import ffi, lib  # pyright: ignore[reportMissingImports]


class ThresholdTanimotoSearch:
    """Fast threshold similarity search using CFFI."""

    def __init__(self, fp_bits: Literal[512, 1024, 2048]) -> None:
        """Initializer."""
        self.fp_bits = fp_bits

        self._target_popcount_bins: list[tuple[list[int], bytes]] = [([], b"")] * (fp_bits + 1)
        self._num_targets = 0

        self._byte_popcount = {
            512: lib.byte_popcount_512,
            1024: lib.byte_popcount_1024,
            2048: lib.byte_popcount_2048,
        }[fp_bits]
        self._threshold_bin_tanimoto_search = {
            512: lib.threshold_bin_tanimoto_search_512,
            1024: lib.threshold_bin_tanimoto_search_1024,
            2048: lib.threshold_bin_tanimoto_search_2048,
        }[fp_bits]
        self._threshold_bin_tanimoto_hit = {
            512: lib.threshold_bin_tanimoto_hit_512,
            1024: lib.threshold_bin_tanimoto_hit_1024,
            2048: lib.threshold_bin_tanimoto_hit_2048,
        }[fp_bits]

    def __len__(self) -> int:
        """Returns the number of targets."""
        return self._num_targets

    def add_target(self, target_fp: ExplicitBitVect) -> int:
        """Adds a target fingerprint.

        Args:
            target_fp (ExplicitBitVect): A target fingerprint.

        Returns:
            int: The index of the added target.
        """
        target_index = len(self)
        target_fp_bytes = DataStructs.BitVectToBinaryText(target_fp)
        popcount = self._byte_popcount(target_fp_bytes)
        target_indices, target_fps_bytes = self._target_popcount_bins[popcount]
        self._target_popcount_bins[popcount] = (
            target_indices + [target_index],
            target_fps_bytes + target_fp_bytes,
        )
        self._num_targets += 1
        return target_index

    def search(
        self,
        query_fp: ExplicitBitVect,
        threshold: float | Fraction,
    ) -> list[tuple[int, float]]:
        """Finds all targets with a similarity >= threshold.

        Args:
            query_fp (ExplicitBitVect): A query fingerprint.
            threshold (float | Fraction): A threshold value for Tanimoto
                similarity. Must not be negative.

        Returns:
            list[tuple[int, float]]: A list of tuples containing the hit
                target indices and their Tanimoto similarity scores.
        """
        if threshold < 0.0:
            raise ValueError(f"`threshold` must not be negative: {threshold}")

        if len(self) == 0:
            return []

        query_fp_bytes = DataStructs.BitVectToBinaryText(query_fp)
        query_popcount = self._byte_popcount(query_fp_bytes)
        min_popcount = math.ceil(query_popcount * threshold)
        max_popcount = (
            min(self.fp_bits, math.floor(query_popcount / threshold))
            if threshold > 0.0
            else self.fp_bits
        )
        threshold = float(threshold)

        max_bin_size = max(len(target_indices) for target_indices, _ in self._target_popcount_bins)
        hit_indices = ffi.new("int[]", max_bin_size)
        hit_scores = ffi.new("double[]", max_bin_size)

        hits = []
        for popcount in range(min_popcount, max_popcount + 1):
            target_indices, target_fps_bytes = self._target_popcount_bins[popcount]
            num_hits = self._threshold_bin_tanimoto_search(
                query_fp_bytes,
                query_popcount,
                threshold,
                len(target_indices),
                target_fps_bytes,
                popcount,
                hit_indices,
                hit_scores,
            )
            for i in range(num_hits):
                hits.append((target_indices[hit_indices[i]], hit_scores[i]))
        return hits

    def hit(self, query_fp: ExplicitBitVect, threshold: float | Fraction) -> bool:
        """Checks if a target with a similarity >= threshold exists.

        Args:
            query_fp (ExplicitBitVect): A query fingerprint.
            threshold (float | Fraction): A threshold value for Tanimoto
                similarity. Must not be negative.

        Returns:
            bool: True if a target with a Tanimoto similarity greater
                than or equal to `threshold` is found, False otherwise.
        """
        if threshold < 0.0:
            raise ValueError(f"`threshold` must not be negative: {threshold}")

        if len(self) == 0:
            return False

        query_fp_bytes = DataStructs.BitVectToBinaryText(query_fp)
        query_popcount = self._byte_popcount(query_fp_bytes)
        min_popcount = math.ceil(query_popcount * threshold)
        max_popcount = (
            min(self.fp_bits, math.floor(query_popcount / threshold))
            if threshold > 0.0
            else self.fp_bits
        )
        threshold = float(threshold)

        for popcount in range(min_popcount, max_popcount + 1):
            target_indices, target_fps_bytes = self._target_popcount_bins[popcount]
            if self._threshold_bin_tanimoto_hit(
                query_fp_bytes,
                query_popcount,
                threshold,
                len(target_indices),
                target_fps_bytes,
                popcount,
            ):
                return True
        return False
