"""Fast top-k similarity search using CFFI."""

from typing import Literal

from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from ._popc import ffi, lib  # pyright: ignore[reportMissingImports]


class TopKTanimotoSearch:
    """Fast top-k similarity search using CFFI."""

    def __init__(self, fp_bits: Literal[512, 1024, 2048]) -> None:
        """Initializer."""
        self.fp_bits = fp_bits

        self._target_fps_bytes = b""
        self._num_targets = 0

        self._topk_tanimoto_search = {
            512: lib.topk_tanimoto_search_512,
            1024: lib.topk_tanimoto_search_1024,
            2048: lib.topk_tanimoto_search_2048,
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
        self._target_fps_bytes += target_fp_bytes
        self._num_targets += 1
        return target_index

    def add_targets(self, target_fps: list[ExplicitBitVect]) -> list[int]:
        """Adds multiple target fingerprints.

        Use this method to add many targets at once for better
        performance.

        Args:
            target_fps (list[ExplicitBitVect]): A list of target
                fingerprints.

        Returns:
            list[int]: A list of indices of the added targets.
        """
        target_indices = [len(self) + i for i in range(len(target_fps))]
        target_fps_bytes = b"".join(DataStructs.BitVectToBinaryText(fp) for fp in target_fps)
        self._target_fps_bytes += target_fps_bytes
        self._num_targets += len(target_fps)
        return target_indices

    def search(self, query_fp: ExplicitBitVect, k: int) -> list[tuple[int, float]]:
        """Finds the top-k targets with the highest Tanimoto similarity.

        Args:
            query_fp: A query fingerprint.
            k: The number of targets to return.

        Returns:
            A list of tuples containing the top-k target indices and
            their Tanimoto similarity scores.
        """
        if k < 1 or k > len(self):
            raise ValueError(f"`k` must be between 1 and the number of targets: {k}")

        query_fp_bytes = DataStructs.BitVectToBinaryText(query_fp)

        hit_indices = ffi.new("int[]", k)
        hit_scores = ffi.new("double[]", k)

        self._topk_tanimoto_search(
            query_fp_bytes,
            k,
            len(self),
            self._target_fps_bytes,
            hit_indices,
            hit_scores,
        )

        return [(hit_indices[i], hit_scores[i]) for i in range(k)]
