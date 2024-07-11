"""Utility functions."""


def round_up(n: int, k: int) -> int:
    """Rounds up an integer to the nearest multiple of k.

    Args:
        n (int): An integer to be rounded up.
        k (int): An integer multiple to round up to.

    Returns:
        The smallest multiple of k that is greater than or equal to n.
    """
    return k * ((n + k - 1) // k)
