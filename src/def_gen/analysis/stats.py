"""Item bootstrap for statistics that are computed from per-item arrays."""

import warnings
from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    num_bootstraps: int = 2000,
    seed: int | None = 42,
    chunk_size: int = 250,
) -> np.ndarray:
    """Compute a 95% percentile interval by resampling the rows of `data`.

    Each row of `data` holds the values of one item, so items are the
    resampling unit. `statistic` receives a stack of resamples with shape
    (num_resamples, num_items, ...) and must reduce axis 1. Resamples in
    which the statistic is NaN are left out of the interval.

    Args:
        data: Array with one row per item.
        statistic: Vectorised statistic, for example
            `lambda s: s.mean(axis=1)`.
        num_bootstraps: Number of resamples.
        seed: Seed of the random generator.
        chunk_size: Number of resamples that are drawn at once, to bound
            memory use.

    Returns:
        An array with the lower and upper bound in its first axis. The
        remaining axes follow the output of `statistic`. Bounds are NaN if
        `data` is empty or every resample gives NaN.

    Examples:
        >>> data = np.array([0.0, 1.0, 1.0, 1.0])
        >>> low, high = bootstrap_ci(data, lambda s: s.mean(axis=1))
        >>> bool(0.0 <= low <= 0.75 <= high <= 1.0)
        True
    """
    data = np.asarray(data, dtype=float)
    num_items = len(data)
    if num_items == 0 or num_bootstraps == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            shape = np.shape(statistic(data[None]))[1:]
        return np.full((2, *shape), np.nan)

    rng = np.random.default_rng(seed)
    replicates = []
    for start in range(0, num_bootstraps, chunk_size):
        size = min(chunk_size, num_bootstraps - start)
        idx = rng.integers(0, num_items, size=(size, num_items))
        replicates.append(np.asarray(statistic(data[idx]), dtype=float))
    values = np.concatenate(replicates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(values, [2.5, 97.5], axis=0)


def ratio_of_sums(sample: np.ndarray) -> np.ndarray:
    """Divide the summed first column by the summed second column per resample.

    Use it with `bootstrap_ci` on per-item counts, such as the number of
    position-1 wins and the number of non-tied pairs of each item.

    Args:
        sample: Array of shape (num_resamples, num_items, 2).

    Returns:
        Array of shape (num_resamples,), NaN where the denominator is 0.

    Examples:
        >>> ratio_of_sums(np.array([[[1, 2], [1, 2]]]))
        array([0.5])
    """
    numerator = sample[..., 0].sum(axis=1)
    denominator = sample[..., 1].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)
