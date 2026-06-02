from collections import defaultdict
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from tqdm import trange


def bootstrap_agreement(
    agreement_fn: Callable,
    df: pd.DataFrame,
    num_bootstraps: int = 1000,
    seed: int | None = 42,
    ci_keys: Iterable | None = None,
) -> dict:
    """Perform bootstrap resampling to compute confidence intervals for agreement statistics on a given agreement function.

    Args:
        df (pd.DataFrame): Dataframe containing rater columns.
        num_bootstraps (int): Number of bootstrap samples.
        seed (int | None): Random seed for reproducibility.
        ci_keys (Iterable | None): Keys to compute confidence intervals for. If None, all keys from agreement_fn result are used.
    Returns:
        dict: Dictionary with 95% confidence intervals for agreement statistics.
    """
    if num_bootstraps < 0:
        raise ValueError("num_bootstraps must be non-negative.")

    # First get the real agreement values to know which keys to track
    initial_res = agreement_fn(df)
    if ci_keys is None:
        ci_keys = list(initial_res.keys())

    rng = np.random.default_rng(seed)
    boot_results = defaultdict(list)

    for _ in trange(num_bootstraps, desc="Bootstrap sampling", leave=False):
        resampled = df.sample(frac=1, replace=True, random_state=rng)
        res = agreement_fn(resampled)
        for key in ci_keys:
            if key not in res:
                raise ValueError(
                    f"Key {key} not found in agreement function result."
                )
            if np.isnan(res[key]):
                raise ValueError(f"{key} is NaN in bootstrap sample.")
            boot_results[key].append(res[key])

    def ci(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return (float(lo), float(hi))

    agreements = {
        k: {
            "value": initial_res[k],
            "ci95": ci(v),
        }
        for k, v in boot_results.items()
    }

    agreements = {
        **agreements,
        "num_bootstraps": num_bootstraps,
        "num_items": len(df),
        "bootstrap_seed": seed,
    }

    return agreements
