"""Surface overlap between writer definitions and reference definitions."""

from collections import Counter
from difflib import SequenceMatcher

import pandas as pd


def normalize_whitespace(text: str) -> str:
    r"""Collapse whitespace runs to single spaces and strip the ends.

    Args:
        text: Input text.

    Returns:
        The normalized text.

    Examples:
        >>> normalize_whitespace("  a \n b\t c ")
        'a b c'
    """
    return " ".join(text.split())


def _char_ngrams(text: str, n: int) -> Counter[str]:
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf(
    hypothesis: str,
    reference: str,
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Compute the character n-gram F-score (chrF) between two strings.

    Whitespace is removed before n-grams are counted. Precision and recall
    are averaged over the orders for which both strings have n-grams. The
    score lies in [0, 1].

    Args:
        hypothesis: The text to score.
        reference: The text to compare against.
        max_order: Highest character n-gram order.
        beta: Weight of recall relative to precision.

    Returns:
        The chrF score. Two empty strings score 1.0. If no n-gram order
        exists for both strings and they differ, the score is 0.0.

    Examples:
        >>> chrf("een hond", "een hond")
        1.0
        >>> chrf("abc", "xyz")
        0.0
    """
    hyp = "".join(hypothesis.split())
    ref = "".join(reference.split())

    precisions: list[float] = []
    recalls: list[float] = []
    for n in range(1, max_order + 1):
        hyp_ngrams = _char_ngrams(hyp, n)
        ref_ngrams = _char_ngrams(ref, n)
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        if hyp_total == 0 or ref_total == 0:
            continue
        matches = sum((hyp_ngrams & ref_ngrams).values())
        precisions.append(matches / hyp_total)
        recalls.append(matches / ref_total)

    if not precisions:
        return 1.0 if hyp == ref else 0.0

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    beta_sq = beta**2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def longest_common_substring_ratio(a: str, b: str) -> float:
    """Length of the longest common substring over the shorter string length.

    Both strings are whitespace-normalized first.

    Args:
        a: First text.
        b: Second text.

    Returns:
        A value in [0, 1], or 0.0 if either string is empty.

    Examples:
        >>> longest_common_substring_ratio("een grote hond", "grote")
        1.0
        >>> longest_common_substring_ratio("abcd", "xbcy")
        0.5
    """
    a = normalize_whitespace(a)
    b = normalize_whitespace(b)
    if not a or not b:
        return 0.0
    match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    )
    return match.size / min(len(a), len(b))


def overlap_table(
    definitions: pd.DataFrame, reference_system: str = "anw"
) -> pd.DataFrame:
    """Compare each system's definitions with the reference definitions.

    Items without a definition in both the system and the reference are not
    used. The number of such items per system is stored in
    `attrs["num_unpaired"]`.

    Args:
        definitions: Long table with columns `item`, `system` and
            `definition`.
        reference_system: Name of the system that holds the references.

    Returns:
        One row per (item, system) with the columns `item`, `system`,
        `exact_match`, `chrf`, `lcs_ratio`, `words_system` and
        `words_reference`.

    Raises:
        ValueError: If an (item, system) pair has more than one definition.
    """
    if definitions.duplicated(["item", "system"]).any():
        raise ValueError("Found multiple definitions for an (item, system).")

    is_reference = definitions["system"] == reference_system
    reference = definitions.loc[is_reference, ["item", "definition"]]
    others = definitions.loc[~is_reference]

    paired = others.merge(
        reference, on="item", how="inner", suffixes=("_system", "_reference")
    )
    system_defs = paired["definition_system"]
    reference_defs = paired["definition_reference"]
    pairs = list(zip(system_defs, reference_defs, strict=True))

    table = pd.DataFrame(
        {
            "item": paired["item"],
            "system": paired["system"],
            "exact_match": [
                normalize_whitespace(s) == normalize_whitespace(r)
                for s, r in pairs
            ],
            "chrf": [chrf(s, r) for s, r in pairs],
            "lcs_ratio": [
                longest_common_substring_ratio(s, r) for s, r in pairs
            ],
            "words_system": [len(s.split()) for s, _ in pairs],
            "words_reference": [len(r.split()) for _, r in pairs],
        }
    )

    unpaired = others.loc[~others["item"].isin(reference["item"])]
    table.attrs["num_unpaired"] = {
        system: int(unpaired["system"].eq(system).sum())
        for system in others["system"].unique()
    }
    return table


def summarize_overlap(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an overlap table to one row per system.

    Args:
        table: Output of `overlap_table`.

    Returns:
        One row per system with `n_items`, `exact_match_rate`, `chrf_mean`,
        `chrf_median`, `lcs_ratio_mean` and `lcs_ratio_median`.
    """
    grouped = table.groupby("system")
    return pd.DataFrame(
        {
            "n_items": grouped.size(),
            "exact_match_rate": grouped["exact_match"].mean(),
            "chrf_mean": grouped["chrf"].mean(),
            "chrf_median": grouped["chrf"].median(),
            "lcs_ratio_mean": grouped["lcs_ratio"].mean(),
            "lcs_ratio_median": grouped["lcs_ratio"].median(),
        }
    ).reset_index()
