"""Tests for the writer/reference overlap measures."""

import pandas as pd
import pytest

from def_gen.analysis.overlap import (
    chrf,
    longest_common_substring_ratio,
    normalize_whitespace,
    overlap_table,
    summarize_overlap,
)


def test_normalize_whitespace():
    assert normalize_whitespace("  a \n b\t\tc  ") == "a b c"
    assert normalize_whitespace("   ") == ""


def test_chrf_identity():
    assert chrf("een kleine hond", "een kleine hond") == 1.0


def test_chrf_hand_computed():
    # hyp "abc", ref "abd", max_order 2:
    # n=1: matches {a, b} = 2, p = 2/3, r = 2/3
    # n=2: matches {ab} = 1, p = 1/2, r = 1/2
    # P = R = (2/3 + 1/2) / 2 = 7/12, so F = 7/12 for any beta.
    assert chrf("abc", "abd", max_order=2) == pytest.approx(7 / 12)


def test_chrf_beta_weights_recall():
    # hyp "ab", ref "abc", max_order 1: p = 1, r = 2/3.
    # F = 5 * 1 * (2/3) / (4 * 1 + 2/3) = (10/3) / (14/3) = 5/7
    assert chrf("ab", "abc", max_order=1) == pytest.approx(5 / 7)


def test_chrf_ignores_whitespace():
    assert chrf("a b c", "abc") == 1.0


def test_chrf_no_effective_order():
    assert chrf("", "") == 1.0
    assert chrf("  ", "") == 1.0
    assert chrf("abc", "") == 0.0
    assert chrf("", "abc") == 0.0


def test_chrf_no_overlap():
    assert chrf("abc", "xyz") == 0.0


def test_lcs_ratio():
    assert longest_common_substring_ratio("abcd", "xbcy") == 0.5
    assert longest_common_substring_ratio("een  grote hond", "grote") == 1.0
    assert longest_common_substring_ratio("", "abc") == 0.0
    assert longest_common_substring_ratio("abc", "  ") == 0.0


def make_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item": ["a", "b", "c", "a", "b", "c", "a", "b"],
            "system": ["anw"] * 3 + ["w1"] * 3 + ["w2"] * 2,
            "definition": [
                "een hond",
                "een kat",
                "een muis",
                "een  hond",
                "een vis",
                "x",
                "een hond",
                "een kat",
            ],
        }
    )


def test_overlap_table():
    table = overlap_table(make_definitions())

    assert list(table.columns) == [
        "item",
        "system",
        "exact_match",
        "chrf",
        "lcs_ratio",
        "words_system",
        "words_reference",
    ]
    assert len(table) == 5
    w1 = table[table["system"] == "w1"].set_index("item")
    assert w1.loc["a", "exact_match"]
    assert not w1.loc["b", "exact_match"]
    assert w1.loc["a", "chrf"] == 1.0
    assert w1.loc["c", "words_system"] == 1
    assert w1.loc["c", "words_reference"] == 2
    assert table.attrs["num_unpaired"] == {"w1": 0, "w2": 0}


def test_overlap_table_counts_unpaired():
    extra = pd.DataFrame(
        {"item": ["z"], "system": ["w1"], "definition": ["iets"]}
    )
    definitions = pd.concat([make_definitions(), extra], ignore_index=True)
    table = overlap_table(definitions)

    assert len(table) == 5
    assert table.attrs["num_unpaired"] == {"w1": 1, "w2": 0}


def test_overlap_table_custom_reference():
    table = overlap_table(make_definitions(), reference_system="w2")
    assert set(table["system"]) == {"anw", "w1"}
    assert len(table) == 4


def test_overlap_table_duplicates_raise():
    definitions = pd.concat(
        [make_definitions(), make_definitions().iloc[[3]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="multiple definitions"):
        overlap_table(definitions)


def test_summarize_overlap():
    summary = summarize_overlap(overlap_table(make_definitions()))

    assert list(summary.columns) == [
        "system",
        "n_items",
        "exact_match_rate",
        "chrf_mean",
        "chrf_median",
        "lcs_ratio_mean",
        "lcs_ratio_median",
    ]
    by_system = summary.set_index("system")
    assert by_system.loc["w2", "n_items"] == 2
    assert by_system.loc["w2", "exact_match_rate"] == 1.0
    assert by_system.loc["w2", "chrf_mean"] == 1.0
    assert by_system.loc["w1", "n_items"] == 3
    assert by_system.loc["w1", "exact_match_rate"] == pytest.approx(1 / 3)
