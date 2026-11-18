import numpy as np
import pandas as pd
import pytest

from def_gen.analysis.bias import length_bias, self_preference


def _pairs(writer: str, judge: str, outcomes: list[str]) -> list[dict]:
    return [
        {
            "item": item,
            "judge": judge,
            "rotation": rotation,
            "system_a": writer,
            "system_b": "anw",
            "position_a": 1 + rotation,
            "outcome": outcome,
            "winner_position": None,
        }
        for item, outcome in enumerate(outcomes)
        for rotation in (0, 1)
    ]


def _definitions(system: str, lengths: list[int]) -> list[dict]:
    return [
        {"item": item, "system": system, "definition": " ".join(["w"] * n)}
        for item, n in enumerate(lengths)
    ]


def test_self_preference_table() -> None:
    pairs = pd.DataFrame(
        _pairs("gemma4-26b-a4b", "gemma-3-27b", ["a", "a", "b", "tie"])
        + _pairs("qwen36-35b-a3b", "gemma-3-27b", ["b", "b", "b", "a"])
    )
    scores = pd.DataFrame(
        [
            {
                "item": item,
                "system": system,
                "judge": "gemma-3-27b",
                "criterion": criterion,
                "score": score,
            }
            for criterion in ("betekenis", "gebruik")
            for system, score in [
                ("anw", 3),
                ("gemma4-26b-a4b", 4),
                ("qwen36-35b-a3b", 2),
            ]
            for item in range(4)
        ]
    )

    out = self_preference(pairs, scores).set_index("writer")

    assert out.loc["gemma4-26b-a4b", "same_family"]
    assert not out.loc["qwen36-35b-a3b", "same_family"]
    assert out.loc["gemma4-26b-a4b", "win_rate"] == pytest.approx(0.5)
    assert out.loc["qwen36-35b-a3b", "win_rate"] == pytest.approx(0.25)
    assert out.loc["gemma4-26b-a4b", "mean_score_diff"] == pytest.approx(1)
    assert out.loc["qwen36-35b-a3b", "mean_score_diff"] == pytest.approx(-1)
    assert (out["n_items"] == 4).all()


def test_length_bias_detects_longer_winners() -> None:
    rng = np.random.default_rng(0)
    length_diff = rng.integers(-6, 7, 300)
    win = rng.random(300) < 1 / (1 + np.exp(-0.4 * length_diff))
    outcomes = np.where(win, "a", "b").tolist()
    outcomes[:5] = ["tie"] * 5
    pairs = pd.DataFrame(_pairs("w", "j", outcomes))
    definitions = pd.DataFrame(
        _definitions("anw", [10] * 300)
        + _definitions("w", (10 + length_diff).tolist())
    )

    out = length_bias(pairs, definitions).iloc[0]

    assert (out["n_items"], out["n_ties"]) == (295, 5)
    assert out["coef_ci_low"] < 0.4 < out["coef_ci_high"]
    assert out["p_value"] < 0.001
    assert out["odds_ratio_per_word"] == pytest.approx(
        np.exp(out["coef_length"])
    )


def test_length_bias_with_two_writers_and_separation() -> None:
    pairs = pd.DataFrame(
        _pairs("w", "j", ["a", "b", "a", "b"])
        + _pairs("v", "j", ["a", "a", "b", "b"])
        + _pairs("w", "sep", ["a", "a", "b", "b"])
    )
    definitions = pd.DataFrame(
        _definitions("anw", [5, 5, 5, 5])
        + _definitions("w", [9, 8, 2, 1])
        + _definitions("v", [6, 4, 7, 3])
    )

    out = length_bias(pairs, definitions).set_index("judge")

    assert out.loc["j", "n_items"] == 8
    assert np.isfinite(out.loc["j", "coef_length"])
    assert np.isnan(out.loc["sep", "coef_length"])


def test_length_bias_requires_definitions() -> None:
    pairs = pd.DataFrame(_pairs("w", "j", ["a"]))
    definitions = pd.DataFrame(_definitions("anw", [3]))

    with pytest.raises(ValueError, match="lack a definition"):
        length_bias(pairs, definitions)
