import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from def_gen.analysis.load import AnalysisData
from def_gen.analysis.report import markdown_table, render_report, run_analysis
from def_gen.definition_generation import JUDGE_FIELDS
from tests.conftest import make_step, simulate_study, write_run


_SPEC = importlib.util.spec_from_file_location(
    "analyze_script",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze.py",
)
assert _SPEC and _SPEC.loader is not None
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)

JUDGES = [
    "Qwen/Qwen3.6-27B-FP8",
    "google/gemma-3-27b-it",
    "openai/gpt-oss-120b",
]
LABELS = ["1 beter dan 2", "2 beter dan 1", "evenwaardig"]
ITEMS = list(range(30))


def _judge_steps(start: int, column: str) -> tuple[list, dict]:
    rng = np.random.default_rng(start)
    steps, columns = [], {}
    for k, model in enumerate(JUDGES):
        name = f"judge-{k}"
        steps.append(
            make_step(
                start + k, name, "judge", model, definition_column=column
            )
        )
        for field in JUDGE_FIELDS:
            columns[f"score_{field}_{name}"] = rng.integers(1, 6, len(ITEMS))
            columns[f"analyse_{field}_{name}"] = ["ok"] * len(ITEMS)
    return steps, columns


def _ranker_steps(
    start: int, prefix: str, candidates: list[str]
) -> tuple[list, dict]:
    rng = np.random.default_rng(start + 100)
    steps, columns = [], {}
    for k, model in enumerate(JUDGES):
        for rotation in (0, 1):
            name = f"{prefix}-{k}-r{rotation}"
            steps.append(
                make_step(
                    start + len(steps),
                    name,
                    "ranker",
                    model,
                    order_rotation=rotation,
                    post_field_renamer={"ranking": f"ranking_{name}"},
                )
            )
            order = candidates if rotation == 0 else candidates[::-1]
            columns[f"volgorde_{name}"] = [order] * len(ITEMS)
            columns[f"ranking_{name}"] = rng.choice(
                LABELS, len(ITEMS), p=[0.5, 0.3, 0.2]
            ).tolist()
    return steps, columns


def _study_dirs(root: Path) -> list[Path]:
    base = {
        "betekenis_pid": ITEMS,
        "originele-definitie": [f"een woord voor {i}" for i in ITEMS],
    }
    dirs = []
    for writer in ["gemma4-26b-a4b", "qwen36-35b-a3b"]:
        texts = [f"{writer[:4]} woord {i}" for i in ITEMS]
        judges, judge_cols = _judge_steps(1, "definitie-v1")
        rankers, ranker_cols = _ranker_steps(
            4, "rank", ["originele-definitie", "definitie-v1"]
        )
        steps = [
            make_step(0, f"definer-{writer}", "definer", "x"),
            *judges,
            *rankers,
        ]
        columns = {**base, "definitie-v1": texts, **judge_cols, **ranker_cols}
        dirs.append(write_run(root / writer, steps, columns))
        base[f"definitie-{writer}"] = texts

    anw_steps, anw_cols = _judge_steps(0, "originele-definitie")
    dirs.append(
        write_run(
            root / "anw",
            anw_steps,
            {
                "betekenis_pid": ITEMS,
                "originele-definitie": base["originele-definitie"],
                **anw_cols,
            },
        )
    )
    h2h_steps, h2h_cols = _ranker_steps(
        0, "h2h", ["definitie-qwen36-35b-a3b", "definitie-gemma4-26b-a4b"]
    )
    dirs.append(
        write_run(root / "head-to-head", h2h_steps, {**base, **h2h_cols})
    )
    return dirs


def test_main_writes_tables_and_report(tmp_path, capsys) -> None:
    dirs = _study_dirs(tmp_path / "runs")
    out = tmp_path / "analysis"

    cli.main(
        run_dirs=[str(d) for d in dirs],
        output_dir=str(out),
        num_bootstraps=20,
        seed=1,
    )

    report = out.joinpath("report.md").read_text(encoding="utf-8")
    assert "## Win rates" in report
    assert "## Bradley-Terry" in report
    assert "## Self-preference" in report
    win_rates = pd.read_csv(out / "win_rates.csv")
    head_to_head = win_rates[win_rates["system_b"] == "qwen36-35b-a3b"]
    assert set(head_to_head["system_a"]) == {"gemma4-26b-a4b"}
    bt = pd.read_csv(out / "bradley_terry.csv")
    assert set(bt["system"]) == {"anw", "gemma4-26b-a4b", "qwen36-35b-a3b"}
    assert len(pd.read_csv(out / "score_differences.csv")) == 3 * 6 * 2
    assert (pd.read_csv(out / "coverage.csv")["n_dropped"] == 0).all()
    assert "dropped" in capsys.readouterr().out


def test_report_without_scores_or_anw_pairs() -> None:
    study = simulate_study(num_items=10)
    data = AnalysisData(
        scores=study.scores[study.scores["system"] == "anw"].head(0),
        pairs=study.pairs.assign(system_b="other"),
        definitions=study.definitions,
        attrition=pd.DataFrame(),
        coverage=pd.DataFrame(),
    )

    tables = run_analysis(data, num_bootstraps=10)
    report = render_report(tables, num_bootstraps=10)

    assert tables["bradley_terry"].empty
    assert tables["score_differences"].empty
    assert "## Win rates" in report
    assert "## Bradley-Terry" not in report
    assert "(no data)" in report


def test_markdown_table_formats_missing_values() -> None:
    table = markdown_table(
        pd.DataFrame({"x": [np.nan, 1.23456, 0.00012, 0.0]}), digits=2
    )

    assert table.splitlines()[2:] == [
        "|  |",
        "| 1.23 |",
        "| 1.2e-04 |",
        "| 0.00 |",
    ]


def test_report_without_pairs() -> None:
    study = simulate_study(num_items=10)
    data = AnalysisData(
        scores=study.scores,
        pairs=study.pairs.head(0),
        definitions=study.definitions,
        attrition=pd.DataFrame(),
        coverage=pd.DataFrame(),
    )

    report = render_report(run_analysis(data, num_bootstraps=10), 10)

    assert "## Win rates" not in report
    assert "## Position bias" not in report
    assert "## Bradley-Terry" not in report
