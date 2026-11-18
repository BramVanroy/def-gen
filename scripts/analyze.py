"""Analyse writer, ANW and head-to-head runs and write CSV tables and a report.

Example:
    uv run python scripts/analyze.py outputs/study/writer-gemma \
        outputs/study/writer-qwen outputs/study/anw \
        outputs/study/head-to-head --output_dir outputs/study/analysis
"""

from pathlib import Path

from def_gen.analysis.load import combine_runs, read_run
from def_gen.analysis.report import render_report, run_analysis


def main(
    run_dirs: list[str],
    output_dir: str,
    num_bootstraps: int = 2000,
    seed: int = 42,
) -> None:
    """Combine the runs, compute all tables and write them to `output_dir`.

    Args:
        run_dirs: Run directories with `pipeline.json` and
            `full_process_done`.
        output_dir: Directory for the CSV files and `report.md`.
        num_bootstraps: Number of bootstrap resamples.
        seed: Seed of the bootstraps.
    """
    data = combine_runs([read_run(d) for d in run_dirs])
    tables = run_analysis(data, num_bootstraps=num_bootstraps, seed=seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.scores.to_csv(out / "scores.csv", index=False)
    data.pairs.to_csv(out / "pairs.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    report = render_report(tables, num_bootstraps=num_bootstraps, seed=seed)
    out.joinpath("report.md").write_text(report, encoding="utf-8")
    dropped = int(data.coverage["n_dropped"].max())
    print(
        f"Wrote {len(tables) + 2} tables and report.md to {out}."
        f" At most {dropped} items were dropped from a comparison."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyse pipeline runs against the ANW definitions."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Run directories (writer runs, the ANW judge run and"
        " head-to-head runs).",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for the results."
    )
    parser.add_argument(
        "--num_bootstraps",
        type=int,
        default=2000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed of the bootstraps."
    )
    main(**vars(parser.parse_args()))
