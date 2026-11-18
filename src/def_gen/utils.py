"""Shared paths and JSON helpers."""

import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[2]


def run_metadata() -> dict[str, Any]:
    """Collect the code version and package versions of the current run.

    Returns:
        A dict with `git_commit` (None outside a git checkout), `versions`
        (None for packages that are not installed) and `timestamp_utc`.
    """
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    versions = {}
    for package in ("def-gen", "llm-annotator", "datasets", "vllm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None

    return {
        "git_commit": git_commit,
        "versions": versions,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON text, raising a ValueError on failure with context.

    Args:
        text: The JSON string to parse.

    Returns:
        Parsed JSON object.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as json_exc:
        snippet = text[:200].replace("\n", " ")
        raise ValueError(
            f"Failed to parse JSON: {json_exc.msg} at pos {json_exc.pos}; snippet={snippet!r}"
        ) from json_exc


class PathCompatibleEncoder(json.JSONEncoder):
    """JSON encoder that converts Path objects to strings."""

    def default(self, obj):
        """Encode Path objects as strings and defer the rest to the base."""
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)
