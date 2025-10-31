import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    create_repo,
)


def expand_inputs(paths_or_globs: Iterable[str]) -> list[Path]:
    """Expand globs and return a de-duplicated list of Paths (files or directories)."""
    out: list[Path] = []
    seen = set()
    for p in paths_or_globs:
        matches = glob.glob(p, recursive=True) or [p]  # if no glob hit, keep as-is
        for m in matches:
            path = Path(m).resolve()
            if path.exists():
                if path not in seen:
                    out.append(path)
                    seen.add(path)
            else:
                print(f"WARNING: Path does not exist and will be skipped: {m}", file=sys.stderr)
    return out


def iter_files_under(folder: Path) -> Iterable[Path]:
    """Yield all regular files under folder (recursively)."""
    for p in folder.rglob("*"):
        if p.is_file():
            yield p


def make_operations(
    items: list[Path],
) -> list[CommitOperationAdd]:
    """
    Create CommitOperationAdd list.
    - Files: placed at repo_root (or "/") with basename (no folder) unless a folder is uploaded.
    - Folders: uploaded preserving their internal structure at repo_root/<folder.name>/...
    """
    ops: list[CommitOperationAdd] = []
    repo_root = ""

    for item in items:
        if item.is_file():
            path_in_repo = f"{repo_root}/{item.name}" if repo_root else item.name
            ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(item)))
        elif item.is_dir():
            base = item.name
            for f in iter_files_under(item):
                rel = f.relative_to(item).as_posix()
                path_in_repo = f"{repo_root}/{base}/{rel}" if repo_root else f"{base}/{rel}"
                ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(f)))
        else:
            print(f"WARNING: Skipping non-regular path: {item}", file=sys.stderr)

    # Deduplicate by path_in_repo (last one wins)
    dedup = {}
    for op in ops:
        dedup[op.path_in_repo] = op
    return list(dedup.values())


def main(
    hub_id: str,
    repo_type: str,
    private: bool,
    files_or_folders: list[str],
):
    # Create repo if it doesn't exist
    create_repo(
        repo_id=hub_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    # Expand inputs (globs supported)
    expanded = expand_inputs(files_or_folders)
    if not expanded:
        raise ValueError("No valid files or folders found to upload.")

    api = HfApi()

    # Build commit operations
    ops = make_operations(expanded)

    if not ops:
        raise ValueError("No files to upload after processing inputs.")

    api.create_commit(
        repo_id=hub_id,
        repo_type=repo_type,
        operations=ops,
        commit_message="Upload files via push_files_to_hub.py script",
    )

    # Success
    print(f"✅ Uploaded {len(ops)} file(s) to {hub_id} ({repo_type}).")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Upload files/folders to a Hugging Face repository.")
    cparser.add_argument("hub_id", help="Repository ID, e.g. 'username/my-dataset' or 'org/repo'")
    cparser.add_argument(
        "files_or_folders",
        nargs="+",
        help="One or more files/folders or glob patterns (e.g. 'data/*.json' 'mydir/').",
    )

    cparser.add_argument(
        "--repo_type",
        default="dataset",
        choices=["model", "dataset", "space"],
        help="Type of repository (default: dataset)",
    )
    cparser.add_argument(
        "--private",
        action="store_true",
        help="If the repo doesn't exist, create it as private (ignored if it already exists).",
    )

    cargs = cparser.parse_args()
    main(**vars(cargs))
