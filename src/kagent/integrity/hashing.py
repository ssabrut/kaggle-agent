"""Hash the files an agent run must not change, and report any difference afterwards."""

import hashlib
import os
import stat
from pathlib import Path

Snapshot = dict[str, dict[str, str]]  # watched name -> {relative path -> sha256}

_SKIP_DIRS = {"__pycache__"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_regular_file(path: Path) -> bool:
    """True for a plain file; False for symlinks, directories and missing paths.

    Host code must check this before reading or uploading anything the sandbox wrote:
    a symlink there would be resolved against the host filesystem.
    """
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def hash_tree(root: Path) -> dict[str, str]:
    """sha256 of every file under root; symlinks hash their target string.

    A missing root hashes as empty.
    """
    if not root.exists():
        return {}
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if _SKIP_DIRS.intersection(rel.parts):
            continue
        if path.is_symlink():
            hashes[rel.as_posix()] = "symlink:" + os.readlink(path)
        elif path.is_file():
            hashes[rel.as_posix()] = hash_file(path)
    return hashes


def snapshot(watched: dict[str, Path]) -> Snapshot:
    return {name: hash_tree(root) for name, root in watched.items()}


def find_changes(before: Snapshot, after: Snapshot) -> list[str]:
    """Human-readable list such as 'public/train.csv modified'; empty means untouched."""
    changes: list[str] = []
    for name, old in before.items():
        new = after[name]
        for rel in sorted(old.keys() | new.keys()):
            if rel not in new:
                changes.append(f"{name}/{rel} removed")
            elif rel not in old:
                changes.append(f"{name}/{rel} added")
            elif old[rel] != new[rel]:
                changes.append(f"{name}/{rel} modified")
    return changes
