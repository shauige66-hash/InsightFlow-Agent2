"""Atomic artifact publication, guarded by the current execution lease."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile


def assert_publication_lease() -> None:
    from core.runtime import RuntimeContextMissing, current_runtime
    try:
        runtime = current_runtime()
    except RuntimeContextMissing:
        return  # standalone document utilities have no executing Run
    if runtime.assert_lease is not None:
        runtime.assert_lease()


def publish_file(staged: Path, destination: Path) -> None:
    assert_publication_lease()
    os.replace(staged, destination)


def atomic_write(path: Path, data: bytes) -> None:
    assert_publication_lease()
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', suffix='.tmp', delete=False) as handle:
            staged = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        publish_file(staged, path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
