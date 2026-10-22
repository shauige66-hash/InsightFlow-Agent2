"""Filesystem resource authorization at the API boundary."""
from __future__ import annotations

import hashlib
from pathlib import Path

import storage
from core.auth import load_api_keys


def owner_upload_dir(upload_root: Path, owner: str | None = None) -> Path:
    identity = owner or storage.current_owner()
    segment = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return Path(upload_root) / "owners" / segment


def validate_sources(paths: list[str], *, project_root: Path, knowledge: bool = False) -> list[str]:
    from fastapi import HTTPException
    project_root = Path(project_root).resolve()
    restricted = bool(load_api_keys())
    suffixes = {".md", ".txt", ".pdf"} if knowledge else {".csv", ".xlsx", ".db", ".sqlite", ".sqlite3"}
    allowed_roots = [owner_upload_dir(project_root / "uploads").resolve(),
                     (project_root / ("knowledge" if knowledge else "data")).resolve()]
    result = []
    for raw in paths:
        path = Path(raw)
        path = (path if path.is_absolute() else project_root / path).resolve()
        if restricted and not any(path.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=404, detail="资源不存在或不属于当前用户")
        if not path.is_file():
            raise HTTPException(status_code=404 if restricted else 400, detail="资源不存在或不是文件")
        if path.suffix.lower() not in suffixes:
            raise HTTPException(status_code=400, detail="不支持的资源格式")
        result.append(str(path))
    return result
