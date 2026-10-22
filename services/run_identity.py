"""Trusted Run identity, request fingerprints and execution fencing."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


class RunConflict(ValueError):
    pass


class LeaseLost(RuntimeError):
    pass


_execution_lease: ContextVar[tuple[str, str] | None] = ContextVar("run_execution_lease", default=None)


def current_owner(owner: str | None = None) -> str:
    if owner is not None:
        return owner
    import storage
    return storage.current_owner()


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def lease_token_for(run_id: str) -> str | None:
    value = _execution_lease.get()
    return value[1] if value and value[0] == run_id else None


@contextmanager
def execution_lease(run_id: str, token: str):
    handle = _execution_lease.set((run_id, token))
    try:
        yield
    finally:
        _execution_lease.reset(handle)
