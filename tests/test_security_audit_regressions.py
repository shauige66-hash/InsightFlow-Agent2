"""Offline regressions for owner scope, dispatch identity, leases and upload limits."""
from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import storage
from storage.config import StorageConfig
from services.run_store import RunStore
from services.run_coordinator import RunCoordinator
from services.run_identity import LeaseLost, RunConflict


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("INSIGHTFLOW_EXECUTION", "inline")
    monkeypatch.delenv("INSIGHTFLOW_API_KEYS", raising=False)
    monkeypatch.setenv("INSIGHTFLOW_TEST_DATABASE_URL", "postgresql+psycopg://test:test@127.0.0.1:1/insightflow_test")
    monkeypatch.setenv("INSIGHTFLOW_TEST_REDIS_URL", "redis://127.0.0.1:1/15")
    storage.configure(StorageConfig(), base_dir=tmp_path)
    handle = storage.set_current_owner("alice")
    yield
    storage.reset_current_owner(handle)
    storage.reset()


def test_run_owner_scope_and_duplicate_identity_preserve_approval(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    first = store.create("run_private", data_sources=["a.csv"], require_plan_approval=True)
    store.record_approval(first.run_id, {"approval_id": "ap1", "decision": "approve"})
    again = store.create(first.run_id, data_sources=["a.csv"], require_plan_approval=True)
    assert again.created_at == first.created_at and len(again.approvals) == 1
    with pytest.raises(RunConflict):
        store.create(first.run_id, data_sources=["b.csv"])
    assert store.get(first.run_id, owner="bob") is None
    assert store.list(owner="bob") == []
    with pytest.raises(RunConflict):
        store.create(first.run_id, data_sources=["a.csv"], owner="bob")


def test_same_approval_cannot_be_changed(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    store.create("run_approval", data_sources=["a.csv"])
    store.record_approval("run_approval", {"approval_id": "ap1", "decision": "approve"})
    with pytest.raises(RunConflict):
        store.record_approval("run_approval", {"approval_id": "ap1", "decision": "reject"})
    assert len(store.get("run_approval").approvals) == 1


def test_atomic_lease_claim_and_old_executor_fencing(tmp_path):
    database = tmp_path / "runs.sqlite"
    store = RunStore(database)
    store.create("run_lease", data_sources=["a.csv"])
    barrier = threading.Barrier(2)
    def claim(token):
        other = RunStore(database)
        barrier.wait()
        return other.claim("run_lease", token, owner="alice")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["lease-one", "lease-two"]))
    assert sorted(results) == [False, True]
    old_token = store.get("run_lease").lease_token
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE runs SET lease_expires_at=? WHERE run_id=?", (time.time()-1, "run_lease"))
    assert store.claim("run_lease", "lease-new", owner="alice", resume=True)
    with pytest.raises(LeaseLost):
        store.assert_lease("run_lease", old_token)
    with pytest.raises(LeaseLost):
        store.update_status("run_lease", "completed", token=old_token)
    assert store.get("run_lease").status == "running"


def test_dispatch_copies_owner_and_only_executes_once(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs.sqlite")
    coordinator = RunCoordinator(store, output_root=tmp_path)
    entered, release = threading.Event(), threading.Event()
    seen = []
    def execute(run_id, **kwargs):
        seen.append(storage.current_owner())
        kwargs["assert_lease"]()
        entered.set()
        assert release.wait(5)
        return {"run_status": "completed"}
    monkeypatch.setattr(coordinator, "_execute", execute)
    try:
        coordinator.dispatch("run_dispatch", data_sources=["a.csv"])
        assert entered.wait(5)
        coordinator.dispatch("run_dispatch", data_sources=["a.csv"])
        release.set()
        coordinator.executor.wait("run_dispatch", 5)
        assert seen == ["alice"]
        assert store.get("run_dispatch").status == "completed"
    finally:
        release.set()
        coordinator.shutdown()


def test_empty_resume_forwards_checkpoint_continuation(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs.sqlite")
    store.create("run_resume", data_sources=["a.csv"])
    store.update_status("run_resume", "interrupted")
    coordinator = RunCoordinator(store, output_root=tmp_path)
    seen = []
    def execute(run_id, **kwargs):
        seen.append(kwargs)
        return {"run_status": "completed"}
    monkeypatch.setattr(coordinator, "_execute", execute)
    try:
        coordinator.resume("run_resume", wait=True)
        assert seen[0]["continue_from_checkpoint"] is True
        assert seen[0]["resume"] is None
    finally:
        coordinator.shutdown()


def test_authenticated_sources_reject_other_owner_and_host_paths(tmp_path, monkeypatch):
    from core.resource_access import owner_upload_dir, validate_sources
    monkeypatch.setenv("INSIGHTFLOW_API_KEYS", "key-alice:alice,key-bob:bob")
    alice = owner_upload_dir(tmp_path / "uploads", "alice") / "a.csv"
    bob = owner_upload_dir(tmp_path / "uploads", "bob") / "b.csv"
    outside = tmp_path / "secret.csv"
    for path in (alice, bob, outside):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a\n1\n", encoding="utf-8")
    assert validate_sources([str(alice)], project_root=tmp_path) == [str(alice.resolve())]
    for path in (bob, outside):
        with pytest.raises(HTTPException) as error:
            validate_sources([str(path)], project_root=tmp_path)
        assert error.value.status_code == 404


def test_signed_artifact_is_owner_scoped_and_html_has_csp(tmp_path, monkeypatch):
    import api
    import api_runs
    from core.auth import install
    monkeypatch.setenv("INSIGHTFLOW_API_KEYS", "key-alice:alice,key-bob:bob")
    store = RunStore(tmp_path / "runs.sqlite")
    store.create("run_artifact", data_sources=["a.csv"])
    folder = tmp_path / "outputs" / "run_artifact"
    folder.mkdir(parents=True)
    (folder / "analysis_report.html").write_text('<html><img src="chart.png"></html>', encoding="utf-8")
    (folder / "chart.png").write_bytes(b"synthetic-image")
    monkeypatch.setattr(api_runs, "run_store", store)
    monkeypatch.setattr(api_runs, "coordinator", SimpleNamespace())
    monkeypatch.setattr(api_runs, "OUTPUT_ROOT", folder.parent)
    app = FastAPI()
    install(app)
    app.get("/runs/{run_id}/artifact-link")(api_runs.artifact_link)
    app.get("/runs/{run_id}/report.html")(api.report_html)
    app.get("/runs/{run_id}/{filename}")(api.run_artifact)
    client = TestClient(app)
    assert client.get("/runs/run_artifact/report.html").status_code == 401
    assert client.get("/runs/run_artifact/artifact-link", headers={"Authorization":"Bearer key-bob"}).status_code == 404
    link = client.get("/runs/run_artifact/artifact-link", headers={"Authorization":"Bearer key-alice"}).json()["url"]
    response = client.get(link)
    assert response.status_code == 200
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/runs/run_artifact/chart.png").content == b"synthetic-image"
    assert client.get("/runs/other_run/chart.png").status_code == 401
    assert client.get("/runs/C:%5Csecret/report.html", headers={"Authorization":"Bearer key-alice"}).status_code == 404


def test_legacy_upload_uses_same_size_limit_and_cleans_partial(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "_PROJECT_DIR", tmp_path)
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 16)
    app = FastAPI()
    app.post("/analyze/upload")(api.analyze_upload)
    response = TestClient(app).post("/analyze/upload", files={"files":("large.csv", b"x"*64, "text/csv")})
    assert response.status_code == 413
    assert not [path for path in (tmp_path / "uploads").rglob("*") if path.is_file()]


def test_queue_claim_token_is_persisted_and_old_token_cannot_finish(monkeypatch):
    from contextlib import contextmanager
    import re
    from services import run_queue
    job = {"status":"queued", "claim_token":None, "attempts":0}
    statements = []
    class Session:
        def execute(self, statement, params):
            sql = str(statement)
            statements.append(sql)
            if "RETURNING run_id" in sql:
                assert re.search(r"claim_token\s*=\s*:claim_token", sql)
                if job["status"] != "queued":
                    return SimpleNamespace(first=lambda: None)
                job.update(status="claimed", claim_token=params["claim_token"], attempts=job["attempts"]+1)
                row = SimpleNamespace(run_id="run_queue",payload={"_owner":"alice"},claim_token=job["claim_token"],attempts=job["attempts"])
                return SimpleNamespace(first=lambda: row)
            assert "claim_token=:claim_token" in sql.replace(" ", "")
            assert ":claim_token IS NULL" not in sql
            if params["claim_token"] == job["claim_token"]:
                job.update(status=params["status"], claim_token=None)
            return SimpleNamespace(rowcount=1)
        def commit(self):
            pass
    @contextmanager
    def session_scope(*args, **kwargs):
        yield Session()
    monkeypatch.setattr(run_queue, "_require_postgres", lambda: None)
    monkeypatch.setattr(storage, "database_url", lambda: "synthetic-no-connection")
    monkeypatch.setattr(storage, "session_scope", session_scope)
    run_id, payload = run_queue.claim("worker")
    assert payload["_job_claim"] and job["claim_token"] == payload["_job_claim"]
    run_queue.finish(run_id, claim_token="old-worker-token")
    assert job["status"] == "claimed"
    with pytest.raises(ValueError):
        run_queue.finish(run_id)
    run_queue.finish(run_id, claim_token=payload["_job_claim"])
    assert job["status"] == "done"


def test_upload_paths_are_independent_of_process_cwd(tmp_path, monkeypatch):
    from pathlib import Path
    import api
    from core.resource_access import validate_sources
    monkeypatch.chdir(tmp_path)
    assert api.UPLOAD_DIR == Path(api.__file__).resolve().parent / "uploads"
    project = tmp_path / "project"
    source = project / "data" / "sample.csv"
    source.parent.mkdir(parents=True)
    source.write_text("a\n1\n", encoding="utf-8")
    monkeypatch.setenv("INSIGHTFLOW_API_KEYS", "key-alice:alice")
    assert validate_sources(["data/sample.csv"], project_root=project) == [str(source.resolve())]


def test_raw_chunked_upload_is_bounded_before_multipart_parsing():
    import asyncio
    from core.upload_limit import UploadBodyLimitMiddleware
    sent = []
    chunks = iter([{"type":"http.request","body":b"abc","more_body":True},
                   {"type":"http.request","body":b"def","more_body":False}])
    async def receive():
        return next(chunks)
    async def send(message):
        sent.append(message)
    async def downstream(scope, receive, send):
        await receive()
        await receive()
        raise AssertionError("oversize upload reached the handler")
    middleware = UploadBodyLimitMiddleware(downstream, limit_getter=lambda:4)
    asyncio.run(middleware({"type":"http","method":"POST","path":"/api/uploads","headers":[]},receive,send))
    assert sent[0]["status"] == 413


def test_sync_worker_execution_renews_lease(tmp_path, monkeypatch):
    import services.run_coordinator as coordinator_module
    monkeypatch.setattr(coordinator_module, "STALE_HEARTBEAT_SECONDS", 0.01)
    store = RunStore(tmp_path / "runs.sqlite")
    coordinator = RunCoordinator(store, output_root=tmp_path)
    seen = []
    def execute(run_id, **kwargs):
        first=store.get(run_id).heartbeat_at
        time.sleep(1.1)
        seen.append(store.get(run_id).heartbeat_at > first)
        kwargs["assert_lease"]()
        return {"run_status":"completed"}
    monkeypatch.setattr(coordinator, "_execute", execute)
    try:
        coordinator.start("run_worker", data_sources=["a.csv"], wait=True)
        assert seen == [True]
    finally:
        coordinator.shutdown()


def test_executor_keeps_bounded_recent_results(tmp_path):
    from services.run_coordinator import RunExecutor
    executor = RunExecutor(RunStore(tmp_path / "runs.sqlite"), max_workers=1)
    try:
        for index in range(150):
            run_id=f"run_{index}"
            executor.submit(run_id, lambda value=index:value)
            assert executor.wait(run_id, 5) == index
        assert executor.wait("run_149", 5) == 149
        assert len(executor._recent) <= 128
    finally:
        executor.shutdown()
