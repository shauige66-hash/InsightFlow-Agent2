"""接口鉴权与归属者隔离。

两个层次都验证：
- 中间件：配了 key 就 401/放行，并把归属者放进上下文；
- 服务：归属者不同，会话与记忆互不可见。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import storage
from core.auth import ApiKeyMiddleware, load_api_keys, resolve_owner


# ---------------------------------------------------------------- 解析与匹配


def test_keys_are_parsed_as_token_owner_pairs() -> None:
    keys = load_api_keys("abc:alice, def:bob")

    assert [(k.token, k.owner) for k in keys] == [("abc", "alice"), ("def", "bob")]


def test_a_malformed_entry_fails_loudly() -> None:
    """写错的配置必须报错——静默忽略等于"看起来有鉴权，实际谁都能进"。"""
    with pytest.raises(ValueError):
        load_api_keys("just-a-token")


def test_empty_configuration_means_no_keys() -> None:
    assert load_api_keys("") == []


def test_the_header_must_be_a_bearer_token() -> None:
    keys = load_api_keys("abc:alice")

    assert resolve_owner(keys, "Bearer abc") == "alice"
    assert resolve_owner(keys, "bearer abc") == "alice"
    assert resolve_owner(keys, "Token abc") is None
    assert resolve_owner(keys, None) is None
    assert resolve_owner(keys, "Bearer wrong") is None


# ---------------------------------------------------------------- 中间件


@pytest.fixture()
def protected_client():
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, keys=load_api_keys("k-alice:alice,k-bob:bob"))

    @app.get("/secret")
    def secret() -> dict[str, str]:
        return {"owner": storage.current_owner()}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ui/")
    def ui() -> dict[str, str]:
        return {"page": "console"}

    return TestClient(app)


def test_a_request_without_a_token_is_rejected(protected_client) -> None:
    response = protected_client.get("/secret")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_a_valid_token_resolves_to_its_owner(protected_client) -> None:
    response = protected_client.get("/secret", headers={"Authorization": "Bearer k-bob"})

    assert response.status_code == 200
    assert response.json() == {"owner": "bob"}


def test_health_and_the_console_stay_public(protected_client) -> None:
    assert protected_client.get("/health").status_code == 200
    assert protected_client.get("/ui/").status_code == 200


def test_the_owner_does_not_leak_into_the_next_request(protected_client) -> None:
    protected_client.get("/secret", headers={"Authorization": "Bearer k-alice"})
    response = protected_client.get("/secret")

    # 上一个请求的归属者必须已经复位，否则未鉴权请求会以 alice 的身份执行
    assert response.status_code == 401


# ---------------------------------------------------------------- 服务隔离


@pytest.fixture()
def isolated_storage(tmp_path):
    """每个用例一个独立的 SQLite，不碰共享库。"""
    from storage.config import StorageConfig

    storage.reset()
    storage.configure(
        StorageConfig(mode="local", local_db_path=str(tmp_path / "owner-test.sqlite"))
    )
    try:
        yield
    finally:
        storage.reset()


def _service():
    from services.conversation_service import ConversationService

    return ConversationService(coordinator=None)


def test_sessions_are_scoped_to_the_request_owner(isolated_storage) -> None:
    service = _service()

    token = storage.set_current_owner("alice")
    try:
        created = service.create_session(title="alice 的会话")
        alice_sessions = service.list_sessions()
    finally:
        storage.reset_current_owner(token)

    token = storage.set_current_owner("bob")
    try:
        bob_sessions = service.list_sessions()
        visible = service.get_session(created["session_id"])
    finally:
        storage.reset_current_owner(token)

    assert [s["session_id"] for s in alice_sessions] == [created["session_id"]]
    assert bob_sessions == []
    assert visible is None


def test_memories_are_scoped_to_the_request_owner(isolated_storage) -> None:
    from services.memory_service import MemoryService

    token = storage.set_current_owner("alice")
    try:
        memory = MemoryService()
        memory.add(kind="preference", key="currency", value="人民币", status="confirmed")
        alice_count = len(memory.list(include_candidates=True))
    finally:
        storage.reset_current_owner(token)

    token = storage.set_current_owner("bob")
    try:
        bob_count = len(MemoryService().list(include_candidates=True))
    finally:
        storage.reset_current_owner(token)

    assert alice_count == 1
    assert bob_count == 0
