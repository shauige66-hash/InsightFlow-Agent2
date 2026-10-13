"""ModelGate 隔离集成（失败链路）。

只打独立测试网关（18000 / 库 modelgate_docker_test），不接触业务实例（8000）。
测试栈的上游是保留域名 `.invalid`，DNS 必然失败，因此不会产生任何真实付费调用；
本文件验证的是失败链路的隔离、追踪与"不该记账就不记账"的边界行为。
成功链路见 test_modelgate_success.py。

本模块自己拉起并拆除测试栈（不叠加覆盖文件，即默认的 `.invalid` 上游），
因此与成功链路测试可以在同一次会话中先后执行。
网关无法安全启动时整体跳过；不会退回业务实例。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from core.config import Settings
from core.llm import CallContext, create_llm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELGATE_ROOT = Path(
    os.getenv("MODELGATE_ROOT", r"G:\AI-Projects\modelgate-original")
)
MODELGATE_COMPOSE = MODELGATE_ROOT / "docker-compose.test.yml"

COMPOSE_PROJECT = "modelgate-docker-test"
GATEWAY_BASE_URL = "http://127.0.0.1:18000/v1"
GATEWAY_ADMIN_URL = "http://127.0.0.1:18000"

# 测试栈自带的固定密钥，与业务环境无关
TEST_CLIENT_KEY = "sk-docker-test-client-key"
TEST_ADMIN_KEY = "sk-docker-test-admin-key"

TEST_SETTINGS = Settings(
    api_key=TEST_CLIENT_KEY,
    base_url=GATEWAY_BASE_URL,
    model="deepseek-chat",
    client_app="insightflow-agent2",
    timeout_seconds=20.0,
)

_DOCKER_FALLBACK = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"


def _docker() -> str | None:
    found = shutil.which("docker")
    if found:
        return found
    return _DOCKER_FALLBACK if os.path.exists(_DOCKER_FALLBACK) else None


def _docker_available() -> bool:
    docker = _docker()
    if docker is None:
        return False
    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _compose(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    docker = _docker()
    assert docker is not None
    return subprocess.run(
        [
            docker,
            "compose",
            "-p",
            COMPOSE_PROJECT,
            "-f",
            str(MODELGATE_COMPOSE),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(MODELGATE_ROOT),
    )


def _http_get(path: str, headers: dict[str, str] | None = None):
    request = urllib.request.Request(GATEWAY_ADMIN_URL + path, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")
    except OSError:
        return 0, ""


def _wait_for(predicate, timeout: float, interval: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(scope="module", autouse=True)
def require_test_gateway() -> None:
    """拉起隔离测试栈，并确认它连的是测试库（不是业务库）。"""
    if not _docker_available():
        pytest.skip("Docker daemon 不可用")
    if not MODELGATE_COMPOSE.exists():
        pytest.skip(f"未找到 ModelGate 测试 compose: {MODELGATE_COMPOSE}")

    up = _compose("up", "-d")
    if up.returncode != 0:
        pytest.fail(
            "启动 ModelGate 测试栈失败：\n"
            f"stdout: {up.stdout[-1200:]}\nstderr: {up.stderr[-1200:]}"
        )

    def ready() -> bool:
        status, body = _http_get("/health/ready")
        if status != 200:
            return False
        try:
            payload = json.loads(body)
        except ValueError:
            return False

        database = payload.get("database", {}).get("name")
        if database not in (None, "modelgate_docker_test"):
            # 明确失败：绝不接受连到非测试库的网关
            pytest.fail(
                f"测试网关连接的库是 {database!r}，不是 modelgate_docker_test"
            )
        return database == "modelgate_docker_test"

    try:
        if not _wait_for(ready, timeout=180):
            logs = _compose("logs", "--tail", "30", "api")
            pytest.fail(
                "ModelGate 测试栈未就绪。\n" f"日志：\n{logs.stdout[-1500:]}"
            )
        yield
    finally:
        _compose("down", "-v", timeout=300)


def fresh_run_id(prefix: str = "iflow") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class TestGatewayReachability:
    def test_openai_compatible_endpoint_requires_auth(self) -> None:
        """未带密钥时网关拒绝，说明请求确实到了网关。"""
        request = urllib.request.Request(
            f"{GATEWAY_ADMIN_URL}/v1/chat/completions",
            data=json.dumps(
                {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with pytest.raises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(request, timeout=20)

        assert captured.value.code == 401


class TestTracingHeaders:
    def test_call_is_recorded_under_run_and_agent(self) -> None:
        """经网关的真实 HTTP 调用必须能按 run_id + agent_name 查回。"""
        agent_name = "planner"
        run_id = fresh_run_id("trace")
        context = CallContext(run_id=run_id, client_app="insightflow-agent2")

        llm = create_llm(TEST_SETTINGS, context, agent_name)

        # 上游是不可解析的保留域名，因此请求会以 502 结束；
        # 关键是网关仍然记录了一次调用。
        with pytest.raises(Exception):
            llm.invoke([HumanMessage(content="统计客户数量")])

        status, body = _http_get(
            f"/admin/runs/{run_id}", {"X-Admin-Key": TEST_ADMIN_KEY}
        )
        assert status == 200, body
        detail = json.loads(body)

        assert detail["totals"]["total_requests"] == 1
        assert detail["totals"]["failure_count"] == 1

        agents = {item["key"] for item in detail["by_agent"]}
        assert agent_name in agents, f"未按 agent 归类: {agents}"

    def test_two_agents_are_attributed_separately(self) -> None:
        """同一次运行里不同 Agent 的调用必须分开统计。"""
        run_id = fresh_run_id("multi")
        context = CallContext(run_id=run_id, client_app="insightflow-agent2")

        for agent_name in ("planner", "executor"):
            llm = create_llm(TEST_SETTINGS, context, agent_name)
            with pytest.raises(Exception):
                llm.invoke([HumanMessage(content="probe")])

        status, body = _http_get(
            f"/admin/runs/{run_id}", {"X-Admin-Key": TEST_ADMIN_KEY}
        )
        assert status == 200
        detail = json.loads(body)

        by_agent = {item["key"]: item["total_requests"] for item in detail["by_agent"]}
        assert by_agent.get("planner") == 1
        assert by_agent.get("executor") == 1
        assert detail["totals"]["total_requests"] == 2


class TestGatewayErrorBoundaries:
    def test_unknown_model_is_rejected(self) -> None:
        """未登记模型返回 400，且不产生调用记录。"""
        run_id = fresh_run_id("unknown")
        settings = Settings(
            api_key=TEST_CLIENT_KEY,
            base_url=GATEWAY_BASE_URL,
            model="not-a-registered-model",
            client_app="insightflow-agent2",
        )
        llm = create_llm(settings, CallContext(run_id=run_id, client_app="x"), "planner")

        with pytest.raises(Exception):
            llm.invoke([HumanMessage(content="probe")])

        status, body = _http_get(
            f"/admin/runs/{run_id}", {"X-Admin-Key": TEST_ADMIN_KEY}
        )
        assert status == 200
        assert json.loads(body)["totals"]["total_requests"] == 0, "未登记模型不应记账"

    def test_bad_client_key_is_rejected(self) -> None:
        settings = Settings(
            api_key="sk-definitely-wrong",
            base_url=GATEWAY_BASE_URL,
            model="deepseek-chat",
            client_app="insightflow-agent2",
        )
        run_id = fresh_run_id("authfail")
        llm = create_llm(settings, CallContext(run_id=run_id, client_app="x"), "planner")

        with pytest.raises(Exception):
            llm.invoke([HumanMessage(content="probe")])

        status, body = _http_get(
            f"/admin/runs/{run_id}", {"X-Admin-Key": TEST_ADMIN_KEY}
        )
        assert status == 200
        assert json.loads(body)["totals"]["total_requests"] == 0
