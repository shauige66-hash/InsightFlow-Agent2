"""ModelGate 成功链路集成。

链路：InsightFlow-Agent2 → 隔离 ModelGate → 本机 Mock Provider → 返回 200。

与 test_modelgate_integration.py 的分工：
- 那个文件用 `.invalid` 上游验证**失败链路**（连接错误、追踪、不记账的边界）；
- 本文件用本机 Mock Provider 验证**成功链路**：HTTP 200、有效响应体、
  token 统计、按 run/agent 聚合。

隔离：沿用 ModelGate 自带的测试 compose（独立项目名、独立库
modelgate_docker_test、端口 18000），再叠加 tests/docker/ 下的覆盖文件把上游
指向 host.docker.internal:9099。不接触业务实例、业务数据库与任何真实模型服务。

Mock Provider 返回的是**固定模拟 usage**，因此 Token 数值只能用于验证
"网关是否正确提取并累加"，不代表真实模型用量。
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from core.config import Settings
from core.llm import CallContext, create_llm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELGATE_ROOT = Path(
    # 默认按"与仓库同级"的目录约定查找；换机器时用 MODELGATE_ROOT 覆盖。
    os.getenv("MODELGATE_ROOT", str(PROJECT_ROOT.parent / "modelgate-original"))
)
MODELGATE_COMPOSE = MODELGATE_ROOT / "docker-compose.test.yml"
SUCCESS_OVERRIDE = PROJECT_ROOT / "tests" / "docker" / "docker-compose.modelgate-success.yml"

COMPOSE_PROJECT = "modelgate-docker-test"
GATEWAY_BASE_URL = "http://127.0.0.1:18000/v1"
GATEWAY_ADMIN_URL = "http://127.0.0.1:18000"

TEST_CLIENT_KEY = "sk-docker-test-client-key"
TEST_ADMIN_KEY = "sk-docker-test-admin-key"

MOCK_PROVIDER_PORT = 9099
MOCK_PROVIDER_HOST = "0.0.0.0"

# Mock Provider 每次返回的固定用量，仅用于验证网关的提取与累加逻辑
MOCK_PROMPT_TOKENS = 12
MOCK_COMPLETION_TOKENS = 8

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
            "-f",
            str(SUCCESS_OVERRIDE),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(MODELGATE_ROOT),
    )


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def _http_get(path: str, headers: dict[str, str] | None = None):
    request = urllib.request.Request(GATEWAY_ADMIN_URL + path, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")
    except OSError:
        return 0, ""


def _wait_for(predicate, timeout: float, interval: float = 1.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(scope="module")
def mock_provider():
    """启动本机 Mock Provider；模块结束后终止。

    绑定 0.0.0.0，这样 Docker 里的 ModelGate 能通过 host.docker.internal 访问到。
    """
    if not _docker_available():
        pytest.skip("Docker daemon 不可用")
    if not MODELGATE_COMPOSE.exists():
        pytest.skip(f"未找到 ModelGate 测试 compose: {MODELGATE_COMPOSE}")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.support.openai_mock_provider",
            "--host",
            MOCK_PROVIDER_HOST,
            "--port",
            str(MOCK_PROVIDER_PORT),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ready = _wait_for(
        lambda: _port_open("127.0.0.1", MOCK_PROVIDER_PORT), timeout=40
    )
    if not ready:
        process.terminate()
        pytest.fail("Mock Provider 未能在 40s 内监听 9099")

    try:
        yield f"http://host.docker.internal:{MOCK_PROVIDER_PORT}"
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        # 汇总 Mock 侧计数，便于确认请求确实到达
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{MOCK_PROVIDER_PORT}/stats", timeout=5
            ) as response:
                print("\n[Mock Provider] 收到请求:", response.read().decode())
        except OSError:
            pass


def _fetch_run_detail(run_id: str) -> dict:
    status, body = _http_get(f"/admin/runs/{run_id}", {"X-Admin-Key": TEST_ADMIN_KEY})
    assert status == 200, body
    return json.loads(body)


@pytest.fixture(scope="module", autouse=True)
def gateway_with_mock_upstream(mock_provider):
    """以成功链路配置启动隔离的 ModelGate 测试栈。

    拆除前采集证据（测试栈会连同数据卷一起删除，留在库里的记录随之消失）。
    证据的落点见 `_evidence_path()`：默认不覆盖已被 Git 跟踪的公开证据文件。
    """
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
            return json.loads(body)["database"].get("name") == "modelgate_docker_test"
        except (ValueError, KeyError):
            return False

    try:
        if not _wait_for(ready, timeout=180, interval=2.0):
            logs = _compose("logs", "--tail", "30", "api")
            pytest.fail(
                "ModelGate 测试栈未就绪或未连到测试库。\n"
                f"日志：\n{logs.stdout[-1500:]}"
            )
        yield
    finally:
        _dump_evidence()
        _compose("down", "-v", timeout=300)


# 取证用：记录本次运行发起过的 run 及其 agent
_EVIDENCE_RUNS: dict[str, list[str]] = {}


def _dump_evidence() -> None:
    """把成功链路的调用与聚合结果写入 docs/evidence/。"""
    if not _EVIDENCE_RUNS:
        return

    entries = []
    for run_id, agents in _EVIDENCE_RUNS.items():
        try:
            detail = _fetch_run_detail(run_id)
        except Exception:  # noqa: BLE001 - 取证失败不应让测试报错
            continue
        entries.append(
            {
                "run_id": run_id,
                "agents_invoked": agents,
                "totals": detail["totals"],
                "by_agent": detail["by_agent"],
                "data_quality": detail["data_quality"],
            }
        )

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{MOCK_PROVIDER_PORT}/stats", timeout=10
        ) as response:
            mock_stats = json.loads(response.read().decode("utf-8"))
    except OSError:
        mock_stats = {}

    payload = {
        "chain": "InsightFlow-Agent2 -> ModelGate(test) -> local mock provider -> 200",
        "gateway": {
            "base_url": GATEWAY_BASE_URL,
            "database": "modelgate_docker_test",
            "compose_project": COMPOSE_PROJECT,
        },
        "mock_upstream": {
            "url": f"http://host.docker.internal:{MOCK_PROVIDER_PORT}",
            "fixed_usage": {
                "prompt_tokens": MOCK_PROMPT_TOKENS,
                "completion_tokens": MOCK_COMPLETION_TOKENS,
            },
            "note": "固定模拟 usage，仅用于验证网关的提取与累加，不代表真实模型用量",
            "requests_received": mock_stats.get("total"),
        },
        "runs": entries,
    }

    out_path = _evidence_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_tracked_by_git(out_path):
        # 取证会带上本次随机的 run_id 与耗时，写回已跟踪的公开证据文件等于制造噪声改动。
        # 需要刷新证据时显式设置 MODELGATE_EVIDENCE_DIR（或直接删掉目标文件）再跑。
        print(f"\n[证据] 跳过写入已跟踪文件 {out_path}（设置 MODELGATE_EVIDENCE_DIR 可指定输出位置）")
        return

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[证据] 已写入 {out_path}")


def _evidence_path() -> Path:
    """证据输出位置。

    默认写仓库的 `docs/evidence/`；该目录里的文件已被 Git 跟踪，
    因此测试运行默认跳过覆盖，只有显式指定 **未被跟踪** 的位置才写入。
    这样公开证据文件不会被一次测试运行改写成带随机 run_id 的内容。
    """
    override = os.getenv("MODELGATE_EVIDENCE_DIR")
    if override:
        return Path(override) / "modelgate-success.json"
    return PROJECT_ROOT / "docs" / "evidence" / "modelgate-success.json"


def _is_tracked_by_git(path: Path) -> bool:
    """目标文件是否已被 Git 跟踪。"""
    git = shutil.which("git")
    if git is None:
        return False
    try:
        result = subprocess.run(
            [git, "-C", str(PROJECT_ROOT), "ls-files", "--error-unmatch", str(path)],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def fresh_run_id(prefix: str = "iflow") -> str:
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def build_settings(agent_suffix: str = "") -> Settings:
    return Settings(
        api_key=TEST_CLIENT_KEY,
        base_url=GATEWAY_BASE_URL,
        model="deepseek-chat",
        client_app="insightflow-agent2",
        timeout_seconds=30.0,
    )


def invoke_agent(run_id: str, agent_name: str) -> str:
    """经网关发起一次真实 HTTP 调用，返回模型回复文本。"""
    _EVIDENCE_RUNS.setdefault(run_id, []).append(agent_name)

    llm = create_llm(
        build_settings(), CallContext(run_id=run_id, client_app="insightflow-agent2"), agent_name
    )
    response = llm.invoke([HumanMessage(content="统计客户总数")])
    return response.content


class TestSuccessfulCallPath:
    def test_single_agent_call_returns_200_and_content(self) -> None:
        """成功链路：HTTP 200 + 有效模型响应。"""
        run_id = fresh_run_id("ok")

        content = invoke_agent(run_id, "planner")

        assert content, "应返回模型内容"
        assert isinstance(content, str)

    def test_call_is_recorded_as_success_with_tokens(self) -> None:
        """成功请求应被记为成功，并提取到 Mock 的固定 usage。"""
        run_id = fresh_run_id("tokens")
        invoke_agent(run_id, "executor")

        status, body = _http_get(
            f"/admin/runs/{run_id}", {"X-Admin-Key": TEST_ADMIN_KEY}
        )
        assert status == 200, body
        detail = json.loads(body)

        assert detail["totals"]["total_requests"] == 1
        assert detail["totals"]["success_count"] == 1
        assert detail["totals"]["failure_count"] == 0
        # 这两个数值来自 Mock Provider 的固定 usage，仅验证网关的提取与累加
        assert detail["totals"]["prompt_tokens"] == MOCK_PROMPT_TOKENS
        assert detail["totals"]["completion_tokens"] == MOCK_COMPLETION_TOKENS
        assert detail["data_quality"]["records_with_unknown_tokens"] == 0

    def test_two_agents_in_one_run_are_attributed_separately(self) -> None:
        """同一 Run 中两个 Agent 的调用分别归属。"""
        run_id = fresh_run_id("multiagent")

        invoke_agent(run_id, "planner")
        invoke_agent(run_id, "executor")

        status, body = _http_get(
            f"/admin/runs/{run_id}", {"X-Admin-Key": TEST_ADMIN_KEY}
        )
        assert status == 200
        detail = json.loads(body)

        assert detail["totals"]["total_requests"] == 2
        assert detail["totals"]["success_count"] == 2

        by_agent = {item["key"]: item for item in detail["by_agent"]}
        assert set(by_agent) == {"planner", "executor"}
        for item in by_agent.values():
            assert item["total_requests"] == 1
            assert item["success_count"] == 1
            assert item["total_tokens"] == MOCK_PROMPT_TOKENS + MOCK_COMPLETION_TOKENS

    def test_run_appears_in_runs_list(self) -> None:
        """成功 Run 应出现在工作流列表接口中。"""
        run_id = fresh_run_id("listed")
        invoke_agent(run_id, "critic")

        status, body = _http_get("/admin/runs?limit=50", {"X-Admin-Key": TEST_ADMIN_KEY})
        assert status == 200
        listing = json.loads(body)

        matching = [item for item in listing["runs"] if item["run_id"] == run_id]
        assert matching, "成功 Run 未出现在列表中"
        assert matching[0]["success_count"] == 1

    def test_client_app_is_attributed(self) -> None:
        """X-Client-App 应被记录，便于区分调用方。"""
        run_id = fresh_run_id("apptag")
        invoke_agent(run_id, "reporter")

        status, body = _http_get(
            f"/admin/runs/{run_id}", {"X-Admin-Key": TEST_ADMIN_KEY}
        )
        detail = json.loads(body)
        assert detail["totals"]["total_requests"] == 1


class TestProviderIsolation:
    def test_upstream_was_reached_via_gateway(self, mock_provider) -> None:
        """确认请求真的经过了 Mock Provider，而不是被别处应答。

        本模块共发起 6 次调用。计数对不上就说明有请求没走到 Mock，
        或走了别的上游。
        """
        with urllib.request.urlopen(
            f"http://127.0.0.1:{MOCK_PROVIDER_PORT}/stats", timeout=10
        ) as response:
            stats = json.loads(response.read().decode("utf-8"))

        assert stats["total"] >= 6, f"Mock Provider 只收到 {stats['total']} 次请求"

    def test_gateway_reports_test_database(self) -> None:
        """再次确认网关连的是测试库，不是业务库。"""
        status, body = _http_get("/health/ready")

        assert status == 200
        assert json.loads(body)["database"]["name"] == "modelgate_docker_test"
