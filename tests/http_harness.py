"""用 **独立进程中的真实 HTTP 服务** 封装启动 / 停止。

与 `TestClient` 的区别：这里是真的 uvicorn + 真 socket，因此可以
强杀进程、再换一个新进程重启，用于验证跨进程恢复。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[int, Any]:
    """发一个 HTTP 请求，返回 (状态码, 解析后的 JSON 或原始文本)。"""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        status = error.code

    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


class ApiServer:
    """一个真实运行中的 API 服务进程。"""

    def __init__(
        self,
        workdir: Path,
        port: int,
        *,
        crash_at: str | None = None,
        log_path: Path | None = None,
        model_factory: str | None = "tests.test_api_workflow:factory",
    ) -> None:
        self.workdir = workdir
        self.port = port
        self.crash_at = crash_at
        self.model_factory = model_factory
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = log_path or (workdir / f"server-{port}.log")
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    def start(self, *, ready_timeout: float = 60.0) -> "ApiServer":
        import os

        env = dict(os.environ)
        env["INSIGHTFLOW_WORKDIR"] = str(self.workdir)
        env["INSIGHTFLOW_MOCK_LLM"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if self.crash_at:
            env["INSIGHTFLOW_CRASH_AT"] = self.crash_at
        else:
            env.pop("INSIGHTFLOW_CRASH_AT", None)
        # 子进程必须用测试提供的脚本化工厂，否则 planner 的目标表由
        # serve_api 自己 import 的工厂决定，测试无法左右它走 SQL 还是 Python。
        if self.model_factory:
            env["INSIGHTFLOW_MODEL_FACTORY"] = self.model_factory
        else:
            env.pop("INSIGHTFLOW_MODEL_FACTORY", None)

        # mock 的目标表/代码由测试的工厂决定，不继承父进程可能残留的覆盖
        env.pop("INSIGHTFLOW_MOCK_TABLE", None)
        env.pop("INSIGHTFLOW_MOCK_CODE", None)
        env.pop("INSIGHTFLOW_CRASH_TRACE", None)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")

        self.process = subprocess.Popen(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "serve_api.py"),
                "--port",
                str(self.port),
                "--workdir",
                str(self.workdir),
                "--mock-llm",
            ],
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(
                    f"服务进程提前退出（{self.process.returncode}）：\n{self.tail()}"
                )
            try:
                status, _body = http("GET", f"{self.base}/health", timeout=3.0)
                if status == 200:
                    return self
            except Exception:  # noqa: BLE001 - 未就绪时继续等
                pass
            time.sleep(0.4)

        self.stop()
        raise AssertionError(f"服务未在 {ready_timeout}s 内就绪：\n{self.tail()}")

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait_death(self, timeout: float = 120.0) -> int:
        """等待服务进程退出（崩溃注入后用它确认真的死了）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return int(self.process.returncode)
            time.sleep(0.3)
        raise AssertionError("服务进程未按预期退出")

    def tail(self, lines: int = 40) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> "ApiServer":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # --- 便捷方法 ---

    def create_run(self, **payload: Any) -> tuple[int, Any]:
        return http("POST", f"{self.base}/runs", payload)

    def get_run(self, run_id: str) -> tuple[int, Any]:
        return http("GET", f"{self.base}/runs/{run_id}")

    def wait_status(
        self, run_id: str, expected: set[str], timeout: float = 240.0
    ) -> dict[str, Any]:
        deadline = time.time() + timeout
        last: dict[str, Any] = {}
        while time.time() < deadline:
            if not self.alive():
                raise AssertionError(
                    f"服务进程已退出，无法继续等待 {run_id}：\n{self.tail()}"
                )
            try:
                _status, body = self.get_run(run_id)
                last = (body or {}).get("run") or {}
                if last.get("status") in expected:
                    return last
            except Exception:  # noqa: BLE001 - 服务可能刚被杀
                pass
            time.sleep(0.4)
        raise AssertionError(f"Run {run_id} 未进入 {expected}：{last}\n{self.tail()}")


def mcp_server_pids() -> list[int]:
    """当前存活的 MCP Server 子进程 pid。"""
    if sys.platform != "win32":
        return []
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'mcp_bridge.server' } | "
            "Select-Object -ExpandProperty ProcessId",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return [int(line) for line in completed.stdout.split() if line.strip().isdigit()]


def isolated_workdir(prefix: str = "p2d-http-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
