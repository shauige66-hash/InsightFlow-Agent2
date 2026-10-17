"""沙箱与既有机制的集成行为。

覆盖：缓存随内容变化失效、多 Run 隔离、超时后容器清理、
以及"绝不回退到同进程执行"这条硬约束。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sandbox.executor import (
    CONTAINER_LABEL,
    CONTAINER_NAME_PREFIX,
    DockerSandboxExecutor,
    SandboxUnavailable,
)
from sandbox.protocol import SandboxJob, SandboxLimits

pytestmark = pytest.mark.sandbox


@pytest.fixture(scope="module", autouse=True)
def require_docker() -> None:
    executor = DockerSandboxExecutor()
    if not executor.available():
        pytest.skip("Docker daemon 不可用")
    if not executor.image_present():
        pytest.skip("沙箱镜像未构建：先运行 sandbox/build_image.py")


def table_for(name: str, values: list[int], content_hash: str | None = None):
    import pandas as pd

    from ingestion.models import TableResource

    return TableResource(
        name=name,
        source_type="csv",
        source_path="/shared/same-path.csv",
        dataframe=pd.DataFrame({"a": values}),
        content_hash=content_hash,
    )


class TestToolCacheFollowsContentHash:
    """工具缓存键必须包含内容哈希，而不是只看路径。"""

    def test_different_content_hash_produces_different_cache_key(self) -> None:
        from tools.tool_registry import _tables_fingerprint

        first = {"t": table_for("t", [1, 2, 3], content_hash="hash-a")}
        second = {"t": table_for("t", [1, 2, 3], content_hash="hash-b")}

        # 路径完全相同，只有内容哈希不同
        assert _tables_fingerprint(first) != _tables_fingerprint(second)

    def test_same_content_hash_reuses_cache_key(self) -> None:
        from tools.tool_registry import _tables_fingerprint

        first = {"t": table_for("t", [1, 2, 3], content_hash="hash-a")}
        second = {"t": table_for("t", [9, 9], content_hash="hash-a")}

        # 哈希相同即视为同一份数据来源，缓存可复用
        assert _tables_fingerprint(first) == _tables_fingerprint(second)

    def test_missing_hash_still_produces_a_key(self) -> None:
        from tools.tool_registry import _tables_fingerprint

        fingerprint = _tables_fingerprint({"t": table_for("t", [1])})

        assert fingerprint and "-" in fingerprint[0]


class TestTwoRunIsolation:
    def test_sandbox_work_dirs_are_separate(self, tmp_path: Path) -> None:
        """两个 Run 的输入输出目录互不重叠。"""
        executor = DockerSandboxExecutor()
        dir_a = tmp_path / "run-a"
        dir_b = tmp_path / "run-b"
        dir_a.mkdir()
        dir_b.mkdir()

        job = SandboxJob(
            mode="python",
            table_names=["t"],
            code="result = len(df)",
            op_label="iso",
        )

        outcome_a = executor.run(
            job, run_id="run-a", run_dir=dir_a, tables={"t": table_for("t", [1, 2])}
        )
        outcome_b = executor.run(
            job, run_id="run-b", run_dir=dir_b, tables={"t": table_for("t", [1, 2, 3])}
        )

        assert outcome_a.success and outcome_b.success
        assert outcome_a.result == 2
        assert outcome_b.result == 3

        # 各自的沙箱目录只在自己的 Run 目录下
        assert (dir_a / "sandbox").exists()
        assert (dir_b / "sandbox").exists()
        assert not (dir_a / "sandbox").samefile(dir_b / "sandbox")

    def test_container_names_are_scoped_per_run(self) -> None:
        executor = DockerSandboxExecutor()

        name_a = executor._container_name("run-a", "op-1")
        name_b = executor._container_name("run-b", "op-1")

        assert name_a != name_b
        assert name_a.startswith(CONTAINER_NAME_PREFIX)

    def test_container_name_does_not_leak_raw_run_id(self) -> None:
        """容器名不能直接使用调用方给的 run_id，避免被注入成非法名称。"""
        executor = DockerSandboxExecutor()

        name = executor._container_name("weird id/with:chars", "op")

        assert "weird" not in name
        assert "/" not in name and ":" not in name


class TestContainerCleanup:
    def test_timeout_leaves_no_running_container(self, tmp_path: Path) -> None:
        executor = DockerSandboxExecutor()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        job = SandboxJob(
            mode="python",
            table_names=["t"],
            code="while True:\n    pass",
            limits=SandboxLimits(timeout_seconds=6),
            op_label="timeout-cleanup",
        )

        outcome = executor.run(
            job, run_id="cleanup-run", run_dir=run_dir, tables={"t": table_for("t", [1])}
        )
        assert outcome.success is False

        # 清理后不应再有带本项目标签的同名前缀容器在运行
        listed = subprocess.run(
            [
                executor._docker, "ps", "-q",
                "--filter", f"label={CONTAINER_LABEL}=1",
            ],
            capture_output=True, text=True, timeout=60,
        )
        running_ids = [line for line in listed.stdout.splitlines() if line.strip()]
        assert running_ids == [], f"超时后仍有容器在运行: {running_ids}"

    def test_cleanup_orphans_is_idempotent(self) -> None:
        executor = DockerSandboxExecutor()

        assert executor.cleanup_orphans("cleanup-run") >= 0
        assert executor.cleanup_orphans("cleanup-run") == 0


class TestNoInProcessFallback:
    """沙箱不可用时必须失败，绝不能回退到同进程 exec。"""

    def test_broken_docker_binary_raises_unavailable(self, tmp_path: Path) -> None:
        executor = DockerSandboxExecutor(docker="not-a-real-docker-binary")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        job = SandboxJob(
            mode="python", table_names=["t"], code="result = 1", op_label="nofallback"
        )

        with pytest.raises((SandboxUnavailable, FileNotFoundError, OSError)):
            executor.run(
                job,
                run_id="nofallback",
                run_dir=run_dir,
                tables={"t": table_for("t", [1])},
            )

    def test_tool_reports_failure_instead_of_executing_locally(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """工具层在沙箱不可用时返回失败字典，且结果里没有本进程算出的值。"""
        import tools.python_tool as python_tool
        from core.runtime import RunRuntime, use_runtime
        from core.llm import build_model_factory
        from ingestion.resources import ResourceResolver

        broken = DockerSandboxExecutor(docker="not-a-real-docker-binary")
        monkeypatch.setattr(python_tool, "get_executor", lambda **_: broken)

        runtime = RunRuntime(
            run_id="nofallback",
            model_factory=build_model_factory("nofallback"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
        )

        with use_runtime(runtime):
            tool = python_tool.create_python_tool({"t": table_for("t", [1, 2, 3])})
            outcome = tool.invoke({"table_names": ["t"], "code": "result = len(df)"})

        assert outcome["success"] is False
        assert outcome["result"] is None
        assert outcome["error"]

    def test_source_contains_no_exec_outside_sandbox(self) -> None:
        """生产执行路径里不存在 exec：只有容器入口可以 exec 模型代码。"""
        root = Path(__file__).resolve().parent.parent

        offenders = []
        for path in (root / "tools").rglob("*.py"):
            if "exec(" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(root)))

        assert offenders == [], f"工具层仍存在同进程 exec: {offenders}"

        # 容器入口是唯一允许的位置
        sandbox_run = (root / "sandbox" / "run.py").read_text(encoding="utf-8")
        assert "exec(compile(" in sandbox_run
