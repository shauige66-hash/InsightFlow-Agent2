"""P2-C 补验收：私有 MCP 授权配置不得被沙箱读取。

背景（实测复现过的真实问题）：`write_authorizations` 原先把配置写在
`<output_root>/<run_id>/mcp/authorizations.json`，而 P2-A 的
`DockerSandboxExecutor` 把整个 run 目录只读挂载为 `/history`。于是容器内存在
`/history/mcp/authorizations.json`，模型代码用被注入的 `pandas` 就能读到
宿主绝对路径、workspace 根目录与数据集元数据。

只读挂载只保证"改不了"，不保证"读不到"。这些用例用**真实容器**验证修复前后
的差别，而不是断言挂载参数。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from core.llm import build_model_factory
from core.runtime import McpSettings, RunRuntime, use_runtime
from ingestion.models import TableResource
from ingestion.resources import ResourceResolver
from mcp_bridge.contract import (
    TOOL_DISCOVER_DATASETS,
    TOOL_GET_SCHEMA,
)
from mcp_bridge.registry import (
    PRIVATE_ROOT_SUFFIX,
    config_path,
    write_authorizations,
)
from sandbox.executor import DockerSandboxExecutor
from sandbox.protocol import SandboxJob, SandboxLimits

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE orders (order_id INTEGER, amount REAL)")
        connection.execute("CREATE TABLE secrets (value TEXT)")
        connection.execute("INSERT INTO orders VALUES (1, 10.0)")
        connection.execute("INSERT INTO secrets VALUES ('top-secret')")
        connection.commit()
    finally:
        connection.close()


def snapshot(name: str, path: Path, table: str) -> dict:
    return {
        "name": name,
        "source_type": "sqlite",
        "source_path": str(path),
        "content_hash": "hash-1",
        "file_size": path.stat().st_size,
        "locator": table,
        "database_table": table,
    }


READ_CONFIG_CODE = (
    "found = {}\n"
    "for path in ['/history/mcp/authorizations.json']:\n"
    "    try:\n"
    "        frame = pd.read_json(path, typ='series')\n"
    "        found[path] = 'LEAKED: ' + str(frame)[:200]\n"
    "    except Exception:\n"
    "        found[path] = 'unreadable'\n"
    "result = found\n"
)


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    """返回 (workspace, run_dir, sqlite 数据库路径)。"""
    run_id = "leak-run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    db = tmp_path / "sales.sqlite"
    build_sqlite(db)
    return tmp_path, run_dir, db


def run_sandbox_code(
    workspace: Path, run_dir: Path, code: str, run_id: str = "leak-run"
):
    runtime = RunRuntime(
        run_id=run_id,
        model_factory=build_model_factory(run_id),
        resolver=ResourceResolver(),
        output_root=workspace,
        mcp=McpSettings(mode="mcp"),
    )
    tables = {
        "t": TableResource(
            name="t",
            source_type="csv",
            source_path=str(workspace / "x.csv"),
            dataframe=pd.DataFrame({"a": [1, 2, 3]}),
        )
    }
    job = SandboxJob(
        mode="python",
        table_names=["t"],
        code=code,
        limits=SandboxLimits(timeout_seconds=60),
        op_label="leak-check",
    )
    with use_runtime(runtime):
        return DockerSandboxExecutor().run(
            job, run_id=run_id, run_dir=run_dir, tables=tables
        )


class TestConfigLocation:
    """配置位置：必须在任何 Run 容器的挂载范围之外。"""

    def test_config_is_not_inside_the_run_directory(self, tmp_path: Path) -> None:
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        config, _refs = write_authorizations(
            run_id="leak-run",
            snapshots={"sales.orders": snapshot("sales.orders", db, "orders")},
            workspace_root=tmp_path,
            output_root=tmp_path,
        )

        run_dir = tmp_path / "leak-run"
        assert not str(config).startswith(str(run_dir)), (
            f"配置在 run 目录内，会被只读挂载进容器：{config}"
        )

    def test_config_uses_the_private_root(self, tmp_path: Path) -> None:
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        config, _refs = write_authorizations(
            run_id="leak-run",
            snapshots={"sales.orders": snapshot("sales.orders", db, "orders")},
            workspace_root=tmp_path,
            output_root=tmp_path,
        )

        assert config.parent.name == "leak-run"
        assert config.parent.parent.name == f"{tmp_path.name}{PRIVATE_ROOT_SUFFIX}"
        # run 目录仍然存在，但里面不该有 mcp/
        assert not (tmp_path / "leak-run" / "mcp").exists()

    def test_runtime_config_path_matches_registry(self, tmp_path: Path) -> None:
        """运行期算出的路径必须与 registry 写出的一致。"""
        runtime = RunRuntime(
            run_id="leak-run",
            model_factory=build_model_factory("leak-run"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            mcp=McpSettings(mode="mcp"),
        )

        assert runtime.mcp_config_path() == config_path(tmp_path, "leak-run")

    def test_private_root_is_outside_output_root(self, tmp_path: Path) -> None:
        """私有目录必须落在 output_root 之外。

        output_root 的整棵树都会被按 Run 只读挂载进沙箱；而 run_id 由调用方
        指定且合法接受 mcp_private，放在里面会造成跨 Run 配置泄漏。
        """
        runtime = RunRuntime(
            run_id="leak-run",
            model_factory=build_model_factory("leak-run"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            mcp=McpSettings(mode="mcp"),
        )

        private_root = runtime.mcp_private_root().resolve()
        output_root = runtime.output_root.resolve()

        assert not private_root.is_relative_to(output_root)
        assert not runtime.mcp_config_path().resolve().is_relative_to(output_root)


class TestSymlinkAndTraversalHardening:
    def test_symlinked_private_dir_is_refused(self, tmp_path: Path) -> None:
        """私有目录被替换成符号链接时必须拒绝写入。"""
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        private_root = tmp_path / "mcp_private"
        try:
            private_root.symlink_to(elsewhere, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("本机不允许创建符号链接")

        with pytest.raises(PermissionError):
            write_authorizations(
                run_id="leak-run",
                snapshots={"sales.orders": snapshot("sales.orders", db, "orders")},
                workspace_root=tmp_path,
                output_root=tmp_path,
            )

        # 没有写到链接指向的位置
        assert not (elsewhere / "leak-run" / "authorizations.json").exists()

    def test_config_write_is_atomic(self, tmp_path: Path) -> None:
        """写出后不应残留临时文件，避免读到半截 JSON。"""
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        config, _refs = write_authorizations(
            run_id="leak-run",
            snapshots={"sales.orders": snapshot("sales.orders", db, "orders")},
            workspace_root=tmp_path,
            output_root=tmp_path,
        )

        leftovers = [p.name for p in config.parent.iterdir() if p.name != config.name]
        assert leftovers == [], f"残留临时文件：{leftovers}"
        json.loads(config.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def require_docker() -> None:
    executor = DockerSandboxExecutor()
    if not executor.available():
        pytest.skip("Docker daemon 不可用")
    if not executor.image_present():
        pytest.skip("沙箱镜像未构建：先运行 sandbox/build_image.py")


@pytest.mark.sandbox
@pytest.mark.usefixtures("require_docker")
class TestSandboxCannotReadTheConfig:
    """真实容器验证：模型代码读不到私有配置。"""

    def test_sandbox_cannot_read_authorizations(
        self, workspace: tuple[Path, Path, Path]
    ) -> None:
        """修复后的位置：容器内的 /history 下不存在该文件。"""
        root, run_dir, db = workspace

        write_authorizations(
            run_id="leak-run",
            snapshots={"sales.orders": snapshot("sales.orders", db, "orders")},
            workspace_root=root,
            output_root=root,
        )

        outcome = run_sandbox_code(root, run_dir, READ_CONFIG_CODE)

        assert outcome.success is True, outcome.error
        assert outcome.result["/history/mcp/authorizations.json"] == "unreadable", (
            f"沙箱仍能读到私有配置：{outcome.result}"
        )

    def test_before_layout_would_have_leaked(
        self, workspace: tuple[Path, Path, Path]
    ) -> None:
        """对照：把配置放回旧位置，沙箱确实能读到。

        这条用例存在的意义是证明上面那条不是因为"路径写错"而通过——
        同一个容器、同一段模型代码，位置换了就真的读不到。
        """
        root, run_dir, db = workspace

        # 手工把配置写到旧位置（run 目录内的 mcp/）
        legacy_dir = run_dir / "mcp"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_payload = {
            "run_id": "leak-run",
            "workspace_root": str(root),
            "datasets": [
                {
                    "dataset_id": "leak-run:sales.orders",
                    "name": "sales.orders",
                    "source_type": "sqlite",
                    "source_path": str(db),
                    "locator": "orders",
                    "content_hash": "hash-1",
                    "database_table": "orders",
                    "columns": [],
                    "row_count": None,
                }
            ],
        }
        (legacy_dir / "authorizations.json").write_text(
            json.dumps(legacy_payload), encoding="utf-8"
        )

        outcome = run_sandbox_code(
            root, run_dir, READ_CONFIG_CODE, run_id="legacy-run"
        )

        assert outcome.success is True, outcome.error
        leaked = outcome.result["/history/mcp/authorizations.json"]
        assert leaked.startswith("LEAKED"), f"对照用例未复现泄漏：{leaked}"
        # 泄漏内容确实包含宿主路径
        assert str(root).split("\\")[0] in leaked or "workspace_root" in leaked

    def test_sandbox_cannot_read_the_private_root(
        self, workspace: tuple[Path, Path, Path]
    ) -> None:
        """私有根目录整体不在容器里，连目录都不存在。"""
        root, run_dir, db = workspace

        write_authorizations(
            run_id="leak-run",
            snapshots={"sales.orders": snapshot("sales.orders", db, "orders")},
            workspace_root=root,
            output_root=root,
        )

        code = (
            "found = {}\n"
            "for path in ['/history/mcp', '/history/mcp_private',"
            " '/mcp_private', '/history/../mcp_private']:\n"
            "    try:\n"
            "        pd.read_json(path + '/authorizations.json', typ='series')\n"
            "        found[path] = 'LEAKED'\n"
            "    except Exception:\n"
            "        found[path] = 'unreadable'\n"
            "result = found\n"
        )

        outcome = run_sandbox_code(root, run_dir, code)

        assert outcome.success is True, outcome.error
        assert all(value == "unreadable" for value in outcome.result.values()), (
            f"存在可读路径：{outcome.result}"
        )


class TestConfigIsNotReturnedToTheModel:
    """模型可见的结果里不能出现私有配置内容。"""

    @pytest.fixture
    def configured(self, tmp_path: Path):
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)
        config, _refs = write_authorizations(
            run_id="leak-run",
            snapshots={"sales.orders": snapshot("sales.orders", db, "orders")},
            workspace_root=tmp_path,
            output_root=tmp_path,
        )
        return tmp_path, db, config

    def test_discover_datasets_has_no_host_paths(self, configured) -> None:
        from mcp_bridge.client import McpStdioClient

        root, db, config = configured

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(TOOL_DISCOVER_DATASETS, {})

        encoded = json.dumps(outcome["result"], ensure_ascii=False)
        assert str(root) not in encoded
        assert str(db) not in encoded
        assert "workspace_root" not in encoded

    def test_get_schema_has_no_host_paths(self, configured) -> None:
        from mcp_bridge.client import McpStdioClient

        root, db, config = configured

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(
                TOOL_GET_SCHEMA, {"dataset_id": "leak-run:sales.orders"}
            )

        encoded = json.dumps(outcome["result"], ensure_ascii=False)
        assert str(root) not in encoded
        assert str(db) not in encoded

    def test_error_messages_do_not_echo_the_config(self, configured) -> None:
        """越权请求的报错也不能把配置内容带出来。"""
        from mcp_bridge.client import McpStdioClient

        _root, _db, config = configured

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(
                TOOL_GET_SCHEMA, {"dataset_id": "other-run:sales.orders"}
            )

        assert outcome["success"] is False
        assert "workspace_root" not in outcome["error"]
        assert "content_hash" not in outcome["error"]

    def test_server_stderr_does_not_print_the_config(self, configured) -> None:
        """诊断日志走 stderr，但也不应把配置内容打出来。"""
        import subprocess
        import sys

        root, db, config = configured

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "mcp_bridge.server",
                "--config",
                str(config),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input="",
            cwd=str(PROJECT_ROOT),
            timeout=120,
        )

        assert str(root) not in completed.stderr
        assert str(db) not in completed.stderr
