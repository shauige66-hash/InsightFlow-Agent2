"""P2-C 补验收：私有配置目录与 Run ID 的命名空间冲突。

背景（真实可复现的跨 Run 泄漏）：

原本私有配置放在 `<output_root>/mcp_private/<run_id>/authorizations.json`，
而 P2-A 把 `<output_root>/<run_id>` 只读挂载为 `/history`。
`api_runs.RUN_ID_PATTERN` 合法接受 `mcp_private` 这个名字，于是当某个 Run 的
`run_id` 就是 `mcp_private` 时：

    该 Run 的 run 目录  ==  <output_root>/mcp_private  ==  私有配置根目录

`/history` 因此会列出**所有** Run 的私有配置，模型代码用一个 pandas 就能读到
别的 Run 的宿主路径、dataset_id 与内容指纹。

修复：私有根目录改为 `<output_root>.mcp_private`，与 output_root **平级**，
从文件系统结构上消除重叠；并拒绝任何被显式配置到 output_root 之内的私有路径。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from core.llm import build_model_factory
from core.runtime import McpSettings, PrivatePathConflict, RunRuntime, use_runtime
from ingestion.models import TableResource
from ingestion.resources import ResourceResolver
from mcp_bridge.registry import (
    PRIVATE_ROOT_SUFFIX,
    ensure_private_root_outside,
    mcp_private_root,
    write_authorizations,
)
from sandbox.executor import DockerSandboxExecutor
from sandbox.protocol import SandboxJob, SandboxLimits

# 与 api_runs.RUN_ID_PATTERN 一致的"合法但危险"的 run_id
RESERVED_LOOKING_RUN_IDS = (
    "mcp_private",
    "MCP_PRIVATE",
    "Mcp_Private",
)


def build_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE orders (order_id INTEGER, amount REAL)")
        connection.execute("INSERT INTO orders VALUES (1, 10.0)")
        connection.commit()
    finally:
        connection.close()


def snapshot(name: str, path: Path) -> dict:
    return {
        "name": name,
        "source_type": "sqlite",
        "source_path": str(path),
        "content_hash": f"hash-{name}",
        "file_size": path.stat().st_size,
        "locator": "orders",
        "database_table": "orders",
    }


def runtime_for(tmp_path: Path, run_id: str) -> RunRuntime:
    return RunRuntime(
        run_id=run_id,
        model_factory=build_model_factory(run_id),
        resolver=ResourceResolver(),
        output_root=tmp_path,
        mcp=McpSettings(mode="mcp"),
    )


class TestPrivateRootIsOutsideOutputRoot:
    def test_default_private_root_is_a_sibling(self, tmp_path: Path) -> None:
        runtime = runtime_for(tmp_path, "run_a")

        private_root = runtime.mcp_private_root()
        output_root = runtime.output_root

        assert private_root.parent == output_root.parent
        assert private_root.name == output_root.name + PRIVATE_ROOT_SUFFIX
        assert not private_root.is_relative_to(output_root)

    def test_registry_default_matches_runtime(self, tmp_path: Path) -> None:
        runtime = runtime_for(tmp_path, "run_a")

        assert runtime.mcp_private_root().name == mcp_private_root(tmp_path).name

    @pytest.mark.parametrize("run_id", RESERVED_LOOKING_RUN_IDS)
    def test_special_run_id_cannot_collide_with_private_root(
        self, tmp_path: Path, run_id: str
    ) -> None:
        """run_id 叫 mcp_private（任意大小写）都不能与私有根目录重合。"""
        runtime = runtime_for(tmp_path, run_id)

        run_dir = runtime.output_dir()
        private_root = runtime.mcp_private_root()

        # 关键：run 目录（会被挂载为 /history）不等于私有根目录
        assert run_dir != private_root
        assert not run_dir.is_relative_to(private_root)

        config = runtime.mcp_config_path()
        assert not config.is_relative_to(run_dir), (
            f"配置落进了会被挂载的 run 目录：{config}"
        )

    @pytest.mark.parametrize("run_id", RESERVED_LOOKING_RUN_IDS)
    def test_config_path_stays_outside_every_run_directory(
        self, tmp_path: Path, run_id: str
    ) -> None:
        runtime = runtime_for(tmp_path, run_id)
        config = runtime.mcp_config_path()

        # 任何已存在的 Run 目录都不应包含该配置
        for candidate in runtime.output_root.iterdir() if runtime.output_root.exists() else []:
            assert not config.is_relative_to(candidate)

    def test_explicit_private_root_inside_output_root_is_refused(
        self, tmp_path: Path
    ) -> None:
        """显式配置到 output_root 之内必须被拒绝——否则重叠问题原样回来。"""
        runtime = RunRuntime(
            run_id="run_a",
            model_factory=build_model_factory("run_a"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            mcp=McpSettings(mode="mcp", private_root=tmp_path / "mcp_private"),
        )

        with pytest.raises(PrivatePathConflict):
            runtime.mcp_private_root()

    def test_explicit_private_root_case_alias_is_refused(self, tmp_path: Path) -> None:
        """大小写不同的别名也要被拒绝（Windows 路径大小写不敏感）。"""
        alias = Path(str(tmp_path).upper()) / "mcp_private"
        runtime = RunRuntime(
            run_id="run_a",
            model_factory=build_model_factory("run_a"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            mcp=McpSettings(mode="mcp", private_root=alias),
        )

        with pytest.raises(PrivatePathConflict):
            runtime.mcp_private_root()

    def test_explicit_private_root_equal_to_output_root_is_refused(
        self, tmp_path: Path
    ) -> None:
        runtime = RunRuntime(
            run_id="run_a",
            model_factory=build_model_factory("run_a"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            mcp=McpSettings(mode="mcp", private_root=tmp_path),
        )

        with pytest.raises(PrivatePathConflict):
            runtime.mcp_private_root()

    def test_registry_guard_rejects_inside_paths(self, tmp_path: Path) -> None:
        with pytest.raises(PermissionError):
            ensure_private_root_outside(tmp_path / "mcp_private", tmp_path)

    def test_registry_guard_allows_sibling_paths(self, tmp_path: Path) -> None:
        sibling = tmp_path.parent / "elsewhere.mcp_private"

        assert ensure_private_root_outside(sibling, tmp_path) == sibling

    def test_traversal_alias_is_refused(self, tmp_path: Path) -> None:
        """用 .. 拼出来的等价路径也要被识别为"在 output_root 之内"。"""
        sneaky = tmp_path / ".." / tmp_path.name / "mcp_private"
        runtime = RunRuntime(
            run_id="run_a",
            model_factory=build_model_factory("run_a"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            mcp=McpSettings(mode="mcp", private_root=sneaky),
        )

        with pytest.raises(PrivatePathConflict):
            runtime.mcp_private_root()


class TestTwoRunsStayIsolated:
    def test_two_runs_write_to_distinct_paths(self, tmp_path: Path) -> None:
        configs = {}
        for name in ("run_alpha", "run_beta"):
            db = tmp_path / f"{name}.sqlite"
            build_sqlite(db)
            config, _refs = write_authorizations(
                run_id=name,
                snapshots={f"{name}.orders": snapshot(f"{name}.orders", db)},
                workspace_root=tmp_path,
                output_root=tmp_path,
            )
            configs[name] = config

        assert configs["run_alpha"] != configs["run_beta"]
        assert configs["run_alpha"].parent.name == "run_alpha"
        assert configs["run_beta"].parent.name == "run_beta"
        # 各自的 dataset_id 只属于自己
        for name, config in configs.items():
            payload = json.loads(config.read_text(encoding="utf-8"))
            ids = [item["dataset_id"] for item in payload["datasets"]]
            assert all(item.startswith(f"{name}:") for item in ids)

    def test_private_subtree_is_not_inside_any_output_root_subdir(
        self, tmp_path: Path
    ) -> None:
        """私有树与 output_root 的目录树完全不重叠。"""
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)
        write_authorizations(
            run_id="run_alpha",
            snapshots={"sales.orders": snapshot("sales.orders", db)},
            workspace_root=tmp_path,
            output_root=tmp_path,
        )

        private_root = mcp_private_root(tmp_path)

        assert private_root.parent == tmp_path.parent
        # output_root 里不存在任何私有配置
        for path in tmp_path.rglob("authorizations.json"):
            pytest.fail(f"output_root 内出现私有配置：{path}")


@pytest.mark.sandbox
class TestSandboxCannotReachOtherRuns:
    """真实容器：伪造 run_id=mcp_private 也读不到别的 Run 的配置。"""

    READ_CODE = (
        "found = {}\n"
        "for target in ['/history/run_alpha/authorizations.json',\n"
        "               '/history/run_beta/authorizations.json',\n"
        "               '/history/mcp_private/run_alpha/authorizations.json']:\n"
        "    try:\n"
        "        frame = pd.read_json(target, typ='series')\n"
        "        found[target] = 'LEAKED: ' + str(frame)[:120]\n"
        "    except Exception:\n"
        "        found[target] = 'unreadable'\n"
        "result = found\n"
    )

    def _prepare(self, tmp_path: Path) -> None:
        for name in ("run_alpha", "run_beta"):
            db = tmp_path / f"{name}.sqlite"
            build_sqlite(db)
            write_authorizations(
                run_id=name,
                snapshots={f"{name}.orders": snapshot(f"{name}.orders", db)},
                workspace_root=tmp_path,
                output_root=tmp_path,
            )

    def _run_sandbox(self, tmp_path: Path, run_id: str):
        runtime = runtime_for(tmp_path, run_id)
        run_dir = runtime.output_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

        tables = {
            "t": TableResource(
                name="t",
                source_type="csv",
                source_path=str(tmp_path / "x.csv"),
                dataframe=pd.DataFrame({"a": [1, 2, 3]}),
            )
        }
        job = SandboxJob(
            mode="python",
            table_names=["t"],
            code=self.READ_CODE,
            limits=SandboxLimits(timeout_seconds=60),
            op_label="overlap-check",
        )
        with use_runtime(runtime):
            return DockerSandboxExecutor().run(
                job, run_id=run_id, run_dir=run_dir, tables=tables
            )

    def test_new_layout_hides_other_runs(self, tmp_path: Path) -> None:
        self._prepare(tmp_path)

        # run_id 就叫 mcp_private：它的 run 目录就是 output_root/mcp_private
        outcome = self._run_sandbox(tmp_path, "mcp_private")

        assert outcome.success is True, outcome.error
        assert all(
            value == "unreadable" for value in outcome.result.values()
        ), f"仍能读到其他 Run 的配置：{outcome.result}"

    def test_old_layout_would_have_leaked(self, tmp_path: Path) -> None:
        """对照：把私有树放回 output_root 之内，同一个容器确实读到它。

        这条用例证明上面那条不是因为"路径写错"而通过。

        注意不能用 `write_authorizations` 构造旧布局——新的守卫会直接拒绝。
        因此这里**手工**把等价内容写到旧位置，模拟修复前的文件系统状态：
        这正是"如果当时没加守卫会怎样"的对照。
        """
        legacy_root = tmp_path / "mcp_private"
        for name in ("run_alpha", "run_beta"):
            db = tmp_path / f"{name}.sqlite"
            build_sqlite(db)

            legacy_dir = legacy_root / name
            legacy_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dir / "authorizations.json").write_text(
                json.dumps(
                    {
                        "run_id": name,
                        "workspace_root": str(tmp_path),
                        "datasets": [
                            {
                                "dataset_id": f"{name}:{name}.orders",
                                "name": f"{name}.orders",
                                "source_type": "sqlite",
                                "source_path": str(db),
                                "locator": "orders",
                                "content_hash": f"hash-{name}",
                                "database_table": "orders",
                                "columns": [],
                                "row_count": None,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        outcome = self._run_sandbox(tmp_path, "mcp_private")

        assert outcome.success is True, outcome.error
        leaked = [
            key
            for key, value in outcome.result.items()
            if isinstance(value, str) and value.startswith("LEAKED")
        ]
        assert leaked, f"对照用例未复现旧布局的泄漏：{outcome.result}"

    def test_normal_run_still_cannot_read_private_root(self, tmp_path: Path) -> None:
        """普通 Run 的 /history 也不含私有配置。"""
        self._prepare(tmp_path)

        outcome = self._run_sandbox(tmp_path, "run_normal")

        assert outcome.success is True, outcome.error
        assert all(value == "unreadable" for value in outcome.result.values())
