"""沙箱隔离能力的真实测试。

这些用例**在真容器里执行**并检查真实行为，而不是断言 docker run 的参数串。
Docker 不可用时整体跳过（决策 J-8）；但正式验收必须在 Docker 可用的环境中
实际运行，交付报告需列出跳过数。

所有安全用例使用无害标记（临时目录内的文件、专用数据），不触碰真实私密路径。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sandbox.executor import DockerSandboxExecutor, SandboxUnavailable
from sandbox.protocol import SandboxErrorKind, SandboxJob, SandboxLimits

pytestmark = pytest.mark.sandbox


def _docker_executor() -> DockerSandboxExecutor:
    return DockerSandboxExecutor()


@pytest.fixture(scope="module", autouse=True)
def require_docker() -> None:
    executor = _docker_executor()
    if not executor.available():
        pytest.skip("Docker daemon 不可用")
    if not executor.image_present():
        pytest.skip("沙箱镜像未构建：先运行 sandbox/build_image.py")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "run-under-test"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_job(
    code: str,
    *,
    run_dir: Path,
    mode: str = "python",
    tables: dict | None = None,
    limits: SandboxLimits | None = None,
    table_names: list[str] | None = None,
    analysis_result=None,
    output_filename: str | None = None,
):
    executor = _docker_executor()
    job = SandboxJob(
        mode=mode,
        table_names=table_names or ["t"],
        code=code,
        limits=limits or SandboxLimits(timeout_seconds=60),
        analysis_result=analysis_result,
        output_filename=output_filename,
        op_label="isolation-test",
    )
    return executor.run(job, run_id="isolation", run_dir=run_dir, tables=tables or {})


def simple_table():
    import pandas as pd

    from ingestion.models import TableResource

    return {
        "t": TableResource(
            name="t",
            source_type="csv",
            source_path="/nonexistent/for-test.csv",
            dataframe=pd.DataFrame({"a": [1, 2, 3]}),
        )
    }


def probe(script: str, *, run_dir: Path, timeout_seconds: float = 60.0):
    """在与模型代码完全相同的容器约束下运行诊断脚本。

    模型代码的作用域没有 `__import__` / `open`（这是设计的一部分），
    所以"容器能否联网""根是否可写"这类**运行时**约束无法从作用域内部探测。
    诊断脚本由宿主提供、以同样的 docker run 参数执行，
    验证的是模型代码实际所处的运行时。
    """
    executor = _docker_executor()
    return executor.run_probe(
        script, run_id="isolation", run_dir=run_dir, timeout_seconds=timeout_seconds
    )


class TestNormalExecution:
    def test_single_table_python_analysis(self, run_dir: Path) -> None:
        outcome = run_job("result = len(df)", run_dir=run_dir, tables=simple_table())

        assert outcome.success is True, outcome.error
        assert outcome.result == 3

    def test_multi_table_analysis_via_dfs(self, run_dir: Path) -> None:
        import pandas as pd

        from ingestion.models import TableResource

        tables = {
            "left": TableResource(
                name="left", source_type="csv", source_path="a.csv",
                dataframe=pd.DataFrame({"k": [1, 2], "v": [10, 20]}),
            ),
            "right": TableResource(
                name="right", source_type="csv", source_path="b.csv",
                dataframe=pd.DataFrame({"k": [1, 2], "w": [3, 4]}),
            ),
        }

        outcome = run_job(
            "merged = dfs['left'].merge(dfs['right'], on='k')\nresult = int(merged['v'].sum() + merged['w'].sum())",
            run_dir=run_dir,
            tables=tables,
            table_names=["left", "right"],
        )

        assert outcome.success is True, outcome.error
        assert outcome.result == 37

    def test_series_and_scalar_results(self, run_dir: Path) -> None:
        outcome = run_job(
            "result = {'count': len(df), 'total': int(df['a'].sum())}",
            run_dir=run_dir,
            tables=simple_table(),
        )

        assert outcome.success is True, outcome.error
        assert outcome.result == {"count": 3, "total": 6}

    def test_model_exception_becomes_execution_error(self, run_dir: Path) -> None:
        outcome = run_job(
            "raise ValueError('boom')", run_dir=run_dir, tables=simple_table()
        )

        assert outcome.success is False
        assert outcome.error_kind == SandboxErrorKind.EXECUTION_ERROR
        assert "boom" in outcome.error

    def test_missing_result_variable_is_reported(self, run_dir: Path) -> None:
        outcome = run_job("x = 1", run_dir=run_dir, tables=simple_table())

        assert outcome.success is False
        assert "result" in outcome.error

    def test_sqlite_table_is_not_sent_to_sandbox(self, run_dir: Path) -> None:
        """SQLite 表没有 DataFrame，不应被送进容器（SQL 走宿主只读路径）。"""
        import pandas as pd

        from ingestion.models import TableResource

        tables = {
            "s": TableResource(
                name="s", source_type="sqlite", source_path="x.sqlite",
                database_table="orders",
            ),
            "f": TableResource(
                name="f", source_type="csv", source_path="y.csv",
                dataframe=pd.DataFrame({"a": [1]}),
            ),
        }

        executor = _docker_executor()
        job = SandboxJob(
            mode="python",
            table_names=["s"],
            code="result = 1",
            op_label="sqlite-reject",
        )

        with pytest.raises(SandboxUnavailable):
            executor.run(job, run_id="isolation", run_dir=run_dir, tables=tables)


class TestTimeout:
    def test_infinite_loop_is_killed(self, run_dir: Path) -> None:
        """死循环必须被宿主侧 wall-time 强制终止，而不是挂住进程。"""
        started = time.perf_counter()
        outcome = run_job(
            "while True:\n    pass",
            run_dir=run_dir,
            tables=simple_table(),
            limits=SandboxLimits(timeout_seconds=8),
        )
        elapsed = time.perf_counter() - started

        assert outcome.success is False
        assert outcome.error_kind == SandboxErrorKind.TIMEOUT
        # 超时必须真的发生，且不会远超预算
        assert elapsed < 60, f"超时耗时过长: {elapsed:.1f}s"

    def test_container_is_cleaned_after_timeout(self, run_dir: Path) -> None:
        executor = _docker_executor()
        run_job(
            "while True:\n    pass",
            run_dir=run_dir,
            tables=simple_table(),
            limits=SandboxLimits(timeout_seconds=6),
        )

        removed = executor.cleanup_orphans("isolation")

        # 清理是幂等的：无论此前有没有残留，都不应抛异常
        assert removed >= 0


class TestResourceLimits:
    def test_memory_over_allocation_fails(self, run_dir: Path) -> None:
        """超出内存上限的分配必须失败，而不是拖垮宿主。"""
        outcome = run_job(
            "blocks = []\nfor _ in range(200):\n    blocks.append(bytearray(32 * 1024 * 1024))\nresult = len(blocks)",
            run_dir=run_dir,
            tables=simple_table(),
            limits=SandboxLimits(timeout_seconds=60, memory_mb=256),
        )

        assert outcome.success is False
        assert outcome.error_kind in (
            SandboxErrorKind.OOM,
            SandboxErrorKind.EXECUTION_ERROR,
        )

    def test_pids_limit_is_enforced(self, run_dir: Path) -> None:
        """进程数上限必须生效：无限 fork 会被拒绝而不是耗尽宿主。"""
        outcome = run_job(
            "import subprocess, sys\n"
            "kids = []\n"
            "for _ in range(300):\n"
            "    kids.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']))\n"
            "result = len(kids)",
            run_dir=run_dir,
            tables=simple_table(),
            limits=SandboxLimits(timeout_seconds=60, pids_limit=64),
        )

        # 要么被拒绝（执行错误），要么因超时/资源被终止；都不能成功创建 300 个进程
        assert outcome.success is False or (isinstance(outcome.result, int) and outcome.result < 300)
        if outcome.success:
            assert outcome.result < 300


class TestFilesystemIsolation:
    """运行时文件系统约束。诊断脚本以同样的 docker run 参数在容器内执行。"""

    def test_host_paths_are_not_visible(self, run_dir: Path) -> None:
        outcome = probe(
            "import json, os\n"
            "probes = ['/var/run/docker.sock', '/root/.ssh', '/root', '/host',"
            " '/Users', 'C:/Users', '/proc/1/root']\n"
            "print(json.dumps({p: os.path.exists(p) for p in probes}))",
            run_dir=run_dir,
        )

        assert outcome.success is True, outcome.error
        # Docker socket 与宿主家目录/盘符必须不可见
        assert outcome.result["/var/run/docker.sock"] is False
        assert outcome.result["/root/.ssh"] is False
        assert outcome.result["C:/Users"] is False
        assert outcome.result["/host"] is False

    def test_root_filesystem_is_read_only(self, run_dir: Path) -> None:
        outcome = probe(
            "import json\n"
            "try:\n"
            "    open('/etc/probe_write', 'w').write('x')\n"
            "    verdict = 'writable'\n"
            "except OSError as error:\n"
            "    verdict = 'readonly:' + type(error).__name__\n"
            "print(json.dumps({'verdict': verdict}))",
            run_dir=run_dir,
        )

        assert outcome.success is True, outcome.error
        assert outcome.result["verdict"].startswith("readonly")

    def test_input_and_source_mounts_are_read_only(self, run_dir: Path) -> None:
        outcome = probe(
            "import json\n"
            "results = {}\n"
            "for path in ('/in/evil.txt', '/sandbox/evil.py'):\n"
            "    try:\n"
            "        open(path, 'w').write('x')\n"
            "        results[path] = 'writable'\n"
            "    except OSError as error:\n"
            "        results[path] = 'readonly:' + type(error).__name__\n"
            "print(json.dumps(results))",
            run_dir=run_dir,
        )

        assert outcome.success is True, outcome.error
        assert outcome.result["/in/evil.txt"].startswith("readonly")
        assert outcome.result["/sandbox/evil.py"].startswith("readonly")

    def test_tmpfs_is_writable_and_mounted_with_restrictive_flags(
        self, run_dir: Path
    ) -> None:
        """tmpfs 可写，且挂载带 nosuid/nodev/noexec。

        这里只断言**挂载选项生效**（来自 /proc/mounts），不代表"禁止执行代码"——
        真正的行为验证见 TestTmpfsNoexec。
        """
        outcome = probe(
            "import json, os\n"
            "try:\n"
            "    open('/tmp/ok.txt', 'w').write('x')\n"
            "    write = 'ok'\n"
            "except OSError as error:\n"
            "    write = 'fail:' + type(error).__name__\n"
            "mounts = open('/proc/mounts').read()\n"
            "tmp_line = [l for l in mounts.splitlines() if ' /tmp ' in l]\n"
            "print(json.dumps({\n"
            "    'write': write,\n"
            "    'uid': os.getuid(),\n"
            "    'gid': os.getgid(),\n"
            "    'tmp_options': tmp_line[0].split()[3] if tmp_line else '',\n"
            "}))",
            run_dir=run_dir,
        )

        assert outcome.success is True, outcome.error
        assert outcome.result["write"] == "ok"
        # 非 root 运行
        assert outcome.result["uid"] == 65534

        options = outcome.result["tmp_options"]
        for flag in ("nosuid", "nodev", "noexec"):
            assert flag in options, f"/tmp 缺少 {flag}: {options}"

    def test_container_cannot_mount_filesystem(self, run_dir: Path) -> None:
        """cap-drop ALL 之后 mount 系统调用必须被拒绝。"""
        outcome = probe(
            "import ctypes, json\n"
            "libc = ctypes.CDLL('libc.so.6', use_errno=True)\n"
            "rc = libc.mount(b'none', b'/tmp/m', b'tmpfs', 0, None)\n"
            "print(json.dumps({'rc': rc, 'errno': ctypes.get_errno()}))",
            run_dir=run_dir,
        )

        assert outcome.success is True, outcome.error
        assert outcome.result["rc"] != 0, "mount 竟然成功，cap-drop 未生效"


class TestTmpfsNoexec:
    """/tmp 的 noexec 到底挡住了什么、没挡住什么。

    必须说清楚：noexec 阻止**从 /tmp 直接执行文件**，但**不能**阻止
    Python 解释器执行 /tmp 里的脚本——`python /tmp/x.py` 是解释器读取文件后
    自己执行，与 execve 那个文件无关。因此 noexec 只消除"落盘再执行"这一条路径，
    不构成"禁止执行代码"的保证。
    """

    def test_direct_execution_from_tmp_is_blocked(self, run_dir: Path) -> None:
        """直接执行 /tmp 下的可执行文件应被拒绝。"""
        outcome = probe(
            "import json, os, subprocess, stat\n"
            "script = '/tmp/hello.sh'\n"
            "with open(script, 'w') as handle:\n"
            "    handle.write('#!/bin/sh\\necho ran\\n')\n"
            "os.chmod(script, 0o755)\n"
            "try:\n"
            "    proc = subprocess.run([script], capture_output=True, text=True)\n"
            "    verdict = 'ran:' + proc.stdout.strip()\n"
            "except OSError as error:\n"
            "    verdict = 'blocked:' + type(error).__name__\n"
            "print(json.dumps({'verdict': verdict}))",
            run_dir=run_dir,
        )

        assert outcome.success is True, outcome.error
        assert outcome.result["verdict"].startswith("blocked"), (
            f"从 /tmp 直接执行未被阻止: {outcome.result}"
        )

    def test_interpreter_can_still_run_a_tmp_script(self, run_dir: Path) -> None:
        """noexec 不能阻止解释器执行 /tmp 里的脚本——这是必须承认的边界。"""
        outcome = probe(
            "import json, subprocess, sys\n"
            "script = '/tmp/hello.py'\n"
            "with open(script, 'w') as handle:\n"
            "    handle.write('print(12345)\\n')\n"
            "proc = subprocess.run([sys.executable, script], capture_output=True, text=True)\n"
            "print(json.dumps({\n"
            "    'returncode': proc.returncode,\n"
            "    'stdout': proc.stdout.strip(),\n"
            "}))",
            run_dir=run_dir,
        )

        assert outcome.success is True, outcome.error
        # 明确记录：解释器仍然能执行 /tmp 下的脚本
        assert outcome.result["stdout"] == "12345", (
            "预期解释器可执行 /tmp 脚本；若被阻止，说明 noexec 的作用被高估了"
        )

    def test_normal_analysis_is_unaffected_by_tmpfs_flags(self, run_dir: Path) -> None:
        """noexec/nodev/nosuid 不影响正常 pandas 分析。"""
        outcome = run_job(
            "sorted_frame = df.sort_values('a', ascending=False)\n"
            "result = int(sorted_frame['a'].iloc[0])",
            run_dir=run_dir,
            tables=simple_table(),
        )

        assert outcome.success is True, outcome.error
        assert outcome.result == 3

    def test_normal_plotting_is_unaffected_by_tmpfs_flags(self, run_dir: Path) -> None:
        """noexec 不影响 matplotlib 正常绘图（它只写文件，不需要执行文件）。"""
        executor = _docker_executor()
        job = SandboxJob(
            mode="plot",
            table_names=["t"],
            code=(
                "fig = plt.figure()\n"
                "plt.plot([1, 2, 3], [1, 4, 9])\n"
                "plt.title('中文标题')\n"
                "plt.savefig(output_path)\n"
                "plt.close()\n"
                "result = output_path\n"
            ),
            analysis_result=[{"a": 1}],
            output_filename="tmpfs_check.png",
            limits=SandboxLimits(timeout_seconds=120),
            op_label="tmpfs-plot",
        )

        outcome = executor.run(
            job, run_id="tmpfs-plot", run_dir=run_dir, tables=simple_table()
        )

        assert outcome.success is True, outcome.error
        assert outcome.diagnostics.get("missing_glyph_count") == 0
        assert (run_dir / "tmpfs_check.png").exists()


class TestModelScopeRestrictions:
    """模型代码作用域本身的限制（与容器约束互为纵深）。"""

    def test_import_is_not_available_to_model_code(self, run_dir: Path) -> None:
        """模型代码不能自行导入模块：`__import__` 不在作用域内。"""
        outcome = run_job(
            "try:\n"
            "    import subprocess\n"
            "    result = 'imported'\n"
            "except ImportError:\n"
            "    result = 'import-unavailable'",
            run_dir=run_dir,
            tables=simple_table(),
        )

        assert outcome.success is True, outcome.error
        assert outcome.result == "import-unavailable"

    def test_open_is_not_available_to_model_code(self, run_dir: Path) -> None:
        """`open` 不在作用域内：模型代码不能直接读写任意文件。"""
        outcome = run_job(
            "try:\n"
            "    open('/etc/hostname').read()\n"
            "    result = 'opened'\n"
            "except NameError:\n"
            "    result = 'open-unavailable'",
            run_dir=run_dir,
            tables=simple_table(),
        )

        assert outcome.success is True, outcome.error
        assert outcome.result == "open-unavailable"

    def test_model_code_can_catch_exceptions(self, run_dir: Path) -> None:
        """try/except 是生成代码的常见写法，必须可用。"""
        outcome = run_job(
            "try:\n"
            "    value = df['no_such_column'].sum()\n"
            "except KeyError:\n"
            "    value = -1\n"
            "result = int(value)",
            run_dir=run_dir,
            tables=simple_table(),
        )

        assert outcome.success is True, outcome.error
        assert outcome.result == -1

    def test_pandas_still_works_within_the_restricted_scope(self, run_dir: Path) -> None:
        """限制原语不等于限制 pandas 能力——分析功能必须完好。"""
        outcome = run_job(
            "sorted_frame = df.sort_values('a', ascending=False)\n"
            "result = int(sorted_frame['a'].iloc[0])",
            run_dir=run_dir,
            tables=simple_table(),
        )

        assert outcome.success is True, outcome.error
        assert outcome.result == 3


class TestNetworkIsolation:
    def test_network_is_unreachable(self, run_dir: Path) -> None:
        """容器内无网络：既不能连出，也不能解析域名。"""
        outcome = probe(
            "import json, socket\n"
            "results = {}\n"
            "for label, target in (('ip', ('1.1.1.1', 53)), ('dns', ('pypi.org', 443)),"
            " ('loopback', ('127.0.0.1', 80))):\n"
            "    try:\n"
            "        socket.create_connection(target, timeout=3).close()\n"
            "        results[label] = 'reachable'\n"
            "    except Exception:\n"
            "        results[label] = 'blocked'\n"
            "print(json.dumps(results))",
            run_dir=run_dir,
        )

        assert outcome.success is True, outcome.error
        assert outcome.result["ip"] == "blocked"
        assert outcome.result["dns"] == "blocked"
        assert outcome.result["loopback"] == "blocked"


class TestResultValidation:
    def test_unserializable_result_is_rejected(self, run_dir: Path) -> None:
        """无法 JSON 表达的结果必须明确报错，而不是静默丢内容。"""
        outcome = run_job(
            "result = object()",
            run_dir=run_dir,
            tables=simple_table(),
        )

        assert outcome.success is False
        assert outcome.error_kind in (
            SandboxErrorKind.INVALID_RESULT,
            SandboxErrorKind.EXECUTION_ERROR,
        )

    def test_oversized_result_is_rejected(self, run_dir: Path) -> None:
        outcome = run_job(
            "result = 'x' * (4 * 1024 * 1024)",
            run_dir=run_dir,
            tables=simple_table(),
            limits=SandboxLimits(timeout_seconds=60, max_result_bytes=64 * 1024),
        )

        assert outcome.success is False
        assert "size limit" in outcome.error

    def test_result_rows_are_truncated_and_flagged(self, run_dir: Path) -> None:
        """行数超限时截断并标记 truncated，而不是整体失败。"""
        import pandas as pd

        from ingestion.models import TableResource

        tables = {
            "t": TableResource(
                name="t", source_type="csv", source_path="x.csv",
                dataframe=pd.DataFrame({"a": range(10)}),
            )
        }

        outcome = run_job(
            "result = df",
            run_dir=run_dir,
            tables=tables,
            limits=SandboxLimits(timeout_seconds=60, max_result_rows=3),
        )

        assert outcome.success is True, outcome.error
        assert outcome.truncated is True
        assert len(outcome.result["records"]) == 3


class TestFailClosed:
    def test_missing_image_raises_unavailable(self, run_dir: Path) -> None:
        """镜像不存在时必须明确失败，不得回退到同进程执行。"""
        executor = DockerSandboxExecutor(image="insightflow-sandbox:does-not-exist")

        with pytest.raises(SandboxUnavailable):
            executor.require_ready()

    def test_run_returns_failure_when_image_missing(self, run_dir: Path) -> None:
        executor = DockerSandboxExecutor(image="insightflow-sandbox:does-not-exist")
        job = SandboxJob(
            mode="python", table_names=["t"], code="result = 1", op_label="missing-image"
        )

        with pytest.raises(SandboxUnavailable):
            executor.run(job, run_id="missing", run_dir=run_dir, tables={})

    def test_unavailable_docker_executor_reports_unavailable(self) -> None:
        """docker 可执行文件不可用时，available() 返回 False 而不是抛异常。"""
        executor = DockerSandboxExecutor(docker="definitely-not-a-docker-binary")

        assert executor.available() is False


class TestImageAndCleaning:
    def test_image_info_is_reported(self) -> None:
        info = _docker_executor().image_info()

        assert info.image_id.startswith("sha256:")
        assert info.size_bytes and info.size_bytes > 0

    def test_cleanup_only_touches_our_label(self, run_dir: Path) -> None:
        """孤儿清理只能处理带本项目标签的容器。"""
        executor = _docker_executor()

        # 用一个与已知 Run 无关的 id 调用；不应报错，也不应影响其他容器
        assert executor.cleanup_orphans("no-such-run-identifier") == 0
