"""宿主侧沙箱执行器。

一次工具调用 = 一个一次性 Docker 容器（P2-A 决策 J-2）。

安全相关的约束全部在**宿主侧**施加，容器内的代码无法放宽：
- 参数以列表传递，绝不使用 shell 字符串拼接；
- 只读挂载源码与输入，可写挂载仅限本次 Run 的输出目录；
- 不传宿主环境变量，不挂 Docker socket，不挂宿主家目录；
- 宿主侧 wall-time 超时，超时后强制 kill 并清理容器。

镜像缺失或 Docker 不可用时抛 `SandboxUnavailable`，调用方据此 fail-closed，
**绝不回退到同进程 exec**。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from core.publication import publish_file

from sandbox.protocol import (
    SandboxErrorKind,
    SandboxJob,
    SandboxLimits,
    SandboxOutcome,
    encode_dataframe,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 所有本项目的沙箱容器都带这个标签前缀，孤儿清理只认它，
# 绝不触碰其他 Docker 服务（包括 ModelGate 的容器）。
CONTAINER_LABEL = "insightflow.sandbox"
CONTAINER_NAME_PREFIX = "iflow-sbx-"

# /tmp 的挂载选项。nosuid/nodev 恒开。
#
# noexec 的**能力边界**必须说清：它阻止从 /tmp 执行文件，但**不能**阻止
# Python 解释器执行 /tmp 里的脚本（`python /tmp/x.py` 是读取文件后自己执行，
# 与 execve 那个文件无关）。因此 noexec 只用于消除"落盘再执行"这一条路径，
# 不构成"禁止执行代码"的保证。
_TMPFS_OPTIONS = ",nosuid,nodev,noexec"

# PNG 头与尺寸校验：宿主发布产物前必须确认它是真实图片，而不是任意字节
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_SHA_PREFIX = 8
MAX_PROCESS_OUTPUT_BYTES = 1024 * 1024


class SandboxLogLimitExceeded(subprocess.SubprocessError):
    """The host stopped collecting an unbounded container log stream."""


class SandboxUnavailable(RuntimeError):
    """Docker 不可用、镜像缺失或容器未能启动。

    属于环境类错误：重试没有意义，调用方应把它当作不可恢复的失败上报。
    """


class SandboxOutputRejected(RuntimeError):
    """staging 中的产物未通过宿主校验，拒绝发布到 Run 路径。"""


def _png_dimensions(path: Path) -> tuple[int, int]:
    """从 PNG 的 IHDR 块读取宽高。

    只解析固定偏移，不依赖图像库：宿主侧校验应当尽量少依赖。
    """
    import struct

    header = path.read_bytes()[:24]
    if len(header) < 24 or not header.startswith(_PNG_MAGIC):
        raise SandboxOutputRejected("not a PNG file")
    return struct.unpack(">II", header[16:24])


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_SHA_PREFIX]


# 真正启动过的容器计数（单调递增）。
#
# 用途只有一个：让测试报告能**实测**区分"确实进入了容器"与"只走了故障/静态
# 分支"的用例，而不是靠人工标注。为此在两次 docker run 的唯一出口自增。
_containers_spawned = 0


def containers_spawned() -> int:
    """返回本进程内实际发起的容器执行次数。"""
    return _containers_spawned


def _count_container_spawn() -> None:
    global _containers_spawned
    _containers_spawned += 1


@dataclass(frozen=True)
class SandboxImageInfo:
    """镜像身份，用于交付报告与可追溯性。"""

    reference: str
    image_id: str
    digest: str | None
    size_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "image_id": self.image_id,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }


class DockerSandboxExecutor:
    """在受限容器中执行 Python 分析或绘图代码。"""

    # 进程级共享配额。
    #
    # 为什么不是实例字段：工具实例可能被按数据集指纹缓存出多个，
    # 每个实例各持一个信号量就无法真正封顶宿主上的容器总数。
    # 把配额放在类上，使"本进程最多同时 N 个沙箱容器"成为全局约束。
    _global_slots: dict[int, threading.BoundedSemaphore] = {}
    _slots_lock = threading.Lock()

    def __init__(
        self,
        *,
        image: str = "insightflow-sandbox:py311",
        docker: str | None = None,
        max_concurrent_containers: int = 2,
        sandbox_source_dir: Path | None = None,
    ) -> None:
        self.image = image
        self._docker = docker or shutil.which("docker") or "docker"
        self._source_dir = sandbox_source_dir or PROJECT_ROOT / "sandbox"
        self._max_concurrent = max_concurrent_containers

        with DockerSandboxExecutor._slots_lock:
            slots = DockerSandboxExecutor._global_slots.get(max_concurrent_containers)
            if slots is None:
                slots = threading.BoundedSemaphore(max_concurrent_containers)
                DockerSandboxExecutor._global_slots[max_concurrent_containers] = slots
        self._slots = slots

        self._image_cache: SandboxImageInfo | None = None

    # ---------- 环境与镜像 ----------

    def _run_docker(
        self, args: list[str], *, timeout: float = 30.0
    ) -> subprocess.CompletedProcess:
        command = [self._docker, *args]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        buffers = [bytearray(), bytearray()]
        exceeded = threading.Event()

        def drain(stream, target):
            try:
                while chunk := stream.read(64 * 1024):
                    remaining = MAX_PROCESS_OUTPUT_BYTES - len(target)
                    target.extend(chunk[:max(0, remaining)])
                    if len(chunk) > remaining:
                        exceeded.set()
                        try:
                            process.kill()
                        except OSError:
                            pass
            finally:
                stream.close()

        readers = [threading.Thread(target=drain, args=(stream, target), daemon=True)
                   for stream, target in zip((process.stdout, process.stderr), buffers)]
        for reader in readers:
            reader.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            for reader in readers:
                reader.join(timeout=2)
        if exceeded.is_set():
            raise SandboxLogLimitExceeded("sandbox log output exceeded host limit")
        return subprocess.CompletedProcess(command, returncode, *(bytes(b).decode("utf-8", errors="replace") for b in buffers))

    def _cleanup_container(self, container_name: str) -> None:
        try:
            self._run_docker(["rm", "-f", container_name], timeout=15)
        except (OSError, subprocess.SubprocessError):
            pass

    def available(self) -> bool:
        """Docker daemon 是否可用。用于 health 与测试门控。"""
        try:
            result = self._run_docker(["info", "--format", "{{.ServerVersion}}"])
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def image_present(self) -> bool:
        """镜像是否已在本地。

        运行期禁止隐式拉取，因此这里只做存在性检查，不触发 pull。
        """
        try:
            result = self._run_docker(["image", "inspect", self.image])
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def image_info(self) -> SandboxImageInfo:
        """读取镜像身份。失败时抛 `SandboxUnavailable`。"""
        if self._image_cache is not None:
            return self._image_cache

        try:
            result = self._run_docker(
                ["image", "inspect", self.image, "--format", "{{json .}}"]
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxUnavailable(f"docker inspect failed: {error}") from error

        if result.returncode != 0:
            raise SandboxUnavailable(
                f"sandbox image not present locally: {self.image}"
            )

        import json as _json

        data = _json.loads(result.stdout.strip())
        digests = data.get("RepoDigests") or []
        info = SandboxImageInfo(
            reference=self.image,
            image_id=data.get("Id", ""),
            digest=digests[0] if digests else None,
            size_bytes=data.get("Size"),
        )
        self._image_cache = info
        return info

    def require_ready(self) -> SandboxImageInfo:
        """执行前的前置检查：daemon 可用且镜像存在。"""
        if not self.available():
            raise SandboxUnavailable("docker daemon is not available")
        return self.image_info()

    # ---------- 孤儿清理 ----------

    def cleanup_orphans(self, run_id: str) -> int:
        """强制移除属于指定 Run 的残留容器。

        只按本项目的标签 + run 短哈希匹配，因此不会影响其他 Docker 服务。
        """
        prefix = f"{CONTAINER_NAME_PREFIX}{_short_hash(run_id)}-"
        try:
            listed = self._run_docker(
                [
                    "ps",
                    "-aq",
                    "--filter",
                    f"label={CONTAINER_LABEL}=1",
                    "--filter",
                    f"name={prefix}",
                ]
            )
        except (OSError, subprocess.SubprocessError):
            return 0

        ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
        if not ids:
            return 0

        try:
            self._run_docker(["rm", "-f", *ids], timeout=60.0)
        except (OSError, subprocess.SubprocessError):
            return 0

        return len(ids)

    # ---------- 执行 ----------

    def run(
        self,
        job: SandboxJob,
        *,
        run_id: str,
        run_dir: Path,
        tables: dict[str, Any] | None = None,
        container_limits: dict[str, Any] | None = None,
    ) -> SandboxOutcome:
        """执行一次沙箱调用。

        `run_dir` 必须是本次 Run 的独立目录；宿主在其中建立 in/out/scratch
        三个子目录并分别挂载。
        """
        started = time.perf_counter()
        image_info = self.require_ready()

        limits = job.limits
        container_limits = container_limits or {}

        # A retry is a new attempt; its scratch files can never be replayed as
        # another attempt's successful result.
        job_root = run_dir / "sandbox" / f"{_short_hash(job.op_label)}-{uuid.uuid4().hex}"
        in_dir = job_root / "in"
        scratch_dir = job_root / "scratch"
        # 容器**唯一**可写的位置。
        #
        # 早先整份 Run 目录都是可写的，实测可以这样篡改历史产物：
        #     df.to_csv('/out/analysis_report.md')      # 覆盖已有报告
        #     df.to_csv('/out/evil_new.csv')            # 凭空新建文件
        # 作用域里没有 open/import 并不能阻止这一点——pandas / matplotlib
        # 自己的方法就会写文件。因此边界必须落在**挂载权限**上：
        # 容器只写 staging，宿主校验后再发布到稳定的 Run 路径。
        staging_dir = job_root / "staging"

        for directory in (in_dir, scratch_dir, staging_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            for directory in (scratch_dir, staging_dir):
                directory.chmod(0o777)  # container UID 65534, private attempt dir

        self._write_inputs(job, in_dir, tables or {})

        container_name = self._container_name(run_id, job.op_label)

        # 并发配额：拿不到就 fail-closed，不无限排队
        acquired = self._slots.acquire(timeout=limits.concurrency_wait_seconds)
        if not acquired:
            return SandboxOutcome(
                success=False,
                result=None,
                error=(
                    "sandbox unavailable: container concurrency limit "
                    f"({self._max_concurrent}) not available within "
                    f"{limits.concurrency_wait_seconds:.0f}s"
                ),
                error_kind=SandboxErrorKind.UNAVAILABLE,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        stop_monitor = threading.Event()
        volume_exceeded = threading.Event()

        def monitor_outputs():
            while not stop_monitor.wait(0.05):
                try:
                    if not self._workspace_within_limits((scratch_dir, staging_dir), limits):
                        volume_exceeded.set()
                        self._cleanup_container(container_name)
                        return
                except OSError:
                    volume_exceeded.set()
                    self._cleanup_container(container_name)
                    return

        monitor = threading.Thread(target=monitor_outputs, daemon=True)
        monitor.start()
        try:
            completed = self._run_container(
                container_name=container_name,
                run_dir=run_dir,
                in_dir=in_dir,
                staging_dir=staging_dir,
                scratch_dir=scratch_dir,
                limits=limits,
            )
        except subprocess.TimeoutExpired:
            self._cleanup_container(container_name)
            return SandboxOutcome(
                success=False,
                result=None,
                error=f"sandbox timeout after {limits.timeout_seconds:.0f}s",
                error_kind=SandboxErrorKind.TIMEOUT,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except SandboxLogLimitExceeded as error:
            self._cleanup_container(container_name)
            return SandboxOutcome(success=False, result=None, error=str(error), error_kind=SandboxErrorKind.OUTPUT_LIMIT)
        finally:
            stop_monitor.set()
            monitor.join(timeout=16)
            self._slots.release()

        duration_ms = int((time.perf_counter() - started) * 1000)
        if volume_exceeded.is_set() or not self._workspace_within_limits((scratch_dir, staging_dir), limits):
            return SandboxOutcome(success=False, result=None, error="sandbox writable output limit exceeded", error_kind=SandboxErrorKind.OUTPUT_LIMIT, duration_ms=duration_ms)
        outcome = self._interpret(completed, scratch_dir, image_info, limits=limits, expected_attempt=job_root.name)
        outcome.duration_ms = duration_ms

        # 产物只在容器返回成功后才发布：宿主先校验 staging 里的文件，
        # 再落到稳定的 Run 路径，避免容器直接写历史产物。
        if outcome.success and job.output_filename:
            try:
                published = self._publish_output(
                    staging_dir=staging_dir,
                    run_dir=run_dir,
                    filename=job.output_filename,
                    limits=limits,
                )
            except SandboxOutputRejected as error:
                return SandboxOutcome(
                    success=False,
                    result=None,
                    error=f"invalid_result: {error}",
                    error_kind=SandboxErrorKind.INVALID_RESULT,
                    duration_ms=duration_ms,
                )

            outcome.result = str(published)
            outcome.diagnostics["published_path"] = str(published)

        return outcome

    @staticmethod
    def _workspace_within_limits(directories, limits) -> bool:
        total = count = 0
        for directory in directories:
            for path in directory.rglob("*"):
                count += 1
                if count > limits.max_output_files or path.is_symlink():
                    return False
                if path.is_file():
                    total += path.stat().st_size
                    if total > limits.max_workspace_bytes:
                        return False
        return True

    def _publish_output(
        self,
        *,
        staging_dir: Path,
        run_dir: Path,
        filename: str,
        limits: Any,
    ) -> Path:
        """校验 staging 中的产物并发布到 Run 的稳定路径。

        校验项：文件名合法、不是符号链接、是普通文件、大小在上限内、
        PNG 头与尺寸合法。任何一项不过就拒绝，不发布、不覆盖既有产物。
        """
        staged = staging_dir / filename

        if not staged.exists():
            raise SandboxOutputRejected(f"staged output missing: {filename}")

        # 符号链接检查必须在读取之前：否则会顺着链接读到挂载外的内容
        if staged.is_symlink():
            raise SandboxOutputRejected(f"staged output is a symlink: {filename}")

        if not staged.is_file():
            raise SandboxOutputRejected(f"staged output is not a regular file: {filename}")

        size = staged.stat().st_size
        if size == 0:
            raise SandboxOutputRejected(f"staged output is empty: {filename}")
        if size > limits.max_output_file_bytes:
            raise SandboxOutputRejected(
                f"staged output too large: {size} > {limits.max_output_file_bytes}"
            )

        header = staged.read_bytes()[:8]
        if not header.startswith(_PNG_MAGIC):
            raise SandboxOutputRejected(f"staged output is not a PNG: {filename}")

        width, height = _png_dimensions(staged)
        if width <= 0 or height <= 0:
            raise SandboxOutputRejected(
                f"staged PNG has invalid dimensions: {width}x{height}"
            )

        destination = run_dir / filename
        # replace replaces a destination symlink itself; it does not follow it.
        # Re-check the execution lease immediately before the atomic publish.
        publish_file(staged, destination)
        return destination

    def _container_name(self, run_id: str, op_label: str) -> str:
        return f"{CONTAINER_NAME_PREFIX}{_short_hash(run_id)}-{_short_hash(op_label)}-{uuid.uuid4().hex[:8]}"

    def _write_inputs(
        self, job: SandboxJob, in_dir: Path, tables: dict[str, Any]
    ) -> None:
        """写出 job.json 与本次调用需要的表。

        只写 `job.table_names` 里实际用到的表——数据集按需传入，不是全量复制。
        """
        data_dir = in_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        dfs_paths: dict[str, str] = {}

        for name in job.table_names:
            if name not in tables:
                raise SandboxUnavailable(f"table not resolvable for sandbox: {name}")

            resource = tables[name]
            frame = getattr(resource, "dataframe", None)

            if frame is None:
                raise SandboxUnavailable(
                    f"table has no DataFrame (sqlite tables are not sent to sandbox): {name}"
                )

            filename = f"data/{_short_hash(name)}.json"
            write_json(encode_dataframe(frame), in_dir / filename)
            dfs_paths[name] = filename

        payload = job.to_dict(dfs_paths)
        payload["attempt_id"] = in_dir.parent.name
        write_json(payload, in_dir / "job.json")

        # 代码单独落文件：容器从只读挂载读取，不经过 shell，也不内联进 JSON
        (in_dir / "code.py").write_text(job.code, encoding="utf-8")

    def run_probe(
        self,
        script: str,
        *,
        run_id: str,
        run_dir: Path,
        timeout_seconds: float = 60.0,
    ) -> SandboxOutcome:
        """在与模型代码**完全相同**的容器约束下运行一段诊断脚本。

        用途：模型代码的执行作用域没有 `__import__`、`open` 等原语，因此
        "容器能否联网""根文件系统是否可写"这类运行时约束无法从作用域内部探测。
        这里用同一套 `docker run` 参数执行宿主自带的诊断脚本，
        验证的是模型代码实际所处的运行时，而不是断言参数串。
        """
        image_info = self.require_ready()

        limits = SandboxLimits(timeout_seconds=timeout_seconds)
        job_root = run_dir / "probe" / _short_hash(script[:64] + str(time.time()))
        in_dir = job_root / "in"
        staging_dir = job_root / "staging"
        scratch_dir = job_root / "scratch"
        for directory in (in_dir, staging_dir, scratch_dir):
            directory.mkdir(parents=True, exist_ok=True)

        (in_dir / "probe.py").write_text(script, encoding="utf-8")

        container_name = self._container_name(run_id, f"probe-{_short_hash(script[:32])}")
        started = time.perf_counter()

        acquired = self._slots.acquire(timeout=limits.concurrency_wait_seconds)
        if not acquired:
            return SandboxOutcome(
                success=False,
                result=None,
                error="sandbox unavailable: container concurrency limit reached",
                error_kind=SandboxErrorKind.UNAVAILABLE,
            )

        try:
            _count_container_spawn()
            completed = self._run_docker(
                self._container_args(
                    container_name=container_name,
                    run_dir=run_dir,
                    in_dir=in_dir,
                    staging_dir=staging_dir,
                    scratch_dir=scratch_dir,
                    limits=limits,
                    entrypoint=["python", "/in/probe.py"],
                ),
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self.cleanup_orphans(run_id)
            return SandboxOutcome(
                success=False,
                result=None,
                error=f"probe timeout after {timeout_seconds:.0f}s",
                error_kind=SandboxErrorKind.TIMEOUT,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        finally:
            self._slots.release()

        duration_ms = int((time.perf_counter() - started) * 1000)

        # 诊断脚本把结论以 JSON 写到 stdout
        import json as _json

        try:
            payload = _json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            payload = None

        return SandboxOutcome(
            success=completed.returncode == 0 and payload is not None,
            result=payload,
            error=(
                None
                if completed.returncode == 0 and payload is not None
                else (completed.stderr or "")[-400:].strip()
                or f"probe exited with code {completed.returncode}"
            ),
            error_kind=None if completed.returncode == 0 else SandboxErrorKind.EXECUTION_ERROR,
            duration_ms=duration_ms,
            container_id=image_info.image_id,
        )

    def _run_container(
        self,
        *,
        container_name: str,
        run_dir: Path,
        in_dir: Path,
        staging_dir: Path,
        scratch_dir: Path,
        limits: Any,
    ) -> subprocess.CompletedProcess:
        """构造并执行 docker run（模型代码路径）。

        显式传入的参数是唯一的信任来源：不使用 shell，不继承宿主环境，
        不挂载任何 runtime 目录之外的东西。
        """
        _count_container_spawn()
        return self._run_docker(
            self._container_args(
                container_name=container_name,
                run_dir=run_dir,
                in_dir=in_dir,
                staging_dir=staging_dir,
                scratch_dir=scratch_dir,
                limits=limits,
                entrypoint=["python", "/sandbox/run.py", "/in/job.json"],
            ),
            timeout=limits.timeout_seconds,
        )

    def _container_args(
        self,
        *,
        container_name: str,
        run_dir: Path,
        in_dir: Path,
        staging_dir: Path,
        scratch_dir: Path,
        limits: Any,
        entrypoint: list[str],
    ) -> list[str]:
        """唯一一份容器参数定义，模型执行与诊断探测共用。

        共用是刻意的：如果诊断用自己的参数集，那它验证的就不是模型代码
        实际所处的运行时。

        挂载模型（写入权限是本阶段的安全边界）：
            /sandbox   只读   执行入口
            /in        只读   job.json、code.py、本次需要的表
            /out       可写   **仅 staging**，宿主校验后才发布到 Run 路径
            /history   只读   Run 已发布的产物，容器可读但改不了
            /scratch   可写   结果 JSON
            /tmp       tmpfs  库缓存
        """
        return [
            "run",
            "--rm",
            "--name",
            container_name,
            "--label",
            f"{CONTAINER_LABEL}=1",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--memory",
            f"{limits.memory_mb}m",
            "--memory-swap",
            f"{limits.memory_mb}m",
            "--cpus",
            str(limits.cpus),
            "--pids-limit",
            str(limits.pids_limit),
            "--ulimit",
            f"fsize={limits.max_workspace_bytes}:{limits.max_workspace_bytes}",
            "--tmpfs",
            f"/tmp:rw{_TMPFS_OPTIONS},size={limits.tmpfs_mb}m",
            "--mount",
            f"type=bind,source={_docker_path(self._source_dir)},target=/sandbox,readonly",
            "--mount",
            f"type=bind,source={_docker_path(in_dir)},target=/in,readonly",
            "--mount",
            f"type=bind,source={_docker_path(staging_dir)},target=/out",
            "--mount",
            f"type=bind,source={_docker_path(run_dir)},target=/history,readonly",
            "--mount",
            f"type=bind,source={_docker_path(scratch_dir)},target=/scratch",
            "--env",
            "MPLCONFIGDIR=/tmp/mpl",
            "--env",
            "XDG_CACHE_HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "PYTHONHASHSEED=0",
            # 显式给 PATH，避免依赖宿主环境；容器内不传任何密钥
            "--env",
            "PATH=/usr/local/bin:/usr/local/sbin:/usr/bin:/bin",
            self.image,
            *entrypoint,
        ]

    def _interpret(
        self,
        completed: subprocess.CompletedProcess,
        scratch_dir: Path,
        image_info: SandboxImageInfo,
        *,
        limits: SandboxLimits | None = None,
        expected_attempt: str | None = None,
    ) -> SandboxOutcome:
        """把容器执行结果归一化。

        Process failure always wins over result files. A successful process
        must also return a bounded, valid result for this exact attempt.
        """
        from sandbox.protocol import read_json

        if completed.returncode != 0:
            return SandboxOutcome(
                success=False, result=None,
                error="sandbox: killed (memory limit exceeded)" if completed.returncode == 137 else f"sandbox exited with code {completed.returncode}",
                error_kind=SandboxErrorKind.OOM if completed.returncode == 137 else SandboxErrorKind.EXECUTION_ERROR,
            )
        limits = limits or SandboxLimits()
        result_path = scratch_dir / "result.json"

        if result_path.exists():
            try:
                if result_path.is_symlink() or not result_path.is_file():
                    raise ValueError("result is not a regular file")
                if result_path.stat().st_size > limits.max_result_bytes + 16 * 1024:
                    raise ValueError("sandbox result exceeded host byte limit")
                payload = read_json(result_path)
                if not isinstance(payload, dict) or type(payload.get("success")) is not bool:
                    raise ValueError("invalid result schema")
                if expected_attempt is not None and payload.get("attempt_id") != expected_attempt:
                    raise ValueError("result attempt does not match current execution")
            except (OSError, ValueError) as error:
                return SandboxOutcome(
                    success=False,
                    result=None,
                    error=f"invalid_result: cannot read sandbox result ({error})",
                    error_kind=SandboxErrorKind.INVALID_RESULT,
                )

            if payload.get("success"):
                return SandboxOutcome(
                    success=True,
                    result=payload.get("result"),
                    error=None,
                    truncated=bool(payload.get("truncated", False)),
                    container_id=image_info.image_id,
                    diagnostics=payload.get("diagnostics") or {},
                )

            return SandboxOutcome(
                success=False,
                result=None,
                error=payload.get("error") or "sandbox reported failure",
                error_kind=payload.get("error_kind") or SandboxErrorKind.EXECUTION_ERROR,
            )

        stderr_tail = (completed.stderr or "")[-400:].strip()
        return SandboxOutcome(
            success=False,
            result=None,
            error=(
                f"sandbox exited with code {completed.returncode}"
                + (f": {stderr_tail}" if stderr_tail else "")
            ),
            error_kind=SandboxErrorKind.EXECUTION_ERROR,
        )


def _docker_path(path: Path) -> str:
    """Docker bind mount 的 source 用原生绝对路径。

    Windows 下 Docker Desktop 接受 `G:\\...` 形式；不做 POSIX 转换，
    避免盘符被错误处理。
    """
    return str(Path(path).resolve())


_default_executor: DockerSandboxExecutor | None = None
_default_lock = threading.Lock()


def get_executor(
    *,
    image: str | None = None,
    max_concurrent_containers: int | None = None,
) -> DockerSandboxExecutor:
    """取进程内共享的执行器。

    Python 与 Plot 复用同一个执行器实例，这样并发配额对两者共同生效；
    显式传参时才新建实例（测试用于隔离）。
    """
    global _default_executor

    if image is not None or max_concurrent_containers is not None:
        return DockerSandboxExecutor(
            image=image or "insightflow-sandbox:py311",
            max_concurrent_containers=max_concurrent_containers or 2,
        )

    with _default_lock:
        if _default_executor is None:
            _default_executor = DockerSandboxExecutor()
        return _default_executor


def reset_default_executor() -> None:
    """供测试隔离使用。"""
    global _default_executor
    with _default_lock:
        _default_executor = None
