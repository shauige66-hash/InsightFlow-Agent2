"""本地演示 / 测试用的 API 服务入口。

与直接 `uvicorn api:app` 的区别：这里在启动前把宿主配置**显式**指向一个
隔离目录，因此可以：

- 把 RunStore、Checkpoint、output_root 都限制在一个临时工作目录里，
  绝不碰用户真实数据或 ModelGate 业务库；
- 用脚本化 Mock 模型跑完整工程链路（不访问任何付费模型）；
- 给跨进程恢复测试提供一个可被强杀、再重启的真实 HTTP 服务。

环境变量
--------
INSIGHTFLOW_WORKDIR
    工作根目录。RunStore / Checkpoint / output_root 全部落在它下面。
    默认 `outputs/demo`。
INSIGHTFLOW_MOCK_LLM
    置 1 时使用脚本化 Mock 模型。**Mock 只用于演示工程链路，
    不能证明真实模型的自主分析能力。**
INSIGHTFLOW_MODEL_FACTORY
    可选，`模块:属性` 形式的模型工厂导入路径（签名同 `build_model_factory`）。
    测试用它把自定义的脚本化模型注入子进程——否则子进程只能用本脚本
    自己 import 的那个工厂，测试无法左右 planner 的目标表与执行代码。
INSIGHTFLOW_CRASH_AT
    故障注入点，仅测试使用。命中即强杀进程，用于制造真实的崩溃现场。
INSIGHTFLOW_HOST / INSIGHTFLOW_PORT
    监听地址，默认 127.0.0.1:8000。

用法：
    python scripts/serve_api.py            # 读环境变量
    python scripts/serve_api.py --port 8100
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_factory(spec: str):
    """把 `模块:属性` 解析成可调用的模型工厂。"""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise SystemExit(
            f"INSIGHTFLOW_MODEL_FACTORY 需要 `模块:属性` 形式，收到：{spec!r}"
        )

    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise SystemExit(f"{module_name} 里没有 {attribute}") from error


def configure(workdir: Path, *, mock_llm: bool) -> None:
    """把 API 模块的宿主级配置指向隔离目录。

    直接改 `api_runs` 的模块级单例，与既有测试的做法一致——
    不新造一套生命周期，只是喂给它不同的路径。
    """
    import api_runs
    from core.runtime import McpSettings
    from services.run_coordinator import RunCoordinator
    from services.run_store import RunStore

    workdir.mkdir(parents=True, exist_ok=True)

    run_db = workdir / "runs" / "runs.sqlite"
    checkpoint = workdir / "checkpoints.sqlite"

    # 存储后端跟着配置走，不能在脚本里写死 SQLite：postgres 模式下若仍然用
    # SQLite，Run 记录会落到工作目录，而 /storage/health 报的是 postgres——
    # 两边不一致，排查时会被误导（真实部署时就踩到过）。
    import storage
    from services import storage_factory

    if storage.current_config().is_postgres:
        store = storage_factory.run_store(workdir)
        checkpoint_arg = None  # 让协调器用默认的 PostgreSQL checkpointer
    else:
        store = RunStore(run_db)
        checkpoint_arg = checkpoint

    factory_spec = os.environ.get("INSIGHTFLOW_MODEL_FACTORY")
    if factory_spec:
        builder = resolve_factory(factory_spec)
    elif mock_llm:
        from scripts.recovery_driver import mock_model_factory

        builder = mock_model_factory
    else:
        from core.llm import build_model_factory

        builder = build_model_factory

    coordinator = RunCoordinator(
        store,
        output_root=workdir,
        checkpointer_path=checkpoint_arg,
        model_factory_builder=builder,
        # Server 模块、超时与私有目录由宿主固定；HTTP 请求只能选模式
        mcp_settings=McpSettings(),
    )

    api_runs.run_store = store
    api_runs.coordinator = coordinator
    api_runs.OUTPUT_ROOT = workdir
    if checkpoint_arg is not None:
        api_runs.CHECKPOINT_PATH = checkpoint_arg


def main() -> int:
    parser = argparse.ArgumentParser(description="InsightFlow 本地演示 API 服务")
    parser.add_argument(
        "--workdir",
        default=os.environ.get("INSIGHTFLOW_WORKDIR", "outputs/demo"),
        help="隔离工作目录（RunStore / Checkpoint / 产物都在它下面）",
    )
    parser.add_argument("--host", default=os.environ.get("INSIGHTFLOW_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("INSIGHTFLOW_PORT", "8000"))
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        default=os.environ.get("INSIGHTFLOW_MOCK_LLM") == "1",
        help="使用脚本化 Mock 模型（不访问任何模型服务）",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    configure(workdir, mock_llm=args.mock_llm)

    import uvicorn

    from api import app

    # 只监听本机：当前 API 没有鉴权，只能作为本地演示服务，
    # 不能直接公网部署。
    print(
        f"InsightFlow API 启动于 http://{args.host}:{args.port}"
        f" | workdir={workdir} | mock_llm={args.mock_llm}",
        flush=True,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
