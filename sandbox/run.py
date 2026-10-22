"""容器内执行入口。

由宿主以 `python /sandbox/run.py /in/job.json` 启动。

挂载布局（宿主持有，容器无法改变）：
    /sandbox  只读   执行入口与共享模块
    /in       只读   job.json 与本次需要的数据集
    /out      可写   绘图产物（对应本次 Run 的目录）
    /scratch  可写   结果 JSON（独立目录，避免与只读输入互相嵌套）
    /tmp      tmpfs  matplotlib 与 fontconfig 的缓存

职责边界：
- 读 job.json，校验结构（宿主生成的，但仍视为不可信输入）；
- 从只读数据目录重建 DataFrame；
- 按 mode 注入符号后执行模型代码；
- 把结果写到 /out 的固定文件名，把异常归一化成同一种结构。

它不决定资源限制、挂载与网络——那些由宿主在 `docker run` 上强制，
容器内的任何代码都无法放宽。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

# 容器内以 /sandbox 为工作目录，且该目录被只读挂载为源码目录
sys.path.insert(0, "/sandbox")

from bootstrap import build_plot_scope, build_python_scope  # noqa: E402
from protocol import decode_dataframe  # noqa: E402

INPUT_DIR = Path("/in")
OUTPUT_DIR = Path("/out")
SCRATCH_DIR = Path("/scratch")
JOB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/in/job.json")
RESULT_PATH = SCRATCH_DIR / "result.json"

PLOT_PATH_MARKERS = ("/", "\\", "..")


def fail(error: str, kind: str = "execution_error") -> dict[str, Any]:
    return {"success": False, "result": None, "error": error, "error_kind": kind}


def load_job() -> dict[str, Any]:
    if not JOB_PATH.exists():
        raise FileNotFoundError(f"job file not found: {JOB_PATH}")
    return json.loads(JOB_PATH.read_text(encoding="utf-8"))


def load_tables(job: dict[str, Any]) -> dict[str, Any]:
    """按 job 指定的表逐个从只读输入目录读回。"""
    tables: dict[str, Any] = {}

    for name, relative in (job.get("dfs") or {}).items():
        path = INPUT_DIR / relative

        # 数据文件名由宿主生成，但仍然限定在 /in 下，避免 ../ 逃逸
        if not path.resolve().is_relative_to(INPUT_DIR.resolve()):
            raise ValueError(f"dataset path escapes input directory: {relative}")
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {relative}")

        tables[name] = decode_dataframe(
            json.loads(path.read_text(encoding="utf-8"))
        )

    return tables


def resolve_output_path(job: dict[str, Any]) -> Path:
    """绘图输出路径固定落在 /out 下，文件名由宿主给出且不含路径分隔符。"""
    filename = job.get("output_filename") or ""
    if not filename or any(marker in filename for marker in PLOT_PATH_MARKERS):
        raise ValueError(f"invalid output filename: {filename!r}")
    return OUTPUT_DIR / filename


def make_jsonable(value: Any) -> Any:
    """把结果转成可 JSON 表达的形式。

    无法表达时抛 `TypeError` 而不是静默丢弃内容——宿主会把它归一化成
    `invalid_result`，让调用方明确知道结果不可用。
    """
    import datetime as _datetime

    import numpy as np
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        from protocol import encode_dataframe

        return encode_dataframe(value)

    if isinstance(value, pd.Series):
        return {
            "__kind__": "series",
            "name": str(value.name) if value.name is not None else None,
            "values": make_jsonable(value.tolist()),
        }

    if isinstance(value, np.ndarray):
        return {"__kind__": "ndarray", "values": value.tolist()}

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (pd.Timestamp, _datetime.datetime, _datetime.date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]

    return value


def enforce_limits(payload: Any, limits: dict[str, Any]) -> tuple[Any, bool]:
    """按行数与字节数限制结果。

    超限时截断并标记，而不是整体失败——截断对分析类结果更可用，
    且宿主能通过 `truncated` 看到发生了截断。
    返回 `(结果, 是否截断)`。
    """
    max_rows = int(limits.get("max_result_rows") or 0)
    max_bytes = int(limits.get("max_result_bytes") or 0)

    truncated = False

    if isinstance(payload, dict) and payload.get("__kind__") == "dataframe":
        records = payload.get("records") or []
        if max_rows and len(records) > max_rows:
            payload = {
                **payload,
                "records": records[:max_rows],
                "row_count": len(records),
            }
            truncated = True

    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    if max_bytes and len(encoded.encode("utf-8")) > max_bytes:
        # 结果过大且无法安全截断时明确报错，交给宿主识别为不可用结果
        raise ValueError(
            f"result exceeds size limit "
            f"({len(encoded.encode('utf-8'))} > {max_bytes} bytes)"
        )

    return payload, truncated


def load_code(job: dict[str, Any]) -> str:
    """从只读输入目录读取模型代码。

    代码以独立文件传入而不是内联在 job.json 里，避免把大段文本塞进 JSON，
    也让"代码是数据、不是命令"这一点在结构上成立。
    """
    relative = job.get("code_path") or "code.py"
    path = INPUT_DIR / relative

    if not path.resolve().is_relative_to(INPUT_DIR.resolve()):
        raise ValueError(f"code path escapes input directory: {relative}")
    if not path.exists():
        raise FileNotFoundError(f"code file not found: {relative}")

    return path.read_text(encoding="utf-8")


def run_python(job: dict[str, Any], tables: dict[str, Any]) -> dict[str, Any]:
    scope = build_python_scope(tables)
    code = load_code(job)
    exec(compile(code, "<model_code>", "exec"), scope, scope)

    if "result" not in scope:
        raise ValueError("Python analysis code must store its final output in 'result'.")

    return {"result": scope["result"], "artifacts": []}


def run_plot(job: dict[str, Any], tables: dict[str, Any]) -> dict[str, Any]:
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文字形来自镜像内的 fonts-noto-cjk；显式把 Noto Sans CJK SC 放在首位，
    # 避免回退到不含 CJK 字形的 DejaVu 而渲染成方框。
    plt.rcdefaults()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    output_path = resolve_output_path(job)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scope = build_plot_scope(
        tables,
        plt=plt,
        analysis_result=job.get("analysis_result"),
        output_path=str(output_path),
    )

    try:
        code = load_code(job)

        # 收集缺字形告警：中文没渲染出来时 matplotlib 会在这里报警，
        # 宿主据此判断"图生成了但字是方框"，而不是只看文件是否存在。
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec(compile(code, "<model_code>", "exec"), scope, scope)

        if "result" not in scope:
            raise ValueError("Plot code must store its final output in 'result'.")

        if not output_path.exists():
            raise ValueError("Plot code did not create the expected image file.")

        missing_glyphs = [
            str(item.message)
            for item in caught
            if "missing from font" in str(item.message).lower()
        ]

        return {
            "result": str(output_path),
            "artifacts": [str(output_path)],
            "model_result": make_jsonable(scope.get("result")),
            "diagnostics": {
                "missing_glyph_count": len(missing_glyphs),
                "output_bytes": output_path.stat().st_size,
            },
        }
    finally:
        plt.close("all")


def main() -> int:
    outcome: dict[str, Any]
    job: dict[str, Any] = {}

    try:
        job = load_job()
        mode = job.get("mode")

        if mode not in ("python", "plot"):
            raise ValueError(f"unsupported mode: {mode!r}")

        tables = load_tables(job)

        if mode == "python":
            executed = run_python(job, tables)
        else:
            executed = run_plot(job, tables)

        payload = make_jsonable(executed["result"])
        payload, truncated = enforce_limits(payload, job.get("limits") or {})

        diagnostics = dict(executed.get("diagnostics") or {})
        # 绘图模式下 result 被约定为图片路径，模型自己写进 result 的值
        # 只能放这里带出来，便于测试与排查。
        if mode == "plot":
            diagnostics["model_result"] = executed.get("model_result")

        outcome = {
            "success": True,
            "result": payload,
            "error": None,
            "truncated": truncated,
            "artifacts": executed.get("artifacts", []),
            "diagnostics": diagnostics,
        }

    except TypeError as error:
        # 结果无法 JSON 表达：明确报错，不静默丢内容
        outcome = fail(f"{type(error).__name__}: {error}", kind="invalid_result")
        outcome["traceback_tail"] = traceback.format_exc()[-1200:]
    except MemoryError:
        outcome = fail("sandbox: memory exhausted by model code", kind="oom")
    except Exception as error:  # noqa: BLE001 - 任何模型代码异常都归一化
        outcome = fail(f"{type(error).__name__}: {error}")
        outcome["traceback_tail"] = traceback.format_exc()[-1200:]

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    outcome["attempt_id"] = job.get("attempt_id")
    temporary_result = RESULT_PATH.with_suffix(".tmp")
    temporary_result.write_text(
        json.dumps(outcome, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temporary_result.replace(RESULT_PATH)

    # stdout 预留给协议类用途，诊断信息走 stderr
    print(json.dumps({"ok": outcome["success"]}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
