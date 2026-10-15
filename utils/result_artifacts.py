"""执行结果的持久化与还原。

`execution_result["result"]` 是模型生成的任意对象，可能是 DataFrame、
ndarray 或包含 Timestamp 的嵌套结构，直接放进 Checkpoint 会导致序列化失败。

策略：小结果内联（便于报告直接使用），大结果落成带指纹的 JSON 产物，
state 里只留**摘要**与引用。这样 checkpoint 体积可控，同时产物可被校验，
避免恢复后引用到被替换过的文件。

由此产生两类消费者，必须区分清楚（P2-B3）：

- **需要真实数据**（绘图、需要落盘渲染的节点）：走 `load_result` 按引用读回，
  读之前校验归属、路径、存在性与 sha256；
- **只需要结果概要的 LLM 节点**：走 `build_preview`，拿到列名、行数、类型、
  有界样本与聚合摘要——**明确是摘要**，不能声称模型看到了完整数据，
  更不能把两万行塞进 Prompt。

两者都不把完整内容写回 LangGraph State。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 内联上限：超过则落盘。按序列化后的字节数判断，而不是行数——
# 一行超宽表的体积可能远超一千行窄表。
INLINE_LIMIT_BYTES = 32 * 1024

# 预览预算：给 LLM 节点的有界样本。两者都要有——
# 行数控制"看几条"，字节数控制"总共多大"，因为单行也可能极宽。
PREVIEW_MAX_RECORDS = 20
PREVIEW_MAX_BYTES = 8 * 1024

# 还原时的内存预算：产物读回后不得超过这个体积。
# 产物本身是受信文件（宿主写入并校验哈希），这里防的是"意外巨大"而不是恶意。
LOAD_MAX_BYTES = 64 * 1024 * 1024


class ResultArtifactError(RuntimeError):
    """结果产物写入或读取失败。"""


@dataclass(frozen=True)
class ResultArtifact:
    """大结果的落盘引用。"""

    path: str
    sha256: str
    size: int
    row_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResultArtifact":
        return cls(
            path=payload["path"],
            sha256=payload["sha256"],
            size=int(payload["size"]),
            row_count=payload.get("row_count"),
        )


def to_jsonable(value: Any) -> Any:
    """把常见分析结果转成 JSON 可表达的形式。

    只处理可预期会出现的类型；未知类型交给调用方按"无法序列化"处理，
    不在这里静默丢弃数据。
    """
    if isinstance(value, pd.DataFrame):
        return {
            "__kind__": "dataframe",
            "records": json.loads(value.to_json(orient="records", date_format="iso")),
            "columns": [str(column) for column in value.columns],
        }

    if isinstance(value, pd.Series):
        return {
            "__kind__": "series",
            "name": str(value.name) if value.name is not None else None,
            "records": json.loads(value.to_json(orient="records", date_format="iso")),
        }

    if isinstance(value, np.ndarray):
        return {"__kind__": "ndarray", "values": value.tolist()}

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    return value


def from_jsonable(value: Any) -> Any:
    """还原 `to_jsonable` 标记过的结构。

    未标记的值原样返回——标量、普通 dict 与 list 不需要还原，
    这条路径同时也服务于"内联结果"。
    """
    if isinstance(value, dict):
        kind = value.get("__kind__")

        if kind == "dataframe":
            columns = value.get("columns") or []
            frame = pd.DataFrame(value.get("records") or [])
            if frame.empty and columns:
                frame = pd.DataFrame(columns=columns)
            return frame

        if kind == "series":
            records = value.get("records") or []
            frame = pd.DataFrame(records)
            series = frame.iloc[:, 0] if len(frame.columns) == 1 else frame
            if isinstance(series, pd.Series) and value.get("name") is not None:
                series = series.rename(value["name"])
            return series

        if kind == "ndarray":
            return np.array(value.get("values") or [])

        # 未识别的标记（例如 externalized 摘要）保持原样，
        # 让调用方自己判断该怎么处理，而不是猜一个内容出来。
        return {key: from_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [from_jsonable(item) for item in value]

    return value


def unwrap_records(value: Any) -> Any:
    """把结果拆成绘图/下游可直接使用的形状。

    绘图工具需要的是"记录序列"或 DataFrame，而不是带 `__kind__` 的信封。
    这里把 dataframe 信封拆成 records 列表、series 信封拆成列表，
    其余原样返回——工具侧的 `build_result_dataframe` 因此不需要认识信封。
    """
    if isinstance(value, dict):
        kind = value.get("__kind__")
        if kind in ("dataframe", "series"):
            return value.get("records") or []
        if kind == "ndarray":
            return value.get("values") or []
    return value


def _row_count(value: Any) -> int | None:
    """结果有多少行。

    既要认原始对象（DataFrame / Series / list），也要认**已经 JSON 化**的信封：
    沙箱执行路径返回的已经是 `{"__kind__": "dataframe", "records": [...]}`，
    只认 DataFrame 会让外置摘要的 row_count 变成 None，
    报告与预览就失去了行数信息。
    """
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return int(len(value))

    if isinstance(value, list):
        return len(value)

    if isinstance(value, dict):
        kind = value.get("__kind__")
        if kind in ("dataframe", "series"):
            records = value.get("records")
            return len(records) if isinstance(records, list) else None
        if kind == "ndarray":
            values = value.get("values")
            return len(values) if isinstance(values, list) else None

    return None


def _columns_of(jsonable: Any) -> list[str]:
    if isinstance(jsonable, dict) and jsonable.get("__kind__") == "dataframe":
        return [str(column) for column in (jsonable.get("columns") or [])]
    if isinstance(jsonable, list) and jsonable and isinstance(jsonable[0], dict):
        return [str(key) for key in jsonable[0]]
    return []


def store_result(
    value: Any,
    *,
    run_id: str,
    task_id: str,
    output_root: Path,
) -> tuple[Any | None, dict[str, Any] | None]:
    """返回 (可内联的结果, 产物引用或 None)。

    结果无法 JSON 编码时抛 `ResultArtifactError`——宁可在写入时明确失败，
    也不要把一个会在 Checkpoint 阶段才崩掉的值留在 state 里。
    """
    jsonable = to_jsonable(value)

    try:
        encoded = json.dumps(jsonable, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as error:
        raise ResultArtifactError(
            f"结果无法序列化（task={task_id}）：{type(error).__name__}"
        ) from error

    if len(encoded.encode("utf-8")) <= INLINE_LIMIT_BYTES:
        return jsonable, None

    directory = output_root / run_id / "results"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task_id}.json"

    payload = json.dumps(jsonable, ensure_ascii=False, default=str).encode("utf-8")
    path.write_bytes(payload)

    artifact = ResultArtifact(
        path=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        row_count=_row_count(value),
    )
    # 内联一个摘要：报告与调试可以看到形状而不必读文件。
    # 同时带上列名与类型标记，让 LLM 节点知道"这是摘要，不是数据"。
    summary: dict[str, Any] = {
        "__kind__": "externalized",
        "row_count": artifact.row_count,
        "size": artifact.size,
        "columns": _columns_of(jsonable),
    }
    return summary, artifact.to_dict()


def _verify_reference(
    artifact: dict[str, Any],
    *,
    run_id: str | None,
    output_root: Path | str | None,
) -> tuple[ResultArtifact, Path]:
    """校验产物引用，返回 (引用, 路径)。

    校验顺序有讲究：先归属、再路径、再存在性、最后内容哈希。
    归属检查放在最前，避免"读到别的 Run 的文件之后才发现不该读"。
    """
    reference = ResultArtifact.from_dict(artifact)

    if not reference.path:
        raise ResultArtifactError("结果产物引用缺少路径")

    path = Path(reference.path)

    if output_root is not None:
        root = Path(output_root).resolve()
        resolved = path.resolve()

        # 路径必须落在 output_root 之内，否则就是越权引用宿主任意路径
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ResultArtifactError(
                f"结果产物路径越界：{path} 不在 {root} 之内"
            ) from error

        # 归属：产物应位于 <output_root>/<run_id>/results/ 下
        if run_id is not None:
            expected = (root / run_id / "results").resolve()
            try:
                resolved.relative_to(expected)
            except ValueError as error:
                raise ResultArtifactError(
                    f"结果产物不属于本次 Run：{path} 不在 {expected} 之内"
                ) from error

    if not path.exists():
        raise ResultArtifactError(f"结果产物已丢失：{path}")

    if not path.is_file():
        raise ResultArtifactError(f"结果产物不是普通文件：{path}")

    if reference.size and path.stat().st_size > LOAD_MAX_BYTES:
        raise ResultArtifactError(
            f"结果产物超出读取预算：{path.stat().st_size} > {LOAD_MAX_BYTES}"
        )

    payload = path.read_bytes()
    current = hashlib.sha256(payload).hexdigest()
    if current != reference.sha256:
        raise ResultArtifactError(
            f"结果产物内容已变化：{path}"
            f"（期望 {reference.sha256[:12]}… 实际 {current[:12]}…）"
        )

    return reference, path


def load_result(
    inline: Any,
    artifact: dict[str, Any] | None,
    *,
    run_id: str | None = None,
    output_root: Path | str | None = None,
) -> Any:
    """还原结果。

    产物缺失、路径越界、不属于本 Run、或指纹不符时明确失败，
    不静默返回空值——静默失败会让报告与绘图用错数据却不报错。

    `run_id` 与 `output_root` 给出时会做归属与路径校验；
    调用方应尽量提供它们（省略只应在没有运行期上下文的场景）。
    """
    if not artifact:
        return from_jsonable(inline)

    _reference, path = _verify_reference(
        artifact, run_id=run_id, output_root=output_root
    )

    return from_jsonable(json.loads(path.read_bytes().decode("utf-8")))


def build_preview(
    inline: Any,
    artifact: dict[str, Any] | None,
    *,
    run_id: str | None = None,
    output_root: Path | str | None = None,
    max_records: int = PREVIEW_MAX_RECORDS,
    max_bytes: int = PREVIEW_MAX_BYTES,
) -> dict[str, Any]:
    """构造给 LLM 节点看的结果预览。

    这是**有界摘要**，不是完整数据：列名、行数、前若干条样本，
    以及"是否被截断"的明确标记。调用方不得据此声称模型看到了完整数据。
    """
    columns: list[str] = []
    total_rows: int | None = None
    truncated = False

    if artifact:
        reference = ResultArtifact.from_dict(artifact)
        total_rows = reference.row_count
        # 摘要里已经带了列名，先用它——不必为了拿列名去读整个产物
        if isinstance(inline, dict):
            columns = [str(column) for column in (inline.get("columns") or [])]

    records: list[Any] = []

    if artifact:
        try:
            loaded = load_result(
                inline, artifact, run_id=run_id, output_root=output_root
            )
        except ResultArtifactError as error:
            # 预览是可降级的：读不到产物时给出摘要并说明原因，
            # 而不是让整个报告生成失败。需要真实数据的节点则会直接抛错。
            return {
                "kind": "externalized",
                "available": False,
                "columns": columns,
                "row_count": total_rows,
                "total_row_count": total_rows,
                "preview_rows": 0,
                "truncated": True,
                "note": f"完整结果不可用：{error}",
            }

        rows = _iter_records(loaded)
        if rows is not None:
            columns = columns or _columns_of_records(rows)
            total_rows = len(rows)
            records, truncated = _bounded(rows, max_records, max_bytes)
        else:
            return {
                "kind": "scalar",
                "available": True,
                "value": loaded,
                "columns": [],
                "row_count": None,
                "total_row_count": None,
                "preview_rows": 0,
                "truncated": False,
            }

        return {
            "kind": "externalized",
            "available": True,
            "columns": columns,
            "row_count": total_rows,
            "total_row_count": total_rows,
            "preview_rows": len(records),
            "sample": records,
            "truncated": truncated,
            "note": (
                "完整结果已外置到产物文件，这里只是有界预览；"
                "不要把它当作完整数据集。"
            ),
        }

    rows = _iter_records(from_jsonable(inline))
    if rows is None:
        return {
            "kind": "scalar",
            "available": True,
            "value": inline,
            "columns": [],
            "row_count": None,
            "total_row_count": None,
            "preview_rows": 0,
            "truncated": False,
        }

    columns = _columns_of_records(rows)
    records, truncated = _bounded(rows, max_records, max_bytes)

    return {
        "kind": "inline",
        "available": True,
        "columns": columns,
        "row_count": len(rows),
        "total_row_count": len(rows),
        "preview_rows": len(records),
        "sample": records,
        "truncated": truncated,
    }


def _iter_records(value: Any) -> list[Any] | None:
    """把结果看成"记录序列"；不是序列时返回 None。"""
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        # Series 的"记录"是 (标签, 值) 对，这样预览仍能反映索引信息
        return [{"index": key, "value": item} for key, item in value.items()]
    if isinstance(value, list):
        return value
    return None


def _columns_of_records(records: list[Any]) -> list[str]:
    if records and isinstance(records[0], dict):
        return [str(key) for key in records[0]]
    return []


def _bounded(
    rows: list[Any], max_records: int, max_bytes: int
) -> tuple[list[Any], bool]:
    """按行数与字节数取前若干条。返回 (样本, 是否被截断)。"""
    selected: list[Any] = []
    total = 0
    truncated = False

    for row in rows:
        if len(selected) >= max_records:
            truncated = True
            break
        size = len(json.dumps(row, ensure_ascii=False, default=str))
        if total + size > max_bytes:
            truncated = True
            break
        selected.append(row)
        total += size

    if len(selected) < len(rows):
        truncated = True

    return selected, truncated
