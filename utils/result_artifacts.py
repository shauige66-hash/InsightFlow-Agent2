"""执行结果的持久化与还原。

`execution_result["result"]` 是模型生成的任意对象，可能是 DataFrame、
ndarray 或包含 Timestamp 的嵌套结构，直接放进 Checkpoint 会导致序列化失败。

策略：小结果内联（便于报告直接使用），大结果落成带指纹的 JSON 产物，
state 里只留引用与摘要。这样 checkpoint 体积可控，同时产物可被校验，
避免恢复后引用到被替换过的文件。

注意：这里不尝试还原"任意对象"。内联结果本身必须可 JSON 编码，
真正大的对象以产物引用表达，读取方需要 JSON 可表达的形式。
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


def _row_count(value: Any) -> int | None:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return int(len(value))
    if isinstance(value, list):
        return len(value)
    return None


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
    # 内联一个摘要，便于报告与调试时看到形状而不必读文件
    summary: dict[str, Any] = {
        "__kind__": "externalized",
        "row_count": artifact.row_count,
        "size": artifact.size,
    }
    return summary, artifact.to_dict()


def load_result(
    inline: Any, artifact: dict[str, Any] | None
) -> Any:
    """还原结果。产物缺失或指纹不符时明确失败，不静默返回空值。"""
    if not artifact:
        return inline

    reference = ResultArtifact.from_dict(artifact)
    path = Path(reference.path)

    if not path.exists():
        raise ResultArtifactError(f"结果产物已丢失：{path}")

    payload = path.read_bytes()
    current = hashlib.sha256(payload).hexdigest()
    if current != reference.sha256:
        raise ResultArtifactError(
            f"结果产物内容已变化：{path}"
            f"（期望 {reference.sha256[:12]}… 实际 {current[:12]}…）"
        )

    return json.loads(payload.decode("utf-8"))
