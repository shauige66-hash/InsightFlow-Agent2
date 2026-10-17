"""按"是否真的进了容器"给沙箱用例分类计数。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe -m pytest -q -m sandbox
    .venv\\Scripts\\python.exe scripts/sandbox_report.py

分类依据是**实测**：conftest 的 hook 记录每个用例期间容器启动计数的增量，
而不是靠用例名或人工标注。因此报告不会与实际行为脱节。

三类定义：
    A. 真实容器执行   —— 该用例期间确实启动了容器（spawned > 0）
    B. 故障模拟       —— 打了 sandbox 标记但没有启动容器，验证 fail-closed /
                         错误分类；需要 Docker 可用（否则会整体跳过）
    C. 静态断言       —— 只做源码、缓存键或参数结构断言，与 Docker 无关
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "tests" / ".sandbox_classification.json"

# 只做源码/缓存键/结构断言的用例：这些天然不需要容器
STATIC_PREFIXES = (
    "tests/test_sandbox_integration.py::TestToolCacheFollowsContentHash",
    "tests/test_error_classification.py::TestErrorKindContractIsStructured::"
    "test_unknown_tool_is_classified",
)


def classify(nodeid: str, record: dict) -> str:
    if record.get("spawned", 0) > 0:
        return "A_real_container"
    for prefix in STATIC_PREFIXES:
        if nodeid.startswith(prefix):
            return "C_static"
    return "B_fault_path"


def main() -> int:
    if not SOURCE.exists():
        print(f"未找到分类结果 {SOURCE}")
        print("请先运行： python -m pytest -q -m sandbox")
        return 2

    data = json.loads(SOURCE.read_text(encoding="utf-8"))

    # 只统计带 sandbox 标记的文件
    sandbox_files = {
        "tests/test_sandbox_isolation.py",
        "tests/test_sandbox_plot.py",
        "tests/test_sandbox_integration.py",
        "tests/test_sandbox_write_restriction.py",
        "tests/test_error_classification.py",
        # 大结果经沙箱绘图，其中一项真实启动容器
        "tests/test_large_result_consumers.py",
    }

    buckets: dict[str, list[str]] = defaultdict(list)
    per_file: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    skipped: list[str] = []
    failed: list[str] = []

    for nodeid, record in data.items():
        path = nodeid.split("::")[0]
        if path not in sandbox_files:
            continue

        bucket = classify(nodeid, record)
        buckets[bucket].append(nodeid)
        per_file[path][bucket] += 1
        if record.get("skipped"):
            skipped.append(nodeid)
        if record.get("failed"):
            failed.append(nodeid)

    total = sum(len(v) for v in buckets.values())

    print("=== 沙箱用例分类（实测口径）===")
    print()
    labels = {
        "A_real_container": "A. 真实容器执行（确实启动了容器）",
        "B_fault_path": "B. 故障模拟（Docker 不可用/镜像缺失/超时/OOM 等，未启动容器）",
        "C_static": "C. 静态断言（源码/缓存键/参数结构，与 Docker 无关）",
    }
    for key in ("A_real_container", "B_fault_path", "C_static"):
        print(f"  {labels[key]:<58} {len(buckets[key]):>3}")

    print()
    print(f"  沙箱用例合计: {total}")
    print(f"  其中跳过: {len(skipped)}    失败: {len(failed)}")

    if skipped:
        print()
        print("  跳过明细:")
        for nodeid in skipped:
            print(f"    - {nodeid}")

    print()
    print("=== 按文件 ===")
    for path in sorted(per_file):
        counts = per_file[path]
        print(
            f"  {path:<46} "
            f"A={counts['A_real_container']:>2}  "
            f"B={counts['B_fault_path']:>2}  "
            f"C={counts['C_static']:>2}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
