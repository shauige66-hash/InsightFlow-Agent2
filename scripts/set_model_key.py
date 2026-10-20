"""安全地把模型 key 写进 .env，并立刻验证。

    python scripts/set_model_key.py

为什么要专门一个脚本：这一步反复出错的方式都很朴素——改到了另一个 .env、
复制时带进了引号或空格、改了没保存。脚本用 getpass 读取（不回显），写死到项目
根目录的 .env，并且只替换 `DEEPSEEK_API_KEY` 那一行，其余内容原样保留。
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
KEY_NAME = "DEEPSEEK_API_KEY"


def _read_key(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{KEY_NAME}="):
            return line.split("=", 1)[1].strip()
    return ""


def _write_key(path: Path, key: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith(f"{KEY_NAME}="):
            lines[index] = f"{KEY_NAME}={key}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{KEY_NAME}={key}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="写入并验证模型 key")
    parser.add_argument("--path", default=str(ENV_PATH), help="调试用：指定 .env 路径")
    parser.add_argument("--no-check", action="store_true", help="只写入，不调用验证")
    args = parser.parse_args()

    path = Path(args.path)
    current = _read_key(path)
    print(f"目标文件：{path}")
    print(f"当前 key：{'（空）' if not current else f'长度 {len(current)}，末尾 {current[-4:]}'}")

    key = getpass.getpass("粘贴新的 API Key（输入不回显）：").strip()
    if not key:
        print("没有输入，未做修改。", file=sys.stderr)
        return 2

    problems = []
    if not key.startswith("sk-"):
        problems.append("通常以 sk- 开头")
    if any(char.isspace() for char in key):
        problems.append("包含空白字符")
    if key.count('"') or key.count("'"):
        problems.append("包含引号")
    if problems:
        print("这个 key 看起来不对：" + "；".join(problems), file=sys.stderr)
        print("已中止，未写入。请确认复制的是完整的 key 本身。", file=sys.stderr)
        return 2

    _write_key(path, key)
    print(f"已写入 {path}（长度 {len(key)}，末尾 {key[-4:]}）")

    if args.no_check:
        return 0

    print("\n正在验证……")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "check_model.py")],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
