"""安全地设置 .env 里的单个键，不动其他内容。

    python scripts/set_env_value.py FEISHU_WEBHOOK_URL

为什么需要它：`.env` 里同时放着模型凭据、存储模式和飞书地址。整体重写很容易
把没注意到的键抹掉（真的发生过：切换模型地址时把飞书 webhook 覆盖没了）。
这个脚本只替换目标那一行，其余原样保留；输入不回显。
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# 键名 -> 是否允许空值（空值表示"关闭这项功能"）
KNOWN = {
    "DEEPSEEK_API_KEY": False,
    "DEEPSEEK_BASE_URL": False,
    "DEEPSEEK_MODEL": False,
    "FEISHU_WEBHOOK_URL": True,
    "FEISHU_SIGN_SECRET": True,
    "FEISHU_CONSOLE_URL": True,
    "INSIGHTFLOW_OWNER": True,
    "INSIGHTFLOW_API_KEYS": True,
    "INSIGHTFLOW_STORAGE": False,
    "INSIGHTFLOW_DATABASE_URL": True,
    "INSIGHTFLOW_REDIS_ENABLED": True,
    "INSIGHTFLOW_REDIS_URL": True,
}


def update(path: Path, key: str, value: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[index] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="设置 .env 中的单个键")
    parser.add_argument("key", help="键名，例如 FEISHU_WEBHOOK_URL")
    parser.add_argument("--path", default=str(ENV_PATH), help="调试用：指定 .env 路径")
    parser.add_argument("--value", default=None, help="非交互模式（脚本/CI 用）")
    args = parser.parse_args()

    key = args.key.strip()
    if key not in KNOWN:
        print(f"未知的键：{key}\n可用的键：{', '.join(sorted(KNOWN))}", file=sys.stderr)
        return 2

    if args.value is not None:
        value = args.value.strip()
    else:
        prompt = f"{key}（输入不回显，留空表示关闭这项功能）："
        value = (getpass.getpass(prompt) if not KNOWN[key] else input(prompt)).strip()

    if not value and not KNOWN[key]:
        print(f"{key} 不能为空。", file=sys.stderr)
        return 2
    if any(char.isspace() for char in value):
        print("值里包含空白字符，已中止（复制时可能带了换行）。", file=sys.stderr)
        return 2

    path = Path(args.path)
    update(path, key, value)
    shown = "（已清空）" if not value else f"长度 {len(value)}"
    print(f"已写入 {path.name}：{key} = {shown}")
    if key.startswith("FEISHU") or key.startswith("DEEPSEEK"):
        print("提示：正在运行的服务需要重启才会读到新的值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
