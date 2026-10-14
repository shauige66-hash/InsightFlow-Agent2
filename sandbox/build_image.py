"""构建并记录沙箱镜像。

构建与运行分离：本脚本需要网络（安装 Debian 包与 PyPI 依赖），
运行期一律 `--network none`，因此镜像必须预构建。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe sandbox/build_image.py
    .venv\\Scripts\\python.exe sandbox/build_image.py --tag insightflow-sandbox:py311

产物：镜像本身，以及 stdout 上的构建记录（镜像 ID、digest、体积、版本、字体告警数）。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_DIR = PROJECT_ROOT / "sandbox"
DEFAULT_TAG = "insightflow-sandbox:py311"

# 验证脚本：构建后立刻在真实沙箱参数下跑一次，确认中文字形可渲染
VERIFY_PAYLOAD = """
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import pandas as pd
import matplotlib.pyplot as plt

families = {f.name for f in font_manager.fontManager.ttflist}
cjk = sorted(n for n in families if "CJK" in n)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    plt.rcdefaults()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure()
    plt.plot([1, 2, 3], [2, 3, 5], label="销售额")
    plt.title("客户数柱状图")
    plt.xlabel("地区")
    plt.ylabel("金额")
    plt.legend()
    fig.savefig("/out/verify.png")
    plt.close(fig)

missing = [w for w in caught if "missing from font" in str(w.message).lower()]
print(json.dumps({
    "matplotlib": matplotlib.__version__,
    "pandas": pd.__version__,
    "cjk_families": cjk[:4],
    "png_exists": Path("/out/verify.png").exists(),
    "missing_glyph_warnings": len(missing),
}))
"""


def docker_available(docker: str) -> bool:
    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def build(docker: str, tag: str) -> int:
    print(f"building {tag} from {SANDBOX_DIR}")
    result = subprocess.run(
        [docker, "build", "-t", tag, str(SANDBOX_DIR)],
        text=True,
    )
    return result.returncode


def base_image_reference() -> str:
    """从 Dockerfile 读出 FROM 行，作为构建溯源的一部分。"""
    for line in (SANDBOX_DIR / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("FROM "):
            return line[len("FROM ") :].strip()
    return "(not found)"


def inspect(docker: str, tag: str) -> None:
    import json
    import time

    result = subprocess.run(
        [docker, "image", "inspect", tag, "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print("inspect failed:", result.stderr[-300:])
        return

    data = json.loads(result.stdout.strip())
    digests = data.get("RepoDigests") or []
    size_mb = (data.get("Size") or 0) / (1024 * 1024)

    print()
    print("=== 镜像记录 ===")
    print(f"  tag        : {tag}")
    print(f"  image id   : {data.get('Id')}")
    print(f"  digest     : {digests[0] if digests else '(本地构建，无 registry digest)'}")
    print(f"  size       : {size_mb:.1f} MB")
    print(f"  base       : {base_image_reference()}")
    # Docker 报告的 Created 是顶层 config 的时间戳，使用构建缓存时会保留旧值，
    # 因此另外记录本次记录生成时间，避免把陈旧时间当成构建时间。
    print(f"  recorded at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"  image Created (可能来自缓存层): {data.get('Created')}")


def verify(docker: str, tag: str) -> int:
    """在真实沙箱参数下验证 CJK 渲染。"""
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="iflow-sandbox-verify-"))
    (work / "out").mkdir(parents=True, exist_ok=True)

    script = work / "verify.py"
    script.write_text(VERIFY_PAYLOAD, encoding="utf-8")

    print()
    print("=== 构建后验证（真实沙箱参数）===")

    result = subprocess.run(
        [
            docker, "run", "--rm",
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "--memory", "1024m",
            "--cpus", "1.0",
            "--pids-limit", "128",
            "--tmpfs", "/tmp:rw,nosuid,size=256m",
            "--mount", f"type=bind,source={script},target=/verify.py,readonly",
            "--mount", f"type=bind,source={work / 'out'},target=/out",
            "--env", "MPLCONFIGDIR=/tmp/mpl",
            "--env", "XDG_CACHE_HOME=/tmp",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            tag,
            "python", "/verify.py",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    print(result.stdout.strip() or "(无 stdout)")
    if result.returncode != 0:
        print("verify failed, stderr tail:", result.stderr[-500:])

    fontconfig_errors = result.stderr.count("Fontconfig error")
    print(f"  fontconfig errors: {fontconfig_errors}")
    print(f"  work dir: {work}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="构建沙箱镜像")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if not docker_available(args.docker):
        print("docker daemon is not available; cannot build sandbox image")
        return 2

    if build(args.docker, args.tag) != 0:
        print("image build failed")
        return 1

    inspect(args.docker, args.tag)

    if not args.skip_verify:
        if verify(args.docker, args.tag) != 0:
            print("post-build verification failed")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
