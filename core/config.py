"""模型配置。

模型访问统一经由 ModelGate（OpenAI 兼容）。这里只负责把环境变量整理成
不可变的 Settings，不在导入阶段做网络探测，也不强制要求密钥存在——
强制校验发生在真正要构造客户端的时候（core/llm.py），
这样没有凭证的环境也能导入并运行使用 Mock 模型的测试。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_CLIENT_APP = "insightflow-agent2"
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    client_app: str = DEFAULT_CLIENT_APP
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv("DEEPSEEK_MODEL", ""),
        client_app=os.getenv("LLM_CLIENT_APP", DEFAULT_CLIENT_APP),
        timeout_seconds=float(
            os.getenv("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        ),
    )


def validate(settings: Settings) -> None:
    """构造真实客户端前校验。缺失时抛错，避免把 401 误判成上游故障。"""
    if not settings.api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is missing（应填入 ModelGate 的 client key）"
        )
    if not settings.model:
        raise ValueError("DEEPSEEK_MODEL is missing")
