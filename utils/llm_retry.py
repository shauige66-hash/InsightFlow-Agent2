"""结构化 JSON 输出的重试。

只处理"模型返回的内容无法解析成预期结构"这一类失败，不负责网络或上游错误——
那些应该由调用方或网关决定是否重试。

两个容易踩的点：

1. 必填字段的检查必须放在重试范围内。否则 JSON 合法但缺字段时会在包装外
   抛 KeyError，整条工作流直接崩掉，而这恰恰是最值得重试的情况。
2. 重试时把上一次的失败原因一并回传，否则 temperature=0 下会得到同样的错误输出，
   重试纯粹是重复消耗。

失败原因写入 `retry_hint` 输入项。调用方的 prompt 需要引用 `{retry_hint}` 才能生效；
未引用时该参数被忽略，行为与不重试提示一致。
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.exceptions import OutputParserException

DEFAULT_MAX_ATTEMPTS = 3

RETRY_HINT_TEMPLATE = (
    "\n上一次的返回无法使用，原因：{error}\n"
    "请只返回符合要求的 JSON，不要包含解释文字或 Markdown 代码块。"
)


class StructuredOutputError(RuntimeError):
    """多次尝试后仍未取得可用的结构化输出。"""


def invoke_json_with_retry(
    chain,
    inputs: dict[str, Any],
    validate: Callable[[dict[str, Any]], Any] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Any:
    """调用 chain 并返回经过校验的结果。

    validate 用于检查并提取必填字段；它抛出的异常与解析失败同等对待，都会触发重试。
    没有 validate 时退化为"必须返回 dict"。
    """
    last_error: Exception | None = None
    current_inputs = {**inputs, "retry_hint": ""}

    for attempt in range(1, max_attempts + 1):
        try:
            result = chain.invoke(current_inputs)

            if not isinstance(result, dict):
                raise ValueError("Structured LLM output must be a JSON object.")

            if validate is None:
                return result

            return validate(result)

        except (OutputParserException, ValueError, KeyError, TypeError) as error:
            last_error = error

            if attempt == max_attempts:
                break

            current_inputs = {
                **inputs,
                "retry_hint": RETRY_HINT_TEMPLATE.format(error=str(error)[:300]),
            }

    raise StructuredOutputError(
        f"结构化输出在 {max_attempts} 次尝试后仍不可用：{last_error}"
    ) from last_error


def invoke_report_with_retry(
    chain,
    inputs: dict[str, Any],
    validate: Callable[[Any], str],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> str:
    """纯文本输出（报告）的重试。

    与结构化输出不同，这里没有 JSON 解析，唯一的失败模式是空内容——
    对下游没有任何意义，值得重试而不是交付一份空报告。
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return validate(chain.invoke(inputs))
        except (OutputParserException, ValueError, TypeError) as error:
            last_error = error
            if attempt == max_attempts:
                break

    raise StructuredOutputError(
        f"文本输出在 {max_attempts} 次尝试后仍不可用：{last_error}"
    ) from last_error

