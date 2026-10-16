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

## 与预算的关系（P3）

**每一次 `chain.invoke` 都是一次真实的上游模型请求**，因此重试会真实地多消耗
预算。这里把计数与判定做成两个回调，而不是让调用方只看"逻辑调用次数"：

- `before_request(attempt)`：每次请求**之前**调用，可用于判预算；
  抛出的异常（`BudgetExceeded`）会直接冒泡，**不**被当成解析失败继续重试；
- `on_request(attempt)`：每次请求**之后**调用，用于计数。

为什么必须在这里做：解析失败的重试发生在 `utils/idempotency.run_agent_call`
的账本区间内部，那里只看到"一次逻辑调用"。不在这里计数，`max_llm_calls`
就会被内部重试绕过——声明上限 2 却可能发出 3 次请求。
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
    before_request: Callable[[int], None] | None = None,
    on_request: Callable[[int], None] | None = None,
) -> Any:
    """调用 chain 并返回经过校验的结果。

    validate 用于检查并提取必填字段；它抛出的异常与解析失败同等对待，都会触发重试。
    没有 validate 时退化为"必须返回 dict"。

    `before_request` / `on_request` 见模块开头：它们让**每一次真实请求**
    都受预算约束并被如实计数，而不是只算一次逻辑调用。
    """
    last_error: Exception | None = None
    current_inputs = {**inputs, "retry_hint": ""}

    for attempt in range(1, max_attempts + 1):
        # 判定与计数都放在请求**之前**：这样即使这次请求失败也已经被计入
        # （"反复失败重发"不能绕过上限），而且计数与判定读到的口径一致。
        if before_request is not None:
            before_request(attempt)
        if on_request is not None:
            on_request(attempt)

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
    before_request: Callable[[int], None] | None = None,
    on_request: Callable[[int], None] | None = None,
) -> str:
    """纯文本输出（报告）的重试。

    与结构化输出不同，这里没有 JSON 解析，唯一的失败模式是空内容——
    对下游没有任何意义，值得重试而不是交付一份空报告。

    回调语义与 `invoke_json_with_retry` 完全一致。
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        # 判定与计数都在请求之前，口径与结构化输出那条一致
        if before_request is not None:
            before_request(attempt)
        if on_request is not None:
            on_request(attempt)

        try:
            raw = chain.invoke(inputs)
            return validate(raw)
        except (OutputParserException, ValueError, TypeError) as error:
            last_error = error
            if attempt == max_attempts:
                break

    raise StructuredOutputError(
        f"文本输出在 {max_attempts} 次尝试后仍不可用：{last_error}"
    ) from last_error

