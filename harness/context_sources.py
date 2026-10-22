"""上下文来源的选择与格式化：长期记忆、多轮历史、知识证据。

三者的优先级必须显式，不能混成一段文字：

1. **系统规则与工具权限**永远不来自这里——记忆只能填充 Prompt 字段，
   不会进入策略、预算或工具白名单。
2. **当前用户指令**优先于一般历史偏好：Prompt 里先给当前指令，再给记忆，
   并明确说明冲突时以当前指令为准。
3. **业务事实（知识证据）**与用户偏好分开呈现：文档里的文字不能被当成
   用户确认过的设置，反之亦然。

所有来源都按预算截断：记忆最多 5 条、每条 300 字符；对话最多 4 条消息 +
3 个上一轮摘要。
"""

from __future__ import annotations

from typing import Any

#: 注入长期记忆的 Agent：需要知道输出偏好与指标定义的那些
MEMORY_AGENTS = frozenset({"planner", "executor", "critic", "refine", "reporter", "visualizer", "plotter"})

#: 注入多轮对话历史的 Agent：只有规划需要理解"刚才的结果"
CONVERSATION_AGENTS = frozenset({"planner"})

MAX_MEMORIES = 5
MAX_MEMORY_CHARS = 300
MAX_RECENT_MESSAGES = 4
MAX_MESSAGE_CHARS = 300
MAX_PREVIOUS_TURNS = 3


def format_memories(memories: list[dict[str, Any]]) -> str:
    """已确认的偏好；候选记忆不会出现在这里。"""
    lines: list[str] = []
    for item in memories[:MAX_MEMORIES]:
        if item.get("status") != "confirmed":
            continue
        value = str(item.get("value") or "").strip()
        if not value:
            continue
        if len(value) > MAX_MEMORY_CHARS:
            value = value[:MAX_MEMORY_CHARS] + "…"
        kind = item.get("kind") or "preference"
        scope = item.get("scope") or "global"
        lines.append(f"- [{kind} / {scope}] {value}")
    return "\n".join(lines)


def format_conversation(context: dict[str, Any]) -> str:
    """最近对话与上一轮结果引用。没有真实结果时不写任何结论。"""
    blocks: list[str] = []

    messages = [
        item
        for item in (context.get("recent_messages") or [])
        if item.get("content")
    ][-MAX_RECENT_MESSAGES:]
    if messages:
        rendered = []
        for item in messages:
            role = "用户" if item.get("role") == "user" else "助手"
            text = str(item["content"]).strip()
            if len(text) > MAX_MESSAGE_CHARS:
                text = text[:MAX_MESSAGE_CHARS] + "…"
            rendered.append(f"{role}：{text}")
        blocks.append("最近对话：\n" + "\n".join(rendered))

    previous = [
        item for item in (context.get("previous_turns") or []) if item.get("summary")
    ][-MAX_PREVIOUS_TURNS:]
    if previous:
        rendered = [
            f"第 {item.get('seq')} 轮（{item.get('status')}）：{item['summary']}"
            for item in previous
        ]
        blocks.append("之前的分析轮次：\n" + "\n".join(rendered))

    return "\n\n".join(blocks)


def inject_context_sources(
    agent: str,
    fields: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """把记忆与对话写进 Agent 的输入字段；返回写入情况。"""
    memory_count = 0
    if agent in MEMORY_AGENTS and _is_placeholder(fields.get("memories")):
        memories = state.get("memories") or []
        if agent in {"reporter", "visualizer", "plotter"}:
            memories = [item for item in memories if item.get("kind") in {"preference", "report_format"}]
            if _is_placeholder(fields.get("current_instruction")):
                fields["current_instruction"] = str(state.get("analysis_focus") or "无")
        text = format_memories(memories)
        if text:
            fields["memories"] = text
            memory_count = text.count("\n- ") + (1 if text.startswith("- ") else 0)

    if agent in CONVERSATION_AGENTS and _is_placeholder(
        fields.get("conversation_history")
    ):
        context = state.get("conversation_context") or {}
        text = format_conversation(context)
        if text:
            fields["conversation_history"] = text

    return {"memory_entries": memory_count}


PLACEHOLDERS = (None, "", "无")


def _is_placeholder(value: Any) -> bool:
    """字段为空或是 Agent 放的占位值时才注入。

    直接判断"字段不存在"会让 Agent 无法为 Prompt 提供默认值，而 PromptTemplate
    缺键会直接报错。
    """
    return value in PLACEHOLDERS


def knowledge_chunk_count(fields: dict[str, Any]) -> int:
    """这个 Agent 实际拿到的知识片段数。"""
    value = fields.get("retrieved_knowledge")
    if isinstance(value, list):
        return len([item for item in value if item])
    if isinstance(value, str) and value.strip():
        return 1
    return 0


__all__ = [
    "CONVERSATION_AGENTS",
    "MEMORY_AGENTS",
    "format_conversation",
    "format_memories",
    "inject_context_sources",
    "knowledge_chunk_count",
]
