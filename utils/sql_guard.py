"""SQL 只读判定。

P2 的沙箱与权限加固尚未实施，因此在 P1 中只读约束必须是硬边界：
非只读语句一律拒绝执行，**不能通过人工审批放行**。审批用来确认"要不要做
这项分析"，而不是用来提升权限。

判定做法是保守的纯词法分析：
- 去掉注释与字符串字面量后再匹配，避免 `/*x*/WITH ... DELETE` 绕过前缀检查；
- 只接受单条语句；
- 出现任何写操作关键字即拒绝；
- 无法确定时拒绝（fail-closed）。

这不是安全沙箱——真正的隔离需要独立的受限进程（P2）。这里的目标是让
"非只读 SQL 不会因为一次误批而被执行"。
"""

from __future__ import annotations

import re

# 写操作与 DDL/DCL 关键字。出现即拒绝，不做任何例外。
_FORBIDDEN = re.compile(
    r"\b("
    r"insert|update|delete|drop|create|alter|replace|truncate|grant|revoke|"
    r"attach|detach|pragma|vacuum|reindex|analyze|begin|commit|rollback|"
    r"savepoint|release|end|load_extension|"
    r"into|set|call|exec|execute"
    r")\b",
    re.IGNORECASE,
)

# `WITH ... SELECT` 合法，但 `WITH ... DELETE` 不合法，因此由 _FORBIDDEN 兜住。
_READ_ONLY_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_QUOTED_IDENTIFIER = re.compile(r'"(?:[^"]|"")*"')


class SqlNotReadOnly(ValueError):
    """语句不是只读查询，且不可通过审批放行。"""


def strip_noise(statement: str) -> str:
    """去掉注释与字面量，只留下结构性关键字。

    这一步是必要的：注释里的关键字可能被误判为写操作，
    而注释也可能被用来隐藏真正的写操作。
    """
    cleaned = _BLOCK_COMMENT.sub(" ", statement)
    cleaned = _LINE_COMMENT.sub(" ", cleaned)
    cleaned = _STRING_LITERAL.sub("''", cleaned)
    cleaned = _QUOTED_IDENTIFIER.sub('""', cleaned)
    return cleaned


def is_read_only_statement(statement: str) -> bool:
    """判断是否可安全执行。无法确定时返回 False。"""
    if not isinstance(statement, str) or not statement.strip():
        return False

    cleaned = strip_noise(statement).strip()

    # 只允许单条语句：分号后再出现内容说明是多语句
    without_trailing = cleaned.rstrip(";").strip()
    if ";" in without_trailing:
        return False

    if not _READ_ONLY_START.match(without_trailing):
        return False

    if _FORBIDDEN.search(without_trailing):
        return False

    return True


def ensure_read_only(statement: str) -> str:
    """只读则原样返回，否则抛错。"""
    if not is_read_only_statement(statement):
        raise SqlNotReadOnly(
            "仅允许只读 SQL（单条 SELECT / WITH 查询）。"
            "非只读语句不能被执行，也不能通过人工审批放行。"
        )
    return statement
