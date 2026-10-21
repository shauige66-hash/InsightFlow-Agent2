"""报告 HTML 的白名单清洗。

报告正文来自模型，模型又读过用户上传的文档和表格——里面任何一段文字都可能带着
`<script>`、`onerror=` 或 `javascript:` 链接。Markdown 转换默认**原样保留**这些
HTML，于是一份"分析报告"就是一个能读到控制台 token 的存储型 XSS 载体。

这里不做"过滤坏词"，而是白名单：只允许报告确实需要的标签，其余整个丢掉；属性只
保留安全的那几个；链接与图片的协议受限。清洗完再包一层 CSP，作为第二道。
"""

from __future__ import annotations

from html import escape
from html.parser import HTMLParser

# 报告会用到的标签。svg / math / iframe / object / embed / form / script / style
# 一律不在其中——它们要么能执行脚本，要么能发起外部请求。
ALLOWED_TAGS = frozenset(
    {
        "p", "br", "hr", "div", "span",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li",
        "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup",
        "strong", "b", "em", "i", "u", "s", "del", "ins", "mark", "small",
        "code", "pre", "blockquote", "a", "img",
        "sup", "sub", "figure", "figcaption",
    }
)

# 允许的属性。刻意没有 style / class / id：样式由模板统一给，避免用样式做覆盖。
ALLOWED_ATTRS = {
    "a": {"href", "title"},
    "img": {"src", "alt", "title", "width", "height"},
    "th": {"colspan", "rowspan", "scope"},
    "td": {"colspan", "rowspan"},
    "colgroup": {"span"},
    "*": set(),
}

ALLOWED_SCHEMES = ("http://", "https://", "mailto:", "#", "/", "./", "../", "data:image/")


_SCHEME = __import__("re").compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


def _safe_url(value: str) -> bool:
    """判断链接/图片地址是否可保留。

    关键是别把相对路径当成非法：报告里的图表就是 `chart_1.png` 这种裸文件名，
    误判会让 PDF 里一张图都没有——写第一版时就踩了，被用例抓住。
    """
    candidate = (value or "").strip()
    if not candidate:
        return False

    lowered = candidate.lower()

    # 显式写出的危险协议直接拒绝；data: 只允许图片（报告里可能内联小图标）
    if lowered.startswith(("javascript:", "vbscript:", "file:", "blob:")):
        return False
    if lowered.startswith("data:"):
        return lowered.startswith("data:image/")

    # 协议相对地址（//host/x）与 Windows UNC（\\\\host）指向外部
    if lowered.startswith("//") or lowered.startswith("\\\\"):
        return False

    if _SCHEME.match(lowered):
        return lowered.startswith(("http://", "https://", "mailto:"))

    # 没有协议：相对路径或锚点，保留
    return True


class _Sanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self._stack: list[str] = []
        self._dropping: list[str] = []

    # -- 标签 --
    def handle_starttag(self, tag: str, attrs) -> None:
        if self._dropping:
            # 被丢弃元素内部的标签不再处理，直到它的结束标签出现
            if tag == self._dropping[-1]:
                self._dropping.pop()
            return

        if tag not in ALLOWED_TAGS:
            # 非空元素才需要等它的结束标签；script/style 这类必须整体吞掉内容
            if tag not in {"br", "hr", "img", "input", "meta", "link", "base"}:
                self._dropping.append(tag)
            return

        allowed = ALLOWED_ATTRS.get(tag, set()) | ALLOWED_ATTRS["*"]
        rendered = []
        for name, value in attrs:
            name = (name or "").lower()
            # 事件属性（onerror/onclick…）在任何标签上都不要
            if name.startswith("on") or name not in allowed:
                continue
            if name in {"href", "src"} and not _safe_url(value or ""):
                continue
            rendered.append(f' {name}="{escape(value or "", quote=True)}"')

        self.parts.append(f"<{tag}{''.join(rendered)}>")
        if tag not in {"br", "hr", "img"}:
            self._stack.append(tag)

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self._dropping:
            if tag == self._dropping[-1]:
                self._dropping.pop()
            return
        if tag in ALLOWED_TAGS and tag in self._stack:
            while self._stack:
                open_tag = self._stack.pop()
                self.parts.append(f"</{open_tag}>")
                if open_tag == tag:
                    break

    # -- 文本 --
    def handle_data(self, data: str) -> None:
        if self._dropping:
            return
        self.parts.append(escape(data, quote=False))

    def handle_entityref(self, name: str) -> None:
        if not self._dropping:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self._dropping:
            self.parts.append(f"&#{name};")

    # -- 注释 / 声明 --
    def handle_comment(self, data: str) -> None:
        return

    def handle_decl(self, decl: str) -> None:
        return

    def handle_pi(self, data: str) -> None:
        return

    def result(self) -> str:
        while self._stack:
            self.parts.append(f"</{self._stack.pop()}>")
        return "".join(self.parts)


def sanitize_report_html(html: str) -> str:
    """把 Markdown 转换后的 HTML 过一遍白名单。"""
    parser = _Sanitizer()
    parser.feed(html or "")
    parser.close()
    return parser.result()


# 第二道防线：即使清洗漏了东西，页面也不允许执行脚本或发外部请求。
# img-src 允许 file: 与 data:，因为报告里的图表是本地相对路径。
REPORT_CSP = (
    "default-src 'none'; "
    "img-src 'self' file: data:; "
    "style-src 'unsafe-inline'; "
    "font-src 'self' data:; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)
