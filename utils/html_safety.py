"""Allowlisted report markup; no model-supplied active content or file URLs."""
from __future__ import annotations

import base64
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlsplit

_TAGS = frozenset('h1 h2 h3 h4 h5 h6 p br hr strong em b i del blockquote ul ol li pre code table thead tbody tfoot tr th td a img div span'.split())
_VOID = frozenset({'br', 'hr', 'img'})
_BLOCKED = frozenset({'script', 'style', 'iframe', 'object', 'embed', 'svg', 'math', 'template', 'textarea', 'title'})
_IMAGE = re.compile(r'^[A-Za-z0-9_-]+\.png$')
MAX_IMAGE_BYTES = 16 * 1024 * 1024


class _ReportMarkup(HTMLParser):
    def __init__(self, image_root: Path | None = None):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.blocked: list[str] = []
        self.image_root = image_root

    def handle_starttag(self, tag, attrs):
        if self.blocked:
            if tag in _BLOCKED:
                self.blocked.append(tag)
            return
        if tag in _BLOCKED:
            self.blocked.append(tag)
            return
        if tag not in _TAGS:
            return
        values = dict(attrs)
        safe: dict[str, str] = {}
        if tag == 'a':
            href = values.get('href') or ''
            try:
                parsed = urlsplit(href)
            except ValueError:
                href = ''
                parsed = urlsplit('')
            if (href.startswith(('https://', 'http://')) and parsed.netloc
                    and not parsed.username and not any(ord(c) < 32 for c in href)):
                safe.update(href=href, rel='noopener noreferrer nofollow', target='_blank')
            elif re.fullmatch(r'#[A-Za-z0-9_-]+', href):
                safe['href'] = href
        if tag == 'img':
            name = values.get('src') or ''
            if not _IMAGE.fullmatch(name):
                return
            if self.image_root is not None:
                path = self.image_root / name
                if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
                    return
                data = path.read_bytes()
                if not data.startswith(b'\x89PNG\r\n\x1a\n'):
                    return
                safe['src'] = 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
            else:
                safe['src'] = name
            safe['alt'] = values.get('alt') or ''
        if values.get('title'):
            safe['title'] = values['title']
        rendered = ''.join(f' {key}="{escape(value, quote=True)}"' for key, value in safe.items())
        self.parts.append(f'<{tag}{rendered}>')

    def handle_endtag(self, tag):
        if self.blocked:
            if tag == self.blocked[-1]:
                self.blocked.pop()
            return
        if tag in _TAGS and tag not in _VOID:
            self.parts.append(f'</{tag}>')

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCKED:
            return
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_data(self, data):
        if not self.blocked:
            self.parts.append(escape(data))


def sanitize_report_html(content: str, *, image_root: Path | None = None) -> str:
    parser = _ReportMarkup(image_root)
    parser.feed(content)
    parser.close()
    return ''.join(parser.parts)

REPORT_CSP = (
    "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
