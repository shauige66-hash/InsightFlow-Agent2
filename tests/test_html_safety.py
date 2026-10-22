"""报告 HTML 的安全边界（IF-04）。

报告正文由模型生成，模型又读过用户上传的文档，因此正文必须按不可信内容处理。
清洗之外还有第二道：CSP，以及导出 PDF 时禁用 JS 并拦截外部请求。
"""

from __future__ import annotations

import pytest

from utils.artifacts import save_html_report
from utils.html_safety import REPORT_CSP, sanitize_report_html


class TestSanitizer:
    def test_script_is_removed_with_its_content(self) -> None:
        assert sanitize_report_html("<script>alert(1)</script><p>ok</p>") == "<p>ok</p>"

    def test_style_is_removed(self) -> None:
        assert "style" not in sanitize_report_html("<style>body{}</style>ok")

    def test_event_handlers_are_removed(self) -> None:
        assert "onerror" not in sanitize_report_html('<img src="x" onerror="alert(1)">')
        assert "onclick" not in sanitize_report_html("<div onclick='steal()'>x</div>")

    def test_javascript_urls_are_removed(self) -> None:
        assert "javascript" not in sanitize_report_html('<a href="javascript:alert(1)">x</a>')

    def test_iframe_and_object_are_removed(self) -> None:
        for payload in ("<iframe src='https://e.x'></iframe>", "<object data='x'></object>"):
            rendered = sanitize_report_html(payload)
            assert "iframe" not in rendered
            assert "object" not in rendered

    def test_report_markup_survives(self) -> None:
        table = "<table><tr><td>1</td></tr></table>"
        assert sanitize_report_html(table) == table

        link = '<a href="https://ok.example" title="t">链接</a>'
        rendered = sanitize_report_html(link)
        assert 'href="https://ok.example"' in rendered
        assert 'title="t"' in rendered and '链接' in rendered

    def test_relative_image_is_kept(self) -> None:
        """报告里的图表是相对路径，必须保留，否则 PDF 里没有图。"""
        assert sanitize_report_html('<img src="chart_1.png" alt="图">') == (
            '<img src="chart_1.png" alt="图">'
        )


class TestReportDocument:
    def test_saved_report_carries_csp_and_no_script(self, tmp_path) -> None:
        path = save_html_report(
            "正文\n\n<script>alert(1)</script>\n\n| a | b |\n|---|---|\n| 1 | 2 |",
            output_path=str(tmp_path / "report.html"),
        )

        html = open(path, encoding="utf-8").read()
        assert "<script>" not in html
        assert REPORT_CSP in html
        assert "<table>" in html

    def test_pdf_export_still_works_for_a_report_with_an_image(self, tmp_path) -> None:
        """清洗与禁网之后，带图的报告仍要能导出 PDF。"""
        import struct
        import zlib

        from utils.artifacts import save_pdf_report

        def _png(width: int = 1, height: int = 1) -> bytes:
            def chunk(kind: bytes, payload: bytes) -> bytes:
                body = kind + payload
                return struct.pack(">I", len(payload)) + body + struct.pack(
                    ">I", zlib.crc32(body) & 0xFFFFFFFF
                )

            header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
            raw = b"".join(b"\x00" + b"\x00\x00\x00\x00" * width for _ in range(height))
            return (
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", header)
                + chunk(b"IDAT", zlib.compress(raw))
                + chunk(b"IEND", b"")
            )

        (tmp_path / "chart_1.png").write_bytes(_png())
        html_path = save_html_report(
            "![图](chart_1.png)", output_path=str(tmp_path / "report.html")
        )

        try:
            pdf = save_pdf_report(html_path, output_path=str(tmp_path / "report.pdf"))
        except RuntimeError as error:  # 没有可用浏览器时按既有降级处理
            pytest.skip(str(error))

        assert open(pdf, "rb").read(4) == b"%PDF"
