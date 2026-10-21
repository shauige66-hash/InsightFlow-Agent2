# 负责把分析过程中生成的报告等产物保存到本地文件。

from pathlib import Path


def save_markdown_report(
    content: str,
    output_path: str = "outputs/analysis_report.md"
) -> str:

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        content,
        encoding="utf-8"
    )

    return str(path)


def save_html_report(
    content: str,
    output_path: str = "outputs/analysis_report.html"
) -> str:

    import markdown

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    body = markdown.markdown(
        content,
        extensions=[
            "tables",
            "fenced_code"
        ]
    )

    # 正文来自模型，模型读过用户上传的文档——必须当不可信内容处理：
    # 不过白名单就直接拼进 HTML，等于交付一个能读到控制台 token 的存储型 XSS。
    from utils.html_safety import REPORT_CSP, sanitize_report_html

    body = sanitize_report_html(body)

    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="UTF-8">

    <meta http-equiv="Content-Security-Policy" content="{REPORT_CSP}">
    <meta name="referrer" content="no-referrer">

    <title>InsightFlow Analysis Report</title>

    <style>
        body {{
            max-width: 1000px;
            margin: 40px auto;
            padding: 0 30px;
            font-family: Arial, sans-serif;
            line-height: 1.7;
        }}

        h1, h2, h3 {{
            margin-top: 32px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}

        th {{
            background: #f5f5f5;
        }}

        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
        }}

        code {{
            background: #f5f5f5;
            padding: 2px 5px;
        }}
    </style>
</head>

<body>

{body}

</body>

</html>
"""

    path.write_text(
        html,
        encoding="utf-8"
    )

    return str(path)



def launch_pdf_browser(playwright):

    browser_options = [
        {
            "name": "Google Chrome",
            "kwargs": {
                "channel": "chrome"
            }
        },
        {
            "name": "Microsoft Edge",
            "kwargs": {
                "channel": "msedge"
            }
        },
        {
            "name": "Playwright Chromium",
            "kwargs": {}
        }
    ]

    errors = []

    for option in browser_options:

        try:
            browser = playwright.chromium.launch(
                **option["kwargs"]
            )

            print(
                f"PDF browser: {option['name']}"
            )

            return browser

        except Exception as error:

            errors.append(
                f"{option['name']}: {error}"
            )

    raise RuntimeError(
        "No supported browser is available "
        "for PDF generation.\n"
        + "\n".join(errors)
    )



def save_pdf_report(
    html_path: str,
    output_path: str = "outputs/analysis_report.pdf"
) -> str:

    from playwright.sync_api import sync_playwright

    source_path = Path(
        html_path
    ).resolve()

    pdf_path = Path(
        output_path
    )

    pdf_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with sync_playwright() as playwright:

        browser = launch_pdf_browser(
            playwright
        )

        page = browser.new_page(java_script_enabled=False)

        # 报告里可能引用了远程图片或链接：导出 PDF 时不让它们真的发出去。
        # 只放行本地文件与内联数据，其余一律 abort——PDF 不该成为外发通道。
        def _local_only(route) -> None:
            url = route.request.url
            if url.startswith("file://") or url.startswith("data:") or url.startswith("about:"):
                route.continue_()
            else:
                route.abort()

        page.route("**/*", _local_only)

        page.goto(
            source_path.as_uri(),
            wait_until="load"
        )

        page.pdf(
            path=str(pdf_path.resolve()),
            format="A4",
            print_background=True,
            margin={
                "top": "15mm",
                "right": "15mm",
                "bottom": "15mm",
                "left": "15mm",
            }
        )

        browser.close()

    return str(pdf_path)

