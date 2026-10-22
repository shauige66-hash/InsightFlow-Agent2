# 负责把分析过程中生成的报告等产物保存到本地文件。

from pathlib import Path
from utils.html_safety import REPORT_CSP, sanitize_report_html
from core.publication import atomic_write, publish_file
import uuid


def save_markdown_report(
    content: str,
    output_path: str = "outputs/analysis_report.md"
) -> str:

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    atomic_write(path, content.encode("utf-8"))

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

    atomic_write(path, html.encode("utf-8"))

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
    staged_pdf = pdf_path.with_name('.' + pdf_path.name + '.' + uuid.uuid4().hex + '.tmp')

    try:
        with sync_playwright() as playwright:

            browser = launch_pdf_browser(
                playwright
            )

            # Re-sanitize even pre-existing reports, inline only validated PNGs from
            # this report's directory, and deny every browser resource request.
            body = sanitize_report_html(
                source_path.read_text(encoding="utf-8"), image_root=source_path.parent
            )
            page = browser.new_page(java_script_enabled=False, service_workers="block")
            page.route("**/*", lambda route: route.abort())
            page.set_content(
                '<!doctype html><html><meta charset="UTF-8">'
                '<style>body{font-family:Arial,sans-serif;line-height:1.7}'
                'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px}'
                'img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
                '<body>' + body + '</body></html>',
                wait_until="load",
            )

            page.pdf(
                path=str(staged_pdf.resolve()),
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

        publish_file(staged_pdf, pdf_path)

    finally:
        staged_pdf.unlink(missing_ok=True)

    return str(pdf_path)

