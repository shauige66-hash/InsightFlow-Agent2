"""Inert tool boundary for workflow/LLM identity tests, never evals model code.

This is deliberately a small whitelist, not a replacement sandbox: unknown code
fails the test. Docker/sandbox integration tests keep their real boundaries.
"""
from __future__ import annotations

import base64
from pathlib import Path
import re

from core.runtime import current_runtime


def invoke_scripted_tool(name, tables, arguments, *, snapshots=None):
    code = str(arguments.get("code", arguments.get("query", ""))).strip()
    raised = re.fullmatch(r"raise ValueError\((['\"])(.*?)\1\)\s*", code, re.DOTALL)
    if raised:
        return {"success": False, "result": None, "error": f"ValueError: {raised.group(2)}", "error_kind": "python_error"}
    if name == "python":
        match = re.fullmatch(r"result\s*=\s*len\(df\)(?:\s*\+\s*(\d+))?", code)
        if not match:
            raise AssertionError(f"Unmodeled code in scripted workflow test: {code!r}")
        resource = tables[arguments["table_names"][0]]
        result = len(resource.dataframe) + int(match.group(1) or 0)
        return {"success": True, "result": result, "error": None, "error_kind": None}
    if name == "plot" and code.startswith("fig = plt.figure()") and "fig.savefig(output_path)" in code:
        path = current_runtime().output_dir() / (arguments["chart_id"] + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="))
        return {"success": True, "result": str(path), "error": None, "error_kind": None}
    raise AssertionError(f"Unmodeled tool in scripted workflow test: {name}: {code!r}")


def no_browser_pdf(*_args, **_kwargs):
    raise RuntimeError("Scripted workflow test does not exercise browser PDF export")


def scripted_pdf(_html_path, *, output_path):
    """Return a valid inert one-page PDF for HTTP artifact routing tests."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(data)
    data.extend(b"xref\n0 5\n0000000000 65535 f \n")
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)
