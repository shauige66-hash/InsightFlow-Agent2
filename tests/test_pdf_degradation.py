"""部署相关的降级行为：没有浏览器时，报告照样产出，只是没有 PDF。"""

from __future__ import annotations

import nodes


def test_pdf_failure_degrades_instead_of_failing_the_run(bound_runtime, monkeypatch) -> None:
    """容器/精简环境里可能一个浏览器都没有。

    PDF 是三种产物里唯一的"锦上添花"：缺了它，Markdown 与 HTML 仍然是一份可交付
    的报告；如果这里抛错，整个分析会在最后一步变成 failed。
    """

    def no_browser(*args, **kwargs):
        raise RuntimeError("Executable doesn't exist: chromium")

    monkeypatch.setattr(nodes, "save_pdf_report", no_browser)

    result = nodes.artifact_node(
        {
            "run_id": "test-run",
            "report_content": "# 分析报告\n\n本期 GMV 环比 +2.0%。",
            "analysis_plan": [{"task_id": "task_1"}],
            "failed_tasks": [],
            "artifacts": [],
        }
    )

    assert result["run_status"] == "completed"
    assert result["analysis_status"] == "completed"
    assert result["report_path"].endswith("analysis_report.md")
    assert not any(str(path).endswith(".pdf") for path in result["artifacts"])
    assert any(str(path).endswith(".html") for path in result["artifacts"])


def test_a_successful_pdf_is_registered(bound_runtime, monkeypatch) -> None:
    def fake_pdf(html_path, output_path):
        with open(output_path, "wb") as handle:
            handle.write(b"%PDF-1.4 fake")
        return output_path

    monkeypatch.setattr(nodes, "save_pdf_report", fake_pdf)

    result = nodes.artifact_node(
        {
            "run_id": "test-run",
            "report_content": "# 分析报告\n\n有结论。",
            "analysis_plan": [{"task_id": "task_1"}],
            "failed_tasks": [],
            "artifacts": [],
        }
    )

    assert any(str(path).endswith(".pdf") for path in result["artifacts"])
