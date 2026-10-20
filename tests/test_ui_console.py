"""操作界面的最小契约。

界面是零构建的静态页 + 两个只读端点。这里只验证"能打开、能列文件、不会越过
既定接口"，不验证浏览器里的交互（那部分靠人工与截图检查）。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import api


def test_root_redirects_to_the_console() -> None:
    client = TestClient(api.app)
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


def test_the_console_page_is_served() -> None:
    client = TestClient(api.app)
    response = client.get("/ui/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "InsightFlow" in response.text


def test_the_page_never_embeds_a_remote_asset() -> None:
    """离线可用是这套界面的前提：不能依赖 CDN。"""
    client = TestClient(api.app)
    body = client.get("/ui/").text

    assert "http://" not in body
    assert "https://" not in body


def test_options_list_the_seed_files() -> None:
    client = TestClient(api.app)
    payload = client.get("/ui-options").json()

    assert "data/customers.csv" in payload["data_sources"]
    assert "knowledge/business_rules.md" in payload["knowledge_sources"]
    # 必须是相对项目根目录的 POSIX 路径：POST /runs 按工作目录解析
    for name in payload["data_sources"] + payload["knowledge_sources"]:
        assert not name.startswith("/")
        assert ":" not in name
        assert ".." not in name
        assert "\\" not in name


def test_options_reach_into_the_case_subdirectory() -> None:
    """`data/retail/` 里的案例数据也要能被选到，否则界面上看不到。"""
    client = TestClient(api.app)
    payload = client.get("/ui-options").json()

    assert "data/retail/orders_eu_2011.csv" in payload["data_sources"]


def test_report_html_is_404_for_an_unknown_run() -> None:
    client = TestClient(api.app)
    response = client.get("/runs/run_does_not_exist/report.html")

    assert response.status_code == 404


class TestRunArtifactRoute:
    """报告用相对路径引用图片，所以这个路由必须能取到它们——但不能变成任意读文件。"""

    def test_a_missing_chart_is_404(self) -> None:
        client = TestClient(api.app)
        assert client.get("/runs/run_does_not_exist/chart_1.png").status_code == 404

    def test_only_images_are_served(self) -> None:
        client = TestClient(api.app)
        assert client.get("/runs/run_x/analysis_report.html").status_code == 404
        assert client.get("/runs/run_x/runs.sqlite").status_code == 404

    def test_path_traversal_is_refused(self) -> None:
        client = TestClient(api.app)
        # 编码后的 ../ 由 Starlette 解码后进入路径参数；即使命中也不得读到目录外文件
        assert client.get("/runs/run_x/..%2F..%2Fapi.py").status_code == 404
