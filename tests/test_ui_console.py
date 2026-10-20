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

    assert "customers.csv" in payload["data_sources"]
    assert "business_rules.md" in payload["knowledge_sources"]
    # 只暴露固定目录里的文件名，不暴露绝对路径
    for name in payload["data_sources"] + payload["knowledge_sources"]:
        assert "/" not in name and "\\" not in name


def test_report_html_is_404_for_an_unknown_run() -> None:
    client = TestClient(api.app)
    response = client.get("/runs/run_does_not_exist/report.html")

    assert response.status_code == 404
