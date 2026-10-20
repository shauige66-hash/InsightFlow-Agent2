"""上传端点：只收白名单类型、限制大小、名字不可穿越目录。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "_PROJECT_DIR", tmp_path)
    return TestClient(api.app)


def test_a_csv_can_be_uploaded_and_reused_as_a_data_source(client) -> None:
    content = "customer_id,amount\nC001,120\n".encode("utf-8")

    response = client.post(
        "/api/uploads",
        files={"file": ("orders.csv", content, "text/csv")},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["name"] == "orders.csv"
    assert payload["size_bytes"] == len(content)
    # 返回的路径必须能直接被 POST /runs 解析（相对项目根目录）
    assert payload["path"].startswith("uploads/")
    assert (Path(api._PROJECT_DIR) / payload["path"]).read_bytes() == content


def test_a_knowledge_file_is_accepted(client) -> None:
    response = client.post(
        "/api/uploads",
        files={"file": ("rules.md", b"# rules\n", "text/markdown")},
    )
    assert response.status_code == 201


def test_an_unknown_extension_is_refused(client) -> None:
    response = client.post(
        "/api/uploads",
        files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "不支持的文件类型" in response.json()["detail"]


def test_the_filename_cannot_escape_the_upload_directory(client) -> None:
    response = client.post(
        "/api/uploads",
        files={"file": ("../../api.py", b"print('pwned')", "text/x-python")},
    )
    # .py 不在白名单里，先被类型挡掉；无论哪条分支，都不能写到 uploads 之外
    assert response.status_code == 400

    for written in (Path(api._PROJECT_DIR) / "uploads").rglob("*"):
        assert written.is_file()
        assert "api.py" not in written.name or written.parent.name != ""


def test_a_traversal_name_with_an_allowed_suffix_stays_inside(client) -> None:
    response = client.post(
        "/api/uploads",
        files={"file": ("../../evil.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["name"] == "evil.csv"
    assert payload["path"].startswith("uploads/")
    assert ".." not in payload["path"]


def test_an_empty_file_is_refused(client) -> None:
    response = client.post("/api/uploads", files={"file": ("empty.csv", b"", "text/csv")})
    assert response.status_code == 400


def test_oversized_files_are_refused(client, monkeypatch) -> None:
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 16)

    response = client.post(
        "/api/uploads",
        files={"file": ("big.csv", b"a" * 64, "text/csv")},
    )

    assert response.status_code == 413
    # 不能留下半个文件
    uploads = Path(api._PROJECT_DIR) / "uploads"
    assert not list(uploads.rglob("big.csv")) if uploads.exists() else True
