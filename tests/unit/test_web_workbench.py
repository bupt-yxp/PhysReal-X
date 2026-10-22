"""The live workbench must validate inputs and isolate each job's outputs."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest
from examples.run_all import _implementation_sha256
from examples.web_app import Workbench, WorkbenchServer, _unpack_upload
from examples.web_inputs import validate_job_request


def _capture(root: Path) -> Path:
    capture = root / "examples/real_assets_and_digital_twin/input/ArtModeling/laptop1"
    (capture / "can.start").mkdir(parents=True)
    (capture / "can.end").mkdir()
    return capture


def test_module_two_submission_uses_validated_upstream_capture(tmp_path: Path) -> None:
    capture = _capture(tmp_path)
    chain, inputs = validate_job_request(
        {
            "module": "physical_asset_building",
            "inputs": {"real_assets_and_digital_twin": {"capture_dir": str(capture)}},
        },
        tmp_path,
        (tmp_path / "examples",),
    )
    assert chain == ("real_assets_and_digital_twin", "physical_asset_building")
    assert inputs[chain[0]]["capture_dir"] == str(capture)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="允许范围"):
        validate_job_request(
            {"module": chain[0], "inputs": {chain[0]: {"capture_dir": str(outside)}}},
            tmp_path,
            (tmp_path / "examples",),
        )


def test_zip_upload_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="非法路径"):
        _unpack_upload(archive, tmp_path / "data")
    assert not (tmp_path / "escape.txt").exists()


def test_notebook_receipt_tracks_inputs_but_not_http_presentation(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (examples / "web_app.py").write_text("page = 1\n")
    (examples / "web_worker.py").write_text("transport = 1\n")
    (examples / "web_inputs.py").write_text("default = 1\n")
    baseline = _implementation_sha256(tmp_path)
    (examples / "web_app.py").write_text("page = 2\n")
    (examples / "web_worker.py").write_text("transport = 2\n")
    assert _implementation_sha256(tmp_path) == baseline
    (examples / "web_inputs.py").write_text("default = 2\n")
    assert _implementation_sha256(tmp_path) != baseline


def test_http_auth_submission_cancel_and_unknown_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path)
    monkeypatch.setattr(Workbench, "_work_loop", lambda self: None)
    app = Workbench(tmp_path, tmp_path / "state", token="test-secret")
    try:
        server = WorkbenchServer(("127.0.0.1", 0), app)
    except PermissionError:
        app.close()
        pytest.skip("当前沙箱禁止绑定本机端口；接口测试可在普通环境执行")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path: str, method: str = "GET", body: dict | None = None, cookie: str = ""):
        request = urllib.request.Request(
            base + path,
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Cookie": cookie} if cookie else {}),
            },
        )
        return urllib.request.urlopen(request, timeout=3)

    try:
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            call("/api/modules")
        assert unauthorized.value.code == 401
        login = urllib.request.Request(
            base + "/api/session",
            method="POST",
            headers={"Authorization": "Bearer test-secret"},
        )
        with urllib.request.urlopen(login, timeout=3) as response:
            cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        with call("/api/modules", cookie=cookie) as response:
            assert len(json.load(response)) == 6
        with pytest.raises(urllib.error.HTTPError) as bad:
            call(
                "/api/jobs",
                "POST",
                {
                    "module": "real_assets_and_digital_twin",
                    "inputs": {"real_assets_and_digital_twin": {"capture_dir": "../outside"}},
                },
                cookie,
            )
        assert bad.value.code == 400
        with call(
            "/api/jobs",
            "POST",
            {
                "module": "real_assets_and_digital_twin",
                "inputs": {"real_assets_and_digital_twin": {"capture_dir": str(capture)}},
            },
            cookie,
        ) as response:
            job = json.load(response)
            assert response.status == 202
        with call(f"/api/jobs/{job['id']}", cookie=cookie) as response:
            assert json.load(response)["status"] == "QUEUED"
        with pytest.raises(urllib.error.HTTPError) as hidden:
            call(f"/api/jobs/{job['id']}/artifacts/art_{'0' * 32}", cookie=cookie)
        assert hidden.value.code == 404
        with call(f"/api/jobs/{job['id']}/cancel", "POST", {}, cookie) as response:
            assert json.load(response)["status"] == "CANCELLED"
    finally:
        server.shutdown()
        server.server_close()
        app.close()
        thread.join(timeout=3)
