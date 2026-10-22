"""Single-host web workbench for executing the six real PhysReal-X modules.

The HTTP process never imports GPU libraries. A single subprocess owns the
shared notebook outputs at a time; every completed job keeps its own result
index while large artifacts remain in the immutable platform store.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import mimetypes
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
import zipfile
from contextlib import suppress
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from examples.web_inputs import module_catalog, validate_job_request

ROOT = Path(__file__).resolve().parents[1]
WEB_FILES = ROOT / "web"
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
ARTIFACT_ID = re.compile(r"art_[0-9a-f]{32}\Z")
MAX_JSON_BYTES = 65536
MAX_UPLOAD_BYTES = 12 * 1024**3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _tail(path: Path, limit: int = 16000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def _unpack_upload(archive: Path, destination: Path) -> Path:
    """Extract a bounded ZIP and reject links, traversal and archive bombs."""
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not members or len(members) > 200000:
            raise ValueError("ZIP 文件数量无效或过多")
        total = 0
        for item in members:
            name = PurePosixPath(item.filename)
            if (
                name.is_absolute()
                or not name.parts
                or any(part in {"", ".", ".."} for part in name.parts)
            ):
                raise ValueError("ZIP 包含非法路径")
            if item.create_system == 3 and ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("ZIP 不允许符号链接")
            total += item.file_size
            if total > MAX_UPLOAD_BYTES or (
                item.file_size > 10**8 and item.compress_size * 100 < item.file_size
            ):
                raise ValueError("ZIP 解压大小或压缩比超出限制")
        for item in members:
            target = destination.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    children = [entry for entry in destination.iterdir() if entry.is_dir()]
    if len(children) == 1 and len(list(destination.iterdir())) == 1:
        return children[0]
    return destination


class Workbench:
    def __init__(
        self,
        root: Path = ROOT,
        state_dir: Path | None = None,
        *,
        token: str = "",
        cell_timeout: int = 7200,
    ) -> None:
        self.root = root.resolve()
        self.state_dir = (state_dir or root / "workspace/web").resolve()
        self.jobs_dir = self.state_dir / "jobs"
        self.uploads_dir = self.state_dir / "uploads"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        extra_roots = tuple(
            Path(item).expanduser().resolve()
            for item in os.environ.get("PHYSREALX_WEB_INPUT_ROOTS", "").split(os.pathsep)
            if item
        )
        self.allowed_roots = (self.root / "examples", self.uploads_dir, *extra_roots)
        self.token = token
        self.cell_timeout = cell_timeout
        self.lock = threading.RLock()
        self.pending: queue.Queue[str | None] = queue.Queue()
        self.stopping = False
        self.active: subprocess.Popen[bytes] | None = None
        self.active_id: str | None = None
        for directory in self.jobs_dir.iterdir():
            record_path = directory / "status.json"
            if record_path.is_file():
                record = _read_json(record_path)
                if record.get("status") in {"QUEUED", "RUNNING", "CANCELLING"}:
                    record.update(
                        status="INTERRUPTED", finished_at=_now(), error="服务重启；请重新提交任务"
                    )
                    _write_json(record_path, record)
        self.worker = threading.Thread(
            target=self._work_loop, daemon=True, name="physrealx-web-worker"
        )
        self.worker.start()

    def submit(self, payload: Any) -> dict[str, Any]:
        chain, inputs = validate_job_request(payload, self.root, self.allowed_roots)
        job_id = uuid.uuid4().hex
        directory = self.jobs_dir / job_id
        directory.mkdir()
        request = {
            "module": payload["module"],
            "chain": chain,
            "inputs": inputs,
            "cell_timeout_seconds": self.cell_timeout,
        }
        record = {
            "id": job_id,
            "module": payload["module"],
            "chain": chain,
            "status": "QUEUED",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
        }
        _write_json(directory / "request.json", request)
        _write_json(directory / "status.json", record)
        self.pending.put(job_id)
        return record

    def get(self, job_id: str) -> dict[str, Any]:
        if not JOB_ID.fullmatch(job_id):
            raise FileNotFoundError("任务不存在")
        directory = self.jobs_dir / job_id
        record = _read_json(directory / "status.json")
        record["log_tail"] = _tail(directory / "run.log")
        result_path = directory / "result.json"
        if record["status"] == "COMPLETED" and result_path.is_file():
            record["result"] = _read_json(result_path)
        elif record["status"] in {"RUNNING", "CANCELLING"}:
            progress_path = self.root / "examples/output/notebook-acceptance.json"
            if progress_path.is_file():
                progress = _read_json(progress_path)
                if progress.get("started_at", "") >= (record["started_at"] or ""):
                    record["progress"] = [
                        {"module": item.get("module"), "status": item.get("status")}
                        for item in progress.get("notebooks", ())
                    ]
            for module in record["chain"]:
                receipt_path = self.root / "examples" / module / "output/execution.json"
                if not receipt_path.is_file():
                    continue
                receipt = _read_json(receipt_path)
                if receipt.get("started_at", "") < (record["started_at"] or ""):
                    continue
                cells = receipt.get("cells", ())
                for item in record.setdefault("progress", []):
                    if item["module"] == module:
                        item["cells_completed"] = sum(
                            cell["status"] == "COMPLETED" for cell in cells
                        )
                        item["cells_total"] = len(cells)
        return record

    def list_jobs(self) -> list[dict[str, Any]]:
        records = []
        for path in self.jobs_dir.glob("*/status.json"):
            try:
                records.append(_read_json(path))
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda item: item["created_at"], reverse=True)[:30]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            record = self.get(job_id)
            if record["status"] == "QUEUED":
                self._update(job_id, status="CANCELLED", finished_at=_now())
            elif record["status"] == "RUNNING":
                self._update(job_id, status="CANCELLING")
                if self.active_id == job_id and self.active is not None:
                    with suppress(ProcessLookupError):
                        os.killpg(self.active.pid, signal.SIGTERM)
            else:
                raise ValueError("任务已结束或正在取消")
            return self.get(job_id)

    def _update(self, job_id: str, **changes: Any) -> None:
        path = self.jobs_dir / job_id / "status.json"
        record = _read_json(path)
        record.update(changes)
        _write_json(path, record)

    def _is_stopping(self) -> bool:
        return self.stopping

    def _work_loop(self) -> None:
        while True:
            job_id = self.pending.get()
            if job_id is None:
                return
            with self.lock:
                if self._is_stopping():
                    return
                if self.get(job_id)["status"] != "QUEUED":
                    continue
                self._update(job_id, status="RUNNING", started_at=_now())
            directory = self.jobs_dir / job_id
            try:
                with (directory / "run.log").open("wb") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "examples.web_worker", str(directory)],
                        cwd=self.root,
                        env={
                            **os.environ,
                            "PYTHONUNBUFFERED": "1",
                            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
                        },
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    with self.lock:
                        self.active, self.active_id = process, job_id
                        if self._is_stopping() or self.get(job_id)["status"] == "CANCELLING":
                            with suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGTERM)
                    return_code = process.wait()
            except OSError as error:
                with self.lock:
                    self._update(job_id, status="FAILED", finished_at=_now(), error=str(error))
                continue
            with self.lock:
                self.active, self.active_id = None, None
                if self._is_stopping():
                    self._update(job_id, status="INTERRUPTED", finished_at=_now())
                elif self.get(job_id)["status"] == "CANCELLING":
                    self._update(job_id, status="CANCELLED", finished_at=_now())
                elif return_code == 0 and (directory / "result.json").is_file():
                    self._update(job_id, status="COMPLETED", finished_at=_now())
                else:
                    self._update(
                        job_id,
                        status="FAILED",
                        finished_at=_now(),
                        error=_tail(directory / "run.log", 2000),
                    )

    def close(self) -> None:
        with self.lock:
            self.stopping = True
            if self.active is not None and self.active.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(self.active.pid, signal.SIGTERM)
        self.pending.put(None)
        self.worker.join(timeout=10)
        if self.worker.is_alive() and self.active is not None:
            with suppress(ProcessLookupError):
                os.killpg(self.active.pid, signal.SIGKILL)
            self.worker.join(timeout=5)


class WorkbenchHandler(BaseHTTPRequestHandler):
    server: WorkbenchServer

    def _authorized(self) -> bool:
        token = self.server.workbench.token
        if not token:
            return True
        supplied = self.headers.get("Authorization", "")
        cookie = self.headers.get("Cookie", "")
        session = next(
            (
                item.strip().partition("=")[2]
                for item in cookie.split(";")
                if item.strip().startswith("prx_session=")
            ),
            "",
        )
        return hmac.compare_digest(supplied, f"Bearer {token}") or hmac.compare_digest(
            session, hashlib.sha256(token.encode()).hexdigest()
        )

    def _json(self, status: int, value: Any, *, cookie: str = "") -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, media_type: str, *, download: bool = False) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'",
        )
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile, length=1024 * 1024)

    def _request_json(self) -> Any:
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ValueError("请求必须使用 application/json")
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > MAX_JSON_BYTES:
            raise ValueError("请求大小无效")
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in {"/", "/app.js", "/app.css"}:
            filename = "index.html" if path == "/" else path[1:]
            media_type = mimetypes.guess_type(filename)[0] or "text/html"
            self._file(WEB_FILES / filename, media_type)
            return
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "需要工作台令牌"})
            return
        app = self.server.workbench
        try:
            if path == "/api/health":
                devices: list[dict[str, Any]] = []
                try:
                    gpu = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,memory.free,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                    if gpu.returncode == 0:
                        for row in gpu.stdout.splitlines():
                            name, free, total = (part.strip() for part in row.split(",", 2))
                            devices.append(
                                {
                                    "name": name,
                                    "free_mib": int(free),
                                    "total_mib": int(total),
                                }
                            )
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    pass
                self._json(
                    200,
                    {
                        "service": "ready",
                        "gpu_available": bool(devices),
                        "gpu_devices": devices,
                    },
                )
            elif path == "/api/modules":
                self._json(200, module_catalog())
            elif path == "/api/jobs":
                self._json(200, app.list_jobs())
            elif match := re.fullmatch(r"/api/jobs/([0-9a-f]{32})", path):
                self._json(200, app.get(match.group(1)))
            elif match := re.fullmatch(
                r"/api/jobs/([0-9a-f]{32})/artifacts/(art_[0-9a-f]{32})", path
            ):
                job = app.get(match.group(1))
                if job["status"] != "COMPLETED":
                    raise FileNotFoundError("任务尚未完成")
                artifact_id = match.group(2)
                item = next(
                    (
                        row
                        for row in job["result"]["artifacts"]
                        if row["artifact_id"] == artifact_id
                    ),
                    None,
                )
                if item is None or not ARTIFACT_ID.fullmatch(artifact_id):
                    raise FileNotFoundError("产物不属于此任务")
                artifact_path = (
                    app.root / ".physrealx/artifacts/objects" / artifact_id[4:6] / artifact_id
                )
                if not artifact_path.is_file() or (
                    item["size_bytes"] is not None
                    and artifact_path.stat().st_size != item["size_bytes"]
                ):
                    raise FileNotFoundError("产物已丢失或大小不符")
                media_type = item["media_type"]
                inline = media_type in {"image/png", "image/jpeg", "image/gif", "application/json"}
                self._file(artifact_path, media_type, download=not inline)
            else:
                self._json(404, {"error": "页面不存在"})
        except (FileNotFoundError, KeyError):
            self._json(404, {"error": "任务或产物不存在"})
        except (OSError, ValueError) as error:
            self._json(400, {"error": str(error)})

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/session" and self.server.workbench.token:
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, f"Bearer {self.server.workbench.token}"):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "令牌错误"})
                return
            digest = hashlib.sha256(self.server.workbench.token.encode()).hexdigest()
            cookie = f"prx_session={digest}; HttpOnly; SameSite=Strict; Path=/; Max-Age=3600"
            self._json(200, {"authenticated": True}, cookie=cookie)
            return
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "需要工作台令牌"})
            return
        app = self.server.workbench
        try:
            if parsed.path == "/api/jobs":
                self._json(HTTPStatus.ACCEPTED, app.submit(self._request_json()))
            elif match := re.fullmatch(r"/api/jobs/([0-9a-f]{32})/cancel", parsed.path):
                self._request_json()
                self._json(200, app.cancel(match.group(1)))
            elif parsed.path == "/api/uploads":
                self._upload(parse_qs(parsed.query).get("kind", [""])[0], app)
            else:
                self._json(404, {"error": "页面不存在"})
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
            self._json(400, {"error": str(error)})
        except FileNotFoundError:
            self._json(404, {"error": "任务不存在"})

    def _upload(self, kind: str, app: Workbench) -> None:
        if (
            kind not in {"capture", "video"}
            or self.headers.get("Content-Type") != "application/zip"
        ):
            raise ValueError("仅接受 capture 或 video 的 ZIP 文件")
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > MAX_UPLOAD_BYTES:
            raise ValueError("ZIP 文件大小超出限制")
        directory = app.uploads_dir / uuid.uuid4().hex
        directory.mkdir()
        archive = directory / "upload.zip"
        extracted = directory / "data"
        extracted.mkdir()
        try:
            remaining = length
            with archive.open("xb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("上传中断")
                    output.write(chunk)
                    remaining -= len(chunk)
            selected = _unpack_upload(archive, extracted)
            if kind == "capture" and not all(
                (selected / state).is_dir() for state in ("can.start", "can.end")
            ):
                raise ValueError("ArtModeling ZIP 缺少 can.start 或 can.end")
            if kind == "video" and not (selected / "img").is_dir():
                raise ValueError("视频 ZIP 缺少 img 目录")
            self._json(201, {"directory": str(selected), "kind": kind})
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            archive.unlink(missing_ok=True)


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], workbench: Workbench) -> None:
        self.workbench = workbench
        super().__init__(address, WorkbenchHandler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cell-timeout", type=int, default=7200)
    args = parser.parse_args(argv)
    if not 0 < args.port < 65536 or args.cell_timeout < 1:
        parser.error("端口或单元格超时无效")
    token = os.environ.get("PHYSREALX_WEB_TOKEN", "")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        parser.error("非本机监听必须设置 PHYSREALX_WEB_TOKEN；建议使用 SSH 端口转发")
    notebook_packages = ("nbformat", "nbclient", "jupyter_client")
    if any(importlib.util.find_spec(name) is None for name in notebook_packages):
        parser.error("当前 Python 缺少 Notebook 执行依赖；请先激活 PhysReal-X Conda 环境")
    from jupyter_client.kernelspec import KernelSpecManager

    try:
        KernelSpecManager().get_kernel_spec("physreal-x")  # type: ignore[no-untyped-call]
    except Exception as error:
        parser.error(f"缺少 physreal-x Jupyter 内核；请先安装项目内核：{error}")
    app = Workbench(token=token, cell_timeout=args.cell_timeout)
    server = WorkbenchServer((args.host, args.port), app)
    print(f"PhysReal-X workbench: http://{args.host}:{args.port}/", flush=True)

    def stop_server(_signal: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
