"""Run one validated workbench request with the canonical notebook executor."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from examples.run_all import execute_notebooks


def _artifact_index(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        artifact_id = value.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id.startswith("art_"):
            found.append(
                {
                    "name": prefix or "artifact",
                    "artifact_id": artifact_id,
                    "media_type": value.get("media_type", "application/json"),
                    "size_bytes": value.get("size_bytes"),
                }
            )
        else:
            for key, child in value.items():
                found.extend(_artifact_index(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_artifact_index(child, f"{prefix}[{index}]"))
    return found


def run_request(job_dir: Path) -> None:
    request_path = job_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    os.environ["PHYSREALX_WEB_INPUTS"] = str(request_path)
    os.environ.setdefault("MUJOCO_GL", "egl")
    chain = tuple(request["chain"])
    report = execute_notebooks(
        root, chain, timeout=int(request["cell_timeout_seconds"]), kernel_name="physreal-x"
    )
    module = request["module"]
    result_path = root / "examples" / module / "output/results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if report["execution_status"] != "COMPLETED" or not result:
        raise RuntimeError("模块执行未完成，不能发布结果")
    snapshot = {
        "module": module,
        "results": result,
        "artifacts": _artifact_index(result),
        "acceptance": {
            "execution_status": report["execution_status"],
            "notebooks": [
                {
                    "module": item["module"],
                    "status": item["status"],
                    "duration_seconds": item.get("duration_seconds"),
                }
                for item in report["notebooks"]
            ],
        },
    }
    temporary = job_dir / "result.tmp"
    temporary.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(job_dir / "result.json")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m examples.web_worker JOB_DIRECTORY")
    run_request(Path(sys.argv[1]).resolve(strict=True))


if __name__ == "__main__":
    main()
