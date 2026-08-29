"""Small shared helpers for independently installed worker processes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from physrealx.external_workers import WorkerInputFile, WorkerRequest


def input_file(request: WorkerRequest, name: str) -> WorkerInputFile:
    """Return one declared input by semantic name."""

    item = next((candidate for candidate in request.inputs if candidate.name == name), None)
    if item is None:
        raise ValueError(f"required worker input is missing: {name}")
    return item


def input_path(request: WorkerRequest, name: str) -> Path:
    """Return a verified materialized input path by semantic name."""

    item = input_file(request, name)
    return (Path.cwd() / item.path).resolve(strict=True)


def config(request: WorkerRequest) -> dict[str, Any]:
    """Copy the JSON-compatible method configuration."""

    return dict(request.operation.method.config)


def report_probe(worker: str, versions: dict[str, str]) -> int:
    """Emit one machine-readable readiness result."""

    print(json.dumps({"worker": worker, "ready": True, "versions": versions}, sort_keys=True))
    return 0


def is_probe(argv: list[str] | None = None) -> bool:
    return "--probe" in (sys.argv[1:] if argv is None else argv)
