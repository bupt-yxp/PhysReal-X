"""Small presentation helpers shared by the six API notebooks.

Business parameters and SDK calls remain visible in each notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from physrealx import ArtifactRef, OperationManifest


class DemoBlocked(RuntimeError):
    """Raised when concrete input evidence or an executable method is absent."""


def output_artifact(operation: OperationManifest, name: str) -> ArtifactRef:
    """Resolve a named, contract-validated operation output."""

    for output in operation.outputs:
        if output.name == name:
            return output.artifact
    raise DemoBlocked(
        f"operation {operation.resource_id}@{operation.version} did not publish {name!r}"
    )


def operation_receipt(operation: OperationManifest) -> Mapping[str, Any]:
    """Return the immutable identity and output evidence of one operation."""

    return {
        "resource_id": operation.resource_id,
        "version": operation.version,
        "method": operation.spec.method,
        "outputs": operation.outputs,
        "metrics": operation.metrics,
    }


def json_value(value: Any) -> Any:
    """Convert public models and paths into deterministic JSON values."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def write_summary(
    path: Path,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Atomically publish one human-readable Demo receipt."""

    result = dict(payload)
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(json_value(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return result
