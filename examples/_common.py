"""Small presentation helpers shared by the six API notebooks.

Business parameters and SDK calls remain visible in each notebook.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
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


def show_metrics(values: Mapping[str, Any]) -> None:
    """Display a compact, escaped table; detailed evidence stays in the run report."""
    from html import escape

    from IPython.display import HTML, display

    rows = []
    for name, value in values.items():
        formatted = f"{value:.4g}" if isinstance(value, float) else str(value)
        rows.append(f"<tr><th>{escape(str(name))}</th><td>{escape(formatted)}</td></tr>")
    display(HTML("<table>" + "".join(rows) + "</table>"))


def show_rgbd(rgb: Any, depth: Any, destination: Path) -> None:
    """Present one captured RGB image and its metric depth; close the figure after display."""
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display

    figure, axes = plt.subplots(1, 2, figsize=(9, 3))
    try:
        axes[0].imshow(rgb)
        axes[0].set_title("Captured RGB")
        valid_depth = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0), depth)
        image = axes[1].imshow(valid_depth, cmap="viridis")
        axes[1].set_title("Depth (m)")
        figure.colorbar(image, ax=axes[1])
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(destination, dpi=140)
        display(figure)
    finally:
        plt.close(figure)


def show_image_strip(images: Mapping[str, Any], destination: Path) -> None:
    """Present already-decoded previews without knowing their reconstruction method."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    if not 1 <= len(images) <= 8:
        raise ValueError("a preview strip requires between one and eight images")
    figure, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3), squeeze=False)
    try:
        for axis, (label, pixels) in zip(axes[0], images.items()):
            axis.imshow(pixels)
            axis.set_title(label)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(destination, dpi=140)
        display(figure)
    finally:
        plt.close(figure)


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


class NotebookRun:
    """Persist execution status; stage bodies keep all SDK calls in the notebook."""

    def __init__(
        self,
        path: Path,
        stages: tuple[str, ...],
        environment: Mapping[str, Any],
        *,
        evidence_scope: str = "software_execution",
    ) -> None:
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("notebook stage names must be nonempty and unique")
        self.path = path
        self.report: dict[str, Any] = {
            "schema": "physrealx.notebook-run/1",
            "execution_status": "RUNNING",
            "environment": dict(environment),
            "evidence_scope": evidence_scope,
            "independent_physical_validation": False,
            "stages": {name: {"status": "PENDING"} for name in stages},
        }
        write_summary(path, self.report)

    @staticmethod
    def _clock() -> float:
        from time import monotonic

        return monotonic()

    @contextmanager
    def stage(self, name: str):
        stages = self.report["stages"]
        if name not in stages:
            raise ValueError(f"unknown notebook stage: {name}")
        order = list(stages)
        if any(stages[key]["status"] != "COMPLETED" for key in order[: order.index(name)]):
            raise DemoBlocked(f"complete earlier notebook stages before {name}")
        # Re-running a cell invalidates downstream receipts, avoiding stale success.
        for key in order[order.index(name) :]:
            stages[key] = {"status": "PENDING"}
        record = stages[name]
        record["status"] = "RUNNING"
        self.report["execution_status"] = "RUNNING"
        start = self._clock()
        write_summary(self.path, self.report)
        try:
            yield record
        except BaseException as error:
            record.update(status="FAILED", error_type=type(error).__name__, error=str(error))
            self.report["execution_status"] = "FAILED"
            raise
        else:
            record["status"] = "COMPLETED"
        finally:
            record["duration_seconds"] = self._clock() - start
            write_summary(self.path, self.report)

    def complete(self) -> Mapping[str, Any]:
        if any(stage["status"] != "COMPLETED" for stage in self.report["stages"].values()):
            raise DemoBlocked("notebook still has failed or pending stages")
        self.report["execution_status"] = "COMPLETED"
        return write_summary(self.path, self.report)


def require_completed(path: Path) -> Mapping[str, Any]:
    """Load a predecessor's explicit result, rejecting absent or interrupted notebook runs."""
    if not path.is_file():
        raise DemoBlocked(f"先完整执行上游 Notebook，缺少验收记录：{path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "physrealx.notebook-run/1"
        or report.get("execution_status") != "COMPLETED"
    ):
        raise DemoBlocked(f"上游 Notebook 尚未完整成功执行：{path}")
    if not report.get("stages") or any(
        item.get("status") != "COMPLETED" for item in report["stages"].values()
    ):
        raise DemoBlocked(f"上游阶段记录不完整：{path}")
    return report
