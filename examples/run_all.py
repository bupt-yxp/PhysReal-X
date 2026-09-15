"""Execute API notebooks and record acceptance outside their business examples."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, cast

MODULES = (
    "real_assets_and_digital_twin",
    "physical_asset_building",
    "scene_and_task_generation",
    "simulation_and_data_generation",
    "data_management_and_training",
    "evaluation_and_feedback",
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_current_receipt(root: Path, module: str) -> dict[str, Any]:
    """Load one completed receipt only when every local evidence file is current."""
    import nbformat

    directory = root / "examples" / module
    source_path = directory / "demo.ipynb"
    execution_path = directory / "output/execution.json"
    executed_path = directory / "output/demo.executed.ipynb"
    results_path = directory / "output/results.json"
    try:
        receipt = cast(dict[str, Any], json.loads(execution_path.read_text()))
        results = json.loads(results_path.read_text())
        source = nbformat.read(source_path, as_version=4)  # type: ignore[no-untyped-call]
        executed = nbformat.read(executed_path, as_version=4)  # type: ignore[no-untyped-call]
    except (FileNotFoundError, json.JSONDecodeError, nbformat.reader.NotJSONError) as error:
        raise RuntimeError(f"{module} has no valid completed execution evidence") from error

    expected_cells = [
        (index, cell.id)
        for index, cell in enumerate(source.cells)
        if cell.cell_type == "code" and cell.source.strip()
    ]
    recorded_cells = [
        (row.get("index"), row.get("id"))
        for row in receipt.get("cells", ())
        if row.get("status") == "COMPLETED"
    ]
    executed_cells = [
        cell for cell in executed.cells if cell.cell_type == "code" and cell.source.strip()
    ]
    valid = (
        receipt.get("module") == module
        and receipt.get("status") == "COMPLETED"
        and receipt.get("source_sha256") == hashlib.sha256(source_path.read_bytes()).hexdigest()
        and receipt.get("code_cells") == len(expected_cells)
        and recorded_cells == expected_cells
        and len(executed_cells) == len(expected_cells)
        and all(cell.execution_count is not None for cell in executed_cells)
        and not any(
            output.output_type == "error" for cell in executed_cells for output in cell.outputs
        )
        and isinstance(results, dict)
        and bool(results)
    )
    if not valid:
        raise RuntimeError(f"{module} execution evidence is stale or incomplete")
    return receipt


def _execute_notebook(root: Path, receipt: dict[str, Any], timeout: int, kernel_name: str) -> None:
    import nbformat
    from nbclient import NotebookClient

    module = receipt["module"]
    directory = root / "examples" / module
    source_path = directory / "demo.ipynb"
    destination = directory / "output/demo.executed.ipynb"
    execution_path = directory / "output/execution.json"
    notebook = None
    start = monotonic()
    receipt.update(status="RUNNING", started_at=datetime.now(timezone.utc).isoformat())
    print(f"RUNNING {module}", flush=True)
    cell_starts: dict[int, float] = {}
    cell_records: dict[int, dict[str, Any]] = {}

    def cell_started(cell: Any, cell_index: int) -> None:
        cell_starts[cell_index] = monotonic()
        cell_records[cell_index]["status"] = "RUNNING"
        _write_json(execution_path, receipt)

    def cell_executed(cell: Any, cell_index: int, execute_reply: Any) -> None:
        cell_records[cell_index].update(
            status="COMPLETED" if execute_reply["content"]["status"] == "ok" else "FAILED",
            duration_seconds=monotonic() - cell_starts[cell_index],
        )
        _write_json(execution_path, receipt)

    try:
        source = source_path.read_bytes()
        notebook = nbformat.reads(  # type: ignore[no-untyped-call]
            source.decode("utf-8"), as_version=4
        )
        # Saved output, execution counts and skip/error tags cannot certify fresh work.
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop("execution", None)
                if cell.source.strip():
                    cell_records[i] = {"index": i, "id": cell.id, "status": "PENDING"}
        if not cell_records:
            raise ValueError("notebook has no executable code cells")
        receipt.update(
            source_sha256=hashlib.sha256(source).hexdigest(),
            cells=list(cell_records.values()),
        )
        NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(root)}},
            allow_errors=False,
            force_raise_errors=True,
            on_cell_execute=cell_started,
            on_cell_executed=cell_executed,
        ).execute()
        if any(row["status"] != "COMPLETED" for row in cell_records.values()):
            raise RuntimeError("not all code cells executed successfully")
        if source_path.read_bytes() != source:
            raise RuntimeError("notebook source changed during execution; rerun for acceptance")
        results = json.loads((directory / "output/results.json").read_text())
        if not isinstance(results, dict) or not results:
            raise ValueError("notebook did not export a nonempty result index")
        receipt.update(
            status="COMPLETED",
            code_cells=len(cell_records),
            executed_notebook=str(destination.relative_to(root)),
            results=str((directory / "output/results.json").relative_to(root)),
        )
        print(f"PASSED {module}", flush=True)
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        for i, row in cell_records.items():
            if row["status"] == "RUNNING":
                row.update(status="FAILED", duration_seconds=monotonic() - cell_starts[i])
        (directory / "output/results.json").unlink(missing_ok=True)
        raise
    finally:
        receipt.update(
            duration_seconds=monotonic() - start,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        if notebook is not None:
            temporary = destination.with_suffix(".tmp.ipynb")
            nbformat.write(notebook, temporary)  # type: ignore[no-untyped-call]
            temporary.replace(destination)
        _write_json(execution_path, receipt)


def execute_notebooks(
    root: Path,
    modules: tuple[str, ...] = MODULES,
    *,
    timeout: int = 3600,
    kernel_name: str = "physreal-x",
    include_current_upstream: bool = False,
) -> dict[str, Any]:
    """Run fresh kernels, stop on failure, and preserve cell-level execution evidence."""
    if not modules or len(set(modules)) != len(modules) or any(m not in MODULES for m in modules):
        raise ValueError("select distinct modules from the six examples")
    if timeout < 1:
        raise ValueError("timeout must be positive")
    root = root.resolve()
    report_path = root / "examples/output/notebook-acceptance.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with (report_path.parent / ".notebook-acceptance.lock").open("a") as lock:
        # Concurrent runners must not invalidate each other's output indices or receipts.
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another notebook acceptance run is active in this workspace"
            ) from error
        upstream: tuple[str, ...] = ()
        if include_current_upstream:
            start = MODULES.index(modules[0])
            if modules != MODULES[start:]:
                raise ValueError("resumed acceptance must execute a contiguous module suffix")
            upstream = MODULES[:start]
        receipts: list[dict[str, Any]] = [
            *(_load_current_receipt(root, module) for module in upstream),
            *({"module": module, "status": "PENDING"} for module in modules),
        ]
        requested_receipts = receipts[len(upstream) :]
        report: dict[str, Any] = {
            "schema": "physrealx.notebook-acceptance/2",
            "execution_status": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable,
            "requested_modules": list(modules),
            "accepted_modules": list(upstream + modules),
            "resumed_from_current_evidence": list(upstream),
            "evidence_scope": (
                "software execution on captures and simulation; "
                "no physical measurement or live robot acceptance"
            ),
            "notebooks": receipts,
        }
        _write_json(report_path, report)
        try:
            # A failed rerun cannot leave old downstream exports looking like this run's results.
            for module in modules:
                output = root / "examples" / module / "output"
                output.mkdir(parents=True, exist_ok=True)
                for name in ("results.json", "execution.json", "demo.executed.ipynb"):
                    (output / name).unlink(missing_ok=True)
            for receipt in requested_receipts:
                receipt["status"] = "RUNNING"
                _write_json(report_path, report)
                try:
                    _execute_notebook(root, receipt, timeout, kernel_name)
                finally:
                    _write_json(report_path, report)
        except BaseException as error:
            report.update(execution_status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        else:
            report["execution_status"] = "COMPLETED"
        finally:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(report_path, report)
        return report


def main() -> None:
    from jupyter_client.kernelspec import KernelSpecManager

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-module", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds allowed per code cell")
    args = parser.parse_args()
    if Path(sys.prefix).name != "PhysReal-X":
        parser.error("activate the existing PhysReal-X Conda environment")
    if args.timeout < 1:
        parser.error("timeout must be positive")
    kernel = KernelSpecManager().get_kernel_spec("physreal-x")  # type: ignore[no-untyped-call]
    if Path(kernel.argv[0]).resolve() != Path(sys.executable).resolve():
        parser.error("physreal-x kernel must use the active Conda interpreter")
    root = Path(__file__).resolve().parents[1]
    report = execute_notebooks(
        root,
        MODULES[args.from_module - 1 :],
        timeout=args.timeout,
        include_current_upstream=args.from_module > 1,
    )
    print(json.dumps({"completed": len(report["notebooks"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
