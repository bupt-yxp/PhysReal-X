"""Execute the six public notebooks in dependency order with the active Conda interpreter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MODULES = (
    "real_assets_and_digital_twin",
    "physical_asset_building",
    "scene_and_task_generation",
    "simulation_and_data_generation",
    "data_management_and_training",
    "evaluation_and_feedback",
)


def main() -> None:
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    from examples._common import write_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-module", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds allowed per code cell")
    args = parser.parse_args()
    if Path(sys.prefix).name != "PhysReal-X":
        parser.error("activate the existing PhysReal-X Conda environment")
    if args.timeout < 1:
        parser.error("timeout must be positive")
    kernel = KernelSpecManager().get_kernel_spec("physreal-x")
    if Path(kernel.argv[0]).resolve() != Path(sys.executable).resolve():
        parser.error("physreal-x kernel must use the active Conda interpreter")
    root = Path(__file__).resolve().parents[1]
    report_path = root / "examples/output/notebook-acceptance.json"
    report: dict[str, object] = {
        "execution_status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "requested_modules": list(MODULES[args.from_module - 1 :]),
        "evidence_scope": "notebook execution; physical and robot evidence scopes remain separate",
        "notebooks": [],
    }
    receipts: list[dict[str, object]] = []
    report["notebooks"] = receipts
    write_summary(report_path, report)
    for module in MODULES[args.from_module - 1 :]:
        directory = root / "examples" / module
        notebook = nbformat.read(directory / "demo.ipynb", as_version=4)
        destination = directory / "output/demo.executed.ipynb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        receipt: dict[str, object] = {"module": module, "status": "RUNNING"}
        receipts.append(receipt)
        write_summary(report_path, report)
        print(f"RUNNING {module}", flush=True)
        try:
            NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="physreal-x",
                resources={"metadata": {"path": str(root)}},
                allow_errors=False,
            ).execute()
            receipt.update(
                status="COMPLETED",
                code_cells=sum(cell.cell_type == "code" for cell in notebook.cells),
                executed_notebook=str(destination.relative_to(root)),
            )
            print(f"PASSED {module}", flush=True)
        except BaseException as error:
            receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            report["execution_status"] = "FAILED"
            raise
        finally:
            temporary = destination.with_suffix(".tmp.ipynb")
            nbformat.write(notebook, temporary)
            temporary.replace(destination)
            write_summary(report_path, report)
    report.update(execution_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat())
    write_summary(report_path, report)
    print(json.dumps({"completed": len(receipts), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
