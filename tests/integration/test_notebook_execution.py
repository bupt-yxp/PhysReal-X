"""Real Jupyter execution guards against stale success and hidden skipped/error cells."""

from __future__ import annotations

import fcntl
import hashlib
import json
import sys
from pathlib import Path

import pytest
from examples.run_all import MODULES, execute_notebooks


@pytest.fixture
def notebooks(tmp_path: Path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    kernels = tmp_path / "jupyter/kernels/test-python"
    kernels.mkdir(parents=True)
    (kernels / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Test Python",
                "language": "python",
            }
        )
    )
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path / "jupyter"))
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "runtime"))
    for module in MODULES:
        output = tmp_path / "examples" / module / "output"
        output.mkdir(parents=True)
        (output / "results.json").write_text('{"old": true}')
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell("# API example"),
                nbformat.v4.new_code_cell("from pathlib import Path\nprint('fresh kernel')"),
                nbformat.v4.new_code_cell(
                    f"Path('examples/{module}/output/results.json').write_text('{{\"value\": 42}}')"
                ),
            ]
        )
        nbformat.write(notebook, output.parent / "demo.ipynb")
    return tmp_path


def test_execution_records_cells_without_notebook_instrumentation(notebooks):
    root = notebooks
    source = root / "examples" / MODULES[0] / "demo.ipynb"
    original = source.read_bytes()
    report = execute_notebooks(root, MODULES[:2], timeout=30, kernel_name="test-python")
    assert report["execution_status"] == "COMPLETED"
    assert source.read_bytes() == original
    assert report["notebooks"][0]["source_sha256"] == hashlib.sha256(original).hexdigest()
    for row in report["notebooks"]:
        assert row["code_cells"] == 2
        assert row["duration_seconds"] > 0
        assert all(c["status"] == "COMPLETED" and c["duration_seconds"] >= 0 for c in row["cells"])
        assert json.loads((root / row["results"]).read_text()) == {"value": 42}
    assert (root / report["notebooks"][0]["executed_notebook"]).is_file()


def test_resumed_execution_accepts_only_current_upstream_evidence(notebooks, monkeypatch):
    import examples.run_all as runner

    root = notebooks
    modules = runner.MODULES[:2]
    monkeypatch.setattr(runner, "MODULES", modules)
    runner.execute_notebooks(root, modules[:1], timeout=30, kernel_name="test-python")
    report = runner.execute_notebooks(
        root,
        modules[1:],
        timeout=30,
        kernel_name="test-python",
    )
    assert report["requested_modules"] == [modules[1]]
    assert report["accepted_modules"] == list(modules)
    assert report["resumed_from_current_evidence"] == [modules[0]]
    assert [row["status"] for row in report["notebooks"]] == ["COMPLETED", "COMPLETED"]

    upstream = root / "examples" / modules[0] / "demo.ipynb"
    upstream.write_text(upstream.read_text() + " ")
    with pytest.raises(RuntimeError, match="stale or incomplete"):
        runner.execute_notebooks(
            root,
            modules[1:],
            timeout=30,
            kernel_name="test-python",
        )


@pytest.mark.parametrize("failure", ["error", "skip", "no_export", "source_change", "timeout"])
def test_failure_cannot_reuse_old_results_or_execute_downstream(notebooks, failure):
    import nbformat

    root = notebooks
    source = root / "examples" / MODULES[0] / "demo.ipynb"
    notebook = nbformat.read(source, as_version=4)
    if failure == "error":
        notebook.cells[1].source = "raise ValueError('invalid input')"
        notebook.cells[1].metadata["tags"] = ["raises-exception"]
    elif failure == "skip":
        notebook.cells[1].source = "print('required step')"
        notebook.cells[1].metadata["tags"] = ["skip-execution"]
        notebook.cells[2].source = "from pathlib import Path\n" + notebook.cells[2].source
    elif failure == "no_export":
        notebook.cells[2].source = "print('export omitted')"
    elif failure == "source_change":
        notebook.cells[
            1
        ].source += f"\np = Path({str(source)!r})\np.write_text(p.read_text() + ' ')"
    elif failure == "timeout":
        notebook.cells[1].source = "import time\ntime.sleep(10)"
    nbformat.write(notebook, source)
    # Even old successful outputs embedded in a skipped cell cannot pass acceptance.
    notebook.cells[1].execution_count = 99
    notebook.cells[1].outputs = [nbformat.v4.new_output("stream", text="old success")]
    nbformat.write(notebook, source)
    from nbclient.exceptions import CellExecutionError, CellTimeoutError

    expected = {
        "error": (CellExecutionError, "invalid input"),
        "skip": (RuntimeError, "not all code cells"),
        "no_export": (FileNotFoundError, "results.json"),
        "source_change": (RuntimeError, "source changed"),
        "timeout": (CellTimeoutError, "timed out"),
    }
    error_type, message = expected[failure]
    with pytest.raises(error_type, match=message):
        execute_notebooks(
            root, MODULES[:2], timeout=1 if failure == "timeout" else 30, kernel_name="test-python"
        )
    report = json.loads((root / "examples/output/notebook-acceptance.json").read_text())
    assert report["execution_status"] == "FAILED"
    assert [r["status"] for r in report["notebooks"]] == ["FAILED", "PENDING"]
    for module in MODULES[:2]:
        assert not (root / "examples" / module / "output/results.json").exists()
    assert (source.parent / "output/demo.executed.ipynb").is_file()
    assert not (root / "examples" / MODULES[1] / "output/demo.executed.ipynb").exists()


def test_concurrent_runner_does_not_invalidate_active_results(notebooks):
    directory = notebooks / "examples/output"
    directory.mkdir()
    with (directory / ".notebook-acceptance.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another notebook acceptance run"):
            execute_notebooks(notebooks, MODULES[:2], kernel_name="test-python")
    assert json.loads(
        (notebooks / "examples" / MODULES[0] / "output/results.json").read_text()
    ) == {"old": True}


@pytest.mark.parametrize("changed", ["results", "executed", "sdk"])
def test_completed_receipt_binds_results_execution_and_sdk(notebooks, changed):
    from examples.run_all import _load_current_receipt

    execute_notebooks(notebooks, MODULES[:1], timeout=30, kernel_name="test-python")
    output = notebooks / "examples" / MODULES[0] / "output"
    if changed == "sdk":
        source = notebooks / "src/physrealx/platform.py"
        source.parent.mkdir(parents=True)
        source.write_text("# implementation changed after execution\n")
    elif changed == "results":
        (output / "results.json").write_text('{"value": 999}')
    else:
        path = output / "demo.executed.ipynb"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="stale or incomplete"):
        _load_current_receipt(notebooks, MODULES[0])


def test_independent_scene_demo_needs_no_real_capture_receipt(notebooks):
    report = execute_notebooks(notebooks, (MODULES[2],), timeout=30, kernel_name="test-python")
    assert report["execution_status"] == "COMPLETED"
    assert report["accepted_modules"] == [MODULES[2]]
    assert report["resumed_from_current_evidence"] == []
    assert (notebooks / "examples" / MODULES[0] / "output/results.json").exists()
    for name in MODULES[3:]:
        assert not (notebooks / "examples" / name / "output/results.json").exists()


def test_downstream_requires_current_actual_dependency_before_removing_outputs(notebooks):
    with pytest.raises(RuntimeError, match="no valid completed execution evidence"):
        execute_notebooks(notebooks, (MODULES[3],), timeout=30, kernel_name="test-python")
    # Missing inputs cannot destroy the previous output before a run can start.
    assert (notebooks / "examples" / MODULES[3] / "output/results.json").exists()


def test_out_of_order_request_is_rejected_without_removing_outputs(notebooks):
    with pytest.raises(ValueError, match="dependency order"):
        execute_notebooks(notebooks, tuple(reversed(MODULES[:2])), kernel_name="test-python")
    assert (notebooks / "examples" / MODULES[0] / "output/results.json").exists()


def test_skipping_intermediate_dependency_is_rejected_before_invalidating_outputs(notebooks):
    with pytest.raises(ValueError, match="include intermediate dependencies"):
        execute_notebooks(notebooks, (MODULES[2], MODULES[4]), kernel_name="test-python")
    assert (notebooks / "examples" / MODULES[3] / "output/results.json").exists()


def test_final_acceptance_rechecks_previously_completed_notebooks(notebooks):
    import nbformat

    source = notebooks / "examples" / MODULES[1] / "demo.ipynb"
    notebook = nbformat.read(source, as_version=4)
    upstream = f"examples/{MODULES[0]}/demo.ipynb"
    notebook.cells[1].source += f"\np = Path({upstream!r})\np.write_text(p.read_text() + ' ')"
    nbformat.write(notebook, source)
    with pytest.raises(RuntimeError, match="stale or incomplete"):
        execute_notebooks(notebooks, MODULES[:2], timeout=30, kernel_name="test-python")
    report = json.loads((notebooks / "examples/output/notebook-acceptance.json").read_text())
    assert report["execution_status"] == "FAILED"
