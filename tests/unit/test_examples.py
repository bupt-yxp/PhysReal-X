"""六模块 Jupyter Demo 的结构与教学性回归测试。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

MODULES = (
    "real_assets_and_digital_twin",
    "physical_asset_building",
    "scene_and_task_generation",
    "simulation_and_data_generation",
    "data_management_and_training",
    "evaluation_and_feedback",
)

EXPECTED_APIS = {
    "real_assets_and_digital_twin": (
        "px.assets.inspect_source(",
        "px.assets.ingest(",
        "px.assets.recover_cameras(",
        "px.assets.align_cameras_metric(",
        "px.assets.enhance_depth(",
        "px.assets.reconstruct_geometry(",
        "px.assets.import_reference(",
        "px.assets.prepare_rgbd(",
    ),
    "physical_asset_building": (
        "px.assets.build_collision(",
        "px.assets.assemble(",
        "px.assets.package(",
        "px.assets.validate(",
    ),
    "scene_and_task_generation": (
        "px.scenes.generate(",
        "px.scenes.compose(",
        "px.tasks.compile(",
        "px.tasks.validate(",
        "px.trajectories.generate(",
    ),
    "simulation_and_data_generation": (
        "px.runs.submit(",
        "px.datasets.get(",
        "px.runs.episodes(",
    ),
    "data_management_and_training": (
        "px.datasets.ingest(",
        "px.datasets.snapshot(",
        "px.training.submit(",
        "px.world_models.train(",
        "px.world_models.rollout(",
    ),
    "evaluation_and_feedback": (
        "px.evaluations.submit(",
        "px.telemetry.record(",
        "px.failures.analyze(",
        "px.feedback.build_dataset(",
    ),
}


def _load_notebook(module_name: str) -> dict[str, Any]:
    source = Path("examples") / module_name / "demo.ipynb"
    return json.loads(source.read_text(encoding="utf-8"))


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


@pytest.mark.parametrize("module_name", MODULES)
def test_module_is_one_self_contained_notebook(module_name: str) -> None:
    module_root = Path("examples") / module_name

    assert module_root.is_dir()
    assert (module_root / "demo.ipynb").is_file()
    assert (module_root / "README.md").is_file()
    assert not (module_root / "run.py").exists()
    assert not (module_root / "_workflow.py").exists()
    assert not (module_root / "_config.py").exists()
    assert not (module_root / "__init__.py").exists()
    assert not (module_root / "configs").exists()
    assert not tuple(module_root.glob("config*.py"))
    assert not tuple(module_root.glob("configs/*.yml"))
    assert not tuple(module_root.glob("configs/*.yaml"))


@pytest.mark.parametrize("module_name", MODULES)
def test_notebook_teaches_and_directly_calls_each_public_api(module_name: str) -> None:
    notebook = _load_notebook(module_name)
    assert notebook["nbformat"] == 4
    markdown = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join(_cell_source(cell) for cell in code_cells)

    assert len(markdown) >= 120
    assert "NotebookRun(" in source and "run.complete()" in source
    assert "method=" in source or "px.methods.bind(" in source
    assert "load_demo_config" not in source
    assert "configs/" not in source
    assert "import _config" not in source
    assert "execute_demo" not in source
    assert "workflow(config)" not in source
    assert "lambda:" not in source
    assert "def " not in source
    for public_call in EXPECTED_APIS[module_name]:
        assert public_call in source
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse(_cell_source(cell))
    tree = ast.parse(source)
    declared = next(
        ast.literal_eval(node.args[1])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "NotebookRun"
    )
    stages = [
        ast.literal_eval(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run"
        and node.func.attr == "stage"
    ]
    assert len(stages) == len(set(stages))
    assert set(stages) == set(declared)


def test_shared_helpers_do_not_hide_business_parameters_or_sdk_calls() -> None:
    source = Path("examples/_common.py").read_text(encoding="utf-8")

    assert "load_demo_config" not in source
    assert "DemoConfig" not in source
    assert "MethodSelection" not in source
    assert "PhysRealX" not in source
    assert "px." not in source
    assert "method=" not in source


def test_artmodeling_input_is_excluded_by_gitignore() -> None:
    source = Path(".gitignore").read_text(encoding="utf-8")

    assert "/examples/real_assets_and_digital_twin/input/ArtModeling/" in source


def test_notebook_reports_failure_and_invalidates_downstream_receipts(tmp_path: Path) -> None:
    from examples._common import DemoBlocked, NotebookRun

    path = tmp_path / "report.json"
    run = NotebookRun(path, ("first", "second"), {"interpreter": "fixture"})
    with run.stage("first") as stage:
        stage["artifact"] = "fixture-evidence"
    with pytest.raises(ValueError, match="bad input"), run.stage("second"):
        raise ValueError("bad input")
    report = json.loads(path.read_text())
    assert report["execution_status"] == "FAILED"
    assert report["stages"]["second"]["error"] == "bad input"
    with pytest.raises(DemoBlocked):
        run.complete()
    with run.stage("second"):
        pass
    assert run.complete()["execution_status"] == "COMPLETED"
    with run.stage("first"):
        pass
    assert json.loads(path.read_text())["stages"]["second"]["status"] == "PENDING"
    with pytest.raises(DemoBlocked):
        run.complete()
