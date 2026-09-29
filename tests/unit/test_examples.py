"""六模块 Jupyter Demo 的结构与教学性回归测试。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from examples.catalog import DEMOS, MODULES, dependencies_for, dependents_of

EXPECTED_APIS = {
    "real_assets_and_digital_twin": (
        "px.assets.inspect_source(",
        "px.assets.ingest(",
        "px.assets.recover_cameras(",
        "px.assets.align_cameras_metric(",
        "px.assets.enhance_depth(",
        "px.assets.reconstruct_geometry(",
        "px.assets.evaluate_reconstruction(",
        "px.assets.import_reference(",
        "px.assets.prepare_rgbd(",
    ),
    "physical_asset_building": (
        "px.assets.build_collision(",
        "px.assets.assemble(",
        "px.assets.package(",
        "px.assets.preview_physical_asset(",
        "px.assets.validate(",
    ),
    "scene_and_task_generation": (
        "px.scenes.generate(",
        "px.scenes.compose(",
        "px.scenes.preview_task(",
        "px.tasks.compile(",
        "px.tasks.validate(",
        "px.trajectories.generate(",
    ),
    "simulation_and_data_generation": (
        "px.runs.submit(",
        "px.datasets.get(",
        "px.datasets.ingest(",
        "px.datasets.preview_episode(",
        "px.runs.episodes(",
    ),
    "data_management_and_training": (
        "px.datasets.snapshot(",
        "px.datasets.preview_snapshot(",
        "px.training.submit(",
        "px.training.prepare_interface(",
        "px.training.preview_interfaces(",
        "px.training.preview_policy(",
        "px.world_models.train(",
        "px.world_models.rollout(",
        "px.world_models.preview_rollout(",
    ),
    "evaluation_and_feedback": (
        "px.evaluations.submit(",
        "px.telemetry.record(",
        "px.failures.analyze(",
        "px.feedback.build_dataset(",
        "px.feedback.analyze_policy_intervention(",
        "px.feedback.assess_policy_update(",
        "px.feedback.plan_physics_calibration(",
        "px.feedback.preview_report(",
        "px.feedback.preview_cycle(",
    ),
}


def test_demo_catalog_matches_actual_data_dependencies():
    assert len(MODULES) == len(set(MODULES)) == 6
    for demo in DEMOS:
        assert (Path("docs") / demo.guide).is_file()
        assert set(demo.dependencies) <= set(MODULES[: MODULES.index(demo.slug)])
    assert dependencies_for(MODULES[3:]) == (MODULES[2],)
    assert dependents_of(MODULES[:1]) == MODULES[:2]
    assert dependents_of((MODULES[2],)) == MODULES[2:]


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
    for obsolete in (
        "NotebookRun",
        "run.stage",
        "stage[",
        "require_completed",
        "operation_receipt",
        "oven4",
        "e_door_197",
    ):
        assert obsolete not in source
    assert "results.json" in source and "px.artifacts.export(" in source
    for implementation_detail in (
        "TemporaryDirectory",
        "urllib.request",
        "Rotation.align_vectors",
        "np.cross(",
        "sys.path",
        "os.environ",
        "TRAINING_FRAMES_PER_STATE",
        "RECONSTRUCTION_METHOD",
        "STATE_STREAMS",
    ):
        assert implementation_detail not in source
    assert f'setup_notebook("{module_name}")' in source
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
        # A locally executed demo may retain user outputs; source structure and
        # fresh-kernel execution are verified independently.
        assert cell["execution_count"] is None or isinstance(cell["execution_count"], int)
        ast.parse(_cell_source(cell))


def test_shared_helpers_do_not_hide_business_parameters_or_sdk_calls() -> None:
    source = Path("examples/_display.py").read_text(encoding="utf-8")

    assert "load_demo_config" not in source
    assert "DemoConfig" not in source
    assert "MethodSelection" not in source
    assert "PhysRealX" not in source
    assert "px." not in source
    assert "method=" not in source


def test_reconstruction_cell_accepts_report_without_method_diagnostics() -> None:
    """Execute the demo presentation with a valid summary and no diagnostic metrics."""
    from collections import Counter
    from types import SimpleNamespace

    notebook = _load_notebook("real_assets_and_digital_twin")
    source = next(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "px.assets.reconstruct_geometry(" in _cell_source(cell)
    )
    report = {
        "evaluation_scope": "held_out_views",
        "motion_group_count": 2,
        "training_frame_ids": [
            f"{state}.{index}" for state in ("can.start.rgb", "can.end.rgb") for index in range(100)
        ],
        "validation_frame_ids": [
            f"{state}.{index}"
            for state in ("can.start.rgb", "can.end.rgb")
            for index in range(100, 150)
        ],
    }
    shown = []
    namespace = {
        "px": SimpleNamespace(
            assets=SimpleNamespace(
                reconstruct_geometry=lambda *args, **kwargs: SimpleNamespace(
                    output=lambda name: name
                )
            ),
            artifacts=SimpleNamespace(get_bytes=lambda ref: json.dumps(report).encode()),
        ),
        "geometry_input": object(),
        "json": json,
        "Counter": Counter,
        "show_metrics": shown.append,
    }
    exec(compile(source, "reconstruction-cell", "exec"), namespace)
    assert namespace["reconstruction_bundle"] == "visual_geometry_bundle"
    assert shown[-1] == {"运动组": 2}


def test_real_asset_demo_exports_one_reconstruction_animation() -> None:
    notebook = _load_notebook("real_assets_and_digital_twin")
    source = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert source.count("show_animation(") == 1
    assert source.count("px.assets.preview_reconstruction(") == 1
    assert "preview_reconstruction_appearance" not in source
    assert "reconstruction-appearance.gif" not in source
    assert '"reconstruction-comparison.gif": reconstruction_animation' in source


def test_scene_task_demo_exports_four_scene_execution_animations() -> None:
    notebook = _load_notebook("scene_and_task_generation")
    source = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert source.count("show_animation(") == 1
    assert source.count("px.scenes.preview_task(") == 1
    assert 'SCENE_TYPES = ("home", "tabletop", "shelf", "industrial")' in source
    assert "for scene_type in SCENE_TYPES:" in source
    assert '"target": "laptop"' in source
    assert '"target": "bottle"' in source
    assert '"target": "cabinet"' in source
    assert '"target": "safe"' in source
    assert '"require_success": False' in source
    assert 'method="SceneSmith"' in source
    assert "px.assets.retrieve_partnet_mobility(" in source
    assert "candidate_count=6" in source
    assert 'report["status"] == "PASSED"' in source
    assert '"candidate_evaluations"' in source
    assert '"retrieval_reports"' in source
    assert "for position_gain, velocity_gain in ((200.0, 20.0), (400.0, 40.0)):" in source
    assert '"cabinet", "relation": "left_of"' in source
    assert '"safe", "relation": "front_of"' in source
    assert "matplotlib" not in source
    assert "trajectory.png" not in source
    assert 'f"task-preview-{scene_type}.gif": preview' in source
    assert '"primary_scenario": "home"' in source
    assert '"task_instance": task_instance.model_dump' in source


def test_simulation_demo_displays_input_and_synchronized_execution() -> None:
    notebook = _load_notebook("simulation_and_data_generation")
    source = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert source.count("show_animation(") == 2
    assert source.count("px.scenes.preview_task(") == 1
    assert source.count("px.datasets.preview_episode(") == 1
    assert '"task-input-preview.gif": input_preview' in source
    assert '"simulation-preview.gif": preview' in source
    assert "pusher/extension" not in source
    assert "registered_artgs" not in source
    assert 'upstream["backend_config"]' not in source


def test_training_demo_exports_dataset_policy_and_world_model_visuals() -> None:
    notebook = _load_notebook("data_management_and_training")
    source = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert source.count("px.datasets.preview_snapshot(") == 1
    assert source.count("px.training.preview_policy(") == 2
    assert source.count("px.training.prepare_interface(") == 2
    assert source.count("px.training.preview_interfaces(") == 1
    assert source.count("px.world_models.preview_rollout(") == 1
    assert "matplotlib" not in source
    for filename in (
        '"dataset-preview.png": dataset_preview',
        '"policy-training.png": policy_preview',
        '"iql-training.png": iql_preview',
        '"training-interfaces.png": interface_preview',
        '"openvla-oft-interface.json": official_interfaces["openvla_oft"]',
        '"td-mpc2-interface.json": official_interfaces["td_mpc2"]',
        '"world-model-preview.gif": world_model_preview',
    ):
        assert filename in source


def test_artmodeling_input_is_excluded_by_gitignore() -> None:
    source = Path(".gitignore").read_text(encoding="utf-8")

    assert "/examples/real_assets_and_digital_twin/input/ArtModeling/" in source


@pytest.mark.parametrize("module", ["../real_assets_and_digital_twin", "/tmp", "missing-example"])
def test_setup_rejects_invalid_example_paths(module):
    from examples._display import setup_notebook

    with pytest.raises(ValueError):
        setup_notebook(module)


def test_setup_rejects_wrong_kernel_before_changing_environment(monkeypatch):
    import os
    import sys

    from examples._display import setup_notebook

    monkeypatch.setattr(sys, "prefix", "/tmp/unrelated-environment")
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="Conda"):
        setup_notebook("real_assets_and_digital_twin")
    assert dict(os.environ) == before


def test_setup_rejects_sdk_from_a_different_checkout(monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    from examples._display import setup_notebook

    monkeypatch.setattr(sys, "prefix", "/tmp/PhysReal-X")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin="/tmp/other/physrealx/__init__.py"),
    )
    with pytest.raises(RuntimeError, match="SDK"):
        setup_notebook("real_assets_and_digital_twin")


def test_setup_is_cwd_independent_and_preserves_user_environment(tmp_path, monkeypatch):
    import os
    import sys

    from examples import _display

    plt = pytest.importorskip("matplotlib.pyplot")
    monkeypatch.setattr(sys, "prefix", "/tmp/PhysReal-X")
    expected = Path(_display.__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    for name, value in {
        "CUDA_VISIBLE_DEVICES": "1",
        "MUJOCO_GL": "egl",
        "OMP_NUM_THREADS": "2",
        "MAX_JOBS": "1",
        "CUDA_HOME": "/custom/cuda",
    }.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)
    path_before = list(sys.path)
    figures = plt.get_fignums()
    monkeypatch.setattr(_display, "show_metrics", lambda values: None)
    first = _display.setup_notebook("real_assets_and_digital_twin")
    assert first == _display.setup_notebook("real_assets_and_digital_twin")
    assert first == (expected, expected / "examples/real_assets_and_digital_twin/output")
    assert dict(os.environ) == before and sys.path == path_before
    assert plt.get_fignums() == figures
