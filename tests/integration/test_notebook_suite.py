"""Opt-in execution of all six real-data notebooks, including native GPU methods."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        os.environ.get("PHYSREALX_REAL_DATA_TESTS") != "1",
        reason="requires local cabinet2 captures, CUDA, and the PhysReal-X Jupyter kernel",
    ),
]


def test_all_six_notebooks_execute_and_publish_actual_evidence() -> None:
    nbformat = pytest.importorskip("nbformat")
    from examples.run_all import MODULES

    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.run_all"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    suite = json.loads((root / "examples/output/notebook-acceptance.json").read_text())
    assert suite["execution_status"] == "COMPLETED"
    assert tuple(row["module"] for row in suite["notebooks"]) == MODULES
    for module in MODULES:
        notebook = nbformat.read(
            root / "examples" / module / "output/demo.executed.ipynb", as_version=4
        )
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code)
        assert not any(item.output_type == "error" for cell in code for item in cell.outputs)
    output = root / "examples/real_assets_and_digital_twin/output"
    report = json.loads((output / "digital-twin-report.json").read_text())
    assert report["execution_status"] == "COMPLETED"
    assert tuple(sorted(report["stages"])) == (
        "cameras",
        "depth",
        "environment",
        "export",
        "input",
        "reconstruction",
        "registration",
    )
    assert all(stage["status"] == "COMPLETED" for stage in report["stages"].values())
    assert Path(report["environment"]["prefix"]).name == "PhysReal-X"
    assert not report["independent_physical_validation"]
    assert len(json.loads((output / "reference-model.json").read_text())["joints"]) == 2
    reconstruction = json.loads((output / "reconstruction-report.json").read_text())
    assert reconstruction["evaluation_scope"] == "held_out_views"
    assert reconstruction["motion_group_count"] == 2
    # Guard against empty/cropped silhouettes looking good on a black background.
    assert reconstruction["metrics"]["validation_fg_psnr"] > 12
    assert reconstruction["metrics"]["validation_alpha_iou"] > 0.75
    learning = json.loads(
        (root / "examples/data_management_and_training/output/report.json").read_text()
    )
    assert learning["stages"]["long_rollout"]["prediction_duration_seconds"] == pytest.approx(5)
    assert (
        learning["stages"]["world_train"]["quality_gate"]
        == "state_latent_rgb_beat_identical_context_persistence"
    )
    simulation = json.loads(
        (root / "examples/simulation_and_data_generation/output/report.json").read_text()
    )
    assert (
        simulation["stages"]["appearance"]["operation"]["metrics"]["physical_motion_span_m"] > 0.1
    )
    feedback = json.loads(
        (root / "examples/evaluation_and_feedback/output/report.json").read_text()
    )
    assert feedback["stages"]["verify"]["success_rates"]["updated"] == 1.0
    assert feedback["stages"]["retrain"]["model"] != feedback["stages"]["input"]["model"]


def test_exported_artgs_bundle_reloads_without_training_directories(tmp_path) -> None:
    from physrealx.integrations.artgs import _SubprocessArtGSRunner

    root = Path(__file__).resolve().parents[2]
    bundle = root / "examples/real_assets_and_digital_twin/output/reconstruction.zip"
    script = """
import json, sys, zipfile
from argparse import Namespace
from pathlib import Path
import torch
from scene import DeformModel, GaussianModel
from scene.artgs import ArtGS
from utils.dual_quaternion import quaternion_translation_to_dual_quaternion, quaternion_to_matrix

destination = Path(sys.argv[2])
with zipfile.ZipFile(sys.argv[1]) as archive:
    config = json.loads(archive.read('model/config.json'))
    for filename in ('point_cloud.ply', 'deform.pth'):
        (destination / filename).write_bytes(archive.read('model/' + filename))
assert not {'source_path', 'model_path', 'coarse_model_path'} & set(config)
model = DeformModel(Namespace(**config))
model.deform.load_state_dict(torch.load(destination / 'deform.pth', weights_only=True))
model.update(20000)
gaussians = GaussianModel(config['sh_degree'])
gaussians.load_ply(str(destination / 'point_cloud.ply'))
with torch.no_grad():
    # Verify an identifiable hinge independently of this prismatic capture.
    half_angle = torch.tensor(0.4, device='cuda', dtype=torch.float64)
    qr = torch.stack((torch.cos(half_angle / 2), half_angle * 0, half_angle * 0,
                      torch.sin(half_angle / 2)))
    half_rotation = quaternion_to_matrix(qr)
    pivot = torch.tensor([0.2, -0.3, 0.0], device='cuda', dtype=torch.float64)
    half_translation = pivot - half_rotation @ pivot
    qrs = torch.stack((torch.tensor([1, 0, 0, 0], device='cuda'), qr))
    shifts = torch.cat((torch.zeros((2, 1), device='cuda'),
                        torch.stack((torch.zeros_like(pivot), half_translation))), dim=1)
    slots = quaternion_translation_to_dual_quaternion(qrs, shifts)
    hinge = ArtGS.get_joint_param(Namespace(get_slot_deform=lambda: slots), ['r'])[0]
    recovered = torch.tensor(hinge['translation'], device='cuda')
    assert torch.allclose(recovered, half_rotation @ half_translation + half_translation)
    assert torch.allclose(torch.tensor(hinge['axis_position'], device='cuda'), pivot)
    translations, rotations = model.deform.interpolate(gaussians, [0, 0.25, 0.5, 0.75, 1])
    assert len(gaussians.get_xyz) > 1000
    assert all(torch.isfinite(v).all() for v in translations + rotations)
    span = float(torch.linalg.vector_norm(translations[-1] - translations[0], dim=-1).max())
    assert 0.01 < span < 1.0
    assert all(torch.allclose(torch.linalg.vector_norm(v, dim=-1),
               torch.ones(len(v), device=v.device), atol=1e-5) for v in rotations)
    print(json.dumps({'gaussians': len(gaussians.get_xyz), 'observed_motion_span_m': span}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(bundle), str(tmp_path)],
        env=_SubprocessArtGSRunner()._environment(2026),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads(completed.stdout.splitlines()[-1])
    destination = root / "examples/real_assets_and_digital_twin/output/reconstruction-reload.json"
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
