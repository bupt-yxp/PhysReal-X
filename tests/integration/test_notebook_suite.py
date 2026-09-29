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
        reason="requires local laptop1 captures, CUDA, and the PhysReal-X Jupyter kernel",
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
    import hashlib

    from physrealx import ArtifactRef, PhysRealX, RunSpec, VersionedRef

    px = PhysRealX.open(root / ".physrealx")
    for receipt in suite["notebooks"]:
        source = root / "examples" / receipt["module"] / "demo.ipynb"
        assert receipt["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert all(row["status"] == "COMPLETED" for row in receipt["cells"])
    results = json.loads((output / "results.json").read_text())
    for filename, ref in results.items():
        assert (output / filename).read_bytes() == px.artifacts.get_bytes(
            ArtifactRef.model_validate(ref)
        )
    assert len(json.loads((output / "reference-model.json").read_text())["joints"]) == 1
    reconstruction = json.loads((output / "reconstruction-report.json").read_text())
    assert reconstruction["evaluation_scope"] == "held_out_views"
    assert reconstruction["motion_group_count"] == 2
    from collections import Counter

    assert Counter(x.rsplit(".", 1)[0] for x in reconstruction["training_frame_ids"]) == {
        "can.start.rgb": 100,
        "can.end.rgb": 100,
    }
    assert Counter(x.rsplit(".", 1)[0] for x in reconstruction["validation_frame_ids"]) == {
        "can.start.rgb": 50,
        "can.end.rgb": 50,
    }
    from physrealx.reconstruction import ReconstructionPrediction

    prediction = ReconstructionPrediction.model_validate(reconstruction["prediction"])
    assert len(prediction.parts) == 2 and prediction.segmented
    evaluation = json.loads((output / "reconstruction-evaluation.json").read_text())
    assert evaluation["status"] == "COMPLETED"
    assert evaluation["protocol"]["name"] == "dtart"
    assert evaluation["coverage"]["evaluated_joints"] == 1
    import math

    for state in ("start", "end"):
        geometry = evaluation["geometry"][state]
        scores = [geometry[k]["cd"] for k in ("whole", "static")]
        scores.extend(row["cd"] for row in geometry["parts"].values())
        assert scores and all(
            value is not None and math.isfinite(value) and value >= 0 for value in scores
        )
    joint = evaluation["joints"]["cover.joint"]
    assert joint["status"] == "measured"
    assert all(math.isfinite(joint[k]) and joint[k] >= 0 for k in ("axis_ang_deg", "part_motion"))
    learning = json.loads(
        (root / "examples/data_management_and_training/output/results.json").read_text()
    )
    rollout = json.loads(
        px.artifacts.get_bytes(ArtifactRef.model_validate(learning["long_prediction"]))
    )
    assert len(rollout["timestamp_seconds"]) == 100
    assert rollout["timestamp_seconds"][-1] == pytest.approx(5)
    assert not rollout["independent_physical_validation"]
    training = json.loads(
        px.artifacts.get_bytes(ArtifactRef.model_validate(learning["world_model_training"]))
    )
    metrics = training["metrics"]
    assert metrics["test_normalized_state_rmse"] < metrics["test_persistence_normalized_state_rmse"]
    for metric in ("latent_mse", "rgb_mse"):
        assert metrics["test_" + metric] <= metrics["test_persistence_" + metric] + 1e-8
    valid_statuses = {"BEATS_PERSISTENCE", "MATCHES_PERSISTENCE", "BELOW_BASELINE"}
    learning_task = px.tasks.get(RunSpec.model_validate(learning["evaluation_run"]).task)
    for key, signal in (
        ("long_prediction_evaluation", "joint.position"),
        (
            "counterfactual_evaluation",
            learning_task.success_conditions[0].signal,
        ),
    ):
        rollout_evaluation = json.loads(
            px.artifacts.get_bytes(ArtifactRef.model_validate(learning[key]))
        )
        evidence = rollout_evaluation["signals"][signal]
        assert evidence["rmse"] and all(math.isfinite(value) for value in evidence["rmse"])
        assert set(evidence["status"]) <= valid_statuses
    simulation = json.loads(
        (root / "examples/simulation_and_data_generation/output/results.json").read_text()
    )
    from physrealx.runtime import load_episode

    dataset = px.datasets.get(VersionedRef.model_validate(simulation["expert_dataset"]))
    episode = load_episode(px.artifacts, next(row for row in dataset.episodes if row.success))
    task = px.tasks.get(episode.task)
    positions = [
        step.observations[task.success_conditions[0].signal].inline[0] for step in episode.steps
    ]
    assert max(positions) - min(positions) > 0.1
    feedback = json.loads(
        (root / "examples/evaluation_and_feedback/output/results.json").read_text()
    )
    initial = px.evaluations.get_report(
        VersionedRef.model_validate(feedback["evaluations"]["initial"])
    )
    updated = px.evaluations.get_report(
        VersionedRef.model_validate(feedback["evaluations"]["updated"])
    )
    assessment = json.loads(
        px.artifacts.get_bytes(ArtifactRef.model_validate(feedback["policy_update_assessment"]))
    )
    expected = (
        "REJECT"
        if updated.success_rate < initial.success_rate
        else "COLLECT_MORE_DATA"
        if updated.success_rate == initial.success_rate or assessment["guardrail_violations"]
        else "PROMOTE"
    )
    assert assessment["status"] == expected
    assert feedback["model"] != feedback["initial_model"]


def test_exported_artgs_bundle_reloads_without_training_directories(tmp_path) -> None:
    from physrealx.integrations.artgs import _SubprocessArtGSRunner

    root = Path(__file__).resolve().parents[2]
    report = json.loads(
        (
            root / "examples/real_assets_and_digital_twin/output/reconstruction-report.json"
        ).read_text()
    )
    if report["method"]["name"] != "artgs":
        pytest.skip("latest exported reconstruction is not ArtGS")
    # ArtGS rendering diagnostics belong to its own native acceptance check.
    assert report["metrics"]["validation_fg_psnr"] > 12
    assert report["metrics"]["validation_alpha_iou"] > 0.75
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
