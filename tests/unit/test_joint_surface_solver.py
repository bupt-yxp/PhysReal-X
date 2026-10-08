"""A synthetic fit checks coupled updates and strict holdout isolation."""

from copy import deepcopy

import numpy as np
import pytest

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise

from physrealx.integrations._joint_surface_solver import (
    _candidate_rank,
    _restoration_contacts,
    solve_joint_surfaces,
)
from physrealx.integrations._mesh_validity import limit_triangle_area_nodes
from physrealx.integrations._surface_query import CpuSurfaceQuery
from physrealx.integrations.video_articulation import (
    FramePoints,
    VideoArticulationRefinementConfig,
    _surface_metrics,
)
from physrealx.reconstruction import joint_transform


def test_candidate_selection_improves_fit_after_physical_separation():
    weights = np.ones(2)
    required = 0.0002
    safe_fit = _candidate_rank(False, np.array([0.0003, 0.0004]), weights, 0.1, required)
    more_reserve = _candidate_rank(False, np.array([0.001, 0.002]), weights, 0.5, required)
    insufficient = _candidate_rank(False, np.array([0.0001, 0.002]), weights, 0.0, required)
    crossing = _candidate_rank(True, np.array([-0.0001, 0.002]), weights, 0.0, required)
    fewer_crossings = _candidate_rank(
        True, np.array([-0.0001, 0.002]), weights, 0.5, required, intersection_count=2
    )
    more_crossings = _candidate_rank(
        True, np.array([-0.00005, 0.002]), weights, 0.0, required, intersection_count=3
    )
    equal_depth_fewer = _candidate_rank(
        True, np.array([-0.00005, 0.002]), weights, 0.5, required, intersection_count=2
    )
    assert more_crossings < fewer_crossings
    assert equal_depth_fewer < more_crossings
    assert safe_fit < more_reserve < insufficient < crossing


def test_restoration_active_set_keeps_safe_pairs_and_distributes_crossings():
    rows = {
        "positions": np.zeros(5),
        "gaps": np.array([-0.004, -0.003, -0.002, -0.001, 0.0003]),
        "parent_faces": np.array([[0, 1, 2], [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),
        "child_faces": np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]),
        "normals": np.tile([0.0, 0.0, 1.0], (5, 1)),
    }
    selected = _restoration_contacts(rows, per_pose=2)
    np.testing.assert_array_equal(selected["gaps"], [-0.004, -0.002, 0.0003])
    assert len(selected["positions"]) == 3
    with pytest.raises(ValueError, match="positive"):
        _restoration_contacts(rows, per_pose=0)


def test_tiny_face_does_not_freeze_an_unrelated_surface_patch():
    import torch

    base = torch.tensor(
        [[0, 0, 0], [0.01, 0, 0], [0.005, 1e-12, 0], [0.03, 0, 0], [0.04, 0, 0], [0.03, 0.01, 0]],
        dtype=torch.float64,
    )
    delta = torch.zeros_like(base)
    delta[2, 1] = -0.001
    delta[3:, 0] = 0.0002
    graph = dict(
        base=base,
        delta=delta,
        faces=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        idx=torch.arange(6)[:, None],
        weights=torch.ones((6, 1)),
    )
    report = limit_triangle_area_nodes(graph, torch.zeros_like(base))
    assert report["restored"] == 3
    torch.testing.assert_close(delta[:3], torch.zeros_like(delta[:3]))
    torch.testing.assert_close(delta[3:, 0], torch.full((3,), 0.0002, dtype=torch.float64))


def test_coupled_shape_and_joint_updates_preserve_input_and_ignore_check_frames():
    meshes = {}
    for name, x in [("body", -0.035), ("cover", 0.035)]:
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.012)
        mesh.apply_translation([x, 0, 0])
        meshes[name] = mesh
    original = {n: m.vertices.copy() for n, m in meshes.items()}
    axis = np.array([0.03, 1.0, 0])
    axis /= np.linalg.norm(axis)
    motion = {
        "axis": axis.tolist(),
        "origin_m": [0.001, 0, 0.001],
        "range": [0.0, 0.7],
        "positions": [0.0, 0.7],
        "kind": "hinge",
        "joint": "hinge",
        "parent": "body",
        "child": "cover",
    }
    obs = {}
    for name, mesh in meshes.items():
        truth = mesh.copy()
        truth.vertices = (truth.vertices - truth.centroid) * 1.025 + truth.centroid
        points = trimesh.sample.sample_surface(truth, 250, seed=19)[0]
        frames = []
        for index, q in enumerate([0.0, 0.35, 0.7]):
            M = (
                joint_transform("hinge", [0, 1.0, 0], [0, 0, 0], q)
                if name == "cover"
                else np.eye(4)
            )
            frames.append(FramePoints(str(index), q, trimesh.transform_points(points, M)))
        obs[name] = {"fit": frames, "check": frames}
    config = VideoArticulationRefinementConfig(
        solver_iterations=30,
        clearance_samples=3,
        remesh_rounds=0,
        hinge_band_m=0.08,
        surface_query_backend="cpu",
    )
    query = CpuSurfaceQuery()
    saved = []
    output, fitted, report = solve_joint_surfaces(
        meshes,
        obs,
        motion,
        config,
        query,
        checkpoint=lambda metrics, candidate, state: saved.append(
            {
                **deepcopy(state),
                "mesh_vertices": {n: m.vertices.copy() for n, m in candidate[0].items()},
            }
        ),
    )
    assert report["geometry_valid"]
    assert any(r["max_displacement_m"] > 1e-6 for r in report["links"].values())
    assert np.linalg.norm(np.array(fitted["origin_m"]) - motion["origin_m"]) > 1e-6
    before = _surface_metrics(meshes["cover"], obs["cover"]["check"], motion, "cover", query)
    after = _surface_metrics(output["cover"], obs["cover"]["check"], fitted, "cover", query)
    assert after["p90_m"] < before["p90_m"]
    assert fitted["range"] == motion["range"] and fitted["positions"] == motion["positions"]
    for name in meshes:
        np.testing.assert_array_equal(meshes[name].vertices, original[name])
    corrupt = deepcopy(obs)
    for value in corrupt.values():
        value["check"] = [FramePoints("wrong", 0, np.ones((250, 3)) * 100)]
    output2, fitted2, _ = solve_joint_surfaces(meshes, corrupt, motion, config, query)
    assert fitted2 == fitted
    for name in output:
        np.testing.assert_array_equal(output2[name].vertices, output[name].vertices)
    # Restart from the third complete outer iteration, including optimizer and
    # selected-candidate state, rather than reinitializing the fit.
    resumed, resumed_motion, resumed_report = solve_joint_surfaces(
        meshes,
        obs,
        motion,
        config,
        query,
        resume=saved[2],
    )
    assert resumed_motion == fitted
    assert resumed_report["history"] == report["history"]
    for name in output:
        np.testing.assert_array_equal(resumed[name].vertices, output[name].vertices)
    initial = deepcopy(saved[2])
    initial["solver_identity"] = "earlier-implementation"
    with pytest.raises(ValueError, match="solver implementation differs"):
        solve_joint_surfaces(meshes, obs, motion, config, query, resume=initial)
    initial.update(initialize_only=True, mesh_faces={n: m.faces for n, m in output.items()})
    initial["adam"] = "must not be reused"
    _initialized, new_motion, new_report = solve_joint_surfaces(
        meshes,
        obs,
        motion,
        config.model_copy(update={"solver_iterations": 5}),
        query,
        resume=initial,
    )
    assert new_report["initialized_from_solver_identity"] == "earlier-implementation"
    assert len(new_report["history"]) == 1 and new_report["history"][0]["iteration"] == 5
    assert new_motion["range"] == motion["range"]
    initial["mesh_faces"]["body"] = np.empty((0, 3), int)
    with pytest.raises(ValueError, match="surface topology differs"):
        solve_joint_surfaces(meshes, obs, motion, config, query, resume=initial)
    corrupt_state = deepcopy(saved[2])
    corrupt_state["parameters"][1][0] = 100
    with pytest.raises(ValueError, match="source trust region"):
        solve_joint_surfaces(meshes, obs, motion, config, query, resume=corrupt_state)


def test_observed_fold_joint_solver_separates_across_the_complete_motion_range():
    from physrealx.integrations._surface_clearance import swept_clearance

    body = trimesh.Trimesh(
        [[-0.02, -0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0], [-0.02, 0.02, 0]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    cover = trimesh.Trimesh(
        [
            [-0.008, -0.008, -0.001],
            [0.008, -0.008, -0.001],
            [-0.008, 0, 0.003],
            [0.008, 0, 0.003],
            [-0.008, 0.008, 0.006],
            [0.008, 0.008, 0.006],
        ],
        [[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]],
        process=False,
    )
    meshes = dict(body=body, cover=cover)
    truth = {n: m.copy() for n, m in meshes.items()}
    truth["cover"].vertices[:, 2] = [0.005, 0.005, 0.006, 0.006, 0.008, 0.008]
    motion = dict(
        parent="body",
        child="cover",
        axis=[1.0, 0, 0],
        origin_m=[0, 0, 0],
        kind="hinge",
        joint="hinge",
        range=[0.0, 0.4],
        positions=[0.0, 0.4],
    )
    obs = {}
    for n, m in truth.items():
        obs[n] = {}
        for partition, seed in [("fit", 5), ("check", 6)]:
            pts = trimesh.sample.sample_surface(m, 300, seed=seed)[0]
            obs[n][partition] = []
            for i, q in enumerate([0.0, 0.2, 0.4]):
                points = (
                    trimesh.transform_points(
                        pts, joint_transform("hinge", motion["axis"], motion["origin_m"], q)
                    )
                    if n == "cover"
                    else pts
                )
                obs[n][partition].append(FramePoints(str(i), q, points))
    cfg = VideoArticulationRefinementConfig(
        solver_iterations=120, clearance_samples=3, remesh_rounds=0, cpu_query_workers=2
    )
    query = CpuSurfaceQuery()
    result, m, report = solve_joint_surfaces(meshes, obs, motion, cfg, query)
    clearance = swept_clearance(
        result["body"],
        result["cover"],
        m,
        query,
        clearance=cfg.minimum_clearance_m,
        samples=17,
        temporal_levels=3,
        max_cells=10000,
    )
    assert report["geometry_valid"]
    assert clearance["status"] == "PASSED"
    assert clearance["minimum_clearance_lower_bound_m"] >= cfg.minimum_clearance_m
    assert m["range"] == motion["range"] and m["positions"] == motion["positions"]
    for name in meshes:
        initial = _surface_metrics(meshes[name], obs[name]["check"], motion, name, query)
        fitted = _surface_metrics(result[name], obs[name]["check"], m, name, query)
        assert fitted["p90_m"] <= initial["p90_m"] + cfg.max_check_degradation_m


def test_restoration_direction_is_not_cancelled_by_soft_fit_regularization():
    """Crossed observations constrain restoration without attracting it backwards."""
    body = trimesh.Trimesh(
        [[-0.01, -0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0], [-0.01, 0.01, 0]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    cover = body.copy()
    cover.vertices[:, 0] *= 0.7
    cover.vertices[:, 2] = [-0.0004, -0.0004, 0.002, 0.002]
    meshes = dict(body=body, cover=cover)
    motion = dict(
        parent="body",
        child="cover",
        kind="hinge",
        joint="hinge",
        axis=[1.0, 0, 0],
        origin_m=[0, 0, 0],
        range=[0.0, 0.1],
        positions=[0.0, 0.1],
    )
    observations = {}
    for name, mesh in meshes.items():
        points = trimesh.sample.sample_surface(mesh, 150, seed=8)[0]
        frames = [
            FramePoints(
                str(q),
                q,
                trimesh.transform_points(
                    points, joint_transform("hinge", motion["axis"], motion["origin_m"], q)
                )
                if name == "cover"
                else points,
            )
            for q in (0.0, 0.1)
        ]
        observations[name] = dict(fit=frames, check=frames)
    config = VideoArticulationRefinementConfig(
        solver_iterations=5,
        clearance_samples=3,
        remesh_rounds=0,
        cpu_query_workers=2,
    )
    checkpoints = []
    for weight in (0.2, 100000.0):
        saved = []
        solve_joint_surfaces(
            meshes,
            observations,
            motion,
            config.model_copy(update={"surface_smoothness": weight}),
            CpuSurfaceQuery(),
            checkpoint=lambda metrics, candidate, state, saved=saved: saved.append(deepcopy(state)),
        )
        assert saved[-1]["history"][-1]["phase"] == "restore_feasibility"
        checkpoints.append(saved[-1])
    for first, second in zip(
        checkpoints[0]["parameters"], checkpoints[1]["parameters"], strict=True
    ):
        np.testing.assert_array_equal(first, second)
    assert any(np.linalg.norm(p) > 1e-8 for p in checkpoints[0]["parameters"])


def test_video_gradient_improves_observation_without_reversing_separation():
    import torch

    from physrealx.integrations._joint_surface_solver import _observed_separation_gradients

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    unused = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    separation = parameter[0] + parameter[1]
    observation = -3 * parameter[0] + 2 * parameter[1]
    _observed_separation_gradients(separation, observation, [parameter, unused])
    direction = -parameter.grad
    assert float(direction @ torch.tensor([1.0, 1.0], dtype=torch.float64)) < 0
    assert float(direction @ torch.tensor([-3.0, 2.0], dtype=torch.float64)) < 0
    torch.testing.assert_close(unused.grad, torch.zeros_like(unused))
