"""Feasibility restoration, persistent branches and adversarial fit proposals."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise

from physrealx.integrations._contact_projection import ContactProjection
from physrealx.integrations._triangle_contacts import swept_constraints
from physrealx.integrations.video_articulation import VideoArticulationRefinementConfig


def fixture():
    body = trimesh.Trimesh(
        [[-0.02, -0.02, 0], [0.02, -0.02, 0], [0, 0.02, 0]], [[0, 1, 2]], process=False
    )
    cover = body.copy()
    cover.apply_translation([0, 0, 0.0001])
    meshes = dict(body=body, cover=cover)
    motion = dict(
        parent="body", child="cover", axis=[1.0, 0, 0], origin_m=[0.0, 0, 0], range=[0.0, 0.4]
    )
    graphs = {
        n: dict(
            editable=np.ones(3, bool),
            observed=np.zeros(3, bool),
            base=torch.tensor(m.vertices),
            observation_normal=np.zeros((3, 3)),
            observation_interval=np.zeros((3, 2)),
        )
        for n, m in meshes.items()
    }
    config = VideoArticulationRefinementConfig()
    projection = ContactProjection(motion, meshes, graphs, config)
    constraints = swept_constraints(body, cover, motion, [0.0], margin=0.0007, budget=100)
    projection.add(constraints, meshes, motion)
    return meshes, motion, projection


def test_hard_projection_restores_gap_then_resists_an_inward_fit_proposal():
    meshes, motion, projection = fixture()
    required = projection.config.minimum_clearance_m
    previous = required - projection.gaps(meshes, motion).min()
    for _ in range(32):
        proposal = [
            np.zeros(3),
            np.zeros(3),
            np.tile([0, 0, 0.001], (3, 1)),
            np.tile([0, 0, -0.001], (3, 1)),
        ]
        update, report = projection.project(meshes, motion, proposal)
        assert update is not None, report
        for (_name, mesh), delta in zip(meshes.items(), update[2:], strict=True):
            mesh.vertices += delta
        violation = max(0.0, required - projection.gaps(meshes, motion).min())
        assert violation <= max(2e-8, previous - 1e-8)
        previous = violation
    assert projection.gaps(meshes, motion).min() >= required - 2e-8
    assert report["maximum_linear_constraint_violation_m"] <= 2e-8


def test_rms_metric_projection_keeps_descent_and_the_safe_gap():
    meshes, motion, projection = fixture()
    meshes["cover"].apply_translation([0, 0, 0.0009])
    projection.refit_supports(meshes, motion)
    proposed = [
        np.zeros(3),
        np.zeros(3),
        np.tile([0, 0, 0.0005], (3, 1)),
        np.tile([0, 0, -0.0005], (3, 1)),
    ]
    metric = [np.ones(3), np.ones(3), np.full((3, 3), 100.0), np.ones((3, 3))]
    step, report = projection.project(
        meshes, motion, proposed, quadratic=True, quadratic_metric=metric
    )
    assert step is not None, report
    gradient_dot_step = sum(
        float((-d * p * s).sum()) for d, p, s in zip(metric, proposed, step, strict=True)
    )
    assert gradient_dot_step < 0
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    assert projection.gaps(meshes, motion).min() >= projection.config.minimum_clearance_m - 2e-9
    metric[2][:] = 0
    with pytest.raises(ValueError, match="finite and positive"):
        projection.project(meshes, motion, proposed, quadratic=True, quadratic_metric=metric)


def test_quadratic_solver_success_does_not_override_original_constraint_residual(monkeypatch):
    import clarabel
    from scipy import sparse

    from physrealx.integrations._contact_projection import _quadratic_projection

    monkeypatch.setattr(
        clarabel,
        "DefaultSolver",
        lambda *a, **k: SimpleNamespace(
            solve=lambda: SimpleNamespace(x=np.zeros(1), status="Solved", iterations=1)
        ),
    )
    value, audit = _quadratic_projection(
        sparse.eye(1, format="csc"),
        np.ones(1),
        np.zeros(1),
        -np.ones(1),
        np.ones(1),
        np.array([0.5]),
        np.array([np.inf]),
    )
    assert value is None
    assert audit["maximum_linear_constraint_violation_m"] == 0.0005


def test_timed_out_quadratic_iterate_needs_explicit_nonlinear_guard(monkeypatch):
    import clarabel
    from scipy import sparse

    from physrealx.integrations._contact_projection import _quadratic_projection

    monkeypatch.setattr(
        clarabel,
        "DefaultSolver",
        lambda *a, **k: SimpleNamespace(
            solve=lambda: SimpleNamespace(x=np.zeros(1), status="MaxTime", iterations=1)
        ),
    )
    args = (
        sparse.eye(1, format="csc"),
        np.ones(1),
        np.zeros(1),
        -np.ones(1),
        np.ones(1),
        np.array([0.5]),
        np.array([np.inf]),
    )
    rejected, audit = _quadratic_projection(*args)
    proposal, guarded_audit = _quadratic_projection(*args, require_feasible=False)
    assert rejected is None and audit["linear_constraints_satisfied"] is False
    np.testing.assert_array_equal(proposal, np.zeros(1))
    assert guarded_audit["maximum_linear_constraint_violation_m"] == 0.0005


def test_restoration_minimizes_motion_only_after_fixing_achieved_separation():
    meshes, motion, projection = fixture()
    initial = projection.gaps(meshes, motion).min()
    required = projection.config.minimum_clearance_m
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
    )
    assert step is not None, report
    assert report["selected_phase"] == "minimum_motion"
    # All three vertices on each side share a planar gap. The minimum total
    # motion is three times the missing gap (plus the numerical solve margin).
    assert sum(np.abs(delta).sum() for delta in step[2:]) <= 3 * (required - initial + 2.1e-6)
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    assert projection.gaps(meshes, motion).min() >= required - 2e-8


def test_rebinding_does_not_flip_a_persistent_separation_direction():
    meshes, motion, projection = fixture()
    original = projection.arrays()[2].copy()
    meshes["cover"].vertices[:, 2] = -0.0001
    constraints = swept_constraints(
        meshes["body"], meshes["cover"], motion, [0.0], margin=0.0007, budget=100
    )
    projection.add(constraints, meshes, motion)
    np.testing.assert_array_equal(projection.arrays()[2], original)
    step, _ = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
    )
    assert step is not None
    assert step[3][:, 2].mean() - step[2][:, 2].mean() > 0


def test_local_validity_limits_reallocate_separation_to_the_other_surface():
    meshes, motion, projection = fixture()
    meshes["cover"].vertices[:, 2] = -0.0015
    # A local validity check permits only 1 µm of the body's 0.5 mm
    # restoration move. Re-solving must involve the other side, not scale
    # every surface by the body's restrictive ratio.
    requested = np.tile([0, 0, -0.0005], (3, 1))
    limited = projection.learn_local_limits("body", requested, requested * 0.002)
    before = projection.gaps(meshes, motion).min()
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
    )
    assert step is not None, report
    assert np.abs(step[2]).max() <= 1.001e-6
    assert step[3][:, 2].min() >= 0.000499
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    assert projection.gaps(meshes, motion).min() > before + 0.000499
    projection.release_local_limits(dict(body=limited, cover=np.zeros(3, bool)))
    np.testing.assert_allclose(projection.local_radii["body"], 1e-6)
    projection.release_local_limits(dict(body=np.zeros(3, bool), cover=np.zeros(3, bool)))
    np.testing.assert_allclose(projection.local_radii["body"], 1.5e-6)


def test_zero_proposal_export_noise_does_not_lock_local_geometry():
    _meshes, _motion, projection = fixture()
    limited = projection.learn_local_limits("body", np.zeros((3, 3)), np.full((3, 3), 2e-8))
    assert not limited.any()
    np.testing.assert_array_equal(projection.local_radii["body"], np.full(3, 0.0005))


def test_crossing_region_reconciles_the_opposite_side_of_a_shared_vertex():
    body = trimesh.Trimesh(
        [[-0.2, -0.2, 0], [0.2, -0.2, 0], [0.2, 0.2, 0], [-0.2, 0.2, 0]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    cover = trimesh.Trimesh(
        [[-0.1, -0.1, -0.0001], [0.1, -0.1, -0.0001], [0.1, 0.1, -0.0001], [-0.1, 0.1, 0.001]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    meshes = dict(body=body, cover=cover)
    motion = dict(parent="body", child="cover", axis=[1.0, 0, 0], origin_m=[0.0, 0, 0])
    graphs = {
        name: dict(
            editable=np.ones(4, bool),
            observed=np.zeros(4, bool),
            base=torch.tensor(mesh.vertices),
            observation_normal=np.zeros((4, 3)),
            observation_interval=np.zeros((4, 2)),
        )
        for name, mesh in meshes.items()
    }
    projection = ContactProjection(motion, meshes, graphs, VideoArticulationRefinementConfig())
    constraints = swept_constraints(body, cover, motion, [0.0], margin=0.0007, budget=100)
    assert np.any(constraints["normals"][:, 2] > 0.9)
    projection.add(constraints, meshes, motion)
    # A valid lateral plane need not point downward. Unresolved neighbours
    # keep the same oriented hemisphere; the independent check below verifies
    # the actual separation rather than prescribing a particular plane angle.
    certified = projection.gaps(meshes, motion) >= projection.config.minimum_clearance_m
    branches = np.array(list(projection.branch_directions.values()))
    alignment = (projection.arrays()[2] * branches).sum(1)
    assert np.all(certified | (alignment >= 1e-6 - 1e-10))
    for _ in range(4):
        projection.refit_supports(meshes, motion)
        step, report = projection.project(
            meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((4, 3)), np.zeros((4, 3))]
        )
        assert step is not None, report
        for mesh, delta in zip(meshes.values(), step[2:], strict=True):
            mesh.vertices += delta
    required = projection.config.minimum_clearance_m
    assert projection.gaps(meshes, motion).min() >= required - 2e-8
    independent = swept_constraints(body, cover, motion, [0.0], margin=required - 2e-8, budget=100)
    assert independent["candidate_pairs"] == 0


def test_large_initial_overlap_makes_bounded_progress_without_relaxing_the_final_gap():
    meshes, motion, projection = fixture()
    meshes["cover"].vertices[:, 2] = -0.004
    required = projection.config.minimum_clearance_m
    before = required - projection.gaps(meshes, motion).min()
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
    )
    assert step is not None and report["restoration_pairs"] == 1
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    assert 0 < required - projection.gaps(meshes, motion).min() < before
    assert report["maximum_linear_constraint_violation_m"] < 1e-8


def test_supporting_planes_follow_bending_without_reversing_the_original_branch():
    meshes, motion, projection = fixture()
    initial = projection.arrays()[2].copy()
    rotation = trimesh.transformations.rotation_matrix(np.pi / 3, [1, 0, 0])
    for mesh in meshes.values():
        mesh.apply_transform(rotation)
    assert projection.gaps(meshes, motion).min() < -0.01
    constraints = swept_constraints(
        meshes["body"], meshes["cover"], motion, [0.0], margin=0.0007, budget=100
    )
    projection.add(constraints, meshes, motion)
    assert projection.gaps(meshes, motion).min() > 0.00009
    assert np.all((projection.arrays()[2] * initial).sum(1) >= 0.3)


def test_time_limit_does_not_bypass_feasibility_checks():
    import highspy

    meshes, motion, projection = fixture()
    with patch(
        "highspy.Highs.getModelStatus",
        return_value=highspy.HighsModelStatus.kTimeLimit,
    ):
        step, report = projection.project(
            meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
        )
    assert step is not None
    assert report["maximum_linear_constraint_violation_m"] < 1e-8


def test_textual_solver_success_with_invalid_constraints_is_rejected():
    meshes, motion, projection = fixture()
    result = SimpleNamespace(value_valid=True, col_value=np.zeros(26))
    with patch("highspy.Highs.getSolution", return_value=result):
        step, report = projection.project(
            meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
        )
    assert step is None
    assert report["maximum_linear_constraint_violation_m"] > 0.0001


def test_self_contact_constraints_preserve_an_unrelated_fit_update():
    lower = np.array([[-0.01, -0.01, 0], [0.01, -0.01, 0], [0, 0.01, 0]])
    body = trimesh.Trimesh(
        np.vstack([lower, lower + np.array([0, 0, 0.001]), lower + np.array([0.05, 0, 0])]),
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        process=False,
    )
    cover = trimesh.Trimesh(lower + np.array([1, 0, 0]), [[0, 1, 2]], process=False)
    meshes = dict(body=body, cover=cover)
    motion = dict(parent="body", child="cover", axis=[1.0, 0, 0], origin_m=[0.0, 0, 0])
    graphs = {
        n: dict(
            editable=np.ones(len(m.vertices), bool),
            observed=np.zeros(len(m.vertices), bool),
            base=torch.tensor(m.vertices),
            observation_normal=np.zeros_like(m.vertices),
            observation_interval=np.zeros((len(m.vertices), 2)),
        )
        for n, m in meshes.items()
    }
    projection = ContactProjection(motion, meshes, graphs, VideoArticulationRefinementConfig())
    projection.add_self_contacts(
        "body", [(np.array([3, 4, 5]), np.array([0, 1, 2]), np.array([0, 0, 1.0]), 1e-6)]
    )
    proposal = np.zeros((9, 3))
    proposal[3:6, 2] = -0.002
    proposal[6:, 0] = 0.002
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), proposal, np.zeros((3, 3))]
    )
    assert step is not None, report
    moved = body.vertices + step[2]
    assert moved[3:6, 2].min() - moved[:3, 2].max() >= 1e-6 - 2e-8
    np.testing.assert_allclose(step[2][6:, 0], 0.002)
    assert report["self_contact_planes"] == 1


def test_projection_expands_an_infeasible_fixed_boundary_instead_of_freezing_a_valid_sheet():
    vertices = np.array([[x * 0.001, y * 0.001, 0.0] for x in range(9) for y in range(2)])
    faces = np.array(
        [[2 * i, 2 * i + 2, 2 * i + 1] for i in range(8)]
        + [[2 * i + 1, 2 * i + 2, 2 * i + 3] for i in range(8)]
    )
    meshes = {
        "body": trimesh.Trimesh(vertices, faces, process=False),
        "cover": trimesh.Trimesh(vertices + np.array([0, 0, 1.0]), faces, process=False),
    }
    motion = dict(parent="body", child="cover", axis=[1.0, 0, 0], origin_m=[0.0, 0, 0])
    graphs = {
        n: dict(
            editable=np.ones(len(vertices), bool),
            observed=np.zeros(len(vertices), bool),
            base=torch.tensor(m.vertices),
            observation_normal=np.zeros_like(vertices),
            observation_interval=np.zeros((len(vertices), 2)),
        )
        for n, m in meshes.items()
    }
    projection = ContactProjection(motion, meshes, graphs, VideoArticulationRefinementConfig())
    projection.add_self_contacts(
        "body", [(np.array([0]), np.array([2]), np.array([-1.0, 0, 0]), 0.0001)]
    )
    projection.local_radii["body"][:8] = 1e-8
    proposal = np.zeros_like(vertices)
    proposal[8:10, 0] = -0.001
    proposal[16:, 2] = 0.0001  # An unrelated valid update must remain available.
    step, report = projection.project(
        meshes,
        motion,
        [np.zeros(3), np.zeros(3), proposal, np.zeros_like(vertices)],
        quadratic=True,
    )
    assert step is not None, report
    assert 0 < report["boundary_expanded_nodes"]["body"] < 6
    moved = meshes["body"].copy()
    moved.vertices += step[2]
    assert np.all(moved.area_faces >= 0.1 * meshes["body"].area_faces)
    np.testing.assert_allclose(step[2][16:, 2], 0.0001)


def test_blocked_worst_patch_does_not_freeze_an_independent_contact_patch():
    meshes, motion, projection = fixture()
    for name, mesh in list(meshes.items()):
        second = mesh.copy()
        second.apply_translation([0.1, 0, 0])
        meshes[name] = trimesh.util.concatenate([mesh, second])
        projection.graphs[name] = dict(
            editable=np.array([False] * 3 + [True] * 3),
            observed=np.zeros(6, bool),
            base=torch.tensor(meshes[name].vertices),
            observation_normal=np.zeros((6, 3)),
            observation_interval=np.zeros((6, 2)),
        )
    meshes["cover"].vertices[:3, 2] = -0.004
    meshes["cover"].vertices[3:, 2] = -0.0001
    projection = ContactProjection(motion, meshes, projection.graphs, projection.config)
    for ids in (np.arange(3), np.arange(3, 6)):
        projection.rows[tuple(ids)] = (ids, ids, np.array([0.0, 0.0, -1.0]), 0.0)
    before = projection.gaps(meshes, motion)
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((6, 3)), np.zeros((6, 3))]
    )
    assert step is not None, report
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    after = projection.gaps(meshes, motion)
    assert after[0] == before[0]
    assert after[1] >= projection.config.minimum_clearance_m
    assert np.max(np.abs(np.concatenate(step[2:]))) <= 0.000500001


def test_small_contact_patch_cannot_trade_required_gap_for_smaller_motion():
    meshes, motion, projection = fixture()
    for name, mesh in list(meshes.items()):
        tiny = mesh.copy()
        tiny.vertices[:, :2] *= 0.001
        tiny.vertices[:, 0] += 0.1
        meshes[name] = trimesh.util.concatenate([mesh, tiny])
        projection.graphs[name] = dict(
            editable=np.ones(6, bool),
            observed=np.zeros(6, bool),
            base=torch.tensor(meshes[name].vertices),
            observation_normal=np.zeros((6, 3)),
            observation_interval=np.zeros((6, 2)),
        )
    projection = ContactProjection(motion, meshes, projection.graphs, projection.config)
    for ids in (np.arange(3), np.arange(3, 6)):
        projection.rows[tuple(ids)] = (ids, ids, np.array([0.0, 0.0, -1.0]), 0.0)
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((6, 3)), np.zeros((6, 3))]
    )
    assert step is not None, report
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    assert np.min(projection.gaps(meshes, motion)) >= projection.config.minimum_clearance_m


def test_self_contact_planes_follow_a_safe_rigid_rotation():
    meshes, motion, projection = fixture()
    lower = meshes["body"].copy()
    upper = lower.copy()
    upper.apply_translation([0, 0, 0.001])
    body = trimesh.util.concatenate([lower, upper])
    projection.graphs["body"] = dict(
        editable=np.ones(6, bool),
        observed=np.zeros(6, bool),
        base=torch.tensor(body.vertices),
        observation_normal=np.zeros((6, 3)),
        observation_interval=np.zeros((6, 2)),
    )
    meshes["body"] = body
    projection = ContactProjection(motion, meshes, projection.graphs, projection.config)
    for a, b in (([0, 1, 2], [3, 4, 5]), ([0], [3]), ([0, 1], [3, 4])):
        projection.add_self_contacts(
            "body", [(np.array(a), np.array(b), np.array([0.0, 0.0, -1.0]), 1e-6)]
        )
    body.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 3, [1, 0, 0]))
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((6, 3)), np.zeros((3, 3))]
    )
    assert step is not None, report
    np.testing.assert_allclose(step[2], 0, atol=1e-10)
    for name, a, b, n, gap in projection.self_planes.values():
        assert (meshes[name].vertices[a] @ n).min() - (meshes[name].vertices[b] @ n).max() >= gap


def test_changing_query_worker_budget_does_not_disable_contact_solves():
    meshes, motion, projection = fixture()
    proposal = [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
    for workers in (1, 4, 2):
        projection.config = projection.config.model_copy(update={"cpu_query_workers": workers})
        step, report = projection.project(meshes, motion, proposal)
        assert step is not None, report
        assert report["maximum_linear_constraint_violation_m"] < 1e-8


def test_restoration_reduces_the_worst_contact_before_optimizing_patch_area():
    meshes, motion, projection = fixture()
    bottom = meshes["body"].copy()
    top = bottom.copy()
    top.vertices[:, :2] *= 0.001
    top.vertices[:, 2] = 0.0003
    meshes["body"] = trimesh.util.concatenate([bottom, top])
    graph = projection.graphs["body"]
    graph.update(
        editable=np.zeros(6, bool),
        observed=np.zeros(6, bool),
        base=torch.tensor(meshes["body"].vertices),
        observation_normal=np.zeros((6, 3)),
        observation_interval=np.zeros((6, 2)),
    )
    meshes["cover"].vertices[:, 2] = 0.0001
    projection = ContactProjection(motion, meshes, projection.graphs, projection.config)
    projection.rows["bottom"] = (np.arange(3), np.arange(3), np.array([0.0, 0.0, -1.0]), 0.0)
    projection.rows["top"] = (np.arange(3, 6), np.arange(3), np.array([0.0, 0.0, 1.0]), 0.0)
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((6, 3)), np.zeros((3, 3))]
    )
    assert step is not None, report
    meshes["cover"].vertices += step[3]
    # Both contacts get the best attainable worst gap, even though one patch
    # carries a million times less area. Area alone favors the larger patch.
    np.testing.assert_allclose(projection.gaps(meshes, motion), 0.00015, atol=2e-8)


def test_secondary_numerical_failure_keeps_the_verified_primary_step():
    from itertools import chain, repeat

    import highspy

    meshes, motion, projection = fixture()
    original_solution = highspy.Highs.getSolution
    reads = 0

    def invalid_after_primary(solver):
        nonlocal reads
        reads += 1
        return (
            original_solution(solver)
            if reads == 1
            else SimpleNamespace(value_valid=True, col_value=np.zeros(solver.getNumCol()))
        )

    with patch(
        "highspy.Highs.getModelStatus",
        side_effect=chain(
            [highspy.HighsModelStatus.kOptimal], repeat(highspy.HighsModelStatus.kUnknown)
        ),
    ), patch("highspy.Highs.getSolution", invalid_after_primary):
        step, report = projection.project(
            meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
        )
    assert step is not None, report
    assert report["selected_phase"] == "worst_gap"
    assert report["secondary_status"] == "Unknown"
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    assert projection.gaps(meshes, motion).min() >= projection.config.minimum_clearance_m - 2e-8


def test_unknown_solver_status_cannot_override_an_independently_verified_primal():
    import highspy

    meshes, motion, projection = fixture()
    with patch("highspy.Highs.getModelStatus", return_value=highspy.HighsModelStatus.kUnknown):
        step, report = projection.project(
            meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
        )
    assert step is not None, report
    assert report["presolve_retries"] == 1
    assert not report["optimality_proven"]
    assert report["maximum_linear_constraint_violation_m"] < 1e-8
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    required = projection.config.minimum_clearance_m
    assert projection.gaps(meshes, motion).min() >= required - 2e-8


def test_contact_projection_rejects_a_locally_collapsing_fit_direction():
    meshes, motion, projection = fixture()
    cover = trimesh.Trimesh(
        [[0, 0, 0.002], [0.0002, 0, 0.002], [0, 0.0002, 0.002]], [[0, 1, 2]], process=False
    )
    meshes["cover"] = cover
    projection.graphs["cover"]["base"] = torch.tensor(cover.vertices)
    proposal = [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
    proposal[3][1, 0] = -0.0002
    step, report = projection.project(meshes, motion, proposal)
    assert step is not None, report
    moved = cover.copy()
    moved.vertices += step[3]
    assert moved.area > 0.1 * cover.area
    assert report["area_constraints"] > 0


def test_joint_linearization_matches_the_actual_normalized_tangent_parameter():
    import highspy
    from scipy import sparse

    from physrealx.reconstruction import joint_transform

    meshes, source, projection = fixture()
    raw_axis = np.array([1.0, 0.12, 0])
    current = dict(source, axis=(raw_axis / np.linalg.norm(raw_axis)).tolist())
    normal = np.array([0.1, 0.2, -1.0])
    normal /= np.linalg.norm(normal)
    projection.rows = {"pair": (np.arange(3), np.arange(3), normal, 0.4)}
    matrices = []
    original = highspy.Highs.passModel

    def capture(solver, problem):
        matrix = problem.a_matrix_
        matrices.append(
            sparse.csc_matrix(
                (matrix.value_, matrix.index_, matrix.start_),
                shape=(problem.num_row_, problem.num_col_),
            )
        )
        return original(solver, problem)

    with patch("highspy.Highs.passModel", capture):
        projection.project(
            meshes, current, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
        )

    def actual_gaps(tangent):
        axis = raw_axis + tangent
        matrix = joint_transform("hinge", axis / np.linalg.norm(axis), source["origin_m"], 0.4)
        right = meshes["cover"].vertices @ matrix[:3, :3].T + matrix[:3, 3]
        return ((meshes["body"].vertices[:, None] - right[None]) * normal).sum(2).ravel()

    epsilon = 1e-6
    expected = np.column_stack(
        [
            (actual_gaps(direction * epsilon) - actual_gaps(-direction * epsilon)) / (2 * epsilon)
            for direction in np.eye(3)
        ]
    )
    np.testing.assert_allclose(matrices[0][:9, :3].toarray(), expected, atol=1e-8, rtol=2e-6)


def test_reserve_slack_cannot_break_an_already_safe_physical_gap():
    meshes, motion, projection = fixture()
    lower = meshes["body"].copy()
    upper = lower.copy()
    upper.apply_translation([0, 0, -0.001])
    meshes["body"] = trimesh.util.concatenate([lower, upper])
    graphs = projection.graphs
    graphs["body"] = dict(
        editable=np.zeros(6, bool),
        observed=np.zeros(6, bool),
        base=torch.tensor(meshes["body"].vertices),
        observation_normal=np.zeros((6, 3)),
        observation_interval=np.zeros((6, 2)),
    )
    meshes["cover"].vertices[:, 2] = 0.0003
    projection = ContactProjection(motion, meshes, graphs, projection.config)
    projection.rows = {
        "safe": (np.arange(3), np.arange(3), np.array([0.0, 0, -1]), 0.0),
        "crossed": (np.arange(3, 6), np.arange(3), np.array([0.0, 0, 1]), 0.0),
    }
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((6, 3)), np.zeros((3, 3))]
    )
    assert step is not None, report
    meshes["cover"].vertices += step[3]
    gaps = projection.gaps(meshes, motion)
    # The opposed fixed sheets cannot both separate. Restoration must retain
    # the existing 0.2 mm safe gap instead of buying a smaller worst deficit
    # by creating a new crossing in the previously safe pair.
    assert gaps[0] >= projection.config.minimum_clearance_m - 2e-9
    assert gaps[1] < 0


def test_joint_trust_regions_are_solved_together_instead_of_clipped_afterwards():
    meshes, source, projection = fixture()
    meshes["cover"].vertices[:, 2] = 0.002
    axis_radius = np.tan(np.radians(projection.config.max_axis_change_deg))
    origin_radius = projection.config.max_origin_displacement_m
    axis_delta = np.array([0, axis_radius * 0.9999, 0])
    origin_delta = np.array([0, origin_radius * 0.9999, 0])
    raw_axis = np.array(source["axis"]) + axis_delta
    current = dict(
        source,
        axis=(raw_axis / np.linalg.norm(raw_axis)).tolist(),
        origin_m=(np.array(source["origin_m"]) + origin_delta).tolist(),
    )
    step, report = projection.project(
        meshes,
        current,
        [np.array([0, 0.002, 0]), np.array([0, 0.0005, 0]), np.zeros((3, 3)), np.zeros((3, 3))],
    )
    assert step is not None, report
    # No post-solve clipping is applied here. The returned joint update itself
    # must respect both accumulated limits and retain the feasible fit movement.
    assert np.linalg.norm(axis_delta + step[0]) <= axis_radius + 1e-10
    assert np.linalg.norm(origin_delta + step[1]) <= origin_radius + 1e-10
    assert step[0][1] > 0.5 * axis_radius * 0.0001
    assert step[1][1] > 0.5 * origin_radius * 0.0001


def test_tight_local_bounds_keep_a_physically_valid_current_state_feasible():
    meshes, motion, projection = fixture()
    meshes["cover"].vertices[:, 2] = projection.config.minimum_clearance_m + 1e-10
    for radii in projection.local_radii.values():
        radii.fill(1e-9)
    # A numerical reserve cannot require more movement than the local trust
    # bounds allow when zero movement already satisfies the physical gap.
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
    )
    assert step is not None, report
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    assert projection.gaps(meshes, motion).min() >= projection.config.minimum_clearance_m - 2e-9


def test_crossing_branch_can_unfold_without_an_artificial_normal_angle_limit():
    meshes, motion, projection = fixture()
    meshes["cover"].vertices[:, 2] = [-0.0001, 0.0001, 0.0001]
    constraints = swept_constraints(
        meshes["body"], meshes["cover"], motion, [0.0], margin=0.0007, budget=100
    )
    projection.add(constraints, meshes, motion)
    assert projection.coherent_keys
    reference = np.array(list(projection.branch_directions.values()))
    meshes["cover"].vertices[:, 2] = 0.001
    rotation = trimesh.transformations.rotation_matrix(np.radians(70), [1, 0, 0])
    for mesh in meshes.values():
        mesh.apply_transform(rotation)
    projection.refit_supports(meshes, motion)
    # A 70-degree rotation stays on the same oriented branch. A narrow
    # 25-degree normal cone falsely reports overlap between these clear planes.
    assert projection.gaps(meshes, motion).min() >= 0.000999
    assert np.all((projection.arrays()[2] * reference).sum(1) > 0)


def test_local_limits_do_not_prevent_restoring_a_numerical_observation_residual():
    meshes, motion, projection = fixture()
    graph = projection.graphs["body"]
    graph["observed"][:] = True
    graph["observation_normal"][:] = [0, 0, 1]
    graph["observation_interval"][:] = [-0.00025, 0.00025]
    meshes["body"].vertices[:, 2] = -0.00025 - 6e-8
    projection.local_radii["body"].fill(1e-9)
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))]
    )
    assert step is not None, report
    moved = meshes["body"].vertices + step[2]
    assert moved[:, 2].min() >= -0.00025 - 1e-10
    assert step[2][:, 2].min() >= 5.99e-8


def test_intersecting_support_can_rotate_within_its_coherent_hemisphere():
    meshes, motion, projection = fixture()
    meshes["cover"].vertices[:, 2] = [-0.0001, 0.0001, 0.0001]
    projection.add(
        swept_constraints(
            meshes["body"], meshes["cover"], motion, [0.0], margin=0.0007, budget=100
        ),
        meshes,
        motion,
    )
    branch = np.array(list(projection.branch_directions.values()))
    rotation = trimesh.transformations.rotation_matrix(np.radians(70), [1, 0, 0])
    for mesh in meshes.values():
        mesh.apply_transform(rotation)
    projection.refit_supports(meshes, motion)
    # The intersection did not deepen when both triangles rotated together.
    # Keeping a 25-degree cone about the old normal invents a centimetre-scale
    # restoration target for this 0.1 mm crossing.
    assert projection.gaps(meshes, motion).min() > -0.0002
    assert np.all((projection.arrays()[2] * branch).sum(1) > 0)


def test_certified_clearance_is_not_overridden_by_a_stale_branch():
    meshes, motion, projection = fixture()
    meshes["cover"].vertices[:, 2] = 0.001
    before = {name: mesh.vertices.copy() for name, mesh in meshes.items()}
    for key, (a, b, normal, q) in list(projection.rows.items()):
        projection.rows[key] = a, b, -normal, q
        projection.branch_directions[key] = -normal
        projection.coherent_keys.add(key)
    assert projection.gaps(meshes, motion).min() < 0
    projection.refit_supports(meshes, motion)
    assert projection.gaps(meshes, motion).min() >= 0.000999
    for name, mesh in meshes.items():
        np.testing.assert_array_equal(mesh.vertices, before[name])


def test_optional_reserve_cannot_block_a_feasible_narrow_gap():
    meshes, motion, previous = fixture()
    lower = meshes["body"].copy()
    upper = lower.copy()
    upper.apply_translation([0, 0, 0.0006])
    meshes["body"] = trimesh.util.concatenate([lower, upper])
    meshes["cover"].vertices[:, 2] = 0.0008
    graphs = previous.graphs
    graphs["body"] = dict(
        editable=np.zeros(6, bool),
        observed=np.zeros(6, bool),
        base=torch.tensor(meshes["body"].vertices),
        observation_normal=np.zeros((6, 3)),
        observation_interval=np.zeros((6, 2)),
    )
    projection = ContactProjection(motion, meshes, graphs, previous.config)
    projection.rows = {
        "lower": (np.arange(3), np.arange(3), np.array([0.0, 0, -1]), 0.0),
        "upper": (np.arange(3, 6), np.arange(3), np.array([0.0, 0, 1]), 0.0),
    }
    # The middle sheet fits at z=0.3 mm with 0.3 mm clearance on BOTH
    # sides. Locking its old 0.7 mm reserve against the lower sheet makes
    # separating from the upper sheet impossible, despite this feasible solution.
    step, report = projection.project(
        meshes, motion, [np.zeros(3), np.zeros(3), np.zeros((6, 3)), np.zeros((3, 3))]
    )
    assert step is not None, report
    meshes["cover"].vertices += step[3]
    assert projection.gaps(meshes, motion).min() >= projection.config.minimum_clearance_m


def test_positive_gap_reserve_is_enforced_without_changing_physical_threshold():
    meshes, motion, projection = fixture()
    meshes["cover"].apply_translation([0, 0, 0.0009])
    projection.refit_supports(meshes, motion)
    threshold = projection.config.minimum_clearance_m
    proposed = [
        np.zeros(3),
        np.zeros(3),
        np.tile([0, 0, 0.001], (3, 1)),
        np.tile([0, 0, -0.001], (3, 1)),
    ]
    step, audit = projection.project(meshes, motion, proposed, quadratic=True, reserve_m=0.00005)
    assert step is not None, audit
    for mesh, delta in zip(meshes.values(), step[2:], strict=True):
        mesh.vertices += delta
    assert projection.gaps(meshes, motion).min() >= threshold + 0.00005 - 1e-9
    assert projection.config.minimum_clearance_m == threshold
    with pytest.raises(ValueError, match="reserve"):
        projection.project(meshes, motion, proposed, reserve_m=-0.001)


@pytest.mark.parametrize("status", ["AlmostSolved", "InsufficientProgress"])
def test_inexact_qp_is_only_a_proposal_and_reports_its_actual_residual(monkeypatch, status):
    import clarabel
    from scipy import sparse

    from physrealx.integrations._contact_projection import _quadratic_projection

    monkeypatch.setattr(
        clarabel,
        "DefaultSolver",
        lambda *a, **k: SimpleNamespace(
            solve=lambda: SimpleNamespace(x=np.zeros(1), status=status, iterations=3)
        ),
    )
    args = (
        sparse.eye(1, format="csc"),
        np.ones(1),
        np.zeros(1),
        -np.ones(1),
        np.ones(1),
        np.array([0.5]),
        np.array([np.inf]),
    )
    strict, strict_audit = _quadratic_projection(*args)
    proposed, audit = _quadratic_projection(*args, require_feasible=False)
    assert strict is None
    assert proposed is not None
    assert not audit["linear_constraints_satisfied"]
    assert (
        audit["maximum_linear_constraint_violation_m"]
        == strict_audit["maximum_linear_constraint_violation_m"]
    )
    assert audit["maximum_linear_constraint_violation_m"] == pytest.approx(0.0005)
