"""An articulated asset must be one surface, not independently fitted frames."""

import copy
import json

import numpy as np
import pytest
import trimesh

from physrealx.integrations._native_untangling import (
    decode_candidate,
    pose_batch,
    read_obj,
    write_case,
)


@pytest.fixture
def source():
    meshes = {
        "body": trimesh.creation.box(extents=[0.3, 0.2, 0.005]),
        "cover": trimesh.creation.box(extents=[0.3, 0.005, 0.2]),
    }
    meshes["cover"].apply_translation([0, 0, 0.1])
    motion = {
        "parent": "body",
        "child": "cover",
        "kind": "hinge",
        "axis": [1.0, 0.0, 0.0],
        "origin_m": [0.0, 0.0, 0.0],
        "range": [-0.2, 0.8],
    }
    return meshes, motion


def test_shared_deformation_roundtrip_preserves_source_and_motion(source):
    meshes, motion = source
    original = {n: m.vertices.copy() for n, m in meshes.items()}
    original_motion = copy.deepcopy(motion)
    batch = pose_batch(meshes, motion, poses=9, hinge_band_m=0.15)
    delta = np.random.default_rng(0).normal(scale=0.0001, size=batch.source_vertices.shape)
    delta[batch.pins] = 0
    result = batch.vertices.copy().reshape(9, -1, 3)
    for i, rotation in enumerate(batch.rotations):
        step = delta.copy()
        step[batch.moving_begin :] = step[batch.moving_begin :] @ rotation.T
        result[i] += step * batch.scale
    decoded = batch.decode(result.reshape(-1, 3), batch.faces)
    np.testing.assert_allclose(decoded, batch.source_vertices + delta, atol=1e-14)
    for n in meshes:
        np.testing.assert_array_equal(meshes[n].vertices, original[n])
    assert motion == original_motion
    assert list(batch.positions[[0, -1]]) == motion["range"]


def test_independent_pose_edit_is_rejected(source):
    batch = pose_batch(*source, poses=3, hinge_band_m=0.15)
    changed = batch.vertices.copy()
    changed[0, 0] += 0.001 * batch.scale
    with pytest.raises(ValueError, match="canonical deformation"):
        batch.decode(changed, batch.faces)


def test_consistent_protected_vertex_edit_is_rejected(source):
    batch = pose_batch(*source, poses=3, hinge_band_m=0.05)
    changed = batch.vertices.copy()
    for p in range(3):
        changed[p * len(batch.source_vertices) + batch.pins[0], 0] += 0.001 * batch.scale
    with pytest.raises(ValueError, match="protected vertices"):
        batch.decode(changed, batch.faces)


def test_missing_or_reordered_triangles_are_rejected(source):
    batch = pose_batch(*source, poses=3, hinge_band_m=0.05)
    for faces in [batch.faces[:-1], batch.faces[::-1]]:
        with pytest.raises(ValueError, match="topology"):
            batch.decode(batch.vertices, faces)


@pytest.mark.parametrize("change", ["axis", "range", "origin_m"])
def test_invalid_joint_rejected_before_launch(source, change):
    meshes, motion = source
    motion[change][0] = float("nan")
    with pytest.raises(ValueError, match="hinge parameters"):
        pose_batch(meshes, motion, poses=3, hinge_band_m=0.05)


def test_isolated_case_and_color_preserving_decode(source, tmp_path):
    meshes, motion = source
    batch = pose_batch(meshes, motion, poses=3, hinge_band_m=0.15)
    config = write_case(batch, tmp_path / "case", iterations=5)
    document = json.loads(config.read_text())
    assert document["solverRunSpec"]["maxIterationCount"] == 5
    vertices, faces = read_obj(config.parent / "input.obj")
    np.testing.assert_allclose(vertices, batch.vertices, atol=1e-15)
    np.testing.assert_array_equal(faces, batch.faces)
    decoded = decode_candidate(batch, meshes, config.parent / "input.obj")
    for n in meshes:
        np.testing.assert_allclose(decoded[n].vertices, meshes[n].vertices, atol=1e-15)
        np.testing.assert_array_equal(
            decoded[n].visual.vertex_colors, meshes[n].visual.vertex_colors
        )
    with pytest.raises(FileExistsError):
        write_case(batch, config.parent)


def test_native_polygon_output_is_not_silently_triangulated(tmp_path):
    path = tmp_path / "invalid.obj"
    path.write_text("v 0 0 0\nf 1 2 3 4\n")
    with pytest.raises(ValueError, match="non-triangular"):
        read_obj(path)


def test_candidate_cannot_self_declare_valid_or_change_hinge(source, tmp_path):
    from physrealx.integrations._native_untangling import load_unfolded_candidate
    from physrealx.integrations.video_articulation import VideoArticulationRefinementConfig
    from physrealx.storage import LocalArtifactStore

    meshes, motion = source
    store = LocalArtifactStore(tmp_path / "artifacts")
    reference = store.put_json({"source": "immutable"})
    candidate = {}
    for name, mesh in meshes.items():
        changed = mesh.copy()
        changed.vertices += [0, 0, 0.02]
        candidate[name] = store.put_bytes(
            changed.export(file_type="glb"), media_type="model/gltf-binary"
        ).model_dump(mode="json")
    document = {
        "schema": "physrealx.articulated-unfolding/1",
        "source_geometry": reference.model_dump(mode="json"),
        "geometries": candidate,
        "motion": copy.deepcopy(motion),
        "provenance": {"geometry_valid": True, "status": "PASSED"},
    }
    _, _, audit = load_unfolded_candidate(
        store,
        store.put_json(document),
        reference,
        meshes,
        motion,
        VideoArticulationRefinementConfig(),
    )
    assert audit["geometry_valid"] is False
    assert audit["links"]["body"]["protected_vertex_displacement_m"] > 0.019
    document["motion"]["range"][1] *= 0.5
    with pytest.raises(ValueError, match="articulation"):
        load_unfolded_candidate(
            store,
            store.put_json(document),
            reference,
            meshes,
            motion,
            VideoArticulationRefinementConfig(),
        )
