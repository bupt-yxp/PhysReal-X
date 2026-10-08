"""Proxy bounds must preserve surfaces, including boundaries and missing regions."""

import json

import numpy as np
import pytest

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise

from physrealx import PhysRealX
from physrealx.errors import DomainValidationError
from physrealx.integrations._collision_shell import (
    approximation_bound,
    certified_surface_budget,
    clearance_error_budget,
    rebuild_collision_shells,
)
from physrealx.integrations._surface_query import CpuSurfaceQuery


def triangle():
    return trimesh.Trimesh([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], [[0, 1, 2]], process=False)


def test_proxy_error_budget_cannot_relax_uncertain_or_small_clearance():
    result = {
        "status": "PASSED",
        "config": {"minimum_clearance_m": 0.0002},
        "swept_surface_clearance": {
            "continuous_surface_clearance_certified": True,
            "minimum_clearance_lower_bound_m": 0.001,
        },
    }
    assert clearance_error_budget(result, 0.0004) < clearance_error_budget(result)
    for gap in (0.0002, -0.001, float("inf"), float("nan")):
        result["swept_surface_clearance"]["minimum_clearance_lower_bound_m"] = gap
        with pytest.raises(DomainValidationError, match="insufficient certified clearance"):
            clearance_error_budget(result)


def test_error_bound_includes_triangle_interiors_and_preserves_open_surface(tmp_path):
    source = triangle()
    original = source.vertices.copy()
    proxy = source.copy()
    proxy.apply_translation([0, 0, 0.0001])
    query = CpuSurfaceQuery()
    check = approximation_bound(proxy, source, query, 0.0005, max_cells=100)
    assert check["status"] == "CERTIFIED"
    assert 0.0001 <= check["upper_bound_m"] <= 0.0005
    # A small shell over a real hole has a finite nonzero error. Vertex-only
    # checking of a coarse hull must not publish that hull as exact coverage.
    missing = trimesh.Trimesh(
        [[0, 0, 0], [0.04, 0, 0], [0, 0.04, 0], [0.1, 0, 0], [0.06, 0, 0], [0.1, 0.04, 0]],
        [[0, 1, 2], [3, 5, 4]],
        process=False,
    )
    assert approximation_bound(source, missing, query, 0.0005)["status"] == "VIOLATION"
    np.testing.assert_array_equal(original, source.vertices)


def test_shell_rebuild_retains_every_face_and_rejects_failed_source(tmp_path):
    px = PhysRealX.open(tmp_path)
    source = triangle().subdivide()
    ref = px.artifacts.put_bytes(source.export(file_type="glb"), media_type="model/gltf-binary")
    result = {
        "schema": "physrealx.video-articulation-refinement/2",
        "status": "PASSED",
        "config": {"minimum_clearance_m": 0.0002},
        "swept_surface_clearance": {
            "minimum_clearance_lower_bound_m": 0.004,
            "continuous_surface_clearance_certified": True,
        },
        "geometries": {"body": ref.model_dump(mode="json")},
    }
    reports = rebuild_collision_shells(px.artifacts, px.artifacts.put_json(result))
    report = json.loads(px.artifacts.get_bytes(reports[0]))
    assert report["preprocessing"]["source_faces_retained"] == len(source.faces)
    assert report["preprocessing"]["source_faces_removed"] == 0
    assert report["metrics"]["source_triangle_ownership_coverage"] == 1
    assert report["gates"]["roundtrip"]
    result["status"] = "FAILED"
    with pytest.raises(DomainValidationError, match="surface separation must pass"):
        rebuild_collision_shells(px.artifacts, px.artifacts.put_json(result))


def test_proxy_budget_tightens_the_surface_proof_without_changing_geometry(tmp_path):
    px = PhysRealX.open(tmp_path)
    body = triangle().subdivide()
    body.vertices /= 1000
    body_ref = px.artifacts.put_bytes(body.export(file_type="glb"), media_type="model/gltf-binary")
    original = px.artifacts.get_bytes(body_ref)
    result = {
        "status": "PASSED",
        "config": {"minimum_clearance_m": 0.0002},
        "swept_surface_clearance": {
            "continuous_surface_clearance_certified": True,
            "minimum_clearance_lower_bound_m": 0.000225,
        },
        "optimized_motion": {
            "parent": "body",
            "child": "cover",
            "axis": [0, 0, 1],
            "origin_m": [0, 0, 0],
            "range": [0, 0.01],
        },
    }
    for gap in (0.0007, 0.0003):
        cover = body.copy()
        cover.apply_translation([0, 0, gap])
        cover_ref = px.artifacts.put_bytes(
            cover.export(file_type="glb"), media_type="model/gltf-binary"
        )
        result["geometries"] = {
            "body": body_ref.model_dump(mode="json"),
            "cover": cover_ref.model_dump(mode="json"),
        }
        if gap > 0.0005:
            budget, proof = certified_surface_budget(px.artifacts, result)
            assert budget >= 8e-5 and proof["status"] == "PASSED"
            assert proof["minimum_clearance_lower_bound_m"] > 0.00052
        else:
            with pytest.raises(DomainValidationError):
                certified_surface_budget(px.artifacts, result)
    assert px.artifacts.get_bytes(body_ref) == original
    assert result["swept_surface_clearance"]["minimum_clearance_lower_bound_m"] == 0.000225
