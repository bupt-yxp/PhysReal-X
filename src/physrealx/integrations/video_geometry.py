"""Bounded, observation-supported surface corrections in fixed articulated coordinates."""

from __future__ import annotations

import io
from importlib import import_module
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_array, read_json
from physrealx.integrations.collision import load_meshes
from physrealx.reconstruction import joint_transform


class VideoGeometryFrame(DomainModel):
    frame_id: str = Field(min_length=1)
    partition: Literal["fit", "check"]
    depth: ArtifactRef
    # Integer PNG: zero is background/hand/unknown, positive IDs map exactly to links.
    link_mask: ArtifactRef
    intrinsics: tuple[float, float, float, float]  # fx, fy, cx, cy, aligned depth pixels
    camera_from_reference: tuple[tuple[float, ...], ...]  # OpenCV camera, 4x4 SE(3)
    positions: dict[str, float]


class VideoGeometryEvidence(DomainModel):
    reference: ArtifactRef
    registration: ArtifactRef
    source_geometry: ArtifactRef
    capture_id: str = Field(min_length=1)
    depth_scale_m: float = Field(gt=0, allow_inf_nan=False)
    link_ids: dict[str, int]
    frames: tuple[VideoGeometryFrame, ...] = Field(min_length=4)


class GeometryRefinementConfig(DomainModel):
    max_displacement_m: float = Field(default=0.003, gt=0, le=0.01)
    correspondence_distance_m: float = Field(default=0.02, gt=0, le=0.1)
    points_per_frame: int = Field(default=2000, ge=100, le=10000)
    smoothness: float = Field(default=0.2, gt=0)
    min_improvement_m: float = Field(default=0.0001, gt=0)


def _observations(
    store: ArtifactStore,
    evidence: VideoGeometryEvidence,
    model: dict[str, Any],
    motion: dict[str, Any],
    config: GeometryRefinementConfig,
) -> dict[str, dict[str, list[Any]]]:
    import numpy as np
    import trimesh
    from PIL import Image

    names = {link["name"] for link in model["links"]}
    if (
        set(evidence.link_ids) != names
        or len(set(evidence.link_ids.values())) != len(names)
        or any(value <= 0 for value in evidence.link_ids.values())
    ):
        raise DomainValidationError("positive unique mask IDs must cover every link")
    if len({f.frame_id for f in evidence.frames}) != len(evidence.frames):
        raise DomainValidationError("video frame IDs must be unique")
    fit_depth = {f.depth.artifact_id for f in evidence.frames if f.partition == "fit"}
    check_depth = {f.depth.artifact_id for f in evidence.frames if f.partition == "check"}
    if fit_depth & check_depth:
        raise DomainValidationError("geometry fit/check must use different frames")
    result: dict[str, dict[str, list[Any]]] = {n: {"fit": [], "check": []} for n in names}
    rng = np.random.default_rng(2026)
    for frame in evidence.frames:
        depth = read_array(store, frame.depth).astype(float) * evidence.depth_scale_m
        mask = np.asarray(Image.open(io.BytesIO(store.get_bytes(frame.link_mask))))
        matrix = np.asarray(frame.camera_from_reference, dtype=float)
        fx, fy, cx, cy = frame.intrinsics
        if (
            depth.ndim != 2
            or mask.shape != depth.shape
            or mask.dtype.kind not in "ui"
            or matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)
            or not np.isfinite(frame.intrinsics).all()
            or min(fx, fy) <= 0
            or not set(np.unique(mask)) <= {0, *evidence.link_ids.values()}
        ):
            raise DomainValidationError(
                "video requires calibrated aligned depth and integer link masks"
            )
        q = frame.positions.get(motion["joint"], float("nan"))
        if not motion["range"][0] <= q <= motion["range"][1]:
            raise DomainValidationError("video position exceeds the observed start/end interval")
        if set(frame.positions) != {motion["joint"]}:
            raise DomainValidationError("video positions must cover the reconstructed joint")
        transforms = {
            motion["parent"]: np.eye(4),
            motion["child"]: joint_transform(
                motion["kind"], motion["axis"], motion["origin_m"], q
            ),
        }
        for name, identifier in evidence.link_ids.items():
            y, x = np.where((mask == identifier) & (depth > 0))
            if len(x) < 100:
                continue
            indices = rng.choice(len(x), min(len(x), config.points_per_frame), replace=False)
            x, y = x[indices], y[indices]
            z = depth[y, x]
            camera_points = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
            points = trimesh.transform_points(
                camera_points, np.linalg.inv(matrix @ transforms[name])
            )
            result[name][frame.partition].append(points)
    return result


def fit_surface(
    mesh: Any, fit: Any, check: Any, config: GeometryRefinementConfig
) -> tuple[Any, dict[str, Any]]:
    """Move vertices only along their normals, bounded and regularized; gate on held-out views."""
    import numpy as np
    from scipy import sparse
    from scipy.optimize import lsq_linear

    trimesh = import_module("trimesh")

    closest, distances, triangles = trimesh.proximity.closest_point(mesh, fit)
    keep = distances < config.correspondence_distance_m
    if np.count_nonzero(keep) < 100:
        return mesh, {"status": "INSUFFICIENT_SURFACE_SUPPORT"}
    closest, points, triangles = closest[keep], fit[keep], triangles[keep]
    faces = mesh.faces[triangles]
    normals = mesh.face_normals[triangles]
    # GLB duplicates vertices at UV/normal seams. Share displacement variables without
    # welding the visual mesh, which would otherwise lose textures or open cracks.
    unique, inverse = np.unique(np.round(mesh.vertices, 9), axis=0, return_inverse=True)
    vertex_normals = np.zeros_like(unique)
    np.add.at(vertex_normals, inverse, mesh.vertex_normals)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-12)
    barycentric = trimesh.triangles.points_to_barycentric(mesh.triangles[triangles], closest)
    mapped_faces = inverse[faces]
    weights = barycentric * np.sum(vertex_normals[mapped_faces] * normals[:, None, :], axis=2)
    rows = np.repeat(np.arange(len(points)), 3)
    matrix = sparse.coo_matrix(
        (weights.ravel(), (rows, mapped_faces.ravel())), shape=(len(points), len(unique))
    ).tocsr()
    edges = np.unique(np.sort(inverse[mesh.edges_unique], axis=1), axis=0)
    smooth = sparse.coo_matrix(
        (np.tile([1.0, -1.0], len(edges)), (np.repeat(np.arange(len(edges)), 2), edges.ravel())),
        shape=(len(edges), len(unique)),
    ).tocsr()
    system = sparse.vstack(
        [matrix, config.smoothness * smooth, 0.1 * sparse.eye(len(unique))]
    ).tocsr()
    target = np.r_[np.sum((points - closest) * normals, axis=1), np.zeros(len(edges) + len(unique))]
    solution = lsq_linear(
        system,
        target,
        bounds=(-config.max_displacement_m, config.max_displacement_m),
        lsmr_tol="auto",
    )
    candidate = mesh.copy()
    candidate.vertices = mesh.vertices + vertex_normals[inverse] * solution.x[inverse, None]
    before = trimesh.proximity.closest_point(mesh, check)[1]
    after = trimesh.proximity.closest_point(candidate, check)[1]
    before_p90, after_p90 = float(np.quantile(before, 0.9)), float(np.quantile(after, 0.9))
    # Retain topology/materials and disallow thin-shell collapse or inverted faces.
    extent_ratio = np.ptp(candidate.vertices, axis=0) / np.maximum(
        np.ptp(mesh.vertices, axis=0), 1e-9
    )
    valid = (
        solution.success
        and np.all(extent_ratio >= 0.8)
        and np.all(extent_ratio <= 1.2)
        and np.all(candidate.area_faces > 1e-12)
        and np.all(np.sum(candidate.face_normals * mesh.face_normals, axis=1) > 0)
        and after_p90 <= before_p90 - config.min_improvement_m
        and float(np.median(after)) <= float(np.median(before))
    )
    return (candidate if valid else mesh), {
        "status": "ACCEPTED" if valid else "REJECTED",
        "check_p90_before_m": before_p90,
        "check_p90_candidate_m": after_p90,
        "max_vertex_displacement_m": float(abs(solution.x).max()),
        "geometry_changed": bool(valid),
        "independent_capture_validation": False,
    }


def refine_video_geometry(
    store: ArtifactStore,
    evidence_ref: ArtifactRef,
    *,
    config: GeometryRefinementConfig,
) -> ArtifactRef:
    """Return corrected geometry while preserving joint axis, origins and endpoints."""
    import numpy as np

    evidence = VideoGeometryEvidence.model_validate_json(store.get_bytes(evidence_ref))
    model = read_json(store, evidence.reference, "physrealx.usd-reference/1")
    registration = read_json(store, evidence.registration, "physrealx.metric-alignment-report/1")
    source_geometry = read_json(
        store,
        evidence.source_geometry,
        "physrealx.registered-reconstruction-geometry/4",
    )
    if registration["source_reference"] != evidence.reference.model_dump(mode="json"):
        raise DomainValidationError("video registration differs from source reference")
    if (
        source_geometry.get("source_reference") != evidence.reference.model_dump(mode="json")
        or source_geometry.get("source_registration")
        != evidence.registration.model_dump(mode="json")
        or set(source_geometry.get("geometries", {})) != {link["name"] for link in model["links"]}
    ):
        raise DomainValidationError("video source geometry differs from reconstruction")
    motion = source_geometry["reconstructed_motion"]
    observations = _observations(store, evidence, model, motion, config)
    geometries, reports = {}, {}
    for link in model["links"]:
        name = link["name"]
        source = ArtifactRef.model_validate(source_geometry["geometries"][name])
        meshes = load_meshes(store, source, process=False)
        samples = observations[name]
        geometries[name] = source.model_dump(mode="json")
        if min(len(samples["fit"]), len(samples["check"])) < 2:
            reports[name] = {"status": "INSUFFICIENT_VIEWS"}
            continue
        if len(meshes) != 1:
            reports[name] = {"status": "MULTI_MESH_REQUIRES_PART_ASSOCIATIONS"}
            continue
        mesh, report = fit_surface(
            meshes[0], np.concatenate(samples["fit"]), np.concatenate(samples["check"]), config
        )
        if report["status"] == "ACCEPTED":
            geometries[name] = store.put_bytes(
                mesh.export(file_type="glb"), media_type="model/gltf-binary"
            ).model_dump(mode="json")
        reports[name] = report
    return store.put_json(
        {
            "schema": "physrealx.video-geometry-refinement/3",
            "source_reference": evidence.reference.model_dump(mode="json"),
            "source_geometry": evidence.source_geometry.model_dump(mode="json"),
            "registration": evidence.registration.model_dump(mode="json"),
            "source_evidence": evidence_ref.model_dump(mode="json"),
            "config": config.model_dump(mode="json"),
            "reconstructed_motion": motion,
            "geometries": geometries,
            "links": reports,
            "dynamics_status": "UNCHANGED_UNKNOWN_EXTERNAL_FORCE",
            "physical_validation": False,
            "contact_status": "REQUIRES_COLLISION_SWEEP",
        }
    )
