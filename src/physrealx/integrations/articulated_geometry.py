"""Joint-local repair of articulated visual and collision geometry."""

from __future__ import annotations

import copy
from importlib import import_module
from typing import Any

from pydantic import Field

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel, GeometryRepresentation, GeometryRole
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.assembly import assembled_asset
from physrealx.integrations.asset_prior import ReferenceAssemblyInput, reference_prior_asset
from physrealx.integrations.collision import _exterior_samples, load_meshes, mesh_reference
from physrealx.integrations.physical_estimation import locked_motion
from physrealx.integrations.physical_feasibility import (
    FeasibilityConfig,
    evaluate_asset_feasibility,
)


class ArticulatedGeometryRefinementConfig(DomainModel):
    """Safety gates for removing a joint-local reconstruction ghost."""

    max_removed_volume_fraction: float = Field(default=0.2, gt=0, le=0.5)
    max_removed_surface_fraction: float = Field(default=0.25, gt=0, le=0.5)
    max_joint_band_fraction: float = Field(default=0.6, gt=0, le=0.8)
    max_surface_error_m: float = Field(default=0.008, gt=0, le=0.05)
    hull_membership_tolerance_m: float = Field(
        default=5e-5, ge=0, le=0.001, allow_inf_nan=False
    )
    clearance_m: float = Field(default=1e-6, ge=0, le=0.001, allow_inf_nan=False)
    seed: int = Field(default=1, ge=1, le=2_147_483_647)


def _radial_distance(points: object, axis: object, origin: object) -> Any:
    import numpy as np

    offsets = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    direction = np.asarray(axis, dtype=float)
    axial = offsets @ direction
    return np.linalg.norm(offsets - axial[:, None] * direction, axis=1)


def _repair_visual_mesh(
    visual: Any,
    offending_parts: list[Any],
    *,
    tolerance_m: float,
) -> tuple[Any, Any]:
    """Remove visual faces inside collision fragments proven invalid by the sweep."""
    import numpy as np
    from scipy.spatial import ConvexHull

    centroids = np.asarray(visual.triangles_center, dtype=float)
    remove = np.zeros(len(visual.faces), dtype=bool)
    for part in offending_parts:
        bounds = np.asarray(part.bounds, dtype=float)
        candidates = np.flatnonzero(
            (~remove)
            & np.all(centroids >= bounds[0] - tolerance_m, axis=1)
            & np.all(centroids <= bounds[1] + tolerance_m, axis=1)
        )
        if not len(candidates):
            continue
        equations = ConvexHull(part.vertices).equations
        remove[candidates] = np.all(
            centroids[candidates] @ equations[:, :3].T + equations[:, 3]
            <= tolerance_m,
            axis=1,
        )
    if not remove.any() or remove.all():
        raise MethodExecutionError(
            "failing collision fragments do not isolate a removable visual surface"
        )
    repaired = visual.copy()
    repaired.update_faces(~remove)
    repaired.remove_unreferenced_vertices()
    components = repaired.split(only_watertight=False)
    if (
        len(repaired.vertices) < 4
        or len(repaired.faces) < 4
        or len(components) != 1
        or not np.isfinite(repaired.vertices).all()
        or np.any(repaired.area_faces <= 0)
    ):
        raise MethodExecutionError("visual repair would fragment or invalidate the moving link")
    return repaired, remove


def _surface_error(
    visual: Any,
    parts: list[Any],
    *,
    seed: int,
) -> float:
    import numpy as np
    import trimesh

    proxy = trimesh.util.concatenate(parts)
    visual_samples, _ = trimesh.sample.sample_surface(visual, 5000, seed=seed)
    proxy_samples = _exterior_samples(parts, proxy, seed)
    visual_distance = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
        proxy, visual_samples
    )[1]
    proxy_distance = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
        visual, proxy_samples
    )[1]
    return max(
        float(np.quantile(distance, 0.95))
        for distance in (visual_distance, proxy_distance)
    )


def refine_articulated_geometry(
    store: ArtifactStore,
    assembly_ref: ArtifactRef,
    feasibility_ref: ArtifactRef,
    *,
    config: ArticulatedGeometryRefinementConfig,
) -> ArtifactRef:
    """Repair a moving-link reconstruction ghost and its matching collision parts.

    The failed full-range collision sweep identifies candidate fragments. The same
    bounded region is removed from the moving visual surface and collision proxy;
    joint endpoints and parent/child collision remain immutable.
    """
    import numpy as np

    trimesh = import_module("trimesh")
    assembly = read_json(store, assembly_ref, "physrealx.asset-assembly-bundle/1")
    feasibility = read_json(
        store, feasibility_ref, "physrealx.physical-feasibility-report/1"
    )
    if (
        feasibility.get("source_assembly") != assembly_ref.model_dump(mode="json")
        or feasibility.get("status") != "FAILED"
    ):
        raise DomainValidationError(
            "articulated geometry refinement requires its failed feasibility report"
        )
    asset = assembled_asset(assembly, store)
    if len(asset.joints) != 1:
        raise DomainValidationError(
            "articulated geometry refinement currently requires one joint"
        )
    joint = asset.joints[0]
    evaluation_config = FeasibilityConfig.model_validate(feasibility["config"])
    before = evaluate_asset_feasibility(store, asset, config=evaluation_config)
    if (
        before["status"] != "FAILED"
        or before["locked_range"] != feasibility.get("locked_range")
        or not np.isclose(before["max_penetration_m"], feasibility["max_penetration_m"])
    ):
        raise DomainValidationError("failed collision evidence is stale or non-reproducible")

    collision_geometries = [
        (index, geometry)
        for index, geometry in enumerate(asset.geometries)
        if geometry.role is GeometryRole.COLLISION and geometry.link == joint.child
    ]
    if (
        len(collision_geometries) != 1
        or collision_geometries[0][1].representation is not GeometryRepresentation.MESH
    ):
        raise DomainValidationError("moving link requires one explicit convex collision mesh")
    geometry_index, child_collision = collision_geometries[0]
    prefix = f"asset/geometry/{geometry_index}/"
    offending: set[int] = set()
    for pair in before["collision_pairs"]:
        if pair["max_penetration_m"] <= config.clearance_m:
            continue
        for side in ("a", "b"):
            name = pair[f"geom_{side}"]
            if pair[f"link_{side}"] == joint.child and name.startswith(prefix):
                try:
                    offending.add(int(name.removeprefix(prefix)))
                except ValueError as error:
                    raise DomainValidationError("collision part name is not canonical") from error
    if not offending:
        raise MethodExecutionError(
            "failure is not caused by removable moving-link collision parts"
        )

    source_input = ArtifactRef.model_validate(assembly["source_input"])
    request = ReferenceAssemblyInput.model_validate(
        read_json(store, source_input, "physrealx.reference-assembly-input/1")
    )
    if request.articulated_geometry_refinement is not None:
        raise DomainValidationError("assembly already contains an articulated geometry repair")
    reports: dict[str, tuple[ArtifactRef, dict[str, Any]]] = {}
    for ref in request.collision_reports:
        document = read_json(store, ref, "physrealx.collision-build-report/1")
        reports[str(document["link"])] = (ref, document)
    if set(reports) != set(asset.links) or joint.child not in reports:
        raise DomainValidationError("collision reports do not cover the articulated asset")
    child_report_ref, child_report = reports[joint.child]
    collision_ref = ArtifactRef.model_validate(child_report["collision_geometry"])
    if collision_ref.artifact_id != child_collision.uri.removeprefix("artifact://"):
        raise DomainValidationError("moving collision geometry differs from its build report")
    parts = load_meshes(store, collision_ref, process=False)
    if max(offending) >= len(parts) or len(offending) >= len(parts):
        raise MethodExecutionError("geometry repair would remove the moving collision body")
    volumes = np.asarray([abs(float(part.volume)) for part in parts])
    if np.any(volumes <= 0) or not np.isfinite(volumes).all():
        raise DomainValidationError("geometry repair requires finite convex volumes")
    removed_volume_fraction = float(volumes[list(offending)].sum() / volumes.sum())
    if removed_volume_fraction > config.max_removed_volume_fraction:
        raise MethodExecutionError(
            "collision failure is not local: removed volume exceeds the configured gate"
        )

    reference = read_json(store, request.reference_model, "physrealx.usd-reference/1")
    authored_joint = next(
        (item for item in reference["joints"] if item["name"] == joint.name), None
    )
    if authored_joint is None:
        raise DomainValidationError("assembly joint is absent from the registered reference")
    vertices = np.concatenate([part.vertices for part in parts])
    offending_parts = [parts[index] for index in sorted(offending)]
    removed_vertices = np.concatenate([part.vertices for part in offending_parts])
    full_radius = float(
        _radial_distance(vertices, authored_joint["axis"], authored_joint["origin_m"]).max()
    )
    removed_radius = float(
        _radial_distance(
            removed_vertices, authored_joint["axis"], authored_joint["origin_m"]
        ).max()
    )
    joint_band_fraction = removed_radius / max(full_radius, 1e-12)
    if joint_band_fraction > config.max_joint_band_fraction:
        raise MethodExecutionError(
            "collision failure extends beyond the configured joint-local repair band"
        )

    child_visual = next(
        (
            geometry
            for geometry in asset.geometries
            if geometry.role is GeometryRole.VISUAL and geometry.link == joint.child
        ),
        None,
    )
    if child_visual is None:
        raise DomainValidationError("moving link has no visual geometry")
    child_visual_ref = mesh_reference(child_visual.uri, store)
    visual_meshes = load_meshes(store, child_visual_ref, process=False)
    if len(visual_meshes) != 1:
        raise DomainValidationError(
            "moving visual link must be one registered mesh before local repair"
        )
    visual = visual_meshes[0]
    original_area = float(visual.area)
    repaired_visual, removed_faces = _repair_visual_mesh(
        visual,
        offending_parts,
        tolerance_m=config.hull_membership_tolerance_m,
    )
    removed_surface_fraction = float(
        visual.area_faces[removed_faces].sum() / max(original_area, 1e-12)
    )
    if removed_surface_fraction > config.max_removed_surface_fraction:
        raise MethodExecutionError(
            "visual failure is not local: removed surface exceeds the configured gate"
        )
    refined_visual_ref = store.put_bytes(
        repaired_visual.export(file_type="glb"), media_type="model/gltf-binary"
    )
    decoded_visual = load_meshes(store, refined_visual_ref, process=False)
    if len(decoded_visual) != 1 or len(decoded_visual[0].faces) != len(repaired_visual.faces):
        raise MethodExecutionError("repaired visual geometry failed GLB roundtrip validation")

    base_geometries = {
        geometry.link: mesh_reference(geometry.uri, store).model_dump(mode="json")
        for geometry in asset.geometries
        if geometry.role is GeometryRole.VISUAL
    }
    if set(base_geometries) != set(asset.links):
        raise DomainValidationError("visual geometry must cover exactly the articulated asset")
    base_geometries[joint.child] = refined_visual_ref.model_dump(mode="json")
    if request.registration is None:
        raise DomainValidationError("articulated geometry repair requires registration")
    registration = read_json(
        store, request.registration, "physrealx.metric-alignment-report/1"
    )
    geometry_refinement_ref = store.put_json(
        {
            "schema": "physrealx.articulated-geometry-refinement/1",
            "status": "PASSED",
            "source_reference": request.reference_model.model_dump(mode="json"),
            "source_geometry": request.reconstructed_geometry.model_dump(mode="json")
            if request.reconstructed_geometry is not None
            else None,
            "upstream_geometry_refinement": request.geometry_refinement.model_dump(mode="json")
            if request.geometry_refinement is not None
            else None,
            "registration": request.registration.model_dump(mode="json"),
            "motion": locked_motion(reference, registration),
            "geometries": base_geometries,
            "moving_link": joint.child,
            "removed_face_count": int(removed_faces.sum()),
            "removed_surface_fraction": removed_surface_fraction,
            "joint": joint.name,
            "locked_range": before["locked_range"],
            "visual_geometry_changed": True,
            "joint_model_changed": False,
        }
    )

    kept = [part for index, part in enumerate(parts) if index not in offending]
    surface_error_m = _surface_error(decoded_visual[0], kept, seed=config.seed)
    if not np.isfinite(surface_error_m) or surface_error_m > config.max_surface_error_m:
        raise MethodExecutionError(
            f"repaired collision surface P95 {surface_error_m:.6g} m exceeds "
            f"tolerance {config.max_surface_error_m:.6g} m"
        )
    collision_scene = trimesh.Scene()
    for index, part in enumerate(kept):
        collision_scene.add_geometry(
            part, node_name=f"{joint.child}_{index}", geom_name=f"convex_{index}"
        )
    refined_collision_ref = store.put_bytes(
        collision_scene.export(file_type="glb"), media_type="model/gltf-binary"
    )
    decoded_collision = load_meshes(store, refined_collision_ref)
    if len(decoded_collision) != len(kept) or any(
        not part.is_volume or not part.is_convex for part in decoded_collision
    ):
        raise MethodExecutionError("repaired collision geometry failed GLB validation")

    refined_report = copy.deepcopy(child_report)
    refined_report["source_geometry"] = refined_visual_ref.model_dump(mode="json")
    refined_report["collision_geometry"] = refined_collision_ref.model_dump(mode="json")
    refined_report["metrics"] = {
        **dict(refined_report.get("metrics", {})),
        "convex_parts": float(len(kept)),
        "removed_parts": float(len(offending)),
        "removed_volume_fraction": removed_volume_fraction,
        "removed_visual_surface_fraction": removed_surface_fraction,
        "joint_band_fraction": joint_band_fraction,
        "bidirectional_surface_p95_m": surface_error_m,
    }
    refined_report["gates"] = {
        **dict(refined_report.get("gates", {})),
        "joint_locality": True,
        "removed_volume": True,
        "removed_visual_surface": True,
        "visual_collision_surface_fidelity": True,
        "parent_child_collision_preserved": True,
    }
    refined_report["articulation_refinement"] = {
        "schema": "physrealx.articulated-geometry-clearance/1",
        "source_collision_report": child_report_ref.model_dump(mode="json"),
        "source_assembly": assembly_ref.model_dump(mode="json"),
        "source_feasibility": feasibility_ref.model_dump(mode="json"),
        "geometry_refinement": geometry_refinement_ref.model_dump(mode="json"),
        "source_geometry_before": child_report["source_geometry"],
        "source_geometry_after": refined_visual_ref.model_dump(mode="json"),
        "joint": joint.name,
        "locked_range": before["locked_range"],
        "removed_part_indices": sorted(offending),
        "policy": "remove_sweep_failing_visual_faces_and_matching_collision_parts",
        "visual_geometry_changed": True,
        "joint_model_changed": False,
        "parent_child_collision_disabled": False,
    }
    refined_report_ref = store.put_json(refined_report)
    replacement_reports = tuple(
        refined_report_ref if ref == child_report_ref else ref
        for ref in request.collision_reports
    )
    candidate_request = request.model_copy(
        update={
            "articulated_geometry_refinement": geometry_refinement_ref,
            "collision_reports": replacement_reports,
        }
    )
    candidate_input = store.put_json(candidate_request.model_dump(mode="json", by_alias=True))
    candidate_asset = reference_prior_asset(candidate_input, store)
    after = evaluate_asset_feasibility(store, candidate_asset, config=evaluation_config)
    before_visual = {
        geometry.link: geometry.uri
        for geometry in asset.geometries
        if geometry.role is GeometryRole.VISUAL
    }
    after_visual = {
        geometry.link: geometry.uri
        for geometry in candidate_asset.geometries
        if geometry.role is GeometryRole.VISUAL
    }
    unchanged_links = set(before_visual) - {joint.child}
    if (
        any(before_visual[link] != after_visual[link] for link in unchanged_links)
        or before_visual[joint.child] == after_visual[joint.child]
        or before["locked_range"] != after["locked_range"]
    ):
        raise MethodExecutionError("geometry repair violated link or joint invariants")
    refinement_passed = (
        after["status"] == "PASSED"
        and after["max_penetration_m"] <= config.clearance_m
    )
    return store.put_json(
        {
            "schema": "physrealx.articulated-geometry-repair/1",
            "status": "PASSED" if refinement_passed else "FAILED",
            "source_assembly": assembly_ref.model_dump(mode="json"),
            "source_feasibility": feasibility_ref.model_dump(mode="json"),
            "candidate_assembly_input": candidate_input.model_dump(mode="json"),
            "geometry_refinement": geometry_refinement_ref.model_dump(mode="json"),
            "refined_visual_geometry": refined_visual_ref.model_dump(mode="json"),
            "collision_reports": [
                ref.model_dump(mode="json") for ref in replacement_reports
            ],
            "joint": joint.name,
            "locked_range": after["locked_range"],
            "before": {
                "max_penetration_m": before["max_penetration_m"],
                "failing_pairs": sum(
                    pair["max_penetration_m"]
                    > evaluation_config.penetration_tolerance_m
                    for pair in before["collision_pairs"]
                ),
                "penetrating_pairs": sum(
                    pair["max_penetration_m"] > config.clearance_m
                    for pair in before["collision_pairs"]
                ),
            },
            "after": {
                "max_penetration_m": after["max_penetration_m"],
                "failing_pairs": sum(
                    pair["max_penetration_m"]
                    > evaluation_config.penetration_tolerance_m
                    for pair in after["collision_pairs"]
                ),
                "penetrating_pairs": sum(
                    pair["max_penetration_m"] > config.clearance_m
                    for pair in after["collision_pairs"]
                ),
            },
            "moving_link": joint.child,
            "removed_part_indices": sorted(offending),
            "removed_face_count": int(removed_faces.sum()),
            "removed_volume_fraction": removed_volume_fraction,
            "removed_surface_fraction": removed_surface_fraction,
            "joint_band_fraction": joint_band_fraction,
            "surface_error_p95_m": surface_error_m,
            "visual_geometry_refined": True,
            "joint_model_unchanged": True,
            "parent_child_collision_enabled": True,
            "config": config.model_dump(mode="json"),
        }
    )
