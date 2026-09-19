"""Joint-local collision refinement driven by a full articulated sweep."""

from __future__ import annotations

import copy
from importlib import import_module
from typing import Any

from pydantic import Field

from physrealx.contracts import ArtifactStore
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    GeometryRepresentation,
    GeometryRole,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.assembly import assembled_asset
from physrealx.integrations.asset_prior import ReferenceAssemblyInput, reference_prior_asset
from physrealx.integrations.collision import load_meshes
from physrealx.integrations.physical_feasibility import (
    FeasibilityConfig,
    evaluate_asset_feasibility,
)


class CollisionRefinementConfig(DomainModel):
    """Safety gates for removing reconstruction-only collision fragments."""

    max_removed_volume_fraction: float = Field(default=0.2, gt=0, le=0.5)
    max_joint_band_fraction: float = Field(default=0.6, gt=0, le=0.8)
    clearance_m: float = Field(default=1e-6, ge=0, le=0.001, allow_inf_nan=False)


def _radial_distance(points: object, axis: object, origin: object) -> Any:
    import numpy as np

    offsets = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    direction = np.asarray(axis, dtype=float)
    axial = offsets @ direction
    return np.linalg.norm(offsets - axial[:, None] * direction, axis=1)


def optimize_articulated_collision(
    store: ArtifactStore,
    assembly_ref: ArtifactRef,
    feasibility_ref: ArtifactRef,
    *,
    config: CollisionRefinementConfig,
) -> ArtifactRef:
    """Remove only failing child-link convex parts inside the local joint band.

    Visual geometry and kinematics remain immutable. Parent/child collision stays
    enabled, so contacts outside the uncertain reconstructed hinge region remain active.
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
        raise DomainValidationError("collision refinement requires its failed feasibility report")
    asset = assembled_asset(assembly, store)
    if len(asset.joints) != 1:
        raise DomainValidationError("collision refinement currently requires one articulated joint")
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
    geometry_index, child_geometry = collision_geometries[0]
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
        raise MethodExecutionError("failure is not caused by removable child-link collision parts")

    source_input = ArtifactRef.model_validate(assembly["source_input"])
    request = ReferenceAssemblyInput.model_validate(
        read_json(store, source_input, "physrealx.reference-assembly-input/1")
    )
    reports: dict[str, tuple[ArtifactRef, dict[str, Any]]] = {}
    for ref in request.collision_reports:
        document = read_json(store, ref, "physrealx.collision-build-report/1")
        reports[str(document["link"])] = (ref, document)
    if set(reports) != set(asset.links) or joint.child not in reports:
        raise DomainValidationError("assembly collision reports do not cover the articulated asset")
    child_report_ref, child_report = reports[joint.child]
    collision_ref = ArtifactRef.model_validate(child_report["collision_geometry"])
    if collision_ref.artifact_id != child_geometry.uri.removeprefix("artifact://"):
        raise DomainValidationError("moving collision geometry differs from its build report")
    parts = load_meshes(store, collision_ref, process=False)
    if max(offending) >= len(parts) or len(offending) >= len(parts):
        raise MethodExecutionError("collision refinement would remove the moving collision body")
    volumes = np.asarray([abs(float(part.volume)) for part in parts])
    if np.any(volumes <= 0) or not np.isfinite(volumes).all():
        raise DomainValidationError("collision refinement requires finite convex volumes")
    removed_volume_fraction = float(volumes[list(offending)].sum() / volumes.sum())
    if removed_volume_fraction > config.max_removed_volume_fraction:
        raise MethodExecutionError(
            "collision failure is not local: required removed volume exceeds the configured gate"
        )

    reference = read_json(store, request.reference_model, "physrealx.usd-reference/1")
    authored_joint = next(
        (item for item in reference["joints"] if item["name"] == joint.name), None
    )
    if authored_joint is None:
        raise DomainValidationError("assembly joint is absent from the registered reference")
    vertices = np.concatenate([part.vertices for part in parts])
    removed_vertices = np.concatenate([parts[index].vertices for index in sorted(offending)])
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
            "collision failure extends beyond the configured joint-local refinement band"
        )

    kept = [part for index, part in enumerate(parts) if index not in offending]
    scene = trimesh.Scene()
    for index, part in enumerate(kept):
        scene.add_geometry(part, node_name=f"{joint.child}_{index}", geom_name=f"convex_{index}")
    refined_geometry = store.put_bytes(
        scene.export(file_type="glb"), media_type="model/gltf-binary"
    )
    decoded = load_meshes(store, refined_geometry)
    if len(decoded) != len(kept) or any(
        not part.is_volume or not part.is_convex for part in decoded
    ):
        raise MethodExecutionError("refined collision geometry failed GLB roundtrip validation")

    refined_report = copy.deepcopy(child_report)
    refined_report["collision_geometry"] = refined_geometry.model_dump(mode="json")
    refined_report["metrics"] = {
        **dict(refined_report.get("metrics", {})),
        "convex_parts": float(len(kept)),
        "removed_parts": float(len(offending)),
        "removed_volume_fraction": removed_volume_fraction,
        "joint_band_fraction": joint_band_fraction,
    }
    refined_report["gates"] = {
        **dict(refined_report.get("gates", {})),
        "joint_locality": True,
        "removed_volume": True,
        "parent_child_collision_preserved": True,
    }
    refined_report["articulation_refinement"] = {
        "schema": "physrealx.articulated-collision-clearance/1",
        "source_collision_report": child_report_ref.model_dump(mode="json"),
        "source_assembly": assembly_ref.model_dump(mode="json"),
        "source_feasibility": feasibility_ref.model_dump(mode="json"),
        "joint": joint.name,
        "locked_range": before["locked_range"],
        "removed_part_indices": sorted(offending),
        "policy": "remove_sweep_failing_child_parts_inside_joint_band",
        "visual_geometry_changed": False,
        "joint_model_changed": False,
        "parent_child_collision_disabled": False,
    }
    refined_report_ref = store.put_json(refined_report)
    replacement_reports = tuple(
        refined_report_ref if ref == child_report_ref else ref for ref in request.collision_reports
    )
    candidate_request = request.model_copy(update={"collision_reports": replacement_reports})
    candidate_input = store.put_json(candidate_request.model_dump(mode="json", by_alias=True))
    candidate_asset = reference_prior_asset(candidate_input, store)
    after = evaluate_asset_feasibility(store, candidate_asset, config=evaluation_config)
    before_visual = {g.name: g.uri for g in asset.geometries if g.role is GeometryRole.VISUAL}
    after_visual = {
        g.name: g.uri for g in candidate_asset.geometries if g.role is GeometryRole.VISUAL
    }
    if before_visual != after_visual or before["locked_range"] != after["locked_range"]:
        raise MethodExecutionError("collision refinement changed visual geometry or joint limits")
    refinement_passed = (
        after["status"] == "PASSED" and after["max_penetration_m"] <= config.clearance_m
    )
    return store.put_json(
        {
            "schema": "physrealx.articulated-collision-refinement/1",
            "status": "PASSED" if refinement_passed else "FAILED",
            "source_assembly": assembly_ref.model_dump(mode="json"),
            "source_feasibility": feasibility_ref.model_dump(mode="json"),
            "candidate_assembly_input": candidate_input.model_dump(mode="json"),
            "collision_reports": [ref.model_dump(mode="json") for ref in replacement_reports],
            "joint": joint.name,
            "locked_range": after["locked_range"],
            "before": {
                "max_penetration_m": before["max_penetration_m"],
                "failing_pairs": sum(
                    pair["max_penetration_m"] > evaluation_config.penetration_tolerance_m
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
                    pair["max_penetration_m"] > evaluation_config.penetration_tolerance_m
                    for pair in after["collision_pairs"]
                ),
                "penetrating_pairs": sum(
                    pair["max_penetration_m"] > config.clearance_m
                    for pair in after["collision_pairs"]
                ),
            },
            "moving_link": joint.child,
            "removed_part_indices": sorted(offending),
            "removed_volume_fraction": removed_volume_fraction,
            "joint_band_fraction": joint_band_fraction,
            "visual_geometry_unchanged": True,
            "joint_model_unchanged": True,
            "parent_child_collision_enabled": True,
            "config": config.model_dump(mode="json"),
        }
    )
