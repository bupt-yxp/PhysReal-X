"""Bind segmented reconstruction meshes to registered articulated link frames."""

from __future__ import annotations

from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.collision import load_meshes
from physrealx.integrations.registration import load_registration
from physrealx.reconstruction import ReconstructionPrediction


def reconstructed_motion(
    reconstruction: dict[str, Any],
    registration: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Return module-one predicted motion in the registered metric frame.

    The supplied USD is evaluation evidence.  Its joint must never replace the
    reconstructed joint that produced the module-one animation.  Registration
    contributes only the rigid capture-to-metric transform and the group/link
    correspondence.
    """
    import numpy as np

    prediction = ReconstructionPrediction.model_validate(reconstruction.get("prediction"))
    if not prediction.segmented or prediction.motion_pair is None:
        raise DomainValidationError("physical asset requires segmented articulated motion")
    mapping = registration.get("motion_groups", {})
    if set(mapping) != set(prediction.parts) or any(len(links) != 1 for links in mapping.values()):
        raise DomainValidationError("each reconstructed motion group must map to one link")
    moving = [(name, part) for name, part in prediction.parts.items() if part.parent is not None]
    if len(moving) != 1:
        raise DomainValidationError("physical asset currently requires one reconstructed joint")
    group, part = moving[0]
    assert part.parent is not None
    parent, child = mapping[part.parent][0], mapping[group][0]
    joints = {joint["name"]: joint for joint in reference["joints"]}
    joint_name = registration.get("observed_joint")
    reference_joint = joints.get(joint_name)
    if (
        reference_joint is None
        or reference_joint["parent"] != parent
        or reference_joint["child"] != child
        or reference_joint["joint_type"] != part.kind
        or tuple(registration.get("state_order", ())) != prediction.motion_pair
    ):
        raise DomainValidationError("reconstructed joint/link mapping differs from registration")
    rotation = np.asarray(registration["rotation"], dtype=float)
    translation = np.asarray(registration["translation_m"], dtype=float)
    axis = rotation @ np.asarray(part.axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    origin = rotation @ np.asarray(part.origin, dtype=float) + translation
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite([*axis, *origin, part.span]).all()
        or not np.isclose(norm, 1.0, atol=1e-6)
        or abs(part.span) < 1e-8
    ):
        raise DomainValidationError("reconstructed motion is not a finite metric rigid joint")
    positions = [0.0, float(part.span)]
    return {
        "joint": joint_name,
        "parent": parent,
        "child": child,
        "kind": part.kind,
        "axis": (axis / norm).tolist(),
        "origin_m": origin.tolist(),
        "states": list(prediction.motion_pair),
        "positions": positions,
        "range": [min(positions), max(positions)],
        "source_group": group,
        "policy": "module_one_predicted_start_end",
    }


def register_reconstruction_geometry(
    store: ArtifactStore,
    reconstruction_ref: ArtifactRef,
    registration_ref: ArtifactRef,
) -> ArtifactRef:
    """Express the module-one reconstruction in registered link rest coordinates.

    Only the input rigid coordinate transforms are applied. No surface is added,
    removed, reshaped, recolored or read from the reference asset. Its already
    predicted joint is transformed with the same rigid registration.  The USD
    joint is retained only for correspondence/evaluation and supplies no motion.
    """
    import numpy as np
    import trimesh

    reconstruction = read_json(
        store, reconstruction_ref, "physrealx.geometry-reconstruction-report/1"
    )
    registration = load_registration(store, registration_ref)
    reference_ref = ArtifactRef.model_validate(registration["source_reference"])
    reference = read_json(store, reference_ref, "physrealx.usd-reference/1")
    if (
        reconstruction.get("unit") != "m"
        or reconstruction.get("coordinate_frame") != registration["source_frame"]
        or reconstruction.get("bundle") != registration["source_visual"]
        or registration.get("target_frame") != reference.get("coordinate_frame")
    ):
        raise DomainValidationError("reconstruction and registration sources differ")
    prediction = ReconstructionPrediction.model_validate(reconstruction.get("prediction"))
    if not prediction.segmented or prediction.motion_pair is None:
        raise DomainValidationError("physical asset requires segmented articulated geometry")
    start = registration["state_order"][0]
    if start != prediction.motion_pair[0] or start not in prediction.states:
        raise DomainValidationError("registration start state differs from reconstruction")
    mapping = registration["motion_groups"]
    if set(mapping) != set(prediction.parts) or any(len(links) != 1 for links in mapping.values()):
        raise DomainValidationError("each reconstructed motion group must map to one link")
    mapped = {group: links[0] for group, links in mapping.items()}
    if set(mapped.values()) != {link["name"] for link in reference["links"]}:
        raise DomainValidationError("reconstruction mapping must cover each reference link once")
    state = prediction.states[start]
    if set(state) != set(prediction.parts):
        raise DomainValidationError("start state must provide every reconstructed part")

    motion = reconstructed_motion(reconstruction, registration, reference)
    capture_to_reference = np.eye(4)
    capture_to_reference[:3, :3] = np.asarray(registration["rotation"], dtype=float)
    capture_to_reference[:3, 3] = np.asarray(registration["translation_m"], dtype=float)
    registered_meshes: dict[str, Any] = {}
    source_geometries: dict[str, Any] = {}
    pose_reprojection_error_m: dict[str, float] = {}
    for group, link in mapped.items():
        meshes = load_meshes(store, state[group], process=False)
        mesh = trimesh.util.concatenate(meshes)
        capture_vertices = np.asarray(mesh.vertices, dtype=float).copy()
        mesh.apply_transform(capture_to_reference)
        expected_reference = trimesh.transform_points(capture_vertices, capture_to_reference)
        reprojected_reference = np.asarray(mesh.vertices)
        reprojection_error = float(
            np.linalg.norm(expected_reference - reprojected_reference, axis=1).max()
        )
        if not np.isfinite(reprojection_error) or reprojection_error > 1e-9:
            raise MethodExecutionError("registering geometry changed an input part pose")
        if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
            raise DomainValidationError("registered reconstruction contains invalid geometry")
        registered_meshes[link] = mesh
        source_geometries[link] = state[group].model_dump(mode="json")
        pose_reprojection_error_m[link] = reprojection_error

    geometries: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for link, mesh in registered_meshes.items():
        artifact = store.put_bytes(mesh.export(file_type="glb"), media_type="model/gltf-binary")
        decoded = trimesh.util.concatenate(load_meshes(store, artifact, process=False))
        source = registered_meshes[link]
        if (
            len(decoded.vertices) != len(source.vertices)
            or not np.array_equal(decoded.faces, source.faces)
            or not np.array_equal(decoded.visual.vertex_colors, source.visual.vertex_colors)
        ):
            raise MethodExecutionError(
                "registered geometry changed reconstruction topology or colors"
            )
        roundtrip_drift = float(np.linalg.norm(decoded.vertices - source.vertices, axis=1).max())
        if not np.isfinite(roundtrip_drift) or roundtrip_drift > 1e-6:
            raise MethodExecutionError("registered geometry roundtrip moved source vertices")
        geometries[link] = artifact.model_dump(mode="json")
        stats[link] = {
            "source_group": next(group for group, name in mapped.items() if name == link),
            "vertices": len(mesh.vertices),
            "source_geometry": source_geometries[link],
            "topology_preserved": True,
            "vertex_colors_preserved": True,
            "triangles": len(mesh.faces),
            "bounds_m": mesh.bounds.tolist(),
            "roundtrip_vertex_drift_m": roundtrip_drift,
            "source_pose_reprojection_error_m": pose_reprojection_error_m[link],
        }
    return store.put_json(
        {
            "schema": "physrealx.registered-reconstruction-geometry/4",
            "status": "COMPLETED",
            "unit": "m",
            "coordinate_frame": "registered_reconstruction_start",
            "source_reconstruction": reconstruction_ref.model_dump(mode="json"),
            "source_registration": registration_ref.model_dump(mode="json"),
            "source_reference": reference_ref.model_dump(mode="json"),
            "source_state": start,
            "source_positions": {motion["joint"]: motion["positions"][0]},
            "reconstructed_motion": motion,
            "geometries": geometries,
            "links": stats,
            "geometry_source": "real_artgs_reconstruction",
            "geometry_policy": "coordinate_transform_only",
            "topology_source": "module_one_reconstruction",
            "pose_policy": "module_one_predicted_start_pose_locked",
            "source_pose_reprojection_error_m": pose_reprojection_error_m,
        }
    )
