"""Bind segmented reconstruction meshes to registered articulated link frames."""

from __future__ import annotations

from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.collision import load_meshes
from physrealx.integrations.physical_estimation import locked_motion
from physrealx.integrations.registration import load_registration
from physrealx.integrations.usd_reference import reference_link_transforms
from physrealx.reconstruction import ReconstructionPrediction


def register_reconstruction_geometry(
    store: ArtifactStore,
    reconstruction_ref: ArtifactRef,
    registration_ref: ArtifactRef,
) -> ArtifactRef:
    """Move observed start-state parts into shared reference-rest coordinates.

    The supplied model contributes only link names, kinematic topology and the metric
    capture-to-reference registration. Published visual geometry remains the measured
    reconstruction and is explicitly unposed from the observed start configuration.
    Vertices retain the shared reference origin; assembly converts them to joint-centred
    link coordinates exactly once.
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

    positions = {joint["name"]: joint["lower_limit"] for joint in reference["joints"]}
    positions[registration["observed_joint"]] = registration["evaluations"][start][
        "joint_position"
    ]
    posed = reference_link_transforms(reference, positions)
    capture_to_reference = np.eye(4)
    capture_to_reference[:3, :3] = np.asarray(registration["rotation"], dtype=float)
    capture_to_reference[:3, 3] = np.asarray(registration["translation_m"], dtype=float)
    geometries: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for group, link in mapped.items():
        meshes = load_meshes(store, state[group], process=False)
        mesh = trimesh.util.concatenate(meshes)
        mesh.apply_transform(capture_to_reference)
        mesh.apply_transform(np.linalg.inv(posed[link]))
        if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
            raise DomainValidationError("registered reconstruction contains invalid geometry")
        artifact = store.put_bytes(mesh.export(file_type="glb"), media_type="model/gltf-binary")
        geometries[link] = artifact.model_dump(mode="json")
        stats[link] = {
            "source_group": group,
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.faces),
            "bounds_m": mesh.bounds.tolist(),
        }
    return store.put_json(
        {
            "schema": "physrealx.registered-reconstruction-geometry/1",
            "status": "COMPLETED",
            "unit": "m",
            "coordinate_frame": "reference_link_rest",
            "source_reconstruction": reconstruction_ref.model_dump(mode="json"),
            "source_registration": registration_ref.model_dump(mode="json"),
            "source_reference": reference_ref.model_dump(mode="json"),
            "source_state": start,
            "source_positions": positions,
            "motion": locked_motion(reference, registration),
            "geometries": geometries,
            "links": stats,
            "geometry_source": "real_artgs_reconstruction",
            "topology_source": "registered_reference_model",
        }
    )
