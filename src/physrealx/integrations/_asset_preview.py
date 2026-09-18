"""Shared metric geometry transforms for platform previews."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import AssetSpec, GeometryRepresentation, GeometryRole, JointType
from physrealx.errors import DomainValidationError
from physrealx.integrations.collision import load_meshes, mesh_reference
from physrealx.reconstruction import joint_transform


def pose_matrix(pose: Any) -> Any:
    import numpy as np
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose.rotation_xyzw).as_matrix()
    matrix[:3, 3] = pose.translation_m
    return matrix


def link_transforms(asset: AssetSpec, positions: Mapping[str, float]) -> dict[str, Any]:
    """Resolve every link pose from named joint positions in asset coordinates."""
    import numpy as np

    unknown = set(positions) - {joint.name for joint in asset.joints}
    if unknown:
        raise DomainValidationError(f"preview contains unknown joints: {sorted(unknown)!r}")
    transforms = {asset.root_link: pose_matrix(asset.root_pose)}
    pending = list(asset.joints)
    while pending:
        ready = [joint for joint in pending if joint.parent in transforms]
        if not ready:
            raise DomainValidationError("physical asset joint graph is disconnected")
        for joint in ready:
            rest = min(max(0.0, joint.lower_limit), joint.upper_limit)
            position = float(positions.get(joint.name, rest))
            tolerance = max(1e-8, (joint.upper_limit - joint.lower_limit) * 1e-3)
            if not (joint.lower_limit - tolerance <= position <= joint.upper_limit + tolerance):
                raise DomainValidationError(
                    f"preview joint position is outside limits: {joint.name}={position}"
                )
            position = min(max(position, joint.lower_limit), joint.upper_limit)
            kind = {
                JointType.FIXED: "fixed",
                JointType.PRISMATIC: "slide",
                JointType.REVOLUTE: "hinge",
            }[joint.joint_type]
            transforms[joint.child] = (
                transforms[joint.parent]
                @ pose_matrix(joint.parent_to_joint)
                @ joint_transform(kind, joint.axis, np.zeros(3), position)
            )
            pending.remove(joint)
    return transforms


def asset_geometry(
    store: ArtifactStore, asset: AssetSpec, *, roles: tuple[GeometryRole, ...]
) -> dict[GeometryRole, list[tuple[str, Any]]]:
    """Load immutable local-space meshes grouped by their declared role."""
    import numpy as np
    import trimesh

    result: dict[GeometryRole, list[tuple[str, Any]]] = {role: [] for role in roles}
    for geometry in asset.geometries:
        if geometry.role not in result:
            continue
        if geometry.representation is GeometryRepresentation.PRIMITIVE:
            meshes = [trimesh.creation.box(extents=geometry.primitive_dimensions_m)]
        elif geometry.representation is GeometryRepresentation.MESH:
            meshes = load_meshes(store, mesh_reference(geometry.uri, store), process=False)
        else:
            raise DomainValidationError("preview supports mesh and primitive geometry")
        local = pose_matrix(geometry.local_pose)
        scale = np.asarray(geometry.scale)
        for mesh in meshes:
            copy = mesh.copy()
            copy.vertices = copy.vertices * scale
            copy.apply_transform(local)
            result[geometry.role].append((geometry.link, copy))
    missing = [role.value for role, meshes in result.items() if not meshes]
    if missing:
        raise DomainValidationError(f"preview requires geometry roles: {missing!r}")
    return result
