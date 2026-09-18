"""Import a local MJCF robot into the canonical articulated-asset contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import (
    AssetEvidenceState,
    AssetSpec,
    BodyPhysicsSpec,
    GeometryRepresentation,
    GeometryRole,
    GeometrySpec,
    JointSpec,
    JointType,
    ParameterSource,
    PoseSpec,
    ScalarParameter,
)
from physrealx.errors import DomainValidationError


def _prior(value: float) -> ScalarParameter:
    return ScalarParameter(
        value=value,
        uncertainty=0.0,
        source=ParameterSource.DEFAULT,
        confidence=1.0,
    )


def _pose(position: object, quaternion_wxyz: object) -> PoseSpec:
    import numpy as np

    position_array = np.asarray(position, dtype=float)
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    return PoseSpec(
        translation_m=tuple(float(value) for value in position_array),
        rotation_xyzw=tuple(float(value) for value in quaternion[[1, 2, 3, 0]]),
    )


def _inertia(model: Any, body_id: int) -> tuple[tuple[float, float, float], ...]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    principal = np.asarray(model.body_inertia[body_id], dtype=float)
    quaternion = np.asarray(model.body_iquat[body_id], dtype=float)
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    matrix = rotation @ np.diag(principal) @ rotation.T
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )


def _mesh(model: Any, geometry_id: int, *, collision: bool) -> object:
    import mujoco
    import numpy as np
    import trimesh

    if model.geom_type[geometry_id] != mujoco.mjtGeom.mjGEOM_MESH:
        raise DomainValidationError("MJCF importer only converts mesh and box geometry")
    mesh_id = int(model.geom_dataid[geometry_id])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            model.mesh_vert[vertex_start : vertex_start + vertex_count],
            dtype=float,
        ),
        faces=np.asarray(
            model.mesh_face[face_start : face_start + face_count],
            dtype=int,
        ),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    if collision and not mesh.is_convex:
        mesh = mesh.convex_hull
    material_id = int(model.geom_matid[geometry_id])
    rgba = model.mat_rgba[material_id] if material_id >= 0 else model.geom_rgba[geometry_id]
    color = np.clip(
        np.rint(np.asarray(rgba, dtype=float) * 255),
        0,
        255,
    ).astype(np.uint8)
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    return mesh


def import_mujoco_model_asset(
    artifacts: ArtifactStore,
    source: str | Path,
    *,
    resource_id: str,
    version: str,
    display_name: str,
    source_name: str,
    source_revision: str,
    license_name: str,
) -> AssetSpec:
    """Compile one local MJCF model and retain its geometry, kinematics and inertia."""
    import mujoco
    import numpy as np

    path = Path(source).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".xml":
        raise DomainValidationError("MuJoCo model source must be a local XML file")
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except ValueError as error:
        raise DomainValidationError(f"MuJoCo rejected the source model: {error}") from error
    body_ids = tuple(range(1, model.nbody))
    links = tuple(model.body(body_id).name for body_id in body_ids)
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise DomainValidationError("MJCF robot bodies must be uniquely named")
    roots = [body_id for body_id in body_ids if int(model.body_parentid[body_id]) == 0]
    if len(roots) != 1:
        raise DomainValidationError("MJCF robot must contain exactly one root body")

    direct_effort_limits: dict[int, float] = {}
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or not model.actuator_forcelimited[actuator_id]:
            continue
        direct_effort_limits[joint_id] = max(
            direct_effort_limits.get(joint_id, 0.0),
            float(np.max(np.abs(model.actuator_forcerange[actuator_id]))),
        )

    geometries: list[GeometrySpec] = []
    for geometry_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geometry_id])
        if body_id == 0:
            continue
        role = (
            GeometryRole.VISUAL
            if not model.geom_contype[geometry_id] and not model.geom_conaffinity[geometry_id]
            else GeometryRole.COLLISION
        )
        name = f"{links[body_id - 1]}_{role.value}_{geometry_id}"
        local_pose = _pose(model.geom_pos[geometry_id], model.geom_quat[geometry_id])
        if model.geom_type[geometry_id] == mujoco.mjtGeom.mjGEOM_BOX:
            geometries.append(
                GeometrySpec(
                    name=name,
                    link=links[body_id - 1],
                    role=role,
                    representation=GeometryRepresentation.PRIMITIVE,
                    uri="primitive://box",
                    local_pose=local_pose,
                    primitive_dimensions_m=tuple(
                        float(value) for value in 2 * model.geom_size[geometry_id]
                    ),
                )
            )
            continue
        mesh = _mesh(model, geometry_id, collision=role is GeometryRole.COLLISION)
        reference = artifacts.put_bytes(
            mesh.export(file_type="glb"),  # type: ignore[attr-defined]
            media_type="model/gltf-binary",
        )
        geometries.append(
            GeometrySpec(
                name=name,
                link=links[body_id - 1],
                role=role,
                representation=GeometryRepresentation.MESH,
                uri=f"artifact://{reference.artifact_id}",
                local_pose=local_pose,
            )
        )

    joints: list[JointSpec] = []
    for body_id in body_ids:
        parent_id = int(model.body_parentid[body_id])
        if parent_id == 0:
            continue
        child = model.body(body_id).name
        parent = model.body(parent_id).name
        joint_count = int(model.body_jntnum[body_id])
        if joint_count > 1:
            raise DomainValidationError("canonical assets require at most one MJCF joint per body")
        pose = _pose(model.body_pos[body_id], model.body_quat[body_id])
        if joint_count == 0:
            joints.append(
                JointSpec(
                    name=f"{child}_fixed",
                    parent=parent,
                    child=child,
                    joint_type=JointType.FIXED,
                    parent_to_joint=pose,
                    lower_limit=0.0,
                    upper_limit=0.0,
                )
            )
            continue
        joint_id = int(model.body_jntadr[body_id])
        if not np.allclose(model.jnt_pos[joint_id], 0.0, atol=1e-10):
            raise DomainValidationError("MJCF joint positions must coincide with body origins")
        mjcf_type = model.jnt_type[joint_id]
        if mjcf_type == mujoco.mjtJoint.mjJNT_HINGE:
            joint_type = JointType.REVOLUTE
        elif mjcf_type == mujoco.mjtJoint.mjJNT_SLIDE:
            joint_type = JointType.PRISMATIC
        else:
            raise DomainValidationError("canonical assets support MJCF hinge and slide joints")
        dof = int(model.jnt_dofadr[joint_id])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        joints.append(
            JointSpec(
                name=model.joint(joint_id).name,
                parent=parent,
                child=child,
                joint_type=joint_type,
                parent_to_joint=pose,
                axis=tuple(float(value) for value in model.jnt_axis[joint_id]),
                lower_limit=lower,
                upper_limit=upper,
                damping=float(model.dof_damping[dof]),
                friction=float(model.dof_frictionloss[dof]),
                effort_limit=direct_effort_limits.get(joint_id, 40.0),
            )
        )

    physics = tuple(
        BodyPhysicsSpec(
            link=model.body(body_id).name,
            mass_kg=_prior(float(model.body_mass[body_id])),
            center_of_mass_m=tuple(float(value) for value in model.body_ipos[body_id]),
            inertia_kg_m2=_inertia(model, body_id),
            static_friction=_prior(1.0),
            dynamic_friction=_prior(1.0),
            restitution=_prior(0.0),
        )
        for body_id in body_ids
    )
    if not geometries:
        raise DomainValidationError("MJCF robot contains no supported geometry")
    home_positions: dict[str, float] = {}
    if model.nkey:
        for joint_id in range(model.njnt):
            address = int(model.jnt_qposadr[joint_id])
            home_positions[model.joint(joint_id).name] = float(model.key_qpos[0, address])
    return AssetSpec(
        resource_id=resource_id,
        version=version,
        display_name=display_name,
        root_link=model.body(roots[0]).name,
        links=links,
        geometries=tuple(geometries),
        joints=tuple(joints),
        physics=physics,
        evidence_state=AssetEvidenceState.PRIOR,
        metadata={
            "asset_kind": "open_source_robot",
            "source": source_name,
            "source_revision": source_revision,
            "source_model": path.name,
            "license": license_name,
            "degrees_of_freedom": sum(joint.joint_type is not JointType.FIXED for joint in joints),
            "home_positions": home_positions,
            "collision_proxy": "source_mesh_convex_hull_when_required",
        },
    )
