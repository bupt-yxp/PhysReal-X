"""MuJoCo adapter: compile immutable articulated assets and execute real dynamics."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore, BackendCapabilities, Frame, MethodProbe, StepResult
from physrealx.domain import (
    AssetSpec,
    DataValue,
    DomainModel,
    GeometryRepresentation,
    GeometryRole,
    JointType,
    JSONValue,
    MethodCapability,
    PoseSpec,
    SceneSpec,
    SignalDType,
    TaskSpec,
)
from physrealx.errors import (
    DomainValidationError,
    InvalidStateTransitionError,
    MethodExecutionError,
)
from physrealx.integrations._artifacts import dependency_probe
from physrealx.integrations.collision import load_meshes, mesh_reference
from physrealx.integrations.sensors import CAMERA_MODALITIES, CameraSpec, MujocoSensors
from physrealx.integrations.visuals import png_bytes
from physrealx.methods import builtin_method


def _numbers(values: Iterable[float]) -> str:
    return " ".join(format(float(v), ".17g") for v in values)


def _pose(pose: PoseSpec) -> dict[str, str]:
    x, y, z, w = pose.rotation_xyzw
    return {"pos": _numbers(pose.translation_m), "quat": _numbers((w, x, y, z))}


class MujocoRandomization(DomainModel):
    mass_scale: tuple[float, float] = (1.0, 1.0)
    damping_scale: tuple[float, float] = (1.0, 1.0)
    friction_scale: tuple[float, float] = (1.0, 1.0)
    initial_position_jitter: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    color_jitter: float = Field(default=0.0, ge=0, le=1)
    light_jitter: float = Field(default=0.0, ge=0, le=1)
    material_shininess_jitter: float = Field(default=0.0, ge=0, le=1)
    material_specular_jitter: float = Field(default=0.0, ge=0, le=1)
    camera_position_jitter_m: float = Field(default=0.0, ge=0, le=1)
    occlusion_probability: float = Field(default=0.0, ge=0, le=1)
    occlusion_fraction: tuple[float, float] = (0.08, 0.2)

    @model_validator(mode="after")
    def validate_intervals(self) -> MujocoRandomization:
        for interval in (self.mass_scale, self.damping_scale, self.friction_scale):
            if (
                not all(math.isfinite(v) for v in interval)
                or interval[0] <= 0
                or interval[0] > interval[1]
            ):
                raise ValueError(
                    "physical randomization needs positive finite ordered scale bounds"
                )
        if not 0 < self.occlusion_fraction[0] <= self.occlusion_fraction[1] <= 0.5:
            raise ValueError("occluder size must occupy a bounded fraction of the camera view")
        return self


class MujocoConfig(DomainModel):
    cameras: dict[str, CameraSpec] = Field(default_factory=dict)
    controlled_joints: tuple[str, ...] = Field(min_length=1)
    timestep_seconds: float = Field(default=0.002, gt=0, le=0.01)
    substeps: int = Field(default=25, ge=1, le=1000)
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    initial_positions: tuple[float, ...]
    passive_positions: dict[str, float] = Field(default_factory=dict)
    effort_limits: tuple[float, ...]
    position_gain: float = Field(default=200.0, gt=0, allow_inf_nan=False)
    velocity_gain: float = Field(default=20.0, ge=0, allow_inf_nan=False)
    randomization: MujocoRandomization = Field(default_factory=MujocoRandomization)

    @model_validator(mode="after")
    def validate_vectors(self) -> MujocoConfig:
        if len(self.cameras) > 8 or any(
            re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None for name in self.cameras
        ):
            raise ValueError("at most eight uniquely named cameras are supported")
        count = len(self.controlled_joints)
        if (
            len(set(self.controlled_joints)) != count
            or len(self.initial_positions) != count
            or len(self.effort_limits) != count
        ):
            raise ValueError("joint names, initial positions and effort limits must align uniquely")
        if any(
            not math.isfinite(v)
            for v in (*self.gravity_m_s2, *self.initial_positions, *self.effort_limits)
        ) or any(v <= 0 for v in self.effort_limits):
            raise ValueError("physical configuration must be finite with positive effort limits")
        if set(self.passive_positions) & set(self.controlled_joints) or any(
            not math.isfinite(value) for value in self.passive_positions.values()
        ):
            raise ValueError("passive positions must be finite and exclude controlled joints")
        return self


@dataclass(frozen=True)
class CompiledScene:
    xml: str
    joints: dict[str, Any]
    files: dict[str, bytes]


def compile_scene(
    scene: SceneSpec, assets: Mapping[str, AssetSpec], store: ArtifactStore, config: MujocoConfig
) -> CompiledScene:
    import numpy as np

    root = ET.Element("mujoco", model=scene.resource_id)
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        inertiafromgeom="false",
        autolimits="true",
        fusestatic="false",
    )
    ET.SubElement(
        root,
        "option",
        timestep=str(config.timestep_seconds),
        gravity=_numbers(config.gravity_m_s2),
        integrator="implicitfast",
    )
    asset_xml, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    files: dict[str, bytes] = {}
    for name, camera in config.cameras.items():
        ET.SubElement(
            world,
            "camera",
            {
                "name": name,
                "fovy": str(camera.fovy_degrees),
                **_pose(camera.pose),
            },
        )
        ET.SubElement(
            world,
            "light",
            pos=_numbers(camera.pose.translation_m),
            directional="true",
            dir="0 0 -1",
        )
        if config.randomization.occlusion_probability:
            occluder = ET.SubElement(world, "body", name=f"occluder/{name}")
            ET.SubElement(
                occluder,
                "geom",
                name=f"occluder/{name}",
                type="box",
                size="0.01 0.01 0.001",
                rgba="0.3 0.3 0.3 0",
                contype="0",
                conaffinity="0",
            )
    if config.cameras:
        visual = ET.SubElement(root, "visual")
        ET.SubElement(
            visual,
            "headlight",
            ambient="0.45 0.45 0.45",
            diffuse="0.75 0.75 0.75",
            specular="0.1 0.1 0.1",
            active="1",
        )
        ET.SubElement(
            visual,
            "global",
            offwidth=str(max(c.width for c in config.cameras.values())),
            offheight=str(max(c.height for c in config.cameras.values())),
        )
    joint_specs: dict[str, Any] = {}
    contacts = ET.SubElement(root, "contact")
    for entity in scene.entities:
        asset = assets[entity.name]
        for left, right in asset.contact_exclusions:
            ET.SubElement(
                contacts, "exclude", body1=f"{entity.name}/{left}", body2=f"{entity.name}/{right}"
            )
        if asset.up_axis != "z":
            raise DomainValidationError("MuJoCo adapter currently requires Z-up assets")
        parent_xml = ET.SubElement(
            world, "body", {"name": f"{entity.name}/frame", **_pose(entity.pose)}
        )
        base = ET.SubElement(
            parent_xml,
            "body",
            {"name": f"{entity.name}/{asset.root_link}", **_pose(asset.root_pose)},
        )
        bodies = {asset.root_link: base}
        remaining = list(asset.joints)
        while remaining:
            ready = [joint for joint in remaining if joint.parent in bodies]
            if not ready:
                raise DomainValidationError("asset joints are not a rooted tree")
            for joint in ready:
                body = ET.SubElement(
                    bodies[joint.parent],
                    "body",
                    attrib={"name": f"{entity.name}/{joint.child}", **_pose(joint.parent_to_joint)},
                )
                bodies[joint.child] = body
                if joint.joint_type is not JointType.FIXED:
                    name = f"{entity.name}/{joint.name}"
                    ET.SubElement(
                        body,
                        "joint",
                        name=name,
                        type="slide" if joint.joint_type is JointType.PRISMATIC else "hinge",
                        axis=_numbers(joint.axis),
                        range=_numbers((joint.lower_limit, joint.upper_limit)),
                        damping=str(joint.damping),
                        frictionloss=str(joint.friction),
                    )
                    joint_specs[name] = joint
                remaining.remove(joint)
        physics = {body.link: body for body in asset.physics}
        for link, body in bodies.items():
            properties = physics[link]
            inertia = properties.inertia_kg_m2
            ET.SubElement(
                body,
                "inertial",
                pos=_numbers(properties.center_of_mass_m),
                mass=str(properties.mass_kg.value),
                fullinertia=_numbers(
                    (
                        inertia[0][0],
                        inertia[1][1],
                        inertia[2][2],
                        inertia[0][1],
                        inertia[0][2],
                        inertia[1][2],
                    )
                ),
            )
        for index, geometry in enumerate(asset.geometries):
            properties = physics[geometry.link]
            # MuJoCo has one sliding-friction coefficient. Preserve explicit dynamic friction;
            # restitution uses the engine contact model and is recorded as an approximation.
            attributes = {
                **_pose(geometry.local_pose),
                "name": f"{entity.name}/geometry/{index}",
                "friction": _numbers((properties.dynamic_friction.value, 0.0, 0.0)),
                "contype": "1" if geometry.role is GeometryRole.COLLISION else "0",
                "conaffinity": "1" if geometry.role is GeometryRole.COLLISION else "0",
                "group": "3" if geometry.role is GeometryRole.COLLISION else "0",
            }
            if geometry.representation is GeometryRepresentation.PRIMITIVE:
                assert geometry.primitive_dimensions_m is not None
                ET.SubElement(
                    bodies[geometry.link],
                    "geom",
                    type="box",
                    size=_numbers(
                        np.array(geometry.primitive_dimensions_m) * np.array(geometry.scale) / 2
                    ),
                    attrib=attributes,
                )
            elif geometry.representation is GeometryRepresentation.MESH:
                meshes = load_meshes(store, mesh_reference(geometry.uri, store))
                for part_index, mesh in enumerate(meshes):
                    if geometry.role is GeometryRole.COLLISION and not mesh.is_convex:
                        raise DomainValidationError(
                            "MuJoCo collision meshes must be explicitly decomposed convex parts"
                        )
                    name = f"{entity.name}/mesh/{index}/{part_index}"
                    mesh_attributes = {}
                    visual_attributes = {}
                    if geometry.role is GeometryRole.VISUAL and mesh.visual.kind == "texture":
                        material = mesh.visual.material
                        if hasattr(material, "to_pbr"):
                            material = material.to_pbr()
                        factor = material.baseColorFactor
                        color = (
                            np.asarray(factor if factor is not None else [255] * 4, dtype=float)
                            / 255
                        )
                        roughness = float(
                            material.roughnessFactor
                            if material.roughnessFactor is not None
                            else 0.5
                        )
                        metallic = float(
                            material.metallicFactor if material.metallicFactor is not None else 0.0
                        )
                        params = {
                            "name": name + "/material",
                            "rgba": _numbers(color),
                            "roughness": str(roughness),
                            "metallic": str(metallic),
                            # Native MuJoCo OpenGL uses scalar highlights; PBR values
                            # are also retained for external physically based renderers.
                            "shininess": str(1 - roughness),
                            "specular": str(0.2 + 0.6 * metallic),
                        }
                        image = material.baseColorTexture
                        if image is not None:
                            uv = np.asarray(mesh.visual.uv)
                            if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
                                raise DomainValidationError(
                                    "textured mesh requires finite per-vertex UVs"
                                )
                            payload = png_bytes(image.convert("RGB"))
                            texture_name = "texture_" + sha256(payload).hexdigest()
                            filename = texture_name + ".png"
                            if filename not in files:
                                files[filename] = payload
                                ET.SubElement(
                                    asset_xml,
                                    "texture",
                                    name=texture_name,
                                    type="2d",
                                    file=filename,
                                )
                            params["texture"] = texture_name
                            mesh_attributes["texcoord"] = _numbers(uv.reshape(-1))
                        ET.SubElement(asset_xml, "material", attrib=params)
                        visual_attributes["material"] = params["name"]
                    ET.SubElement(
                        asset_xml,
                        "mesh",
                        attrib={
                            "name": name,
                            "vertex": _numbers(mesh.vertices.reshape(-1)),
                            "face": " ".join(str(int(v)) for v in mesh.faces.reshape(-1)),
                            "scale": _numbers(geometry.scale),
                            **mesh_attributes,
                        },
                    )
                    ET.SubElement(
                        bodies[geometry.link],
                        "geom",
                        type="mesh",
                        mesh=name,
                        attrib={
                            **attributes,
                            **visual_attributes,
                            "name": f"{attributes['name']}/{part_index}",
                        },
                    )
            else:
                raise DomainValidationError("MuJoCo requires explicit primitive or mesh geometry")
    if not set(config.controlled_joints) <= set(joint_specs):
        raise DomainValidationError("controlled joint does not exist in the composed scene")
    actuator = ET.SubElement(root, "actuator")
    for name, limit in zip(config.controlled_joints, config.effort_limits):
        declared = joint_specs[name].effort_limit
        if declared is not None and limit > declared:
            raise DomainValidationError("backend effort limit exceeds asset limit")
        ET.SubElement(
            actuator,
            "motor",
            name=f"motor/{name}",
            joint=name,
            gear="1",
            ctrllimited="true",
            ctrlrange=_numbers((-limit, limit)),
        )
    return CompiledScene(ET.tostring(root, encoding="unicode"), joint_specs, files)


class MujocoSession:
    def __init__(
        self,
        scene: SceneSpec,
        assets: Mapping[str, AssetSpec],
        task: TaskSpec,
        config: MujocoConfig,
        store: ArtifactStore,
    ) -> None:
        import mujoco
        import numpy as np

        self._config, self._task, self._store = config, task, store
        declared_controlled = task.metadata.get("controlled_joints")
        controlled_entities = (
            {str(name).split("/", 1)[0] for name in declared_controlled}
            if isinstance(declared_controlled, list)
            else set()
        )
        self._contact_entity_pair = (
            frozenset((*controlled_entities, task.subject_entity))
            if task.metadata.get("evidence_scope") == "robot_contact"
            and len(controlled_entities) == 1
            else None
        )
        self._sensors: MujocoSensors | None = None
        compiled = compile_scene(scene, assets, store, config)
        joints = compiled.joints
        try:
            self.model = mujoco.MjModel.from_xml_string(compiled.xml, assets=compiled.files)
        except ValueError as error:
            raise DomainValidationError(f"MuJoCo rejected the compiled asset: {error}") from error
        self.data = mujoco.MjData(self.model)
        self._body_ids = [
            self.model.body(f"{entity.name}/{link}").id
            for entity in scene.entities
            for link in assets[entity.name].links
        ]
        self._joint_indices = {
            name: (
                int(self.model.joint(name).qposadr[0]),
                int(self.model.joint(name).dofadr[0]),
                "m" if joint.joint_type is JointType.PRISMATIC else "rad",
            )
            for name, joint in joints.items()
        }
        if any(
            name not in joints or not joints[name].lower_limit <= value <= joints[name].upper_limit
            for name, value in config.passive_positions.items()
        ):
            raise DomainValidationError("passive initial positions exceed known joint limits")
        self._nominal = {
            name: getattr(self.model, name).copy()
            for name in (
                "body_mass",
                "body_inertia",
                "dof_damping",
                "dof_frictionloss",
                "geom_friction",
                "geom_rgba",
                "mat_rgba",
                "mat_shininess",
                "mat_specular",
                "light_diffuse",
                "cam_pos",
                "body_pos",
                "body_quat",
                "geom_size",
            )
        }
        self._qpos = [int(self.model.joint(name).qposadr[0]) for name in config.controlled_joints]
        self._dofs = [int(self.model.joint(name).dofadr[0]) for name in config.controlled_joints]
        self._limits = [
            (joints[name].lower_limit, joints[name].upper_limit)
            for name in config.controlled_joints
        ]
        kinds = {joints[name].joint_type for name in config.controlled_joints}
        if len(kinds) != 1:
            raise DomainValidationError("one action vector requires homogeneous joint units")
        self._position_unit = "m" if JointType.PRISMATIC in kinds else "rad"
        self._force_unit = "N" if JointType.PRISMATIC in kinds else "N*m"
        if task.action.signal.shape != (len(config.controlled_joints),):
            raise DomainValidationError("action shape differs from controlled joint count")
        expected_unit = (
            self._force_unit if task.action.kind == "joint_effort" else self._position_unit
        )
        if task.action.signal.unit != expected_unit:
            raise DomainValidationError("action units do not match the actual joint command")
        if any(not lo <= q <= hi for q, (lo, hi) in zip(config.initial_positions, self._limits)):
            raise DomainValidationError("initial joint position exceeds asset limits")
        self._closed, self._initialized, self._steps = False, False, 0
        self._target = np.array(config.initial_positions)
        try:
            self._sensors = MujocoSensors(
                self.model, dict(config.cameras), task.observations, store
            )
            mujoco.mj_forward(self.model, self.data)
            self._frame()
        except BaseException:
            self.close()
            raise

    @property
    def dt_seconds(self) -> float:
        return self._config.timestep_seconds * self._config.substeps

    def _ensure_open(self) -> None:
        if self._closed:
            raise InvalidStateTransitionError("MuJoCo session is closed")

    def reset(self, seed: int) -> Frame:
        import mujoco
        import numpy as np

        self._ensure_open()
        rng = np.random.default_rng(seed)
        config = self._config.randomization
        for name, values in self._nominal.items():
            getattr(self.model, name)[:] = values
        masses = rng.uniform(*config.mass_scale, size=len(self._body_ids))
        self.model.body_mass[self._body_ids] *= masses
        self.model.body_inertia[self._body_ids] *= masses[:, None]
        self.model.dof_damping[:] *= rng.uniform(*config.damping_scale, size=self.model.nv)
        self.model.dof_frictionloss[:] *= rng.uniform(*config.friction_scale, size=self.model.nv)
        self.model.geom_friction[:, 0] *= rng.uniform(*config.friction_scale, size=self.model.ngeom)
        unbound = self.model.geom_matid < 0
        for index, colors in enumerate((self.model.geom_rgba, self.model.mat_rgba)):
            jitter = rng.uniform(
                -config.color_jitter, config.color_jitter, size=colors[:, :3].shape
            )
            if index == 0:
                jitter[~unbound] = 0
            colors[:, :3] = np.clip(colors[:, :3] + jitter, 0, 1)
        for name, bound in (
            ("mat_shininess", config.material_shininess_jitter),
            ("mat_specular", config.material_specular_jitter),
        ):
            values = getattr(self.model, name)
            values[:] = np.clip(values + rng.uniform(-bound, bound, size=values.shape), 0, 1)
        self.model.light_diffuse[:] = np.clip(
            self.model.light_diffuse
            + rng.uniform(
                -config.light_jitter, config.light_jitter, size=self.model.light_diffuse.shape
            ),
            0,
            1,
        )
        self.model.cam_pos[:] += rng.uniform(
            -config.camera_position_jitter_m,
            config.camera_position_jitter_m,
            size=self.model.cam_pos.shape,
        )
        if config.occlusion_probability:
            from scipy.spatial.transform import Rotation

            for name, camera in self._config.cameras.items():
                body_id = self.model.body(f"occluder/{name}").id
                geom_id = self.model.geom(f"occluder/{name}").id
                camera_id = self.model.camera(name).id
                rotation = Rotation.from_quat(camera.pose.rotation_xyzw).as_matrix()
                distance = 0.3
                half_height = distance * math.tan(math.radians(camera.fovy_degrees) / 2)
                half_width = half_height * camera.width / camera.height
                offset = np.array(
                    [
                        rng.uniform(-0.7, 0.7) * half_width,
                        rng.uniform(-0.7, 0.7) * half_height,
                        -distance,
                    ]
                )
                self.model.body_pos[body_id] = self.model.cam_pos[camera_id] + rotation @ offset
                x, y, z, w = camera.pose.rotation_xyzw
                self.model.body_quat[body_id] = (w, x, y, z)
                fraction = rng.uniform(*config.occlusion_fraction)
                self.model.geom_size[geom_id] = (
                    half_width * fraction,
                    half_height * fraction,
                    0.001,
                )
                self.model.geom_rgba[geom_id, :3] = rng.uniform(0.1, 0.9, 3)
                self.model.geom_rgba[geom_id, 3] = float(
                    rng.random() < config.occlusion_probability
                )
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        for name, value in self._config.passive_positions.items():
            self.data.qpos[self._joint_indices[name][0]] = value
        for index, (q, (lo, hi)) in enumerate(zip(self._config.initial_positions, self._limits)):
            lower = max(lo, q - config.initial_position_jitter)
            upper = min(hi, q + config.initial_position_jitter)
            self.data.qpos[self._qpos[index]] = rng.uniform(lower, upper)
        self._target[:] = self._config.initial_positions
        if self._sensors is not None:
            self._sensors.reset()
        self._steps, self._initialized = 0, True
        mujoco.mj_forward(self.model, self.data)
        return self._frame()

    def step(self, action: DataValue) -> StepResult:
        import mujoco
        import numpy as np

        self._ensure_open()
        if not self._initialized:
            raise InvalidStateTransitionError("MuJoCo session must be reset before stepping")
        action.assert_matches(self._task.action.signal)
        if action.inline is None:
            raise DomainValidationError("MuJoCo requires inline joint commands")
        command = np.asarray(action.inline, dtype=float)
        if (
            not np.isfinite(command).all()
            or (command < self._task.action.lower).any()
            or (command > self._task.action.upper).any()
        ):
            raise DomainValidationError("joint command exceeds the task action bounds")
        if self._task.action.kind == "joint_delta":
            self._target = self.data.qpos[self._qpos] + command
        elif self._task.action.kind == "joint_position":
            self._target = command
        if self._task.action.kind != "joint_effort" and any(
            not lo <= q <= hi for q, (lo, hi) in zip(self._target, self._limits)
        ):
            raise DomainValidationError("commanded joint target exceeds physical limits")
        for _ in range(self._config.substeps):
            effort = (
                command
                if self._task.action.kind == "joint_effort"
                else (
                    self._config.position_gain * (self._target - self.data.qpos[self._qpos])
                    - self._config.velocity_gain * self.data.qvel[self._dofs]
                )
            )
            self.data.ctrl[:] = np.clip(
                effort, -np.array(self._config.effort_limits), self._config.effort_limits
            )
            mujoco.mj_step(self.model, self.data)
            if (
                not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
                or any(w.number for w in self.data.warning)
            ):
                raise MethodExecutionError(
                    "MuJoCo produced non-finite dynamics or a solver warning"
                )
        self._steps += 1
        # mj_step integrates qpos after its position stage. Refresh all derived poses,
        # contacts and sensor quantities before publishing the new timestamp.
        mujoco.mj_forward(self.model, self.data)
        return StepResult(
            frame=self._frame(),
            reward=0.0,
            terminated=False,
            truncated=self._steps >= self._task.max_steps,
        )

    def _frame(self) -> Frame:
        import mujoco
        import numpy as np

        values: dict[str, DataValue] = {}
        signals = {
            "joint.position": (self.data.qpos[self._qpos], self._position_unit),
            "joint.velocity": (self.data.qvel[self._dofs], f"{self._position_unit}/s"),
            "joint.effort": (self.data.qfrc_actuator[self._dofs], self._force_unit),
        }
        world_signals = {
            "body.position": (self.data.xpos[self._body_ids], "m"),
            "body.orientation": (self.data.xquat[self._body_ids][:, [1, 2, 3, 0]], "1"),
        }
        wrench_signals = {"body.external_force", "body.external_torque", "body.wrench_origin"}
        if any(signal.name in wrench_signals for signal in self._task.observations):
            mujoco.mj_rnePostConstraint(self.model, self.data)
            world_signals.update(
                {
                    "body.external_force": (self.data.cfrc_ext[self._body_ids, 3:], "N"),
                    "body.external_torque": (self.data.cfrc_ext[self._body_ids, :3], "N*m"),
                    "body.wrench_origin": (
                        self.data.subtree_com[self.model.body_rootid[self._body_ids]],
                        "m",
                    ),
                }
            )
        for name, (qpos, dof, unit) in self._joint_indices.items():
            prefix = "joint." + name.replace("/", ".")
            signals[prefix + ".position"] = (self.data.qpos[qpos : qpos + 1], unit)
            signals[prefix + ".velocity"] = (self.data.qvel[dof : dof + 1], unit + "/s")
            signals[prefix + ".effort"] = (
                self.data.qfrc_actuator[dof : dof + 1],
                "N" if unit == "m" else "N*m",
            )
        if any(signal.name == "contact.force" for signal in self._task.observations):
            forces = np.zeros(6)
            magnitude = 0.0
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                entities = frozenset(
                    (
                        self.model.geom(int(contact.geom1)).name.split("/", 1)[0],
                        self.model.geom(int(contact.geom2)).name.split("/", 1)[0],
                    )
                )
                if self._contact_entity_pair is not None and entities != self._contact_entity_pair:
                    continue
                mujoco.mj_contactForce(self.model, self.data, index, forces)
                magnitude += float(np.linalg.norm(forces[:3]))
            signals["contact.force"] = (np.array([magnitude]), "N")
        timestamp = self._steps * self.dt_seconds
        if self._sensors is not None:
            values.update(self._sensors.read(self.data, timestamp))
        for signal in self._task.observations:
            if signal.name in values:
                continue
            if signal.name in world_signals:
                data, unit = world_signals[signal.name]
                if signal.dtype not in {SignalDType.FLOAT32, SignalDType.FLOAT64}:
                    raise DomainValidationError("body signals require floating-point observations")
                value = DataValue(
                    dtype=signal.dtype,
                    shape=data.shape,
                    unit=unit,
                    frame="world",
                    timestamp_seconds=timestamp,
                    inline=tuple(float(v) for v in data.reshape(-1)),
                )
                value.assert_matches(signal)
                values[signal.name] = value
                continue
            if signal.name not in signals or signal.dtype not in {
                SignalDType.FLOAT32,
                SignalDType.FLOAT64,
            }:
                raise DomainValidationError(f"MuJoCo cannot produce observation {signal.name}")
            data, unit = signals[signal.name]
            if signal.unit != unit or signal.frame != "joint":
                raise DomainValidationError(
                    "MuJoCo observation units or frame disagree with physical signals"
                )
            value = DataValue(
                dtype=signal.dtype,
                shape=(len(data),),
                unit=unit,
                frame="joint",
                timestamp_seconds=self._steps * self.dt_seconds,
                inline=tuple(float(v) for v in data),
            )
            value.assert_matches(signal)
            values[signal.name] = value
        metadata: dict[str, JSONValue] = {
            "body_pose_order": [self.model.body(index).name for index in self._body_ids],
            "segmentation_labels": {
                str(index): self.model.body(index).name or "background_or_world"
                for index in range(self.model.nbody)
            },
            "camera_axes": "OpenCV optical x-right y-down z-forward; pose camera-to-world",
            "external_wrench": "world axes, torque about body.wrench_origin (root subtree COM)",
            "events": "positive/negative log-intensity crossings accumulated between frames",
        }
        return Frame(observations=values, state=dict(values), metadata=metadata)

    def close(self) -> None:
        self._closed = True
        if self._sensors is not None:
            self._sensors.close()
            self._sensors = None


class MujocoBackend:
    descriptor = builtin_method(
        MethodCapability.SIMULATION_RUN,
        "mujoco",
        input_contract="SceneSpec/1+TaskSpec/1+AssetSpec/1",
        output_contract="SimulationSession/1",
        python=">=3.10,<3.13",
        version="3.0.0",
    )

    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    def probe(self) -> MethodProbe:
        return dependency_probe(
            {
                "numpy": ("numpy", "1.26.4"),
                "mujoco": ("mujoco", "3.3.7"),
                "trimesh": ("trimesh", "4.5.1"),
            }
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rigid_body=True,
            articulation=True,
            deterministic_reset=True,
            contact_force=True,
            observation_keys=(
                "joint.position",
                "joint.velocity",
                "joint.effort",
                "contact.force",
                "body.position",
                "body.orientation",
                "body.external_force",
                "body.external_torque",
                "body.wrench_origin",
            ),
            observation_patterns=(
                *(f"camera.*.{modality}" for modality in CAMERA_MODALITIES),
                "joint.*.position",
                "joint.*.velocity",
                "joint.*.effort",
            ),
            action_kinds=("joint_delta", "joint_position", "joint_effort"),
        )

    def create_session(
        self,
        scene: SceneSpec,
        assets: Mapping[str, AssetSpec],
        task: TaskSpec,
        config: Mapping[str, JSONValue],
    ) -> MujocoSession:
        if task.action.kind not in self.capabilities().action_kinds:
            raise DomainValidationError("unsupported MuJoCo action kind")
        return MujocoSession(
            scene, assets, task, MujocoConfig.model_validate(config), self.artifacts
        )
