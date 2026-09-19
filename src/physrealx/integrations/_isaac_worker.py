"""Private Isaac Sim 4.5 worker. Run only with the distribution's python.sh."""

from __future__ import annotations

import json
import socket
import struct
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any


def send(connection: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, allow_nan=False).encode()
    connection.sendall(struct.pack("!I", len(data)) + data)


def receive(connection: socket.socket) -> dict[str, Any]:
    def read(count: int) -> bytes:
        data = bytearray()
        while len(data) < count:
            chunk = connection.recv(count - len(data))
            if not chunk:
                raise EOFError
            data.extend(chunk)
        return bytes(data)

    count = struct.unpack("!I", read(4))[0]
    if count > 1024 * 1024:
        raise ValueError("oversized worker command")
    result: dict[str, Any] = json.loads(read(count))
    return result


class Simulation:
    def __init__(self, request: dict[str, Any]) -> None:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from omni.physx import get_physx_simulation_interface
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        self.config, self.task = request["config"], request["task"]
        self.output_directory = Path(request["output_directory"])
        self.body_paths: dict[str, str] = {}
        self.body_coms: dict[str, Any] = {}
        self.contact_loads: dict[str, Any] = {}
        self.cameras: dict[str, Any] = {}
        self.labels: dict[str, str] = {"0": "background_or_world"}
        self.label_paths: dict[str, int] = {}
        self.randomized: dict[str, Any] = {}
        dt = self.config["timestep_seconds"]
        self.world = World(
            physics_dt=dt,
            rendering_dt=dt,
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            set_defaults=False,
        )
        stage = self.world.stage
        gravity = np.asarray(self.config["gravity_m_s2"], dtype=float)
        magnitude = float(np.linalg.norm(gravity))
        scene = UsdPhysics.Scene.Get(stage, self.world.get_physics_context().prim_path)
        scene.CreateGravityMagnitudeAttr(magnitude)
        scene.CreateGravityDirectionAttr(
            Gf.Vec3f(*(gravity / magnitude if magnitude else [0, 0, -1]))
        )
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx_scene.CreateEnableGPUDynamicsAttr(False)
        physx_scene.CreateBroadphaseTypeAttr("MBP")
        physx_scene.CreateEnableEnhancedDeterminismAttr(True)
        self.articulations: dict[str, Any] = {}
        self.joints: dict[str, Any] = {}
        self.bodies: dict[str, Any] = {}
        self.body_entities: dict[str, str] = {}
        self.efforts: dict[str, float] = {}
        self.contact_force = 0.0
        self.initialized = False
        for index, entry in enumerate(request["entities"]):
            prefix = f"/World/entity_{index}"
            prim = UsdGeom.Xform.Define(stage, prefix).GetPrim()
            prim.GetReferences().AddReference(entry["path"], "/Asset")
            pose = entry["pose"]
            xform = UsdGeom.Xformable(prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(*pose["translation_m"]))
            x, y, z, w = pose["rotation_xyzw"]
            xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            asset, name = entry["asset"], entry["name"]
            root_path = f"{prefix}/link_{asset['links'].index(asset['root_link'])}"
            movable = any(j["joint_type"] != "fixed" for j in asset["joints"])
            for link_index, link in enumerate(asset["links"]):
                body_path = f"{prefix}/link_{link_index}"
                body_prim = stage.GetPrimAtPath(body_path)
                if movable:
                    UsdPhysics.RigidBodyAPI(body_prim).CreateKinematicEnabledAttr(False)
                PhysxSchema.PhysxContactReportAPI.Apply(body_prim).CreateThresholdAttr(0.0)
                self.body_entities[body_path] = name
                key = f"{name}/{link}"
                self.body_paths[key] = body_path
                self.body_coms[key] = np.array(
                    next(p for p in asset["physics"] if p["link"] == link)["center_of_mass_m"]
                )
                label = len(self.body_paths)
                self.labels[str(label)] = key
                self.label_paths[body_path] = label
                # Read-only views: World must not reset articulated links independently.
                self.bodies[f"{name}/{link}"] = SingleRigidPrim(
                    prim_path=body_path, name=f"body_{index}_{link_index}"
                )
            if not movable:
                continue
            # Fixed base: constrain the root at its authored WORLD pose (not at world origin).
            root_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(root_path)
            )
            fixed = UsdPhysics.FixedJoint.Define(stage, prefix + "/base_anchor")
            fixed.CreateBody1Rel().SetTargets([root_path])
            fixed.CreateLocalPos0Attr(Gf.Vec3f(root_matrix.ExtractTranslation()))
            q = root_matrix.ExtractRotationQuat()
            fixed.CreateLocalRot0Attr(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
            UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
            for joint_index, joint in enumerate(asset["joints"]):
                if joint["joint_type"] == "fixed":
                    continue
                joint_prim = stage.GetPrimAtPath(f"{prefix}/joint_{joint_index}")
                drive = UsdPhysics.DriveAPI.Get(
                    joint_prim, "angular" if joint["joint_type"] == "revolute" else "linear"
                )
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(0.0)
                physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
                physx_joint.CreateJointFrictionAttr(0.0)
                if joint["velocity_limit"] is not None:
                    scale = 180 / np.pi if joint["joint_type"] == "revolute" else 1
                    physx_joint.CreateMaxJointVelocityAttr(joint["velocity_limit"] * scale)
                self.joints[f"{name}/{joint['name']}"] = (name, f"joint_{joint_index}", joint)
            self.articulations[name] = self.world.scene.add(
                SingleArticulation(
                    prim_path=prefix + "/base_anchor",
                    name=f"articulation_{index}",
                )
            )
        controlled_entities = {name.split("/", 1)[0] for name in self.config["controlled_joints"]}
        self.contact_pair = (
            frozenset((*controlled_entities, self.task["subject_entity"]))
            if self.task["metadata"].get("evidence_scope") == "robot_contact"
            and len(controlled_entities) == 1
            else None
        )
        self.subscription = get_physx_simulation_interface().subscribe_full_contact_report_events(
            self._contacts
        )
        self._setup_sensors()
        self._capture_nominal()
        self.world.reset()
        for name, (entity, usd_name, spec) in tuple(self.joints.items()):
            art = self.articulations[entity]
            self.joints[name] = (entity, art.get_dof_index(usd_name), spec)
        self.world.pause()

    def _setup_sensors(self) -> None:
        import numpy as np
        import omni.replicator.core as rep
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom, UsdLux

        stage = self.world.stage
        used = {
            s["name"].split(".")[1]
            for s in self.task["observations"]
            if s["name"].startswith("camera.")
        }
        if not used:
            return
        # Cameras share lighting; authored meshes and PBR textures stay intact.
        dome = UsdLux.DomeLight.Define(stage, "/World/lighting/ambient")
        dome.CreateIntensityAttr(500.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/lighting/key")
        sun.CreateIntensityAttr(1500.0)
        sun.CreateAngleAttr(10.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35, 25, 0))
        for name, spec in self.config["cameras"].items():
            if name not in used:
                continue
            camera = Camera(
                prim_path=f"/World/cameras/{name}", resolution=(spec["width"], spec["height"])
            )
            camera.set_world_pose(
                position=np.array(spec["pose"]["translation_m"]),
                orientation=np.array(spec["pose"]["rotation_xyzw"])[[3, 0, 1, 2]],
                camera_axes="usd",
            )
            camera.initialize()
            # Camera setters use cm for lens attributes (USD stores tenths of cm).
            focal = 2.0
            vertical = 2 * focal * np.tan(np.deg2rad(spec["fovy_degrees"]) / 2)
            camera.set_focal_length(focal)
            camera.set_horizontal_aperture(vertical * spec["width"] / spec["height"])
            camera.set_vertical_aperture(vertical)
            camera.set_clipping_range(min(0.01, spec["max_depth_m"] / 100), spec["max_depth_m"])
            annotators = {}
            for modality, annotator in (
                ("rgb", "rgb"),
                ("depth", "distance_to_image_plane"),
                ("mask", "instance_id_segmentation"),
            ):
                annotation = rep.AnnotatorRegistry.get_annotator(annotator, device="cpu")
                annotation.attach([camera.get_render_product_path()])
                annotators[modality] = annotation
            self.cameras[name] = (camera, annotators)
            if self.config["randomization"]["occlusion_probability"]:
                path = f"/World/occluders/{name}"
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr(2.0)
                transform = UsdGeom.Xformable(cube)
                transform.AddTranslateOp()
                transform.AddOrientOp()
                transform.AddScaleOp()
                label = len(self.labels)
                self.labels[str(label)] = f"occluder/{name}"
                self.label_paths[path] = label

    def _capture_nominal(self) -> None:
        from pxr import UsdPhysics, UsdShade

        self.nominal: list[Any] = []
        stage = self.world.stage
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.MassAPI):
                api = UsdPhysics.MassAPI(prim)
                self.nominal.append(
                    (
                        "mass",
                        api.GetMassAttr(),
                        api.GetMassAttr().Get(),
                        api.GetDiagonalInertiaAttr(),
                        api.GetDiagonalInertiaAttr().Get(),
                    )
                )
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                api = UsdPhysics.MaterialAPI(prim)
                for attr in (api.GetStaticFrictionAttr(), api.GetDynamicFrictionAttr()):
                    self.nominal.append(("friction", attr, attr.Get()))
            display = prim.GetAttribute("primvars:displayColor")
            if display and display.Get() is not None:
                self.nominal.append(("display_color", display, display.Get()))
            shader = UsdShade.Shader(prim)
            if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface":
                diffuse = shader.GetInput("diffuseColor")
                if diffuse and not diffuse.HasConnectedSource():
                    self.nominal.append(("color", diffuse.GetAttr(), diffuse.Get()))
                roughness = shader.GetInput("roughness")
                if roughness and not roughness.HasConnectedSource():
                    self.nominal.append(("roughness", roughness.GetAttr(), roughness.Get()))
            if (
                shader
                and shader.GetIdAttr().Get() == "UsdUVTexture"
                and prim.GetName() == "BaseColor"
            ):
                scale = shader.GetInput("scale")
                self.nominal.append(("color", scale.GetAttr(), scale.Get()))
            if shader and shader.GetIdAttr().Get() == "UsdUVTexture" and prim.GetName() == "ORM":
                scale = shader.GetInput("scale")
                self.nominal.append(("roughness_texture", scale.GetAttr(), scale.Get()))
            if prim.GetPath().pathString.startswith("/World/lighting/"):
                attr = prim.GetAttribute("inputs:intensity")
                if attr:
                    self.nominal.append(("light", attr, attr.Get()))
        # Perturb dielectric Fresnel reflectance through IOR, preserving metallic workflow.
        if self.config["randomization"]["material_specular_jitter"]:
            from pxr import Sdf

            for prim in stage.Traverse():
                shader = UsdShade.Shader(prim)
                if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface":
                    ior = shader.CreateInput("ior", Sdf.ValueTypeNames.Float)
                    if ior.Get() is None:
                        ior.Set(1.5)
                    self.nominal.append(("specular", ior.GetAttr(), ior.Get()))

    def _randomize(self, rng: Any, seed: int) -> None:
        import numpy as np
        from pxr import Gf, UsdGeom
        from scipy.spatial.transform import Rotation

        config = self.config["randomization"]
        changes: dict[str, Any] = {}
        friction_material_scales: dict[str, float] = {}
        for kind, attr, nominal, *rest in self.nominal:
            if kind == "mass":
                scale = rng.uniform(*config["mass_scale"])
                attr.Set(float(nominal * scale))
                inertia_attr, inertia = rest
                inertia_attr.Set(inertia * scale)
                changes[str(attr.GetPath())] = float(nominal * scale)
            elif kind in {"friction", "light"}:
                if kind == "friction":
                    material = str(attr.GetPrim().GetPath())
                    if material not in friction_material_scales:
                        friction_material_scales[material] = float(
                            rng.uniform(*config["friction_scale"])
                        )
                    scale = friction_material_scales[material]
                else:
                    scale = rng.uniform(1 - config["light_jitter"], 1 + config["light_jitter"])
                attr.Set(float(nominal * scale))
                changes[str(attr.GetPath())] = float(nominal * scale)
            elif kind == "display_color":
                tint = rng.uniform(-config["color_jitter"], config["color_jitter"], 3)
                colors = np.clip(np.asarray(nominal) + tint, 0, 1)
                attr.Set(type(nominal)([Gf.Vec3f(*row) for row in colors]))
                changes[str(attr.GetPath())] = {"additive_rgb": tint.tolist()}
            elif kind == "specular":
                f0 = ((nominal - 1) / (nominal + 1)) ** 2
                bound = config["material_specular_jitter"]
                root = float(np.sqrt(np.clip(f0 + rng.uniform(-bound, bound), 0, 0.99)))
                ior_value = (1 + root) / (1 - root)
                attr.Set(ior_value)
                changes[str(attr.GetPath())] = ior_value
            elif kind == "roughness_texture":
                scale_value = np.array(nominal)
                bound = config["material_shininess_jitter"]
                scale_value[1] = np.clip(scale_value[1] + rng.uniform(-bound, bound), 0, 1)
                attr.Set(type(nominal)(*scale_value))
                changes[str(attr.GetPath())] = scale_value.tolist()
            elif kind == "roughness":
                value = float(
                    np.clip(
                        nominal
                        + rng.uniform(
                            -config["material_shininess_jitter"],
                            config["material_shininess_jitter"],
                        ),
                        0,
                        1,
                    )
                )
                attr.Set(value)
                changes[str(attr.GetPath())] = value
            else:
                bound = config["color_jitter"]
                color = np.array(nominal, dtype=float)
                color[:3] = np.clip(color[:3] + rng.uniform(-bound, bound, 3), 0, 1)
                attr.Set(type(nominal)(*color))
                changes[str(attr.GetPath())] = color.tolist()
        self.damping_scale = {
            name: float(rng.uniform(*config["damping_scale"])) for name in self.joints
        }
        self.friction_scale = {
            name: float(rng.uniform(*config["friction_scale"])) for name in self.joints
        }
        camera_positions = {}
        for name, (camera, _) in self.cameras.items():
            spec = self.config["cameras"][name]
            position = np.array(spec["pose"]["translation_m"]) + rng.uniform(
                -config["camera_position_jitter_m"], config["camera_position_jitter_m"], 3
            )
            camera.set_world_pose(position=position, camera_axes="usd")
            camera_positions[name] = position.tolist()
            if config["occlusion_probability"]:
                cube = UsdGeom.Cube.Get(self.world.stage, f"/World/occluders/{name}")
                rotation = Rotation.from_quat(spec["pose"]["rotation_xyzw"])
                half_height = 0.3 * np.tan(np.deg2rad(spec["fovy_degrees"]) / 2)
                half_width = half_height * spec["width"] / spec["height"]
                fraction = rng.uniform(*config["occlusion_fraction"])
                offset = [
                    rng.uniform(-0.7, 0.7) * half_width,
                    rng.uniform(-0.7, 0.7) * half_height,
                    -0.3,
                ]
                translation = position + rotation.apply(offset)
                prim = cube.GetPrim()
                prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*translation))
                x, y, z, w = spec["pose"]["rotation_xyzw"]
                prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
                prim.GetAttribute("xformOp:scale").Set(
                    Gf.Vec3f(half_width * fraction, half_height * fraction, 0.001)
                )
                cube.CreateDisplayColorAttr([Gf.Vec3f(*rng.uniform(0.1, 0.9, 3))])
                visible = rng.random() < config["occlusion_probability"]
                cube.CreateVisibilityAttr(
                    UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
                )
                changes[str(prim.GetPath())] = {
                    "position": translation.tolist(),
                    "fraction": float(fraction),
                    "visible": bool(visible),
                }
        self.randomized = {
            "seed": seed,
            "usd_attributes": changes,
            "damping_scale": self.damping_scale,
            "joint_friction_scale": self.friction_scale,
            "camera_positions": camera_positions,
        }
        self.sensor_warmup = True

    def _render_sensors(self) -> None:
        import numpy as np
        from scipy.spatial.transform import Rotation

        if not self.cameras:
            return
        # Render the frozen physics state. Extra RTX updates never integrate dynamics.
        physics_time = self.world.current_time
        for _ in range(12 if self.sensor_warmup else 3):
            self.world.render()
        if self.world.current_time != physics_time:
            raise RuntimeError("rendering advanced physics time")
        self.sensor_warmup = False
        for name, (camera, annotators) in self.cameras.items():
            spec = self.config["cameras"][name]
            rgb = np.asarray(annotators["rgb"].get_data())[..., :3].copy().astype(np.uint8)
            depth = np.asarray(annotators["depth"].get_data()).copy().astype(np.float32)
            if rgb.shape != (spec["height"], spec["width"], 3) or depth.shape != rgb.shape[:2]:
                raise ValueError(f"native camera did not produce synchronized images: {name}")
            depth[~np.isfinite(depth) | (depth <= 0) | (depth > spec["max_depth_m"])] = 0
            segmentation = annotators["mask"].get_data()
            raw = segmentation["data"]
            mask = np.zeros(depth.shape, dtype=np.int32)
            for index, path in segmentation["info"]["idToLabels"].items():
                matches = [
                    label
                    for prefix, label in self.label_paths.items()
                    if path == prefix or path.startswith(prefix + "/")
                ]
                if matches:
                    mask[raw == int(index)] = matches[0]
                elif int(index) > 0 and path not in {"BACKGROUND", "UNLABELLED"}:
                    raise ValueError(f"unmapped segmentation prim: {path}")
            position, q = camera.get_world_pose(camera_axes="usd")
            orientation = Rotation.from_matrix(
                Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix() @ np.diag([1, -1, -1])
            ).as_quat()
            np.savez(
                self.output_directory / f"camera_{name}.npz",
                rgb=rgb,
                depth=depth,
                mask=mask,
                intrinsics=np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64),
                position=np.asarray(position, dtype=np.float64),
                orientation=orientation.astype(np.float64),
            )

    def _contacts(self, headers: Any, data: Any, friction_anchors: Any) -> None:
        import numpy as np
        from pxr import PhysicsSchemaTools

        for header in headers:
            paths = [
                str(PhysicsSchemaTools.intToSdfPath(body))
                for body in (header.actor0, header.actor1)
            ]
            pair = frozenset(self.body_entities.get(path, "") for path in paths)
            selected = self.contact_pair is None or pair == self.contact_pair
            for samples, start, count, normal in (
                (data, header.contact_data_offset, header.num_contact_data, True),
                (
                    friction_anchors,
                    header.friction_anchors_offset,
                    header.num_friction_anchors_data,
                    False,
                ),
            ):
                for index in range(start, start + count):
                    contact = samples[index]
                    force = (
                        np.asarray(contact.impulse, dtype=float) / self.config["timestep_seconds"]
                    )
                    if selected and normal:
                        self.contact_force += float(np.linalg.norm(force))
                    # Full reports supply normal points and tangential friction anchors separately.
                    for path, sign in zip(paths, (1, -1)):
                        if path in self.contact_loads:
                            self.contact_loads[path][:3] += sign * force
                            self.contact_loads[path][3:] += np.cross(
                                np.asarray(contact.position), sign * force
                            )

    def _com_world(self, name: str) -> Any:
        from scipy.spatial.transform import Rotation

        position, q = self.bodies[name].get_world_pose()
        return position + Rotation.from_quat(q[[1, 2, 3, 0]]).apply(self.body_coms[name])

    def _clear_contacts(self) -> None:
        import numpy as np

        self.contact_force = 0.0
        # Store torque about world origin; shift to the reported COM after integration.
        self.contact_loads = {path: np.zeros(6) for path in self.body_paths.values()}

    def _apply_wrenches(self) -> None:
        import carb
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, Sdf

        interface = get_physx_simulation_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        for name, wrench in self.config["external_wrenches"].items():
            path = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(self.body_paths[name]))
            interface.apply_force_at_pos(
                stage_id, path, carb.Float3(*wrench["force_n"]), carb.Float3(*self._com_world(name))
            )
            interface.apply_torque(stage_id, path, carb.Float3(*wrench["torque_nm"]))

    def reset(self, seed: int) -> dict[str, Any]:
        import numpy as np

        self.world.stop()
        rng = np.random.default_rng(seed)
        self._randomize(rng, seed)
        self.world.reset()
        for body in self.bodies.values():
            body.initialize()
        self.randomized["runtime_body_masses_kg"] = {
            name: float(body.get_mass()) for name, body in self.bodies.items()
        }
        for entity, articulation in self.articulations.items():
            positions = np.zeros(articulation.num_dof, dtype=np.float32)
            initial = dict(zip(self.config["controlled_joints"], self.config["initial_positions"]))
            initial.update(self.config["passive_positions"])
            for name, (owner, index, spec) in self.joints.items():
                if owner == entity:
                    q = initial.get(name, 0.0)
                    jitter = (
                        self.config["randomization"]["initial_position_jitter"]
                        if name in self.config["controlled_joints"]
                        else 0
                    )
                    positions[index] = rng.uniform(
                        max(spec["lower_limit"], q - jitter), min(spec["upper_limit"], q + jitter)
                    )
            articulation.set_joint_positions(positions)
            articulation.set_joint_velocities(np.zeros_like(positions))
            articulation.set_joint_efforts(np.zeros_like(positions))
        self.world.physics_sim_view.update_articulations_kinematic()
        self._clear_contacts()
        self.efforts = dict.fromkeys(self.joints, 0.0)
        self.initialized = True
        return self.frame()

    def step(self, command: list[float]) -> dict[str, Any]:
        import numpy as np

        if not self.initialized:
            raise ValueError("reset is required")
        controlled = self.config["controlled_joints"]
        command_array = np.asarray(command, dtype=float)
        target = command_array.copy()
        if self.task["action"]["kind"] == "joint_delta":
            target += np.array([self.position(name) for name in controlled])
        if self.task["action"]["kind"] != "joint_effort":
            for name, value in zip(controlled, target):
                spec = self.joints[name][2]
                if not spec["lower_limit"] <= value <= spec["upper_limit"]:
                    raise ValueError(f"target exceeds physical limits: {name}")
        for _ in range(self.config["substeps"]):
            applied = {}
            for entity, articulation in self.articulations.items():
                q, v = articulation.get_joint_positions(), articulation.get_joint_velocities()
                forces = np.zeros_like(q)
                for joint_name, (owner, index, spec) in self.joints.items():
                    if owner == entity:
                        forces[index] = -spec["damping"] * self.damping_scale[joint_name] * v[
                            index
                        ] - spec["friction"] * self.friction_scale[joint_name] * np.sign(v[index])
                applied[entity] = forces
            for i, name in enumerate(controlled):
                entity, index, spec = self.joints[name]
                articulation = self.articulations[entity]
                force = (
                    command_array[i]
                    if self.task["action"]["kind"] == "joint_effort"
                    else (
                        self.config["position_gain"]
                        * (target[i] - articulation.get_joint_positions()[index])
                        - self.config["velocity_gain"] * articulation.get_joint_velocities()[index]
                    )
                )
                limit = self.config["effort_limits"][i]
                if spec["effort_limit"] is not None:
                    limit = min(limit, spec["effort_limit"])
                force = float(np.clip(force, -limit, limit))
                applied[entity][index] += force
                self.efforts[name] = force
            for entity, forces in applied.items():
                self.articulations[entity].set_joint_efforts(forces)
            self._clear_contacts()
            self._apply_wrenches()
            self.world.step(render=False)
        return self.frame()

    def position(self, name: str) -> float:
        entity, index, _ = self.joints[name]
        return float(self.articulations[entity].get_joint_positions()[index])

    def frame(self) -> dict[str, Any]:
        import numpy as np

        signals = {}

        def put(name: str, data: Any, unit: str, frame: str = "joint") -> None:
            values = np.asarray(data, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite dynamics: {name}")
            signals[name] = {
                "values": values.reshape(-1).tolist(),
                "shape": list(values.shape),
                "unit": unit,
                "frame": frame,
            }

        for name, (entity, index, spec) in self.joints.items():
            unit = "m" if spec["joint_type"] == "prismatic" else "rad"
            prefix = "joint." + name.replace("/", ".")
            put(prefix + ".position", [self.position(name)], unit)
            put(
                prefix + ".velocity",
                [self.articulations[entity].get_joint_velocities()[index]],
                unit + "/s",
            )
            put(prefix + ".effort", [self.efforts[name]], "N" if unit == "m" else "N*m")
        for suffix in ("position", "velocity", "effort"):
            selected = [
                signals["joint." + name.replace("/", ".") + "." + suffix]
                for name in self.config["controlled_joints"]
            ]
            put("joint." + suffix, [v["values"][0] for v in selected], selected[0]["unit"])
        poses = [body.get_world_pose() for body in self.bodies.values()]
        put("body.position", [pose[0] for pose in poses], "m", "world")
        put("body.orientation", [pose[1][[1, 2, 3, 0]] for pose in poses], "1", "world")
        origins, forces, torques = [], [], []
        for name, path in self.body_paths.items():
            origin = self._com_world(name)
            load = self.contact_loads[path]
            external = self.config["external_wrenches"].get(
                name, {"force_n": [0, 0, 0], "torque_nm": [0, 0, 0]}
            )
            origins.append(origin)
            forces.append(load[:3] + external["force_n"])
            torques.append(load[3:] - np.cross(origin, load[:3]) + external["torque_nm"])
        put("body.external_force", forces, "N", "world")
        put("body.external_torque", torques, "N*m", "world")
        put("body.wrench_origin", origins, "m", "world")
        put("contact.force", [self.contact_force], "N")
        self._render_sensors()
        return {
            "signals": signals,
            "cameras": list(self.cameras),
            "metadata": {
                "body_pose_order": list(self.bodies),
                "segmentation_labels": self.labels,
                "camera_axes": "OpenCV optical x-right y-down z-forward; pose camera-to-world",
                "depth": "optical z in meters; zero denotes invalid/out-of-range",
                "external_wrench": (
                    "normal and friction contact plus applied loads, world axes; torque about "
                    "body.wrench_origin (body COM); excludes gravity/internal joint forces"
                ),
                "events": "positive/negative log-intensity crossings accumulated between frames",
                "camera_renderer": "Isaac Sim RTX / native USD materials",
                "randomization": self.randomized,
                "physics_engine": "PhysX / Isaac Sim 4.5.0",
                "contact_force": (
                    "sum of normal-contact impulse magnitudes / physics dt at final substep"
                ),
                "joint_effort": "applied actuator effort; excludes passive damping/friction",
            },
        }


def main() -> None:
    # This directory contains usd.py, which must not shadow Kit's usd.schema namespace.
    sys.path = [
        entry for entry in sys.path if Path(entry).resolve() != Path(__file__).parent.resolve()
    ]
    connection = socket.socket(fileno=int(sys.argv[2]))
    application = None
    simulation = None
    try:
        from isaacsim import SimulationApp

        application = SimulationApp(
            {
                "headless": True,
                "multi_gpu": False,
                "sync_loads": True,
                "disable_viewport_updates": True,
            }
        )
        request = json.loads(Path(sys.argv[1]).read_text())
        simulation = Simulation(request)
        send(connection, {"ready": True})
        while True:
            message = receive(connection)
            if message["command"] == "reset":
                send(connection, simulation.reset(message["seed"]))
            elif message["command"] == "step":
                send(connection, simulation.step(message["action"]))
            else:
                raise ValueError("unknown worker command")
    except EOFError:
        pass
    except BaseException as error:
        traceback.print_exc()
        with suppress(OSError):
            send(connection, {"error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()
        if simulation is not None:
            simulation.world.stop()
            simulation.subscription = None
        if application is not None:
            application.close()


if __name__ == "__main__":
    main()
