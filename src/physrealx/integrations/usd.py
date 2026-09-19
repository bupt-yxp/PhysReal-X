"""Self-contained OpenUSD physics packaging and numerical round-trip checks."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    AssetSpec,
    GeometryRepresentation,
    GeometryRole,
    GeometrySpec,
    JointType,
    MethodCapability,
    OperationSpec,
    canonical_json_bytes,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import dependency_probe, input_artifact, publish, read_json
from physrealx.integrations.assembly import assembled_asset
from physrealx.integrations.collision import load_meshes, mesh_reference
from physrealx.integrations.visuals import validate_usd_visual, write_usd_visual
from physrealx.methods import builtin_method


def _geometry_meshes(geometry: GeometrySpec, artifacts: ArtifactStore) -> list[Any]:
    if geometry.representation is GeometryRepresentation.PRIMITIVE:
        import trimesh

        meshes = [trimesh.creation.box(extents=geometry.primitive_dimensions_m)]
    elif geometry.representation is GeometryRepresentation.MESH:
        meshes = load_meshes(artifacts, mesh_reference(geometry.uri, artifacts))
    else:
        raise DomainValidationError("USD physics requires mesh or box geometry")
    if geometry.role is GeometryRole.COLLISION and any(not mesh.is_convex for mesh in meshes):
        raise DomainValidationError(
            "USD collision meshes require explicitly decomposed convex parts"
        )
    return meshes


def _link_poses(asset: AssetSpec) -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    def matrix(pose: Any) -> Any:
        value = np.eye(4)
        value[:3, :3] = Rotation.from_quat(pose.rotation_xyzw).as_matrix()
        value[:3, 3] = pose.translation_m
        return value

    result = {asset.root_link: matrix(asset.root_pose)}
    pending = list(asset.joints)
    while pending:
        for joint in tuple(pending):
            if joint.parent in result:
                result[joint.child] = result[joint.parent] @ matrix(joint.parent_to_joint)
                pending.remove(joint)
    return result


def validate_usd_stage(stage: Any, asset: AssetSpec, artifacts: ArtifactStore) -> None:
    """Verify the portable USD representation against the complete source asset."""
    import numpy as np
    from pxr import Ar, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation

    if (
        stage is None
        or UsdGeom.GetStageMetersPerUnit(stage) != 1.0
        or UsdGeom.GetStageUpAxis(stage) != "Z"
    ):
        raise MethodExecutionError("USD roundtrip changed units or up axis")
    if stage.GetRootLayer().GetExternalReferences():
        raise MethodExecutionError("USD package contains external references")
    root = stage.GetPrimAtPath("/Asset")
    encoded = root.GetAttribute("physrealx:assetSpec").Get()
    if not isinstance(encoded, str) or AssetSpec.model_validate_json(encoded) != asset:
        raise MethodExecutionError("USD source asset identity changed")
    appearance = asset.metadata.get("registered_appearance")
    if appearance is not None:
        from physrealx.integrations.registration import load_registration

        registration = load_registration(artifacts, ArtifactRef.model_validate(appearance))
        if root.GetAttribute("physrealx:registeredAppearance").Get() != canonical_json_bytes(
            registration
        ).decode("utf-8"):
            raise MethodExecutionError("USD roundtrip changed appearance registration")
        bundle = root.GetAttribute("physrealx:gaussianBundle").Get()
        resolved = (
            Ar.GetResolver().OpenAsset(Ar.ResolvedPath(bundle.resolvedPath)) if bundle else None
        )
        if resolved is None or bytes(resolved.GetBuffer()) != artifacts.get_bytes(
            ArtifactRef.model_validate(registration["source_visual"])
        ):
            raise MethodExecutionError("USD roundtrip lost the registered Gaussian model")
    poses = _link_poses(asset)
    paths = {link: f"/Asset/link_{index}" for index, link in enumerate(asset.links)}
    for link, path in paths.items():
        expected_filtered = {
            paths[right] for left, right in asset.contact_exclusions if left == link
        }
        actual_filtered = {
            str(item)
            for item in UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(path))
            .GetFilteredPairsRel()
            .GetTargets()
        }
        if actual_filtered != expected_filtered:
            raise MethodExecutionError("USD roundtrip changed contact exclusions")
    for body in asset.physics:
        prim = stage.GetPrimAtPath(paths[body.link])
        if not np.allclose(
            np.array(UsdGeom.Xformable(prim).GetLocalTransformation()),
            poses[body.link].T,
            atol=1e-7,
        ):
            raise MethodExecutionError("USD roundtrip changed body transform")
        mass = UsdPhysics.MassAPI(prim)
        axes = mass.GetPrincipalAxesAttr().Get()
        rotation = Rotation.from_quat([*axes.GetImaginary(), axes.GetReal()]).as_matrix()
        inertia = rotation @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ rotation.T
        if (
            not math.isclose(mass.GetMassAttr().Get(), body.mass_kg.value, rel_tol=1e-6)
            or not np.allclose(mass.GetCenterOfMassAttr().Get(), body.center_of_mass_m, atol=1e-7)
            or not np.allclose(inertia, body.inertia_kg_m2, rtol=1e-6, atol=1e-9)
        ):
            raise MethodExecutionError("USD roundtrip changed body mass, center or inertia")
        material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(f"{paths[body.link]}/Material"))
        for attribute, expected in (
            (material.GetStaticFrictionAttr(), body.static_friction.value),
            (material.GetDynamicFrictionAttr(), body.dynamic_friction.value),
            (material.GetRestitutionAttr(), body.restitution.value),
        ):
            if not math.isclose(attribute.Get(), expected, rel_tol=1e-6, abs_tol=1e-8):
                raise MethodExecutionError("USD roundtrip changed contact material")
        for index, geometry in enumerate(g for g in asset.geometries if g.link == body.link):
            for part_index, original in enumerate(_geometry_meshes(geometry, artifacts)):
                points = original.vertices * np.array(geometry.scale)
                points = (
                    points @ Rotation.from_quat(geometry.local_pose.rotation_xyzw).as_matrix().T
                    + geometry.local_pose.translation_m
                )
                mesh = UsdGeom.Mesh(
                    stage.GetPrimAtPath(f"{paths[body.link]}/mesh_{index}_{part_index}")
                )
                if (
                    not mesh
                    or not np.allclose(np.array(mesh.GetPointsAttr().Get()), points, atol=1e-6)
                    or not np.array_equal(
                        np.array(mesh.GetFaceVertexIndicesAttr().Get()), original.faces.reshape(-1)
                    )
                    or not np.array_equal(
                        np.array(mesh.GetFaceVertexCountsAttr().Get()),
                        np.full(len(original.faces), 3),
                    )
                ):
                    raise MethodExecutionError("USD mesh roundtrip changed geometry")
                if geometry.role is GeometryRole.VISUAL:
                    validate_usd_visual(stage, mesh, original)
                if (
                    geometry.role is GeometryRole.COLLISION
                    and not UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
                ):
                    raise MethodExecutionError("USD roundtrip disabled collision geometry")
    for index, joint in enumerate(asset.joints):
        prim = stage.GetPrimAtPath(f"/Asset/joint_{index}")
        usd_joint = UsdPhysics.Joint(prim)
        if (
            [str(path) for path in usd_joint.GetBody0Rel().GetTargets()] != [paths[joint.parent]]
            or [str(path) for path in usd_joint.GetBody1Rel().GetTargets()] != [paths[joint.child]]
            or not np.allclose(
                usd_joint.GetLocalPos0Attr().Get(), joint.parent_to_joint.translation_m, atol=1e-7
            )
        ):
            raise MethodExecutionError("USD roundtrip changed joint topology or origin")
        if joint.joint_type is not JointType.FIXED:
            expected_type = (
                "PhysicsPrismaticJoint"
                if joint.joint_type is JointType.PRISMATIC
                else "PhysicsRevoluteJoint"
            )
            if prim.GetTypeName() != expected_type:
                raise MethodExecutionError("USD roundtrip changed joint type")
            q = usd_joint.GetLocalRot1Attr().Get()
            axis = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).apply([1, 0, 0])
            scale = 180.0 / math.pi if joint.joint_type is JointType.REVOLUTE else 1.0
            if (
                not np.allclose(axis, joint.axis, atol=1e-6)
                or not math.isclose(
                    prim.GetAttribute("physics:lowerLimit").Get(),
                    joint.lower_limit * scale,
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
                or not math.isclose(
                    prim.GetAttribute("physics:upperLimit").Get(),
                    joint.upper_limit * scale,
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
            ):
                raise MethodExecutionError("USD roundtrip changed joint axis or limits")


def author_asset_stage(
    asset: AssetSpec, artifacts: ArtifactStore
) -> tuple[Any, dict[str, bytes], int]:
    """Author the shared, lossless USD asset used by packaging and simulation."""
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from scipy.spatial.transform import Rotation

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    root.CreateAttribute("physrealx:assetSpec", Sdf.ValueTypeNames.String, custom=True).Set(
        canonical_json_bytes(asset).decode()
    )
    UsdPhysics.Scene.Define(stage, "/PhysicsScene").CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    stage.GetPrimAtPath("/PhysicsScene").GetAttribute("physics:gravityMagnitude").Set(9.81)
    poses = _link_poses(asset)
    paths = {link: f"/Asset/link_{index}" for index, link in enumerate(asset.links)}
    mesh_count = 0
    texture_files: dict[str, bytes] = {}
    for body in asset.physics:
        xform = UsdGeom.Xform.Define(stage, paths[body.link])
        xform.AddTransformOp().Set(Gf.Matrix4d(poses[body.link].T.tolist()))
        rigid = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(body.link == asset.root_link)
        mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass.CreateMassAttr(body.mass_kg.value)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*body.center_of_mass_m))
        eigenvalues, eigenvectors = np.linalg.eigh(body.inertia_kg_m2)
        if eigenvalues[-1] > eigenvalues[:2].sum() + 1e-9:
            raise DomainValidationError("body inertia violates the physical triangle inequality")
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1
        quaternion = Rotation.from_matrix(eigenvectors).as_quat()
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*eigenvalues.tolist()))
        mass.CreatePrincipalAxesAttr(
            Gf.Quatf(float(quaternion[3]), Gf.Vec3f(*quaternion[:3].tolist()))
        )
        material = UsdShade.Material.Define(stage, f"{paths[body.link]}/Material")
        physical_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physical_material.CreateStaticFrictionAttr(body.static_friction.value)
        physical_material.CreateDynamicFrictionAttr(body.dynamic_friction.value)
        physical_material.CreateRestitutionAttr(body.restitution.value)
        for index, geometry in enumerate(g for g in asset.geometries if g.link == body.link):
            for part_index, mesh in enumerate(_geometry_meshes(geometry, artifacts)):
                points = mesh.vertices * np.array(geometry.scale)
                points = (
                    points @ Rotation.from_quat(geometry.local_pose.rotation_xyzw).as_matrix().T
                    + geometry.local_pose.translation_m
                )
                path = f"{paths[body.link]}/mesh_{index}_{part_index}"
                usd_mesh = UsdGeom.Mesh.Define(stage, path)
                usd_mesh.CreatePointsAttr(points.tolist())
                usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
                usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
                usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                if geometry.role is GeometryRole.VISUAL:
                    write_usd_visual(stage, usd_mesh, mesh, texture_files)
                if geometry.role is GeometryRole.COLLISION:
                    UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim()).CreateCollisionEnabledAttr(
                        True
                    )
                    UsdPhysics.MeshCollisionAPI.Apply(usd_mesh.GetPrim()).CreateApproximationAttr(
                        "convexHull"
                    )
                    UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(
                        material, materialPurpose="physics"
                    )
                    usd_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
                mesh_count += 1
    for index, joint in enumerate(asset.joints):
        path = f"/Asset/joint_{index}"
        cls = {
            JointType.PRISMATIC: UsdPhysics.PrismaticJoint,
            JointType.REVOLUTE: UsdPhysics.RevoluteJoint,
            JointType.FIXED: UsdPhysics.FixedJoint,
        }[joint.joint_type]
        usd_joint = cls.Define(stage, path)
        usd_joint.CreateBody0Rel().SetTargets([paths[joint.parent]])
        usd_joint.CreateBody1Rel().SetTargets([paths[joint.child]])
        # USD joint axes are tokens. Encode an arbitrary recovered axis in the joint frames.
        axis_rotation, _ = Rotation.align_vectors([joint.axis], [[1.0, 0.0, 0.0]])
        parent_rotation = Rotation.from_quat(joint.parent_to_joint.rotation_xyzw) * axis_rotation
        for creator, rotation in (
            (usd_joint.CreateLocalRot0Attr, parent_rotation),
            (usd_joint.CreateLocalRot1Attr, axis_rotation),
        ):
            q = rotation.as_quat()
            creator(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3].tolist())))
        usd_joint.CreateLocalPos0Attr(Gf.Vec3f(*joint.parent_to_joint.translation_m))
        usd_joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        if joint.joint_type is not JointType.FIXED:
            usd_joint.CreateAxisAttr("X")
            scale = 180.0 / math.pi if joint.joint_type is JointType.REVOLUTE else 1.0
            usd_joint.CreateLowerLimitAttr(joint.lower_limit * scale)
            usd_joint.CreateUpperLimitAttr(joint.upper_limit * scale)
            drive = UsdPhysics.DriveAPI.Apply(
                usd_joint.GetPrim(),
                "angular" if joint.joint_type is JointType.REVOLUTE else "linear",
            )
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(joint.damping / scale)
            # Portable USD Physics has no Coulomb joint-friction field. Preserve its
            # explicit unit and value as custom evidence; never silently discard it.
            usd_joint.GetPrim().CreateAttribute(
                "physrealx:coulombFriction", Sdf.ValueTypeNames.Double, custom=True
            ).Set(joint.friction)
    for left, right in asset.contact_exclusions:
        UsdPhysics.FilteredPairsAPI.Apply(
            stage.GetPrimAtPath(paths[left])
        ).CreateFilteredPairsRel().AddTarget(paths[right])
    return stage, texture_files, mesh_count


class UsdPackageAdapter:
    descriptor = builtin_method(
        MethodCapability.ASSET_PACKAGE,
        "openusd",
        input_contract="AssetAssemblyBundle/1",
        output_contract="UsdAssetBundle/1+AssetPackageManifest/1+UsdRoundTripValidationReport/1",
        python=">=3.10,<3.13",
        version="2.0.0",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe(
            {
                "pxr.Usd": ("usd-core", "24.11"),
                "scipy": ("scipy", "1.13.1"),
                "trimesh": ("trimesh", "4.5.1"),
            }
        )

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        from pxr import Sdf, Usd

        if spec.method.config:
            raise DomainValidationError("OpenUSD packaging has no implicit configuration")
        source = input_artifact(spec, "asset", artifacts)
        document = read_json(artifacts, source, "physrealx.asset-assembly-bundle/1")
        asset = assembled_asset(document, artifacts)
        stage, texture_files, mesh_count = author_asset_stage(asset, artifacts)
        with tempfile.TemporaryDirectory(prefix="physrealx-usd-") as directory:
            output = Path(directory) / "asset.usdc"
            package_files = dict(texture_files)
            appearance = asset.metadata.get("registered_appearance")
            if appearance is not None:
                from physrealx.integrations.registration import load_registration

                registration = load_registration(artifacts, ArtifactRef.model_validate(appearance))
                package_files["appearance/artgs.zip"] = artifacts.get_bytes(
                    ArtifactRef.model_validate(registration["source_visual"])
                )
                stage.GetDefaultPrim().CreateAttribute(
                    "physrealx:registeredAppearance", Sdf.ValueTypeNames.String, custom=True
                ).Set(canonical_json_bytes(registration).decode("utf-8"))
                stage.GetDefaultPrim().CreateAttribute(
                    "physrealx:gaussianBundle", Sdf.ValueTypeNames.Asset, custom=True
                ).Set(Sdf.AssetPath("appearance/artgs.zip"))
            for name, payload in package_files.items():
                destination = Path(directory) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            if not stage.GetRootLayer().Export(str(output)):
                raise MethodExecutionError("USD export failed")
            package_path = Path(directory) / "asset.usdz"
            # All dependencies were authored here with relative content-addressed
            # paths. The native writer preserves them and applies USDZ alignment,
            # without dependency localization rewriting paths through a global temp.
            with Usd.ZipFileWriter.CreateNew(str(package_path)) as writer:
                if not writer.AddFile(str(output), "asset.usdc"):
                    raise MethodExecutionError("USDZ root layer publication failed")
                for name in sorted(package_files):
                    if not writer.AddFile(str(Path(directory) / name), name):
                        raise MethodExecutionError("USDZ texture publication failed")
            reopened = Usd.Stage.Open(str(package_path))
            validate_usd_stage(reopened, asset, artifacts)
            usd_ref = artifacts.put_file(package_path, media_type="model/vnd.usdz+zip")
        metrics = {
            "links": float(len(asset.links)),
            "joints": float(len(asset.joints)),
            "meshes": float(mesh_count),
            "textures": float(len(texture_files)),
            "registered_appearance": float(appearance is not None),
        }
        manifest = {
            "schema": "physrealx.asset-package-manifest/1",
            "roundtrip_status": "PASSED",
            "self_contained": True,
            "source_assembly": source.model_dump(mode="json"),
            "usd_asset": usd_ref.model_dump(mode="json"),
            "asset_spec": asset.model_dump(mode="json"),
            "physics_mapping": {
                "joint_coulomb_friction": "physrealx:coulombFriction custom attribute"
            },
        }
        report = {
            "schema": "physrealx.usd-roundtrip-validation-report/1",
            "status": "PASSED",
            "source_assembly_artifact_id": source.artifact_id,
            "usd_artifact_id": usd_ref.artifact_id,
            "method": self.descriptor.model_dump(mode="json"),
            "metrics": metrics,
            "validation_scope": "serialization_roundtrip",
            "physical_validation": False,
        }
        return publish(
            spec,
            artifacts,
            {"asset_package_manifest": manifest, "roundtrip_validation_report": report},
            artifacts={"usd_asset_bundle": usd_ref},
            metrics=metrics,
        )
