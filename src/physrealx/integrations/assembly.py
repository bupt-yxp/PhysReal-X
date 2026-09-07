"""Assemble complete measured assets without inferring missing body physics."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    AssetEvidenceState,
    AssetSpec,
    BodyPhysicsSpec,
    DomainModel,
    GeometryRepresentation,
    GeometryRole,
    GeometrySpec,
    JointSpec,
    JointType,
    MethodCapability,
    OperationSpec,
    ParameterSource,
    ScalarParameter,
)
from physrealx.errors import DomainValidationError
from physrealx.evidence import ScientificEvidenceValidator
from physrealx.integrations._artifacts import input_artifact, publish, read_json
from physrealx.methods import builtin_method


class AssetAssemblyInput(DomainModel):
    schema_id: Literal["physrealx.asset-assembly-input/1"] = Field(
        default="physrealx.asset-assembly-input/1", alias="schema"
    )
    resource_id: str
    version: str
    display_name: str
    camera_rig: ArtifactRef
    articulation: ArtifactRef
    collision_reports: tuple[ArtifactRef, ...] = Field(min_length=1)
    body_physics: ArtifactRef
    joint_physics: tuple[ArtifactRef, ...] = Field(min_length=1)


def assembled_asset(document: Mapping[str, Any], artifacts: ArtifactStore) -> AssetSpec:
    """Validate assembly according to its declared evidence origin before packaging."""
    if document.get("evidence_scope") == "usd_reference_with_explicit_physics_priors":
        from physrealx.integrations.asset_prior import reference_prior_asset

        asset = reference_prior_asset(
            ArtifactRef.model_validate(document.get("source_input")), artifacts
        )
        if (
            asset != AssetSpec.model_validate(document.get("asset_spec"))
            or document.get("independent_physical_validation") is not False
        ):
            raise DomainValidationError(
                "reference assembly differs from its immutable prior inputs"
            )
        return asset
    return ScientificEvidenceValidator(artifacts).assembled_asset(document)


class AssetAssemblyAdapter:
    descriptor = builtin_method(
        MethodCapability.ASSET_ASSEMBLE,
        "measured_asset_assembly",
        input_contract="AssetAssemblyInput/1",
        output_contract="AssetAssemblyBundle/1",
    )

    def probe(self) -> MethodProbe:
        return MethodProbe(True, "Core asset graph and evidence validators available")

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.config:
            raise DomainValidationError("asset assembly has no implicit configuration")
        source = input_artifact(spec, "evidence", artifacts)
        request = AssetAssemblyInput.model_validate(
            read_json(artifacts, source, "physrealx.asset-assembly-input/1")
        )
        validator = ScientificEvidenceValidator(artifacts)
        camera = read_json(artifacts, request.camera_rig, "physrealx.camera-rig-calibration/1")
        if (
            camera.get("scale_status") != "resolved"
            or camera.get("coordinate_convention", {}).get("translation_unit") != "m"
        ):
            raise DomainValidationError("asset assembly requires resolved metric camera evidence")
        articulation = read_json(artifacts, request.articulation, "physrealx.articulation-model/1")
        if articulation.get("status") not in {"IDENTIFIED", "VALIDATED"}:
            raise DomainValidationError("articulation has not been identified")
        joints = [JointSpec.model_validate(j) for j in articulation.get("joints", [])]
        links = articulation.get("links", [])
        body_document = read_json(
            artifacts, request.body_physics, "physrealx.identified-body-physics/1"
        )
        if (
            body_document.get("status") not in {"IDENTIFIED", "VALIDATED"}
            or body_document.get("evidence_state") != "identified"
        ):
            raise DomainValidationError("body physics must be measured or identified")
        validator.validate_body_physics_evidence(body_document)
        bodies = []
        for body in body_document["bodies"]:
            converted = {
                field: ScalarParameter(
                    value=body[field]["value"],
                    source=ParameterSource(body[field]["source"]),
                    uncertainty=body[field]["uncertainty"],
                    confidence=body[field]["confidence"],
                )
                for field in ("mass_kg", "static_friction", "dynamic_friction", "restitution")
            }
            bodies.append(
                BodyPhysicsSpec(
                    link=body["link"],
                    center_of_mass_m=body["center_of_mass_m"],
                    inertia_kg_m2=body["inertia_kg_m2"],
                    **converted,
                )
            )
        estimates = {}
        for ref in request.joint_physics:
            estimate = read_json(artifacts, ref, "physrealx.identified-joint-physics/1")
            if estimate.get("status") != "IDENTIFIED":
                raise DomainValidationError("joint physics has not been identified")
            validator.validate_joint_physics_evidence(estimate)
            name = estimate["joint"]["name"]
            if name in estimates:
                raise DomainValidationError("duplicate joint physics evidence")
            estimates[name] = estimate
        movable = {joint.name for joint in joints if joint.joint_type is not JointType.FIXED}
        if set(estimates) != movable:
            raise DomainValidationError("identified physics must cover exactly the movable joints")
        assembled_joints = []
        for joint in joints:
            if joint.joint_type is JointType.FIXED:
                assembled_joints.append(joint)
                continue
            expected = "slide" if joint.joint_type is JointType.PRISMATIC else "hinge"
            estimate = estimates[joint.name]
            if estimate["joint"]["type"] != expected:
                raise DomainValidationError("joint physics type conflicts with articulation")
            values = estimate["parameters"]
            assembled_joints.append(
                joint.model_copy(
                    update={
                        "damping": values["viscous_damping"]["value"],
                        "friction": values["coulomb_friction"]["value"],
                    }
                )
            )
        geometries = []
        visual_meshes: list[dict[str, Any]] = []
        collision_meshes: list[dict[str, Any]] = []
        for ref in request.collision_reports:
            report = read_json(artifacts, ref, "physrealx.collision-build-report/1")
            if report.get("status") != "PASSED" or report.get("unit") != "m":
                raise DomainValidationError("collision report did not pass in metre units")
            validator.validate_collision_evidence(report)
            link = report["link"]
            for role, key, records in (
                (GeometryRole.VISUAL, "source_geometry", visual_meshes),
                (GeometryRole.COLLISION, "collision_geometry", collision_meshes),
            ):
                mesh = ArtifactRef.model_validate(report[key])
                geometries.append(
                    GeometrySpec(
                        name=f"{link}.{role.value}",
                        link=link,
                        role=role,
                        representation=GeometryRepresentation.MESH,
                        uri=f"artifact://{mesh.artifact_id}",
                    )
                )
                records.append(
                    {"link": link, "artifact": mesh.model_dump(mode="json"), "units": "meters"}
                )
        evidence = (
            request.camera_rig,
            request.body_physics,
            *request.joint_physics,
            *request.collision_reports,
            source,
        )
        asset = AssetSpec(
            resource_id=request.resource_id,
            version=request.version,
            display_name=request.display_name,
            root_link=articulation.get("root_link", links[0] if links else ""),
            links=tuple(links),
            geometries=tuple(geometries),
            joints=tuple(assembled_joints),
            physics=tuple(bodies),
            evidence_state=AssetEvidenceState.IDENTIFIED,
            evidence=(request.articulation, *evidence),
        )
        # Effective joint inertia is evidence for validation; it is never substituted for link mass.
        physics = artifacts.put_json(
            {
                "schema": "physrealx.identified-physics/1",
                "status": "IDENTIFIED",
                "evidence_state": "identified",
                "bodies": body_document["bodies"],
                "joints": [j.model_dump(mode="json") for j in assembled_joints],
            }
        )
        bundle = {
            "schema": "physrealx.asset-assembly-bundle/1",
            "resource_id": asset.resource_id,
            "version": asset.version,
            "root_link": asset.root_link,
            "up_axis": "Z",
            "metric_scale": {
                "status": "resolved",
                "meters_per_unit": 1.0,
                "evidence_artifact_id": request.camera_rig.artifact_id,
            },
            "visual_meshes": visual_meshes,
            "collision_meshes": collision_meshes,
            "articulation_model": request.articulation.model_dump(mode="json"),
            "identified_physics": physics.model_dump(mode="json"),
            "evidence": [ref.model_dump(mode="json") for ref in evidence],
            "asset_spec": asset.model_dump(mode="json"),
        }
        validator.assembled_asset(bundle)
        return publish(
            spec,
            artifacts,
            {"asset_assembly": bundle},
            metrics={"links": float(len(asset.links)), "joints": float(len(asset.joints))},
        )
