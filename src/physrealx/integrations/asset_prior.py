"""Construct simulation assets from supplied geometry and explicit, unmeasured priors."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

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
    PoseSpec,
    ScalarParameter,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import dependency_probe, input_artifact, publish, read_json
from physrealx.integrations.collision import load_meshes
from physrealx.methods import builtin_method


class PhysicsPrior(DomainModel):
    density_kg_m3: float = Field(gt=0, allow_inf_nan=False)
    static_friction: float = Field(ge=0, allow_inf_nan=False)
    dynamic_friction: float = Field(ge=0, allow_inf_nan=False)
    restitution: float = Field(ge=0, le=1, allow_inf_nan=False)
    joint_damping: float = Field(ge=0, allow_inf_nan=False)
    joint_friction: float = Field(ge=0, allow_inf_nan=False)
    effort_limit: float = Field(gt=0, allow_inf_nan=False)
    joint_models_adjacent_contact: bool

    @model_validator(mode="after")
    def validate_friction(self) -> PhysicsPrior:
        if self.dynamic_friction > self.static_friction:
            raise ValueError("dynamic friction cannot exceed static friction")
        return self


class ReferenceAssemblyInput(DomainModel):
    schema_id: Literal["physrealx.reference-assembly-input/1"] = Field(
        default="physrealx.reference-assembly-input/1", alias="schema"
    )
    resource_id: str
    version: str
    display_name: str
    reference_model: ArtifactRef
    collision_reports: tuple[ArtifactRef, ...] = Field(min_length=1)
    physics_prior: PhysicsPrior
    registration: ArtifactRef | None = None


def reference_prior_asset(source: ArtifactRef, artifacts: ArtifactStore) -> AssetSpec:
    """Recompute the canonical asset from immutable inputs, including its prior provenance."""
    import numpy as np

    request = ReferenceAssemblyInput.model_validate(
        read_json(artifacts, source, "physrealx.reference-assembly-input/1")
    )
    reference = read_json(artifacts, request.reference_model, "physrealx.usd-reference/1")
    if reference.get("unit") != "m" or reference.get("coordinate_frame") != "reference_z_up":
        raise DomainValidationError("reference assembly requires canonical metric Z-up geometry")
    links = {item["name"]: item for item in reference["links"]}
    if len(links) != len(reference["links"]):
        raise DomainValidationError("reference link names must be unique")
    origins = {reference["root_link"]: np.zeros(3)}
    for joint in reference["joints"]:
        if joint["child"] in origins:
            raise DomainValidationError("reference links cannot have multiple parent joints")
        origins[joint["child"]] = np.asarray(joint["origin_m"], dtype=float)
    if set(origins) != set(links):
        raise DomainValidationError("reference joints must cover every link")
    reports: dict[str, Any] = {}
    collision_volumes: dict[str, float] = {}
    for ref in request.collision_reports:
        report = read_json(artifacts, ref, "physrealx.collision-build-report/1")
        name = report.get("link")
        if not isinstance(name, str) or name not in links or name in reports:
            raise DomainValidationError("collision reports must bind unique reference links")
        if (
            report.get("status") != "PASSED"
            or report.get("unit") != "m"
            or report["source_geometry"] != links[name]["geometry"]
        ):
            raise DomainValidationError("collision evidence differs from the supplied visual mesh")
        collision = load_meshes(artifacts, ArtifactRef.model_validate(report["collision_geometry"]))
        if any(not mesh.is_volume or not mesh.is_convex for mesh in collision):
            raise DomainValidationError("collision artifact does not contain closed convex parts")
        reports[name] = report
        collision_volumes[name] = sum(float(mesh.volume) for mesh in collision)
    if set(reports) != set(links):
        raise DomainValidationError("collision geometry must cover exactly the reference links")

    def prior(value: float) -> ScalarParameter:
        return ScalarParameter(
            value=value, source=ParameterSource.DEFAULT, uncertainty=abs(value), confidence=0.0
        )

    geometries, bodies = [], []
    for name, link in links.items():
        visual = load_meshes(artifacts, ArtifactRef.model_validate(link["geometry"]))
        vertices = np.concatenate([mesh.vertices for mesh in visual])
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        dimensions = upper - lower
        # This is an explicit shape-and-density prior, never a measured body tensor.
        volume = collision_volumes[name]
        if volume <= 0 or not np.isfinite(volume) or np.any(dimensions <= 0):
            raise DomainValidationError("collision proxy has no finite positive volume")
        mass = volume * request.physics_prior.density_kg_m3
        moments = mass * (np.sum(dimensions**2) - dimensions**2) / 12
        bodies.append(
            BodyPhysicsSpec(
                link=name,
                mass_kg=prior(mass),
                center_of_mass_m=tuple((lower + upper) / 2 - origins[name]),
                inertia_kg_m2=tuple(tuple(row) for row in np.diag(moments)),
                static_friction=prior(request.physics_prior.static_friction),
                dynamic_friction=prior(request.physics_prior.dynamic_friction),
                restitution=prior(request.physics_prior.restitution),
            )
        )
        for role, key in (
            (GeometryRole.VISUAL, "source_geometry"),
            (GeometryRole.COLLISION, "collision_geometry"),
        ):
            ref = ArtifactRef.model_validate(reports[name][key])
            geometries.append(
                GeometrySpec(
                    name=f"{name}.{role.value}",
                    link=name,
                    role=role,
                    representation=GeometryRepresentation.MESH,
                    uri=f"artifact://{ref.artifact_id}",
                    local_pose=PoseSpec(translation_m=tuple(-origins[name])),
                )
            )
    joints = []
    for joint in reference["joints"]:
        kind = {"slide": JointType.PRISMATIC, "hinge": JointType.REVOLUTE}.get(joint["joint_type"])
        if kind is None:
            raise DomainValidationError("reference joint type must be slide or hinge")
        joints.append(
            JointSpec(
                name=joint["name"],
                parent=joint["parent"],
                child=joint["child"],
                joint_type=kind,
                parent_to_joint=PoseSpec(
                    translation_m=tuple(origins[joint["child"]] - origins[joint["parent"]])
                ),
                axis=joint["axis"],
                lower_limit=joint["lower_limit"],
                upper_limit=joint["upper_limit"],
                damping=request.physics_prior.joint_damping,
                friction=request.physics_prior.joint_friction,
                effort_limit=request.physics_prior.effort_limit,
            )
        )
    appearance = {}
    appearance_evidence: tuple[ArtifactRef, ...] = ()
    if request.registration is not None:
        from physrealx.integrations.registration import load_registration

        registered = load_registration(artifacts, request.registration)
        if registered["source_reference"] != request.reference_model.model_dump(mode="json"):
            raise DomainValidationError("appearance registration is bound to another USD reference")
        appearance = {"registered_appearance": request.registration.model_dump(mode="json")}
        appearance_evidence = (request.registration,)
    return AssetSpec(
        resource_id=request.resource_id,
        version=request.version,
        display_name=request.display_name,
        root_link=reference["root_link"],
        links=tuple(links),
        geometries=tuple(geometries),
        joints=tuple(joints),
        physics=tuple(bodies),
        contact_exclusions=tuple((j.parent, j.child) for j in joints)
        if request.physics_prior.joint_models_adjacent_contact
        else (),
        evidence_state=AssetEvidenceState.PRIOR,
        evidence=(
            source,
            request.reference_model,
            *request.collision_reports,
            *appearance_evidence,
        ),
        metadata={
            **appearance,
            "evidence_scope": "usd_reference_with_explicit_physics_priors",
            "independent_physical_validation": False,
            "mass_model": "convex_collision_volume_sum_times_assumed_density",
            "inertia_model": "uniform_bounding_box_at_assumed_mass",
            "physics_prior": request.physics_prior.model_dump(mode="json"),
        },
    )


class ReferenceAssemblyAdapter:
    descriptor = builtin_method(
        MethodCapability.ASSET_ASSEMBLE,
        "reference_asset_assembly",
        version="2.1.0",
        input_contract="AssetAssemblyInput/1",
        output_contract="AssetAssemblyBundle/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe({"numpy": ("numpy", "1.26.4"), "trimesh": ("trimesh", "4.5.1")})

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.config:
            raise DomainValidationError(
                "reference assembly parameters belong in its immutable input"
            )
        source = input_artifact(spec, "evidence", artifacts)
        asset = reference_prior_asset(source, artifacts)
        return publish(
            spec,
            artifacts,
            {
                "asset_assembly": {
                    "schema": "physrealx.asset-assembly-bundle/1",
                    "evidence_scope": "usd_reference_with_explicit_physics_priors",
                    "source_input": source.model_dump(mode="json"),
                    "asset_spec": asset.model_dump(mode="json"),
                    "independent_physical_validation": False,
                }
            },
            metrics={"links": float(len(asset.links)), "joints": float(len(asset.joints))},
        )
