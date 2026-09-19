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
    reconstructed_geometry: ArtifactRef | None = None
    collision_reports: tuple[ArtifactRef, ...] = Field(min_length=1)
    physics_prior: PhysicsPrior | None = None
    predicted_physics: ArtifactRef | None = None
    geometry_refinement: ArtifactRef | None = None
    articulated_geometry_refinement: ArtifactRef | None = None
    registration: ArtifactRef | None = None

    @model_validator(mode="after")
    def one_prior(self) -> ReferenceAssemblyInput:
        if (self.physics_prior is None) == (self.predicted_physics is None):
            raise ValueError("assembly requires exactly one explicit or predicted prior")
        return self


def reference_prior_asset(source: ArtifactRef, artifacts: ArtifactStore) -> AssetSpec:
    """Recompute the canonical asset from immutable inputs, including its prior provenance."""
    import numpy as np

    request = ReferenceAssemblyInput.model_validate(
        read_json(artifacts, source, "physrealx.reference-assembly-input/1")
    )
    reference = read_json(artifacts, request.reference_model, "physrealx.usd-reference/1")
    geometry_evidence: tuple[ArtifactRef, ...] = ()
    if request.reconstructed_geometry is not None:
        reconstructed = read_json(
            artifacts,
            request.reconstructed_geometry,
            "physrealx.registered-reconstruction-geometry/1",
        )
        if (
            request.registration is None
            or reconstructed.get("source_reference")
            != request.reference_model.model_dump(mode="json")
            or reconstructed.get("source_registration")
            != request.registration.model_dump(mode="json")
            or reconstructed.get("coordinate_frame") != "reference_link_rest"
            or set(reconstructed.get("geometries", {}))
            != {link["name"] for link in reference["links"]}
        ):
            raise DomainValidationError("reconstructed geometry is bound to another source")
        for link in reference["links"]:
            link["geometry"] = reconstructed["geometries"][link["name"]]
        geometry_evidence = (request.reconstructed_geometry,)
    prediction = None
    predicted_document: dict[str, Any] = {}
    prediction_evidence: tuple[ArtifactRef, ...] = ()
    if request.predicted_physics is not None:
        from physrealx.integrations.physical_estimation import PhysicsPrediction, locked_motion

        predicted_document = read_json(
            artifacts, request.predicted_physics, "physrealx.visual-physics-prior/1"
        )
        if (
            predicted_document["source_reference"]
            != request.reference_model.model_dump(mode="json")
            or predicted_document.get("source_geometry")
            != (
                request.reconstructed_geometry.model_dump(mode="json")
                if request.reconstructed_geometry is not None
                else None
            )
            or request.registration is None
            or predicted_document["registration"] != request.registration.model_dump(mode="json")
            or predicted_document.get("evidence_state") != "prior"
        ):
            raise DomainValidationError("predicted physics is bound to another source")
        registration = read_json(
            artifacts, request.registration, "physrealx.metric-alignment-report/1"
        )
        motion = locked_motion(reference, registration)
        if predicted_document["motion"] != motion:
            raise DomainValidationError("predicted physics changed the observed endpoints")
        prediction = PhysicsPrediction.model_validate(predicted_document["prediction"])
        if set(prediction.bodies) != {v["name"] for v in reference["links"]} or set(
            prediction.joints
        ) != {v["name"] for v in reference["joints"]}:
            raise DomainValidationError("predicted physics must cover every link and joint")
        prediction_evidence = (request.predicted_physics,)
    refinement_evidence: tuple[ArtifactRef, ...] = ()
    if request.geometry_refinement is not None:
        from physrealx.integrations.physical_estimation import locked_motion

        refined = read_json(
            artifacts, request.geometry_refinement, "physrealx.video-geometry-refinement/1"
        )
        if request.registration is None:
            raise DomainValidationError("geometry refinement requires registered endpoints")
        registration = read_json(
            artifacts, request.registration, "physrealx.metric-alignment-report/1"
        )
        if (
            refined["source_reference"] != request.reference_model.model_dump(mode="json")
            or refined.get("source_geometry")
            != (
                request.reconstructed_geometry.model_dump(mode="json")
                if request.reconstructed_geometry is not None
                else None
            )
            or refined["registration"] != request.registration.model_dump(mode="json")
            or refined["motion"] != locked_motion(reference, registration)
            or set(refined["geometries"]) != {link["name"] for link in reference["links"]}
        ):
            raise DomainValidationError("geometry refinement must preserve reference and motion")
        for link in reference["links"]:
            link["geometry"] = refined["geometries"][link["name"]]
        refinement_evidence = (request.geometry_refinement,)
    articulation_evidence: tuple[ArtifactRef, ...] = ()
    if request.articulated_geometry_refinement is not None:
        from physrealx.integrations.physical_estimation import locked_motion

        refined = read_json(
            artifacts,
            request.articulated_geometry_refinement,
            "physrealx.articulated-geometry-refinement/1",
        )
        if request.registration is None:
            raise DomainValidationError("articulated geometry refinement requires registration")
        registration = read_json(
            artifacts, request.registration, "physrealx.metric-alignment-report/1"
        )
        expected_upstream = (
            request.geometry_refinement.model_dump(mode="json")
            if request.geometry_refinement is not None
            else None
        )
        if (
            refined.get("status") != "PASSED"
            or refined.get("visual_geometry_changed") is not True
            or refined.get("joint_model_changed") is not False
            or refined["source_reference"] != request.reference_model.model_dump(mode="json")
            or refined.get("source_geometry")
            != (
                request.reconstructed_geometry.model_dump(mode="json")
                if request.reconstructed_geometry is not None
                else None
            )
            or refined.get("upstream_geometry_refinement") != expected_upstream
            or refined["registration"] != request.registration.model_dump(mode="json")
            or refined["motion"] != locked_motion(reference, registration)
            or set(refined["geometries"]) != {link["name"] for link in reference["links"]}
        ):
            raise DomainValidationError("articulated geometry refinement changed its sources")
        for link in reference["links"]:
            link["geometry"] = refined["geometries"][link["name"]]
        articulation_evidence = (request.articulated_geometry_refinement,)
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
        refinement = report.get("articulation_refinement")
        if refinement is not None:
            source_report_ref = ArtifactRef.model_validate(
                refinement.get("source_collision_report")
            )
            source_report = read_json(
                artifacts, source_report_ref, "physrealx.collision-build-report/1"
            )
            if (
                source_report.get("link") != name
                or source_report.get("source_geometry")
                != refinement.get("source_geometry_before")
                or refinement.get("source_geometry_after") != links[name]["geometry"]
                or request.articulated_geometry_refinement is None
                or refinement.get("geometry_refinement")
                != request.articulated_geometry_refinement.model_dump(mode="json")
                or refinement.get("visual_geometry_changed") is not True
                or refinement.get("joint_model_changed") is not False
                or refinement.get("parent_child_collision_disabled") is not False
            ):
                raise DomainValidationError("collision refinement changed immutable asset evidence")
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

    # ``reference_link_rest`` keeps every unposed part in the shared reference frame;
    # it does not recenter vertices on a joint. A child link frame is joint-centred, so
    # the same -origin transform is required for either geometry source. Together with
    # parent_to_joint this exactly reproduces the pivot transform used by registration.
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
        body_prediction = prediction.bodies[name] if prediction is not None else None
        if body_prediction is not None:
            mass = body_prediction.mass_kg.value
        else:
            assert request.physics_prior is not None
            mass = volume * request.physics_prior.density_kg_m3

        def body_parameter(key: str, prediction: Any = body_prediction) -> ScalarParameter:
            if prediction is None:
                assert request.physics_prior is not None
                return prior(float(getattr(request.physics_prior, key)))
            estimate = getattr(prediction, key)
            return ScalarParameter(
                value=estimate.value,
                source=ParameterSource.INFERRED,
                confidence=0.0,
                uncertainty=max(estimate.value - estimate.lower, estimate.upper - estimate.value),
            )

        moments = mass * (np.sum(dimensions**2) - dimensions**2) / 12
        bodies.append(
            BodyPhysicsSpec(
                link=name,
                mass_kg=body_parameter("mass_kg") if body_prediction is not None else prior(mass),
                center_of_mass_m=tuple((lower + upper) / 2 - origins[name]),
                inertia_kg_m2=tuple(tuple(row) for row in np.diag(moments)),
                static_friction=body_parameter("static_friction"),
                dynamic_friction=body_parameter("dynamic_friction"),
                restitution=body_parameter("restitution"),
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
        if prediction is not None:
            candidate = prediction.joints[joint["name"]]
            lower_limit, upper_limit = predicted_document["motion"]["range"]
            damping, friction, effort = (
                candidate.damping.value,
                candidate.friction.value,
                candidate.effort_limit.value,
            )
        else:
            assert request.physics_prior is not None
            lower_limit, upper_limit = joint["lower_limit"], joint["upper_limit"]
            damping, friction, effort = (
                request.physics_prior.joint_damping,
                request.physics_prior.joint_friction,
                request.physics_prior.effort_limit,
            )
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
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                damping=damping,
                friction=friction,
                effort_limit=effort,
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
        if request.physics_prior is not None and request.physics_prior.joint_models_adjacent_contact
        else (),
        evidence_state=AssetEvidenceState.PRIOR,
        evidence=(
            source,
            request.reference_model,
            *request.collision_reports,
            *geometry_evidence,
            *appearance_evidence,
            *prediction_evidence,
            *refinement_evidence,
            *articulation_evidence,
        ),
        metadata={
            **appearance,
            "evidence_scope": (
                "real_reconstruction_geometry_with_registered_topology_and_physics_priors"
                if request.reconstructed_geometry is not None
                else "usd_reference_with_explicit_physics_priors"
            ),
            "independent_physical_validation": False,
            "geometry_source": (
                "articulation_refined_real_reconstruction"
                if request.articulated_geometry_refinement is not None
                else "real_artgs_reconstruction"
                if request.reconstructed_geometry is not None
                else "reference_model"
            ),
            "mass_model": "vlm_visual_prior"
            if prediction is not None
            else "convex_collision_volume_sum_times_assumed_density",
            "inertia_model": "uniform_bounding_box_at_assumed_mass",
            "physics_prior": request.predicted_physics.model_dump(mode="json")
            if request.predicted_physics is not None
            else request.physics_prior.model_dump(mode="json")
            if request.physics_prior
            else {},
        },
    )


class ReferenceAssemblyAdapter:
    descriptor = builtin_method(
        MethodCapability.ASSET_ASSEMBLE,
        "reference_asset_assembly",
        version="2.2.0",
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
                    "evidence_scope": asset.metadata["evidence_scope"],
                    "source_input": source.model_dump(mode="json"),
                    "asset_spec": asset.model_dump(mode="json"),
                    "independent_physical_validation": False,
                }
            },
            metrics={"links": float(len(asset.links)), "joints": float(len(asset.joints))},
        )
