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
    articulated_collision_refinement: ArtifactRef | None = None
    registration: ArtifactRef | None = None

    @model_validator(mode="after")
    def one_prior(self) -> ReferenceAssemblyInput:
        if (self.physics_prior is None) == (self.predicted_physics is None):
            raise ValueError("assembly requires exactly one explicit or predicted prior")
        return self


def _refined_asset_version(version: str) -> str:
    """Publish collision-refined geometry as the next immutable patch version."""
    try:
        major, minor, patch = (int(part) for part in version.split("."))
    except (TypeError, ValueError) as error:
        raise DomainValidationError(
            "reference assembly version must use semantic versioning"
        ) from error
    return f"{major}.{minor}.{patch + 1}"


def _articulated_source(
    artifacts: ArtifactStore,
    request: ReferenceAssemblyInput,
    refined: dict[str, Any],
) -> dict[str, tuple[ArtifactRef, dict[str, Any]]]:
    """Bind a repair to its original canonical assembly and failed sweep."""
    assembly_ref = ArtifactRef.model_validate(refined.get("source_assembly"))
    assembly = read_json(artifacts, assembly_ref, "physrealx.asset-assembly-bundle/1")
    source_input = ArtifactRef.model_validate(assembly.get("source_input"))
    original = ReferenceAssemblyInput.model_validate(
        read_json(artifacts, source_input, "physrealx.reference-assembly-input/1")
    )
    mutable_fields = {"articulated_collision_refinement", "collision_reports"}
    if original.articulated_collision_refinement is not None or original.model_dump(
        exclude=mutable_fields
    ) != request.model_dump(exclude=mutable_fields):
        raise DomainValidationError("geometry repair source assembly changed immutable inputs")
    canonical = reference_prior_asset(source_input, artifacts)
    if (
        canonical != AssetSpec.model_validate(assembly.get("asset_spec"))
        or assembly.get("independent_physical_validation") is not False
    ):
        raise DomainValidationError("geometry repair source assembly differs from its inputs")
    feasibility = read_json(
        artifacts,
        ArtifactRef.model_validate(refined.get("source_feasibility")),
        "physrealx.physical-feasibility-report/1",
    )
    if (
        feasibility.get("source_assembly") != assembly_ref.model_dump(mode="json")
        or feasibility.get("status") not in {"PASSED", "FAILED"}
        or feasibility.get("physical_validation") is not False
        or feasibility.get("adjacent_contact_filter_disabled_for_audit") is not True
        or feasibility.get("locked_range") != refined.get("locked_range")
        or canonical.contact_exclusions
    ):
        raise DomainValidationError(
            "geometry refinement is not bound to its source collision sweep"
        )
    return {
        str(report["link"]): (ref, report)
        for ref in original.collision_reports
        for report in [read_json(artifacts, ref, "physrealx.collision-build-report/1")]
    }


def reference_prior_asset(source: ArtifactRef, artifacts: ArtifactStore) -> AssetSpec:
    """Recompute the canonical asset from immutable inputs, including its prior provenance."""
    import numpy as np

    request = ReferenceAssemblyInput.model_validate(
        read_json(artifacts, source, "physrealx.reference-assembly-input/1")
    )
    reference = read_json(artifacts, request.reference_model, "physrealx.usd-reference/1")
    reconstructed: dict[str, Any] = {}
    motion: dict[str, Any] = {}
    geometry_evidence: tuple[ArtifactRef, ...] = ()
    if request.reconstructed_geometry is not None:
        from physrealx.integrations.reconstructed_asset import reconstructed_motion

        reconstructed = read_json(
            artifacts,
            request.reconstructed_geometry,
            "physrealx.registered-reconstruction-geometry/4",
        )
        registration_document = (
            read_json(
                artifacts,
                request.registration,
                "physrealx.metric-alignment-report/1",
            )
            if request.registration is not None
            else {}
        )
        source_reconstruction = read_json(
            artifacts,
            ArtifactRef.model_validate(reconstructed.get("source_reconstruction")),
            "physrealx.geometry-reconstruction-report/1",
        )
        motion = reconstructed_motion(source_reconstruction, registration_document, reference)
        source_positions = {motion["joint"]: motion["positions"][0]}
        if (
            request.registration is None
            or reconstructed.get("source_reference")
            != request.reference_model.model_dump(mode="json")
            or reconstructed.get("source_registration")
            != request.registration.model_dump(mode="json")
            or reconstructed.get("coordinate_frame") != "registered_reconstruction_start"
            or set(reconstructed.get("geometries", {}))
            != {link["name"] for link in reference["links"]}
            or reconstructed.get("source_positions") != source_positions
            or reconstructed.get("reconstructed_motion") != motion
            or reconstructed.get("pose_policy") != "module_one_predicted_start_pose_locked"
            or reconstructed.get("geometry_policy") != "coordinate_transform_only"
            or any(
                not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 <= value <= 1e-9
                for value in reconstructed.get("source_pose_reprojection_error_m", {}).values()
            )
            or set(reconstructed.get("source_pose_reprojection_error_m", {}))
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
        from physrealx.integrations.physical_estimation import PhysicsPrediction

        predicted_document = read_json(
            artifacts, request.predicted_physics, "physrealx.visual-physics-prior/3"
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
        if predicted_document["reconstructed_motion"] != motion:
            raise DomainValidationError("predicted physics changed reconstructed motion")
        prediction = PhysicsPrediction.model_validate(predicted_document["prediction"])
        if set(prediction.bodies) != {v["name"] for v in reference["links"]} or set(
            prediction.joints
        ) != {motion["joint"]}:
            raise DomainValidationError("predicted physics must cover every link and joint")
        prediction_evidence = (request.predicted_physics,)
    refinement_evidence: tuple[ArtifactRef, ...] = ()
    if request.geometry_refinement is not None:
        refined = read_json(
            artifacts, request.geometry_refinement, "physrealx.video-geometry-refinement/3"
        )
        if request.registration is None:
            raise DomainValidationError("geometry refinement requires registered endpoints")
        if (
            refined["source_reference"] != request.reference_model.model_dump(mode="json")
            or refined.get("source_geometry")
            != (
                request.reconstructed_geometry.model_dump(mode="json")
                if request.reconstructed_geometry is not None
                else None
            )
            or refined["registration"] != request.registration.model_dump(mode="json")
            or refined["reconstructed_motion"] != motion
            or set(refined["geometries"]) != {link["name"] for link in reference["links"]}
        ):
            raise DomainValidationError("geometry refinement must preserve reference and motion")
        for link in reference["links"]:
            link["geometry"] = refined["geometries"][link["name"]]
        refinement_evidence = (request.geometry_refinement,)
    articulation_evidence: tuple[ArtifactRef, ...] = ()
    articulation_sources: dict[str, tuple[ArtifactRef, dict[str, Any]]] = {}
    refined_collision_links: set[str] = set()
    if request.articulated_collision_refinement is not None:
        refined = read_json(
            artifacts,
            request.articulated_collision_refinement,
            "physrealx.articulated-collision-refinement/4",
        )
        if request.registration is None:
            raise DomainValidationError("articulated collision refinement requires registration")
        expected_upstream = (
            request.geometry_refinement.model_dump(mode="json")
            if request.geometry_refinement is not None
            else None
        )
        original_geometries = {link["name"]: link["geometry"] for link in reference["links"]}
        refined_collision_links = set(refined.get("links", ()))
        expected_links = {motion["parent"], motion["child"]}
        if (
            refined.get("status") != "CANDIDATE"
            or refined.get("evidence_scope") != "collision_consistency_hypothesis"
            or refined.get("physical_validation") is not False
            or refined.get("independent_observation_validation") is not False
            or refined.get("selection_policy") != "adaptive_source_surface_support_refit"
            or refined.get("source_face_ownership_preserved") is not True
            or refined.get("visual_geometry_changed") is not False
            or refined.get("collision_geometry_changed") is not True
            or refined.get("joint_model_changed") is not False
            or refined.get("parent_child_collision_disabled") is not False
            or refined_collision_links != expected_links
            or refined.get("joint") != motion["joint"]
            or refined.get("locked_range") != motion["range"]
            or refined["source_reference"] != request.reference_model.model_dump(mode="json")
            or refined.get("source_geometry")
            != (
                request.reconstructed_geometry.model_dump(mode="json")
                if request.reconstructed_geometry is not None
                else None
            )
            or refined.get("upstream_geometry_refinement") != expected_upstream
            or refined["registration"] != request.registration.model_dump(mode="json")
            or refined["reconstructed_motion"] != motion
            or refined.get("visual_geometries") != original_geometries
            or set(refined.get("source_collision_reports", {})) != expected_links
            or set(refined.get("refined_collision_geometries", {})) != expected_links
        ):
            raise DomainValidationError("articulated collision refinement changed its sources")
        articulation_sources = _articulated_source(artifacts, request, refined)
        articulation_evidence = (request.articulated_collision_refinement,)
    if reference.get("unit") != "m" or reference.get("coordinate_frame") != "reference_z_up":
        raise DomainValidationError("reference assembly requires canonical metric Z-up geometry")
    links = {item["name"]: item for item in reference["links"]}
    if len(links) != len(reference["links"]):
        raise DomainValidationError("reference link names must be unique")
    joint_records = (
        [
            {
                "name": motion["joint"],
                "parent": motion["parent"],
                "child": motion["child"],
                "joint_type": motion["kind"],
                "axis": motion["axis"],
                "origin_m": motion["origin_m"],
                "lower_limit": motion["range"][0],
                "upper_limit": motion["range"][1],
            }
        ]
        if reconstructed
        else reference["joints"]
    )
    origins = {reference["root_link"]: np.zeros(3)}
    for joint in joint_records:
        if joint["child"] in origins:
            raise DomainValidationError("reference links cannot have multiple parent joints")
        origins[joint["child"]] = np.asarray(joint["origin_m"], dtype=float)
    if set(origins) != set(links):
        raise DomainValidationError("asset joints must cover every link")
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
        if articulation_sources:
            source_ref = articulation_sources[name][0]
            if name in refined_collision_links:
                if ref == source_ref or refinement is None:
                    raise DomainValidationError(
                        "refined articulated link requires new bound collision evidence"
                    )
            elif ref != source_ref or refinement is not None:
                raise DomainValidationError(
                    "collision refinement changed an unrelated link's evidence"
                )
        elif refinement is not None:
            raise DomainValidationError("collision refinement has no bound refinement document")
        if refinement is not None:
            source_report_ref = ArtifactRef.model_validate(
                refinement.get("source_collision_report")
            )
            source_report = read_json(
                artifacts, source_report_ref, "physrealx.collision-build-report/1"
            )
            if (
                name not in refined_collision_links
                or name not in articulation_sources
                or source_report_ref != articulation_sources[name][0]
                or refined.get("source_collision_reports", {}).get(name)
                != source_report_ref.model_dump(mode="json")
                or source_report.get("link") != name
                or source_report.get("status") != "PASSED"
                or refinement.get("schema") != "physrealx.articulated-collision-clearance/4"
                or refinement.get("source_assembly") != refined.get("source_assembly")
                or refinement.get("source_feasibility") != refined.get("source_feasibility")
                or refinement.get("joint") != refined.get("joint")
                or refinement.get("locked_range") != refined.get("locked_range")
                or source_report.get("source_geometry") != refinement.get("source_geometry")
                or refinement.get("source_geometry") != links[name]["geometry"]
                or request.articulated_collision_refinement is None
                or refinement.get("collision_refinement")
                != request.articulated_collision_refinement.model_dump(mode="json")
                or report.get("collision_geometry")
                != refined.get("refined_collision_geometries", {}).get(name)
                or refinement.get("visual_geometry_changed") is not False
                or refinement.get("collision_geometry_changed") is not True
                or refinement.get("joint_model_changed") is not False
                or refinement.get("parent_child_collision_disabled") is not False
            ):
                raise DomainValidationError("collision refinement changed immutable asset evidence")
        collision = load_meshes(artifacts, ArtifactRef.model_validate(report["collision_geometry"]))
        if any(not mesh.is_volume or not mesh.is_convex for mesh in collision):
            raise DomainValidationError("collision artifact does not contain closed convex parts")
        if refinement is not None:
            from physrealx.integrations._collision_refit import SurfaceCoverage, bind_surface
            from physrealx.integrations.articulated_geometry import (
                ArticulatedCollisionRefinementConfig,
            )

            config = ArticulatedCollisionRefinementConfig.model_validate(refined.get("config"))
            source_parts = load_meshes(
                artifacts,
                ArtifactRef.model_validate(articulation_sources[name][1]["collision_geometry"]),
                process=False,
            )
            source_visual = load_meshes(
                artifacts, ArtifactRef.model_validate(links[name]["geometry"]), process=False
            )
            if len(source_visual) != 1:
                raise DomainValidationError("surface fidelity requires one immutable visual mesh")
            from physrealx.integrations._collision_refit import SurfacePatch

            source_patches = bind_surface(source_visual[0], source_parts)
            failed = read_json(
                artifacts,
                ArtifactRef.model_validate(refined["source_feasibility"]),
                "physrealx.physical-feasibility-report/1",
            )
            contact_indices = {
                int(pair[f"geom_{side}"].rsplit("/", 1)[-1])
                for pair in failed["collision_pairs"]
                for side in ("a", "b")
                if pair[f"link_{side}"] == name
                and pair["max_penetration_m"] > failed["config"]["penetration_tolerance_m"]
            }
            audit = SurfaceCoverage(
                source_visual[0],
                source_patches,
                contact_indices,
                cell_size_m=config.patch_size_m,
                seed=config.seed,
            )
            observed, _ = audit.evaluate(
                [SurfacePatch(part, np.empty(0, dtype=int)) for part in collision],
                max_error_m=config.max_local_error_m,
                coverage_distance_m=config.coverage_distance_m,
                min_coverage=config.min_patch_coverage,
            )
            expected_fidelity = refined.get("fidelity", {}).get(name, {})
            numeric_agrees = all(
                type(expected_fidelity.get(key)) in (int, float)
                and np.isclose(value, expected_fidelity[key], rtol=1e-5, atol=1e-7)
                for key, value in observed.items()
                if key != "status"
            )
            if (
                not numeric_agrees
                or observed["status"] != "PASSED"
                or refined.get("fidelity", {}).get(name, {}).get("status") != "PASSED"
            ):
                raise DomainValidationError(
                    "refitted collision geometry fails local/contact surface coverage"
                )
        reports[name] = report
        collision_volumes[name] = sum(
            float(mesh.volume) for mesh in (source_parts if refinement is not None else collision)
        )
    if set(reports) != set(links):
        raise DomainValidationError("collision geometry must cover exactly the reference links")

    def prior(value: float) -> ScalarParameter:
        return ScalarParameter(
            value=value, source=ParameterSource.DEFAULT, uncertainty=abs(value), confidence=0.0
        )

    # Registered reconstruction meshes share the module-one start frame. A child link
    # frame is joint-centred, so -origin on geometry plus parent_to_joint exactly
    # reproduces the predicted pivot without changing a vertex or the start pose.
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
    for joint in joint_records:
        lower_limit, upper_limit = joint["lower_limit"], joint["upper_limit"]
        if prediction is not None:
            candidate = prediction.joints[joint["name"]]
            damping, friction, effort = (
                candidate.damping.value,
                candidate.friction.value,
                candidate.effort_limit.value,
            )
        else:
            assert request.physics_prior is not None
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
        version=(
            _refined_asset_version(request.version)
            if request.articulated_collision_refinement is not None
            else request.version
        ),
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
                reconstructed.get("geometry_source", "real_artgs_reconstruction")
                + "_with_refined_collision_proxy"
                if request.articulated_collision_refinement is not None
                and request.reconstructed_geometry is not None
                else reconstructed.get("geometry_source", "real_artgs_reconstruction")
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
            "joint_limit_source": (
                "module_one_reconstruction_start_end" if reconstructed else "reference_model"
            ),
            "reconstructed_motion": motion if reconstructed else {},
        },
    )


class ReferenceAssemblyAdapter:
    descriptor = builtin_method(
        MethodCapability.ASSET_ASSEMBLE,
        "reference_asset_assembly",
        version="3.0.0",
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
