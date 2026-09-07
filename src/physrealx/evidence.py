"""Scientific evidence validation shared by readiness gates and executable adapters."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, AssetEvidenceState, AssetSpec, GeometryRole, PoseSpec
from physrealx.robot_agent import RobotInterfaceSpec

_MAX_EVIDENCE_JSON_BYTES = 64 * 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = member
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    if len(payload) > _MAX_EVIDENCE_JSON_BYTES:
        raise ValueError(f"{label} exceeds the evidence JSON size limit")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{label} is not unambiguous finite UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    pending: list[object] = [document]
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return document


class ScientificEvidenceValidator:
    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    @staticmethod
    def _artifact_reference(value: object, label: str) -> ArtifactRef:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an ArtifactRef object")
        try:
            return ArtifactRef.model_validate(value)
        except ValueError as error:
            raise ValueError(f"{label} must be a valid ArtifactRef") from error

    @staticmethod
    def _artifact_id(value: object, label: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be an artifact identifier")
        try:
            return ArtifactRef(
                artifact_id=value,
                size_bytes=0,
                media_type="application/octet-stream",
            ).artifact_id
        except ValueError as error:
            raise ValueError(f"{label} must be an artifact identifier") from error

    def _require_available_artifacts(
        self,
        references: list[ArtifactRef],
        label: str,
    ) -> None:
        missing = sorted(
            reference.artifact_id
            for reference in references
            if not self.artifacts.verify(reference)
        )
        if missing:
            raise ValueError(f"{label} has unavailable artifacts: {', '.join(missing)}")

    def validate_geometry_evidence(self, document: Mapping[str, Any]) -> None:
        self._artifact_id(document.get("source_camera_artifact_id"), "source_camera_artifact_id")
        self._artifact_id(document.get("source_depth_artifact_id"), "source_depth_artifact_id")
        method = document.get("method")
        if not isinstance(method, dict) or any(
            not isinstance(method.get(field), str) or not method[field].strip()
            for field in ("name", "version", "source_revision")
        ):
            raise ValueError("geometry report method identity must be complete")
        gates = document.get("gates")
        if (
            not isinstance(gates, dict)
            or not gates
            or any(value is not True for value in gates.values())
        ):
            raise ValueError("geometry report must pass every declared gate")
        metrics = document.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not metrics
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
        ):
            raise ValueError("geometry report metrics must be non-empty and finite")
        raw_meshes = document.get("meshes")
        if not isinstance(raw_meshes, list) or not raw_meshes:
            raise ValueError("geometry report meshes must be a non-empty array")
        links: set[str] = set()
        references: list[ArtifactRef] = []
        for index, raw in enumerate(raw_meshes):
            if not isinstance(raw, dict):
                raise ValueError(f"meshes[{index}] must be an object")
            link = raw.get("link")
            if not isinstance(link, str) or not link.strip() or link in links:
                raise ValueError("geometry report mesh links must be non-empty and unique")
            links.add(link)
            reference = self._artifact_reference(raw.get("artifact"), f"meshes[{index}].artifact")
            if not reference.media_type.startswith("model/"):
                raise ValueError("geometry report meshes must reference model/* artifacts")
            references.append(reference)
        if len({reference.artifact_id for reference in references}) != len(references):
            raise ValueError("geometry report mesh artifact IDs must be unique")
        self._require_available_artifacts(references, "geometry report")

    def validate_collision_evidence(self, document: Mapping[str, Any]) -> None:
        link = document.get("link")
        if not isinstance(link, str) or not link.strip():
            raise ValueError("collision report link must be non-empty")
        source = self._artifact_reference(
            document.get("source_geometry"),
            "collision report source_geometry",
        )
        collision = self._artifact_reference(
            document.get("collision_geometry"),
            "collision report collision_geometry",
        )
        if not source.media_type.startswith("model/"):
            raise ValueError("collision report source_geometry must be a model/* artifact")
        if collision.media_type != "model/gltf-binary":
            raise ValueError("collision report collision_geometry must be a GLB artifact")
        if source.artifact_id == collision.artifact_id:
            raise ValueError("collision geometry must be distinct from its visual source mesh")
        method = document.get("method")
        if not isinstance(method, dict) or any(
            not isinstance(method.get(field), str) or not method[field].strip()
            for field in ("name", "version", "source_revision")
        ):
            raise ValueError("collision report method identity must be complete")
        gates = document.get("gates")
        if (
            not isinstance(gates, dict)
            or not gates
            or any(value is not True for value in gates.values())
        ):
            raise ValueError("collision report must pass every declared gate")
        metrics = document.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not metrics
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
        ):
            raise ValueError("collision report metrics must be non-empty and finite")
        self._require_available_artifacts([source, collision], "collision report")

    def _identified_parameter(
        self,
        value: object,
        label: str,
        *,
        unit: str,
        minimum: float,
        maximum: float | None = None,
    ) -> tuple[float, str]:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an identified parameter object")
        uncertainty = value.get("uncertainty")
        confidence = value.get("confidence")
        if (
            isinstance(uncertainty, bool)
            or not isinstance(uncertainty, (int, float))
            or not math.isfinite(uncertainty)
            or uncertainty < 0
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or value.get("source") not in {"measured", "inferred"}
        ):
            raise ValueError(
                f"{label} requires explicit uncertainty, confidence and measured/inferred source"
            )
        raw_number = value.get("value")
        if (
            isinstance(raw_number, bool)
            or not isinstance(raw_number, (int, float))
            or not math.isfinite(float(raw_number))
        ):
            raise ValueError(f"{label}.value must be finite")
        number = float(raw_number)
        if number < minimum or (maximum is not None and number > maximum):
            raise ValueError(f"{label}.value is outside its physical range")
        if value.get("unit") != unit:
            raise ValueError(f"{label}.unit must be {unit!r}")
        evidence_artifact_id = self._artifact_id(
            value.get("evidence_artifact_id"), f"{label}.evidence_artifact_id"
        )
        return number, evidence_artifact_id

    @staticmethod
    def _finite_vector(value: object, label: str, length: int) -> tuple[float, ...]:
        if (
            not isinstance(value, list)
            or len(value) != length
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in value
            )
        ):
            raise ValueError(f"{label} must contain {length} finite values")
        return tuple(float(item) for item in value)

    def validate_body_physics_evidence(self, document: Mapping[str, Any]) -> None:
        raw_bodies = document.get("bodies")
        if not isinstance(raw_bodies, list) or not raw_bodies:
            raise ValueError("body physics must contain at least one identified body")
        links: set[str] = set()
        for index, body in enumerate(raw_bodies):
            if not isinstance(body, dict):
                raise ValueError(f"bodies[{index}] must be an object")
            link = body.get("link")
            if not isinstance(link, str) or not link.strip() or link in links:
                raise ValueError("body physics links must be non-empty and unique")
            links.add(link)
            label = f"bodies[{index}]"
            body_artifact_id = self._artifact_id(
                body.get("evidence_artifact_id"), f"{label}.evidence_artifact_id"
            )
            _, mass_artifact_id = self._identified_parameter(
                body.get("mass_kg"),
                f"{label}.mass_kg",
                unit="kg",
                minimum=1e-12,
            )
            self._finite_vector(body.get("center_of_mass_m"), f"{label}.center_of_mass_m", 3)
            inertia_rows = body.get("inertia_kg_m2")
            if not isinstance(inertia_rows, list) or len(inertia_rows) != 3:
                raise ValueError(f"{label}.inertia_kg_m2 must be a 3x3 matrix")
            inertia = tuple(
                self._finite_vector(row, f"{label}.inertia_kg_m2[{row_index}]", 3)
                for row_index, row in enumerate(inertia_rows)
            )
            if any(
                abs(inertia[row][column] - inertia[column][row]) > 1e-9
                for row in range(3)
                for column in range(3)
            ):
                raise ValueError(f"{label}.inertia_kg_m2 must be symmetric")
            first_minor = inertia[0][0]
            second_minor = inertia[0][0] * inertia[1][1] - inertia[0][1] * inertia[1][0]
            determinant = (
                inertia[0][0] * (inertia[1][1] * inertia[2][2] - inertia[1][2] * inertia[2][1])
                - inertia[0][1] * (inertia[1][0] * inertia[2][2] - inertia[1][2] * inertia[2][0])
                + inertia[0][2] * (inertia[1][0] * inertia[2][1] - inertia[1][1] * inertia[2][0])
            )
            if first_minor <= 0.0 or second_minor <= 0.0 or determinant <= 0.0:
                raise ValueError(f"{label}.inertia_kg_m2 must be positive definite")
            static_friction, static_artifact_id = self._identified_parameter(
                body.get("static_friction"),
                f"{label}.static_friction",
                unit="1",
                minimum=0.0,
            )
            dynamic_friction, dynamic_artifact_id = self._identified_parameter(
                body.get("dynamic_friction"),
                f"{label}.dynamic_friction",
                unit="1",
                minimum=0.0,
            )
            _, restitution_artifact_id = self._identified_parameter(
                body.get("restitution"),
                f"{label}.restitution",
                unit="1",
                minimum=0.0,
                maximum=1.0,
            )
            if dynamic_friction > static_friction:
                raise ValueError(f"{label} dynamic friction cannot exceed static friction")
            if {
                mass_artifact_id,
                static_artifact_id,
                dynamic_artifact_id,
                restitution_artifact_id,
            } != {body_artifact_id}:
                raise ValueError(f"{label} parameters must share its measured evidence artifact ID")
            if not self.artifacts.verify(body_artifact_id):
                raise ValueError(f"{label} measured evidence artifact ID is unavailable")

    def validate_joint_physics_evidence(self, document: Mapping[str, Any]) -> None:
        joint = document.get("joint")
        parameters = document.get("parameters")
        if not isinstance(joint, dict):
            raise ValueError("joint physics must identify one joint")
        if not isinstance(joint.get("name"), str) or not joint["name"].strip():
            raise ValueError("joint physics joint.name must be non-empty")
        if joint.get("type") not in {"hinge", "slide"}:
            raise ValueError("joint physics joint.type must be hinge or slide")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("joint physics parameters must be non-empty")
        evidence_artifact_id = self._artifact_id(
            document.get("evidence_artifact_id"),
            "joint physics evidence_artifact_id",
        )
        if not self.artifacts.verify(evidence_artifact_id):
            raise ValueError("joint physics measured evidence artifact ID is unavailable")
        expected_units = (
            {
                "effective_inertia": "kg",
                "viscous_damping": "N*s/m",
                "coulomb_friction": "N",
            }
            if joint["type"] == "slide"
            else {
                "effective_inertia": "kg*m^2",
                "viscous_damping": "N*m*s/rad",
                "coulomb_friction": "N*m",
            }
        )
        if set(parameters) != set(expected_units):
            raise ValueError("joint physics must identify inertia, damping and Coulomb friction")
        for name, unit in expected_units.items():
            parameter = parameters[name]
            if not isinstance(parameter, dict):
                raise ValueError(f"joint physics parameter {name!r} must be an object")
            raw_value = parameter.get("value")
            raw_spread = parameter.get("multi_start_spread")
            confidence = parameter.get("confidence_interval")
            minimum = 1e-12 if name == "effective_inertia" else 0.0
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
                or float(raw_value) < minimum
                or parameter.get("unit") != unit
                or isinstance(raw_spread, bool)
                or not isinstance(raw_spread, (int, float))
                or not math.isfinite(float(raw_spread))
                or float(raw_spread) < 0.0
                or not isinstance(confidence, dict)
            ):
                raise ValueError(f"joint physics parameter {name!r} is not a finite estimate")
            lower = confidence.get("lower")
            upper = confidence.get("upper")
            level = confidence.get("level")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in (lower, upper, level)
            ):
                raise ValueError(f"joint physics parameter {name!r} confidence interval is invalid")
            assert isinstance(lower, (int, float)) and not isinstance(lower, bool)
            assert isinstance(upper, (int, float)) and not isinstance(upper, bool)
            assert isinstance(level, (int, float)) and not isinstance(level, bool)
            if (
                not 0.0 < float(level) < 1.0
                or float(lower) > float(raw_value)
                or float(raw_value) > float(upper)
            ):
                raise ValueError(f"joint physics parameter {name!r} confidence interval is invalid")

    def validate_asset_assembly(self, document: Mapping[str, Any]) -> None:
        for field in ("resource_id", "version", "root_link"):
            if not isinstance(document.get(field), str) or not document[field].strip():
                raise ValueError(f"asset assembly {field} must be non-empty")
        if document.get("up_axis") not in {"Y", "Z"}:
            raise ValueError("asset assembly up_axis must be Y or Z")
        metric_scale = document.get("metric_scale")
        if not isinstance(metric_scale, dict) or metric_scale.get("status") != "resolved":
            raise ValueError("asset assembly metric_scale must be resolved")
        meters_per_unit = metric_scale.get("meters_per_unit")
        if (
            isinstance(meters_per_unit, bool)
            or not isinstance(meters_per_unit, (int, float))
            or not math.isfinite(float(meters_per_unit))
            or float(meters_per_unit) <= 0.0
        ):
            raise ValueError("asset assembly meters_per_unit must be positive and finite")
        scale_artifact_id = self._artifact_id(
            metric_scale.get("evidence_artifact_id"),
            "asset assembly metric scale evidence_artifact_id",
        )
        references: list[ArtifactRef] = []
        geometry_links: dict[str, set[str]] = {"visual_meshes": set(), "collision_meshes": set()}
        for field in geometry_links:
            records = document.get(field)
            if not isinstance(records, list) or not records:
                raise ValueError(f"asset assembly {field} must be non-empty")
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError(f"asset assembly {field}[{index}] must be an object")
                link = record.get("link")
                if not isinstance(link, str) or not link.strip() or link in geometry_links[field]:
                    raise ValueError(f"asset assembly {field} links must be non-empty and unique")
                geometry_links[field].add(link)
                if record.get("units") != "meters":
                    raise ValueError(f"asset assembly {field}[{index}] units must be meters")
                references.append(
                    self._artifact_reference(
                        record.get("artifact"),
                        f"asset assembly {field}[{index}].artifact",
                    )
                )
        if geometry_links["visual_meshes"] != geometry_links["collision_meshes"]:
            raise ValueError("every assembled link requires visual and collision geometry")
        articulation_ref = self._artifact_reference(
            document.get("articulation_model"),
            "asset assembly articulation_model",
        )
        physics_ref = self._artifact_reference(
            document.get("identified_physics"),
            "asset assembly identified_physics",
        )
        raw_evidence = document.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValueError("asset assembly evidence closure must be non-empty")
        evidence_refs = [
            self._artifact_reference(item, f"asset assembly evidence[{index}]")
            for index, item in enumerate(raw_evidence)
        ]
        if scale_artifact_id not in {item.artifact_id for item in evidence_refs}:
            raise ValueError("metric scale evidence is outside the asset assembly closure")
        references.extend((articulation_ref, physics_ref, *evidence_refs))
        if len({reference.artifact_id for reference in references}) != len(references):
            raise ValueError("asset assembly artifact IDs must be unique")
        self._require_available_artifacts(references, "asset assembly")
        articulation = strict_json_object(
            self.artifacts.get_bytes(articulation_ref),
            "asset assembly articulation model",
        )
        physics = strict_json_object(
            self.artifacts.get_bytes(physics_ref),
            "asset assembly identified physics",
        )
        if articulation.get("schema") != "physrealx.articulation-model/1" or articulation.get(
            "status"
        ) not in {"IDENTIFIED", "VALIDATED"}:
            raise ValueError("asset assembly articulation model is not identified")
        links = articulation.get("links")
        joints = articulation.get("joints")
        if not isinstance(links, list) or not links or len(set(links)) != len(links):
            raise ValueError("asset assembly articulation links must be non-empty and unique")
        if not isinstance(joints, list):
            raise ValueError("asset assembly articulation joints must be an array")
        joint_names = {
            joint.get("name")
            for joint in joints
            if isinstance(joint, dict) and isinstance(joint.get("name"), str)
        }
        if not joint_names or len(joint_names) != len(joints):
            raise ValueError("asset assembly articulation joints must be non-empty and named")
        if set(links) != geometry_links["visual_meshes"]:
            raise ValueError("asset assembly geometry does not cover every articulation link")
        if (
            physics.get("schema") != "physrealx.identified-physics/1"
            or physics.get("status") not in {"IDENTIFIED", "VALIDATED"}
            or physics.get("evidence_state") not in {"identified", "validated"}
        ):
            raise ValueError("asset assembly full physics is not identified")
        bodies = physics.get("bodies")
        physics_joints = physics.get("joints")
        body_links = (
            {
                body.get("link")
                for body in bodies
                if isinstance(body, dict) and isinstance(body.get("link"), str)
            }
            if isinstance(bodies, list)
            else set()
        )
        physics_joint_names = (
            {
                joint.get("name")
                for joint in physics_joints
                if isinstance(joint, dict) and isinstance(joint.get("name"), str)
            }
            if isinstance(physics_joints, list)
            else set()
        )
        if body_links != set(links) or physics_joint_names != joint_names:
            raise ValueError("asset assembly physics must cover every body and joint exactly")

    def assembled_asset(self, document: Mapping[str, Any]) -> AssetSpec:
        """Read the same full asset representation consumed by USD and simulation."""
        self.validate_asset_assembly(document)
        asset = AssetSpec.model_validate(document.get("asset_spec"))
        if asset.evidence_state is not AssetEvidenceState.IDENTIFIED:
            raise ValueError("assembly must preserve identified evidence state")
        if asset.contact_exclusions:
            raise ValueError("measured assembly cannot introduce unaudited contact exclusions")
        if any(not self.artifacts.verify(ref) for ref in asset.evidence):
            raise ValueError("assembled asset has unavailable evidence")
        if (
            asset.resource_id != document.get("resource_id")
            or asset.version != document.get("version")
            or asset.root_link != document.get("root_link")
        ):
            raise ValueError("assembled asset identity disagrees with its manifest")
        expected = {}
        for field, role in (
            ("visual_meshes", GeometryRole.VISUAL),
            ("collision_meshes", GeometryRole.COLLISION),
        ):
            for record in document[field]:
                expected[(record["link"], role)] = ArtifactRef.model_validate(
                    record["artifact"]
                ).artifact_id
        if len(asset.geometries) != len(expected):
            raise ValueError("assembled asset geometry closure differs from its manifest")
        for geometry in asset.geometries:
            if (
                geometry.uri != f"artifact://{expected.get((geometry.link, geometry.role))}"
                or geometry.scale != (1.0, 1.0, 1.0)
                or geometry.local_pose != PoseSpec()
            ):
                raise ValueError(
                    "assembled asset geometry or coordinate frames differ from evidence"
                )
        physical = strict_json_object(
            self.artifacts.get_bytes(ArtifactRef.model_validate(document["identified_physics"])),
            "assembled physics",
        )
        if [joint.model_dump(mode="json") for joint in asset.joints] != physical["joints"]:
            raise ValueError("assembled joint physics differs from evidence")
        for body in asset.physics:
            original = next(
                (item for item in physical["bodies"] if item["link"] == body.link), None
            )
            if (
                original is None
                or list(body.center_of_mass_m) != original["center_of_mass_m"]
                or [list(row) for row in body.inertia_kg_m2] != original["inertia_kg_m2"]
            ):
                raise ValueError("assembled body tensor differs from measured evidence")
            for field in ("mass_kg", "static_friction", "dynamic_friction", "restitution"):
                if getattr(body, field).value != original[field]["value"]:
                    raise ValueError("assembled body parameter differs from measured evidence")
        return asset

    def validate_robot_target(self, document: Mapping[str, Any]) -> None:
        try:
            target = RobotInterfaceSpec.model_validate_json(
                json.dumps(document, allow_nan=False, sort_keys=True)
            )
        except ValueError as error:
            raise ValueError("robot target does not satisfy RobotInterfaceSpec/1") from error
        self._require_available_artifacts(
            [target.urdf, target.controller_configuration],
            "robot target",
        )

    @staticmethod
    def metric_alignment_evidence_error(document: Mapping[str, Any]) -> str | None:
        """Reject marker detections that do not carry independent metric metrology."""

        camera_artifact_id = document.get("camera_rig_artifact_id")
        try:
            if not isinstance(camera_artifact_id, str):
                raise ValueError
            ArtifactRef(
                artifact_id=camera_artifact_id,
                size_bytes=0,
                media_type="application/json",
            )
        except ValueError:
            return "camera_rig_artifact_id must bind the exact unresolved camera artifact"
        for field in ("component_id", "source_frame", "target_frame"):
            if not isinstance(document.get(field), str) or not document[field].strip():
                return f"field {field!r} must be a non-empty string"
        if document["source_frame"] == document["target_frame"]:
            return "source_frame and target_frame must differ"
        reference = document.get("metric_reference")
        provenance = document.get("provenance")
        if not isinstance(reference, dict) or not isinstance(provenance, dict):
            return "metric_reference and provenance must be objects"
        for field in ("kind", "identifier", "source"):
            if not isinstance(reference.get(field), str) or not reference[field].strip():
                return f"metric_reference.{field} must be a non-empty string"
        value_m = reference.get("value_m")
        if (
            isinstance(value_m, bool)
            or not isinstance(value_m, (int, float))
            or not math.isfinite(float(value_m))
            or float(value_m) <= 0.0
        ):
            return "metric_reference.value_m must be a positive finite metric value"
        for field in ("capture_id", "measurement_method", "source_uri"):
            if not isinstance(provenance.get(field), str) or not provenance[field].strip():
                return f"provenance.{field} must be a non-empty string"
        if "://" not in provenance["source_uri"]:
            return "provenance.source_uri must be absolute"
        marker_evidence = "marker" in reference["kind"].lower() or any(
            token in provenance["measurement_method"].lower() for token in ("apriltag", "aruco")
        )
        if marker_evidence:
            for field in ("marker_family", "layout", "layout_source_uri"):
                if not isinstance(reference.get(field), str) or not reference[field].strip():
                    return f"metric_reference.{field} is required for marker evidence"
            marker_size = reference.get("marker_size_m")
            if (
                isinstance(marker_size, bool)
                or not isinstance(marker_size, (int, float))
                or not math.isfinite(float(marker_size))
                or float(marker_size) <= 0.0
                or not math.isclose(float(marker_size), float(value_m), rel_tol=0.0, abs_tol=1e-12)
            ):
                return "marker_size_m must be positive and equal metric_reference.value_m"
            if "://" not in reference["layout_source_uri"]:
                return "metric_reference.layout_source_uri must be absolute"
        identifiers: dict[str, set[str]] = {}
        for split, minimum in (("train_correspondences", 4), ("holdout_correspondences", 3)):
            records = document.get(split)
            if not isinstance(records, list) or len(records) < minimum:
                return f"{split} requires at least {minimum} measured 3D correspondences"
            split_ids: set[str] = set()
            for record in records:
                if not isinstance(record, dict):
                    return f"{split} entries must be objects"
                identifier = record.get("id")
                if not isinstance(identifier, str) or not identifier.strip():
                    return f"{split} entries require non-empty ids"
                for field in ("source_xyz", "target_xyz_m"):
                    point = record.get(field)
                    if (
                        not isinstance(point, list)
                        or len(point) != 3
                        or any(
                            isinstance(item, bool)
                            or not isinstance(item, (int, float))
                            or not math.isfinite(float(item))
                            for item in point
                        )
                    ):
                        return f"{split}.{field} must contain three finite coordinates"
                split_ids.add(identifier)
            if len(split_ids) != len(records):
                return f"{split} ids must be unique"
            identifiers[split] = split_ids
        if identifiers["train_correspondences"] & identifiers["holdout_correspondences"]:
            return "training and holdout correspondence ids must be disjoint"
        train_records = document["train_correspondences"]
        holdout_records = document["holdout_correspondences"]
        assert isinstance(train_records, list) and isinstance(holdout_records, list)
        for coordinate in ("source_xyz", "target_xyz_m"):
            train_points = {
                tuple(record[coordinate]) for record in train_records if isinstance(record, dict)
            }
            holdout_points = {
                tuple(record[coordinate]) for record in holdout_records if isinstance(record, dict)
            }
            if train_points & holdout_points:
                return f"training and holdout {coordinate} coordinates must be disjoint"
        return None
