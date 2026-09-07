"""Observable, bounded single-joint inverse dynamics with independent holdout."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import ArtifactRef, DomainModel, MethodCapability, OperationSpec
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import (
    dependency_probe,
    input_artifact,
    publish,
    read_array,
    read_json,
)
from physrealx.methods import builtin_method


class JointIdentificationEvidence(DomainModel):
    schema_id: Literal["physrealx.joint-system-identification-evidence/1"] = Field(
        default="physrealx.joint-system-identification-evidence/1", alias="schema"
    )
    joint_name: str = Field(min_length=1)
    joint_type: Literal["slide", "hinge"]
    # NPY columns: time seconds, q, qdot, qddot, applied generalized force.
    train: ArtifactRef
    holdout: ArtifactRef
    train_capture_id: str = Field(min_length=1)
    holdout_capture_id: str = Field(min_length=1)
    position_unit: Literal["m", "rad"]
    force_unit: Literal["N", "N*m"]
    gravity_compensated: Literal[True]
    prior_parameters: tuple[float, float, float]

    @model_validator(mode="after")
    def validate_units(self) -> JointIdentificationEvidence:
        import math

        if (
            self.train.artifact_id == self.holdout.artifact_id
            or self.train_capture_id == self.holdout_capture_id
        ):
            raise ValueError("identification requires independent train and holdout captures")
        if (self.position_unit, self.force_unit) != (
            ("m", "N") if self.joint_type == "slide" else ("rad", "N*m")
        ):
            raise ValueError("joint trajectory units do not match joint type")
        if (
            any(not math.isfinite(v) or v < 0 for v in self.prior_parameters)
            or self.prior_parameters[0] <= 0
        ):
            raise ValueError("prior inertia must be positive and all prior parameters finite")
        return self


class IdentificationConfig(DomainModel):
    lower_bounds: tuple[float, float, float] = (0.001, 0.0, 0.0)
    upper_bounds: tuple[float, float, float] = (1000.0, 1000.0, 1000.0)
    min_velocity: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    max_condition_number: float = Field(default=10000.0, gt=1, allow_inf_nan=False)
    max_holdout_rmse: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    min_holdout_improvement: float = Field(default=0.05, gt=0, lt=1)
    max_kinematic_relative_error: float = Field(default=0.15, gt=0, lt=1)
    max_relative_ci_width: float = Field(default=0.5, gt=0, allow_inf_nan=False)


class JointPhysicsAdapter:
    descriptor = builtin_method(
        MethodCapability.PHYSICS_CALIBRATE,
        "bounded_inverse_dynamics",
        input_contract="JointSystemIdentificationEvidence/1",
        output_contract="IdentifiedJointPhysics/1+PhysicsIdentificationReport/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe({"numpy": ("numpy", "1.26.4"), "scipy": ("scipy", "1.13.1")})

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np
        from scipy.optimize import lsq_linear
        from scipy.stats import t

        config = IdentificationConfig.model_validate(spec.method.config)
        reference = input_artifact(spec, "evidence", artifacts)
        evidence = JointIdentificationEvidence.model_validate(
            read_json(artifacts, reference, "physrealx.joint-system-identification-evidence/1")
        )
        lower, upper = np.array(config.lower_bounds), np.array(config.upper_bounds)
        if (
            not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or (lower < 0).any()
            or (upper <= lower).any()
            or lower[0] <= 0
        ):
            raise DomainValidationError("identification parameter bounds are invalid")
        designs = []
        for ref in (evidence.train, evidence.holdout):
            values = read_array(artifacts, ref).astype(float)
            if (
                values.ndim != 2
                or values.shape[1] != 5
                or len(values) < 16
                or (np.diff(values[:, 0]) <= 0).any()
            ):
                raise DomainValidationError(
                    "trajectory requires 16 ordered rows of time, q, qdot, qddot, force"
                )
            for position_column, velocity_column in ((1, 2), (2, 3)):
                derivative = np.gradient(values[:, position_column], values[:, 0], edge_order=2)[
                    2:-2
                ]
                observed = values[2:-2, velocity_column]
                rms_scale = max(config.min_velocity, float(np.sqrt(np.mean(observed**2))))
                error = float(np.sqrt(np.mean((derivative - observed) ** 2))) / rms_scale
                if error > config.max_kinematic_relative_error:
                    raise DomainValidationError(
                        "trajectory position, velocity and acceleration are inconsistent"
                    )
            moving = values[np.abs(values[:, 2]) >= config.min_velocity]
            if len(moving) < 16:
                raise MethodExecutionError("NOT_IDENTIFIABLE: insufficient moving samples")
            designs.append(
                (np.column_stack((moving[:, 3], moving[:, 2], np.sign(moving[:, 2]))), moving[:, 4])
            )
        x, y = designs[0]
        norms = np.linalg.norm(x, axis=0)
        if (norms == 0).any() or np.linalg.matrix_rank(x) < 3:
            raise MethodExecutionError("NOT_IDENTIFIABLE: rank-deficient excitation")
        condition = float(np.linalg.cond(x / norms))
        if condition > config.max_condition_number:
            raise MethodExecutionError("NOT_IDENTIFIABLE: ill-conditioned excitation")
        result = lsq_linear(x, y, bounds=(lower, upper), tol=1e-12)
        parameters = result.x
        if (
            not result.success
            or (parameters - lower <= 1e-7 * np.maximum(1, np.abs(parameters))).any()
            or (upper - parameters <= 1e-7 * np.maximum(1, np.abs(parameters))).any()
        ):
            raise MethodExecutionError("NOT_IDENTIFIABLE: solution reaches parameter bounds")
        residual = x @ parameters - y
        covariance = float(residual @ residual / (len(y) - 3)) * np.linalg.inv(x.T @ x)
        width = t.ppf(0.975, len(y) - 3) * np.sqrt(np.diag(covariance))
        if (
            not np.isfinite(width).all()
            or (2 * width > config.max_relative_ci_width * parameters).any()
        ):
            raise MethodExecutionError("NOT_IDENTIFIABLE: parameter uncertainty is too large")
        test_x, test_y = designs[1]
        rmse = float(np.sqrt(np.mean((test_x @ parameters - test_y) ** 2)))
        prior_rmse = float(
            np.sqrt(np.mean((test_x @ np.array(evidence.prior_parameters) - test_y) ** 2))
        )
        if (
            rmse > config.max_holdout_rmse
            or prior_rmse <= 1e-12
            or rmse >= prior_rmse * (1 - config.min_holdout_improvement)
        ):
            raise MethodExecutionError(
                "NOT_IDENTIFIABLE: independent holdout does not improve the prior"
            )
        units = (
            ("kg", "N*s/m", "N")
            if evidence.joint_type == "slide"
            else ("kg*m^2", "N*m*s/rad", "N*m")
        )
        estimates = {
            name: {
                "value": float(value),
                "unit": unit,
                "confidence_interval": {
                    "level": 0.95,
                    "lower": float(value - spread),
                    "upper": float(value + spread),
                },
                "multi_start_spread": 0.0,
            }
            for name, value, unit, spread in zip(
                ("effective_inertia", "viscous_damping", "coulomb_friction"),
                parameters,
                units,
                width,
            )
        }
        identified = {
            "schema": "physrealx.identified-joint-physics/1",
            "status": "IDENTIFIED",
            "evidence_state": "identified",
            "joint": {"name": evidence.joint_name, "type": evidence.joint_type},
            "parameters": estimates,
            "evidence_artifact_id": reference.artifact_id,
        }
        metrics = {
            "holdout_rmse": rmse,
            "prior_holdout_rmse": prior_rmse,
            "condition_number": condition,
        }
        report = {
            "schema": "physrealx.physics-identification-report/1",
            "status": "PASSED",
            "method": self.descriptor.model_dump(mode="json"),
            "metrics": metrics,
            "evidence_artifact_id": reference.artifact_id,
            "solver": "strictly_convex_bounded_linear_least_squares",
            "uncertainty_model": "independent_homoscedastic_force_residuals",
            "multi_start_policy": "full_column_rank_implies_unique_global_minimum",
        }
        return publish(
            spec,
            artifacts,
            {"identified_joint_physics": identified, "identification_report": report},
            metrics=metrics,
        )
