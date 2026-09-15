"""RGB-guided metric depth completion with independent measured validation."""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    MethodCapability,
    OperationSpec,
    canonical_json_bytes,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import (
    dependency_probe,
    input_artifact,
    publish,
    read_array,
    read_json,
)
from physrealx.methods import builtin_method

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class DepthFrame(DomainModel):
    frame_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    rgb: ArtifactRef
    depth: ArtifactRef
    depth_scale_m: float = Field(gt=0, allow_inf_nan=False)
    reference_depth_m: ArtifactRef | None = None
    capture_id: str = Field(min_length=1)
    reference_capture_id: str | None = Field(default=None, min_length=1)
    view: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def independent_reference(self) -> DepthFrame:
        if (self.reference_depth_m is None) != (self.reference_capture_id is None):
            raise ValueError("reference depth and capture identity must be declared together")
        if self.reference_depth_m is not None and (
            self.depth.artifact_id == self.reference_depth_m.artifact_id
            or self.capture_id == self.reference_capture_id
        ):
            raise ValueError("depth validation requires a separate reference capture")
        return self


class CalibratedRGBDBundle(DomainModel):
    schema_id: Literal["physrealx.calibrated-rgbd-bundle/1"] = Field(
        default="physrealx.calibrated-rgbd-bundle/1", alias="schema"
    )
    camera_rig: ArtifactRef
    validation_scope: Literal["independent_capture", "masked_observed_pixels"] = (
        "independent_capture"
    )
    frames: tuple[DepthFrame, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_frames(self) -> CalibratedRGBDBundle:
        if any(
            (frame.reference_depth_m is not None)
            != (self.validation_scope == "independent_capture")
            for frame in self.frames
        ):
            raise ValueError("depth references must match the explicitly selected evaluation scope")
        if len({frame.frame_id for frame in self.frames}) != len(self.frames):
            raise ValueError("depth frame IDs must be unique")
        return self


class DepthConfig(DomainModel):
    radius: int = Field(default=3, ge=1, le=15)
    color_sigma: float = Field(default=25.0, gt=0, allow_inf_nan=False)
    max_neighbor_spread_m: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    max_rmse_m: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    min_reference_pixels: int = Field(default=16, ge=3)


def complete_depth(
    depth: object, rgb: object, config: DepthConfig, *, require_complete: bool = True
) -> object:
    import numpy as np

    depth = np.asarray(depth, dtype=float)
    rgb = np.asarray(rgb, dtype=float)
    valid = depth > 0
    missing_y, missing_x = np.nonzero(~valid)
    if not len(missing_y):
        return depth.astype(np.float32)
    weighted = np.zeros(len(missing_y), dtype=float)
    weights = np.zeros(len(missing_y), dtype=float)
    low = np.full(len(missing_y), np.inf)
    high = np.full(len(missing_y), -np.inf)
    r = config.radius
    padded_depth = np.pad(depth, r)
    padded_rgb = np.pad(rgb, ((r, r), (r, r), (0, 0)))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            neighbor = padded_depth[missing_y + r + dy, missing_x + r + dx]
            color = padded_rgb[missing_y + r + dy, missing_x + r + dx]
            support = neighbor > 0
            color_delta = rgb[missing_y, missing_x] - color
            weight = np.exp(-np.sum(color_delta**2, axis=1) / (2 * config.color_sigma**2))
            weight *= np.exp(-(dx * dx + dy * dy) / (2 * r * r)) * support
            weighted += weight * neighbor
            weights += weight
            low = np.minimum(low, np.where(support, neighbor, np.inf))
            high = np.maximum(high, np.where(support, neighbor, -np.inf))
    fill = (weights > 1e-12) & ((high - low) <= config.max_neighbor_spread_m)
    result = depth.copy()
    result[missing_y[fill], missing_x[fill]] = weighted[fill] / weights[fill]
    if require_complete and not (result > 0).all():
        raise MethodExecutionError("depth holes lack local support or cross a depth discontinuity")
    return result.astype(np.float32)


def local_depth_range(
    depth: object, radius: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return positive-neighbor minima and maxima using a zero-copy window view."""
    import numpy as np

    depth = np.asarray(depth, dtype=float)
    size = 2 * radius + 1
    positive = np.pad(np.where(depth > 0, depth, np.inf), radius, constant_values=np.inf)
    padded = np.pad(depth, radius)
    low = np.lib.stride_tricks.sliding_window_view(positive, (size, size)).min(axis=(-2, -1))
    high = np.lib.stride_tricks.sliding_window_view(padded, (size, size)).max(axis=(-2, -1))
    return low, high


class DepthEnhancementAdapter:
    descriptor = builtin_method(
        MethodCapability.DEPTH_ENHANCE,
        "rgb_guided_fill",
        input_contract="CalibratedRGBDBundle/1",
        output_contract="EnhancedDepthBundle/1+DepthEnhancementReport/1",
        python=">=3.10,<3.13",
        version="1.1.0",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe({"numpy": ("numpy", "1.26.4")})

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np

        config = DepthConfig.model_validate(spec.method.config)
        source = input_artifact(spec, "rgbd", artifacts)
        bundle = CalibratedRGBDBundle.model_validate(
            read_json(artifacts, source, "physrealx.calibrated-rgbd-bundle/1")
        )
        camera = read_json(artifacts, bundle.camera_rig, "physrealx.camera-rig-calibration/1")
        if (
            camera.get("scale_status") != "resolved"
            or camera.get("coordinate_convention", {}).get("translation_unit") != "m"
        ):
            raise DomainValidationError("depth requires a camera rig with resolved metre units")
        payloads = []
        records = []
        squared_error, pixel_count, filled_count, remaining_count = 0.0, 0, 0, 0
        independent = bundle.validation_scope == "independent_capture"
        generator = np.random.default_rng(0)
        for frame in bundle.frames:
            depth = read_array(artifacts, frame.depth).astype(float) * frame.depth_scale_m
            rgb = read_array(artifacts, frame.rgb)
            truth = (
                read_array(artifacts, frame.reference_depth_m)
                if frame.reference_depth_m is not None
                else depth
            )
            if (
                depth.ndim != 2
                or not depth.size
                or rgb.shape != (*depth.shape, 3)
                or truth.shape != depth.shape
            ):
                raise DomainValidationError("RGB, depth and registered reference dimensions differ")
            if (depth < 0).any() or (truth < 0).any() or (rgb < 0).any() or (rgb > 255).any():
                raise DomainValidationError("depth must be nonnegative and RGB must be in [0, 255]")
            if independent:
                enhanced = np.asarray(complete_depth(depth, rgb, config))
                mask = truth > 0
                prediction = enhanced
            else:
                # Mask previously observed interior pixels before inference. These are
                # self-supervised checks, explicitly not an independent sensor capture.
                interior = depth > 0
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    interior &= np.roll(depth > 0, (dy, dx), (0, 1))
                interior[[0, -1], :] = False
                interior[:, [0, -1]] = False
                # Evaluate small holes on locally continuous surfaces. Discontinuities
                # remain unfilled, and their coverage is reported separately.
                low, high = local_depth_range(depth, config.radius)
                interior &= high - low <= config.max_neighbor_spread_m
                available = np.flatnonzero(interior)
                if len(available) < config.min_reference_pixels * 4:
                    raise MethodExecutionError(
                        "insufficient observed pixels for masked depth validation"
                    )
                count = min(4096, max(config.min_reference_pixels, len(available) // 100))
                selected = generator.choice(available, count, replace=False)
                mask = np.zeros(depth.shape, dtype=bool)
                mask.flat[selected] = True
                withheld = depth.copy()
                withheld[mask] = 0
                prediction = np.asarray(
                    complete_depth(withheld, rgb, config, require_complete=False)
                )
                supported = mask & (prediction > 0)
                if supported.sum() / mask.sum() < 0.9:
                    raise MethodExecutionError("masked depth completion has insufficient coverage")
                mask = supported
                enhanced = np.asarray(complete_depth(depth, rgb, config, require_complete=False))
                if np.mean(enhanced > 0) < 0.8:
                    raise MethodExecutionError(
                        "enhanced depth retains less than 80% valid coverage"
                    )
            if int(mask.sum()) < config.min_reference_pixels:
                raise MethodExecutionError("insufficient depth evaluation pixels")
            error = float(np.sum((prediction[mask] - truth[mask]) ** 2))
            if float(np.sqrt(error / mask.sum())) > config.max_rmse_m:
                raise MethodExecutionError(
                    f"depth frame {frame.frame_id} failed {bundle.validation_scope} RMSE"
                )
            squared_error += error
            pixel_count += int(mask.sum())
            filled_count += int(((depth == 0) & (enhanced > 0)).sum())
            remaining_count += int((enhanced == 0).sum())
            stream = io.BytesIO()
            np.save(stream, enhanced, allow_pickle=False)
            payload = stream.getvalue()
            name = f"depth/{frame.frame_id}.npy"
            payloads.append((name, payload))
            records.append(
                {
                    "frame_id": frame.frame_id,
                    "rgb": frame.rgb.model_dump(mode="json"),
                    "view": frame.view,
                    "depth": {
                        "path": name,
                        "size_bytes": len(payload),
                        "media_type": "application/x-npy",
                        "dtype": "float32",
                        "unit": "m",
                        "invalid_value": None if independent else 0,
                    },
                }
            )
        metrics = {
            "holdout_rmse_m": (squared_error / pixel_count) ** 0.5,
            "filled_pixels": float(filled_count),
            "remaining_invalid_pixels": float(remaining_count),
            "evaluation_pixels": float(pixel_count),
        }
        manifest: dict[str, Any] = {
            "schema": "physrealx.enhanced-depth-bundle/1",
            "status": "PASSED" if independent else "SELF_SUPERVISED",
            "evaluation_scope": bundle.validation_scope,
            "evaluation_region": "registered_reference"
            if independent
            else "locally_continuous_observed_pixels",
            "source_camera_artifact_id": bundle.camera_rig.artifact_id,
            "source_rgbd_artifact_id": source.artifact_id,
            "depth_encoding": {"unit": "m"},
            "frames": records,
        }
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in [("manifest.json", canonical_json_bytes(manifest)), *payloads]:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload)
        report = {
            "schema": "physrealx.depth-enhancement-report/1",
            "status": "PASSED" if independent else "SELF_SUPERVISED",
            "evaluation_scope": bundle.validation_scope,
            "source_camera_artifact_id": bundle.camera_rig.artifact_id,
            "source_rgbd_artifact_id": source.artifact_id,
            "method": self.descriptor.model_dump(mode="json"),
            "metrics": metrics,
            "gates": {
                "positive_metric_depth": remaining_count == 0,
                "independent_reference": independent,
                "masked_observed_validation": not independent,
            },
        }
        return publish(
            spec,
            artifacts,
            {"enhancement_report": report},
            artifacts={
                "enhanced_depth": artifacts.put_bytes(
                    archive_bytes.getvalue(), media_type="application/zip"
                ),
            },
            metrics=metrics,
        )
