"""Decode imported RGB-D and resolve its scale against an explicit USD reference.

Depth-to-RGB registration and the support-plane geometry are checked on the
actual capture. Model-conditioned scale is always labelled as such; it is not
independent metrology or a replacement for a calibrated sensor certificate.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.digital_twin import ObservationSetManifest
from physrealx.domain import ArtifactRef, DomainModel, MethodCapability, OperationSpec
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import dependency_probe, input_artifact, publish, read_json
from physrealx.methods import builtin_method

_MAX_FRAMES_PER_STREAM = 512


def put_array(artifacts: ArtifactStore, array: Any) -> ArtifactRef:
    import numpy as np

    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return artifacts.put_bytes(stream.getvalue(), media_type="application/x-npy")


def capture_frames(
    observations: ObservationSetManifest,
    artifacts: ArtifactStore,
    stream_name: str,
    images: list[dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], Any, Any, Any]]:
    import cv2
    import numpy as np

    streams = {stream.name: stream for stream in observations.streams}
    rgb_stream = streams.get(stream_name)
    depth_stream = streams.get(stream_name.removesuffix(".rgb") + ".depth")
    mask_stream = streams.get(stream_name.removesuffix(".rgb") + ".mask")
    if (
        rgb_stream is None
        or depth_stream is None
        or mask_stream is None
        or depth_stream.encoding != "npy.depth.uint16"
        or mask_stream.encoding != "png.mask.uint8"
        or depth_stream.frame_count != rgb_stream.frame_count
        or mask_stream.frame_count != rgb_stream.frame_count
        or len(depth_stream.artifacts) != rgb_stream.frame_count
    ):
        raise DomainValidationError(
            "capture requires paired imported RGB, uint16 NPY depth and frame masks"
        )
    for image in images:
        index = image["source_frame_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < rgb_stream.frame_count
        ):
            raise DomainValidationError("camera frame index is outside its source capture")
        ref = rgb_stream.artifacts[index]
        if ref.artifact_id != image["source_artifact_id"]:
            raise DomainValidationError("camera image is bound to a different capture artifact")
        rgb = cv2.imdecode(np.frombuffer(artifacts.get_bytes(ref), np.uint8), cv2.IMREAD_COLOR)
        try:
            depth = np.load(
                io.BytesIO(artifacts.get_bytes(depth_stream.artifacts[index])),
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            raise DomainValidationError("cannot decode imported depth frame") from error
        mask = cv2.imdecode(
            np.frombuffer(artifacts.get_bytes(mask_stream.artifacts[index]), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if (
            rgb is None
            or depth.dtype != np.uint16
            or depth.ndim != 2
            or mask is None
            or rgb.shape[:2] != depth.shape
            or mask.shape != depth.shape
            or not set(np.unique(mask)).issubset({0, 255})
            or not 0 < int(np.count_nonzero(mask)) < mask.size
        ):
            raise DomainValidationError(
                "RGB/depth/mask must be registered, nonempty and use binary mask values 0/255"
            )
        depth = depth.copy()
        depth[depth == 65535] = 0
        yield image, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth, mask


def camera_rays(camera: dict[str, Any], xy: Any) -> Any:
    """Use COLMAP's actual distortion model, including non-pinhole cameras."""
    import numpy as np
    import pycolmap

    model = pycolmap.Camera(
        model=camera["model"],
        width=camera["width"],
        height=camera["height"],
        params=camera["params"],
    )
    rays = np.asarray(model.cam_from_img(np.asarray(xy, dtype=float)))
    if rays.shape != (len(xy), 2) or not np.isfinite(rays).all():
        raise DomainValidationError("camera model cannot unproject the requested pixels")
    return np.column_stack([rays, np.ones(len(rays))])


def support_geometry(
    rgb: Any, depth: Any, camera: dict[str, Any], *, stride: int = 4
) -> dict[str, Any] | None:
    """Find the visible object above a plane supported by detected table markers."""
    import cv2
    import numpy as np

    corners, _, _ = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    ).detectMarkers(rgb)
    if len(corners) < 3:
        return None
    table_mask = np.zeros(depth.shape, dtype=np.uint8)
    for polygon in corners:
        cv2.fillConvexPoly(table_mask, polygon[0].astype(np.int32), (1,))
    hull = cv2.convexHull(np.concatenate(corners).reshape(-1, 2).astype(np.float32))
    roi = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(roi, hull.astype(np.int32), (1,))
    y, x = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    xy = np.column_stack([x.ravel(), y.ravel()])
    z = depth[y, x].ravel().astype(float)
    points = camera_rays(camera, xy) * z[:, None]
    good = z > 0
    table = points[good & (table_mask[y, x].ravel() > 0)]
    if len(table) < 40:
        return None
    center = np.median(table, axis=0)
    for _ in range(3):
        _, singular, vt = np.linalg.svd(table - center, full_matrices=False)
        if singular[1] <= singular[0] * 1e-4:
            return None
        normal = vt[-1]
        normal *= 1 if normal @ center < 0 else -1
        offset = float(np.median(table @ normal))
        residual = np.abs(table @ normal - offset)
        keep = residual <= max(float(np.median(residual)) * 3, 1e-3)
        if keep.sum() < 40:
            return None
        table = table[keep]
        center = np.median(table, axis=0)
    heights = points @ normal - offset
    # Relative thresholds work before the sensor's length unit is known.
    noise = float(np.sqrt(np.mean((table @ normal - offset) ** 2)))
    mask = good & (roi[y, x].ravel() > 0) & (heights > max(6 * noise, abs(offset) * 0.04))
    if int(mask.sum()) < 100:
        return None
    height = float(np.percentile(heights[mask], 90))
    top = mask & (np.abs(heights - height) <= max(3 * noise, height * 0.04))
    if top.sum() < 60 or noise / height > 0.06:
        return None
    return {
        "points": points,
        "xy": xy,
        "mask": mask,
        "shape": y.shape,
        "height_native": float(np.median(heights[top])),
        "plane_normal": normal,
        "plane_offset": offset,
        "plane_rmse_native": noise,
        "marker_count": len(corners),
    }


class ModelScaleConfig(DomainModel):
    min_support_frames: int = Field(default=4, ge=4)
    min_observations: int = Field(default=30, ge=12)
    max_height_relative_error: float = Field(default=0.08, gt=0, le=0.2)
    max_depth_relative_error: float = Field(default=0.06, gt=0, le=0.15)
    min_inlier_fraction: float = Field(default=0.85, gt=0.5, le=1)


class ModelScaleAdapter:
    descriptor = builtin_method(
        MethodCapability.CAMERA_ALIGN_METRIC,
        "usd_depth_scale",
        input_contract="CameraRigCalibration/1+MetricAlignmentEvidence/1",
        output_contract="CameraRigCalibration/1+MetricAlignmentReport/1",
        python=">=3.10,<3.13",
        version="1.2.0",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe(
            {
                "numpy": ("numpy", "1.26.4"),
                "cv2": ("opencv-python", "4.10.0.84"),
                "pycolmap": ("pycolmap", "4.1.1"),
            }
        )

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np

        config = ModelScaleConfig.model_validate(spec.method.config)
        camera_ref = input_artifact(spec, "camera_rig", artifacts)
        evidence_ref = input_artifact(spec, "evidence", artifacts)
        camera = read_json(artifacts, camera_ref, "physrealx.camera-rig-calibration/1")
        evidence = read_json(artifacts, evidence_ref, "physrealx.model-scale-input/1")
        if camera.get("scale_status") != "unresolved":
            raise DomainValidationError("model scale requires unresolved cameras")
        observation_ref = ArtifactRef.model_validate(evidence["observations"])
        observations = ObservationSetManifest.model_validate_json(
            artifacts.get_bytes(observation_ref)
        )
        if camera.get("source_observations", {}).get("artifact_id") != observation_ref.artifact_id:
            raise DomainValidationError("scale observations differ from camera recovery input")
        reference = read_json(
            artifacts,
            ArtifactRef.model_validate(evidence["reference_model"]),
            "physrealx.usd-reference/1",
        )
        origin = reference.get("source_observations", {})
        if (
            origin.get("resource_id") != observations.resource_id
            or origin.get("version") != observations.version
            or reference.get("source_usd")
            not in [item.artifact.model_dump(mode="json") for item in observations.attachments]
        ):
            raise DomainValidationError(
                "USD reference is not bound to the same observation package"
            )
        height_m = float(reference["dimensions_m"][2])
        if not np.isfinite(height_m) or height_m <= 0 or reference.get("unit") != "m":
            raise DomainValidationError("reference model has no positive metre height")
        selected_stream_names = {stream["stream_name"] for stream in camera["streams"]}
        depth_units = {
            stream.unit
            for stream in observations.streams
            if stream.name.removesuffix(".depth") + ".rgb" in selected_stream_names
            and stream.name.endswith(".depth")
        }
        if len(depth_units) != 1:
            raise DomainValidationError(
                "metric alignment requires one shared depth unit across capture streams"
            )
        estimates = {}
        for stream in camera["streams"]:
            if len(stream["components"]) != 1:
                raise DomainValidationError(
                    "model scale requires one connected component per stream"
                )
            component = stream["components"][0]
            calibration = {c["camera_id"]: c for c in component["cameras"]}
            track_observations: dict[str, list[Any]] = {}
            for track in component["fiducial_tracks"]:
                for observation in track["observations"]:
                    track_observations.setdefault(observation["image_name"], []).append(
                        (track["xyz"], observation["xy"])
                    )
            heights, records = [], []
            for image, rgb, depth, _ in capture_frames(
                observations, artifacts, stream["stream_name"], component["images"]
            ):
                geometry = support_geometry(rgb, depth, calibration[image["camera_id"]])
                if geometry:
                    heights.append((image["name"], geometry["height_native"]))
                pose = np.asarray(image["cam_from_world"])
                for xyz, xy in track_observations.get(image["name"], []):
                    x, y = np.rint(xy).astype(int)
                    if x < 3 or y < 3 or x >= depth.shape[1] - 3 or y >= depth.shape[0] - 3:
                        continue
                    patch = depth[y - 2 : y + 3, x - 2 : x + 3]
                    patch = patch[patch > 0]
                    if len(patch) < 12:
                        continue
                    observed = float(np.median(patch))
                    predicted = float((pose[:, :3] @ xyz + pose[:, 3])[2])
                    if predicted > 0:
                        records.append((image["name"], predicted, observed))
            if len(heights) < config.min_support_frames or len(records) < config.min_observations:
                raise MethodExecutionError(
                    f"insufficient support-plane/marker depth evidence for {stream['stream_name']}"
                )
            # Split by full views, preventing repeated marker corners from leaking across folds.
            height_train = np.array([h for i, (_, h) in enumerate(heights) if i % 3 != 0])
            height_test = np.array([h for i, (_, h) in enumerate(heights) if i % 3 == 0])
            native_height = float(np.median(height_train))
            height_error = float(np.median(np.abs(height_test / native_height - 1)))
            if height_error > config.max_height_relative_error:
                raise MethodExecutionError(
                    "USD-conditioned height scale is inconsistent across held-out views"
                )
            image_names = sorted({name for name, _, _ in records})
            holdout = set(image_names[::3])
            train = np.array([(p, z) for name, p, z in records if name not in holdout])
            test = np.array([(p, z) for name, p, z in records if name in holdout])
            if min(len(train), len(test)) < 12:
                raise MethodExecutionError(
                    "depth-scale fit requires disjoint populated view splits"
                )
            native_scale = float(np.median(train[:, 1] / train[:, 0]))
            error = np.abs(test[:, 1] - test[:, 0] * native_scale) / test[:, 1]
            fraction = float(np.mean(error <= config.max_depth_relative_error))
            if fraction < config.min_inlier_fraction:
                raise MethodExecutionError(
                    "RGB/depth registration failed held-out scale consistency"
                )
            estimates[stream["stream_name"]] = {
                "component_id": component["component_id"],
                "native_height": native_height,
                "native_scale": native_scale,
                "support_frames": len(heights),
                "support_views": [name for name, _ in heights],
                "height_holdout_relative_error": height_error,
                "depth_holdout_relative_error_median": float(np.median(error)),
                "depth_holdout_inlier_fraction": fraction,
                "depth_holdout_rmse_native": float(
                    np.sqrt(np.mean((test[:, 1] - test[:, 0] * native_scale) ** 2))
                ),
                "fit_views": sorted(set(image_names) - holdout),
                "holdout_views": sorted(holdout),
            }
        # All streams originate from one capture package and declare one depth unit.  Its
        # metre conversion is therefore shared.  Articulation may change the object's
        # observed height between states, so applying the complete USD height to every
        # state independently is physically invalid.  The largest endpoint observation
        # anchors the supplied model envelope; marker similarities validate the remaining
        # streams without inventing state-specific sensor units.
        depth_scale = height_m / max(item["native_height"] for item in estimates.values())
        for item in estimates.values():
            item["depth_scale_m"] = depth_scale
            item["scale_m"] = item["native_scale"] * depth_scale
            item["depth_holdout_rmse_m"] = item.pop("depth_holdout_rmse_native") * depth_scale
        # Keep explicitly connected streams in a single metric reconstruction frame.
        anchor = camera["streams"][0]["stream_name"]
        anchor_scale = estimates[anchor]["scale_m"]
        transforms = {anchor: (np.eye(3), np.zeros(3), 1.0)}
        for alignment in camera.get("fiducial_alignments", []):
            source, target = alignment["source_stream_name"], alignment["target_stream_name"]
            matrix = np.asarray(alignment["target_from_source"], dtype=float)
            alpha = float(alignment["scale"])
            if (
                matrix.shape != (3, 4)
                or not np.isfinite(matrix).all()
                or not np.isfinite(alpha)
                or alpha <= 0
            ):
                raise DomainValidationError(
                    "cross-stream transform must be a finite positive similarity"
                )
            rotation = matrix[:, :3] / alpha
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
                np.linalg.det(rotation), 1, atol=1e-6
            ):
                raise DomainValidationError("cross-stream similarity requires a proper rotation")
            translation = matrix[:, 3]
            if target == anchor:
                transforms[source] = (rotation, translation, alpha)
            elif source == anchor:
                transforms[target] = (rotation.T, -rotation.T @ translation / alpha, 1 / alpha)
        if set(transforms) != set(estimates):
            raise MethodExecutionError(
                "capture streams are not connected by observed fiducial tracks"
            )
        for stream in camera["streams"]:
            name = stream["stream_name"]
            rotation, translation, alpha = transforms[name]  # anchor = alpha * R * source + t
            scale = anchor_scale * alpha
            relative_error = abs(scale / estimates[name]["scale_m"] - 1)
            if relative_error > config.max_height_relative_error:
                raise MethodExecutionError(
                    "shared depth scale conflicts with cross-stream marker alignment"
                )
            estimates[name]["cross_stream_relative_error"] = relative_error
            # Anchor is a measured capture frame; no claim of a recovered USD pose is made.
            for image in stream["components"][0]["images"]:
                pose = np.asarray(image["cam_from_world"])
                new_rotation = pose[:, :3] @ rotation.T
                new_translation = scale * pose[:, 3] - new_rotation @ (anchor_scale * translation)
                image["cam_from_world"] = np.column_stack([new_rotation, new_translation]).tolist()
            for track in stream["components"][0].get("fiducial_tracks", []):
                track["xyz"] = (
                    anchor_scale * (alpha * rotation @ np.asarray(track["xyz"]) + translation)
                ).tolist()
        camera.pop("fiducial_alignments", None)
        camera.update(
            scale_status="resolved",
            unit="m",
            coordinate_frame="capture_metric",
            coordinate_convention={
                "frame": "capture_metric",
                "pose": "world_to_camera",
                "translation_unit": "m",
                "quaternion_order": "wxyz",
            },
            source_camera_artifact_id=camera_ref.artifact_id,
            metric_alignment_evidence_artifact_id=evidence_ref.artifact_id,
            metric_evidence_scope="model_conditioned_shared_depth_unit",
            reference_height_m=height_m,
            depth_calibrations=estimates,
        )
        metrics = {
            "stream_count": float(len(estimates)),
            "max_height_holdout_relative_error": max(
                v["height_holdout_relative_error"] for v in estimates.values()
            ),
            "max_depth_holdout_relative_error_median": max(
                v["depth_holdout_relative_error_median"] for v in estimates.values()
            ),
        }
        report = {
            "schema": "physrealx.metric-alignment-report/1",
            "status": "MODEL_CONDITIONED",
            "evidence_scope": "usd_height_shared_capture_depth_and_markers",
            "independent_metrology": False,
            "source_camera_artifact_id": camera_ref.artifact_id,
            "evidence_artifact_id": evidence_ref.artifact_id,
            "reference_model": evidence["reference_model"],
            "reference_height_m": height_m,
            "streams": estimates,
            "metrics": metrics,
        }
        return publish(
            spec,
            artifacts,
            {"camera_rig_calibration": camera, "metric_alignment_report": report},
            metrics=metrics,
        )


def capture_support_region(camera: dict[str, Any]) -> dict[str, Any]:
    """Fit one observed support plane and footprint in the shared metric frame.

    Per-view marker visibility constrains scale fitting, not which registered
    views may reconstruct the object. A three-marker image hull can cut through
    the object; the union of reconstructed table markers avoids that clipping.
    """
    import numpy as np
    from scipy.spatial import ConvexHull

    height = camera.get("reference_height_m")
    if not isinstance(height, (float, int)) or not np.isfinite(height) or height <= 0:
        raise DomainValidationError("capture support requires an explicit metre reference height")
    components = [part for stream in camera["streams"] for part in stream["components"]]
    points = np.asarray([track["xyz"] for part in components for track in part["fiducial_tracks"]])
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise DomainValidationError("capture support requires four finite reconstructed markers")
    threshold = height * 0.02
    best = np.zeros(len(points), bool)
    generator = np.random.default_rng(0)
    for _ in range(256):
        a, b, c = points[generator.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < height**2 * 1e-6:
            continue
        inliers = np.abs((points - a) @ (normal / length)) <= threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(4, len(points) * 0.8):
        raise DomainValidationError("table markers do not support a consistent plane")
    points = points[best]
    origin = points.mean(0)
    _, singular, axes = np.linalg.svd(points - origin, full_matrices=False)
    if singular[1] < height * 0.05:
        raise DomainValidationError("table marker footprint is geometrically degenerate")
    normal = axes[2]
    centers = []
    for part in components:
        for image in part["images"]:
            pose = np.asarray(image["cam_from_world"])
            centers.append(-pose[:, :3].T @ pose[:, 3])
    side = (np.asarray(centers) - origin) @ normal
    normal *= 1 if np.median(side) > 0 else -1
    if np.mean((np.asarray(centers) - origin) @ normal > threshold) < 0.9:
        raise DomainValidationError("capture cameras must lie above the observed support plane")
    footprint = ConvexHull((points - origin) @ axes[:2].T)
    return {
        "origin": origin,
        "normal": normal,
        "basis": axes[:2],
        "footprint": footprint.equations,
        "height_m": height,
        "plane_rmse_m": float(np.sqrt(np.mean(((points - origin) @ normal) ** 2))),
    }


def _metric_object_depth_validity(
    depth_m: Any,
    rays: Any,
    object_mask: Any,
    cam_from_world: Any,
    support: dict[str, Any],
) -> Any:
    """Validate foreground depth in the observed metric support frame.

    The RGB mask defines object membership; this function only decides whether
    a positive depth measurement places that pixel inside the physically
    declared metric object-height envelope.  Camera distance is deliberately not
    bounded, so valid captures remain valid at any working distance.
    """
    import numpy as np

    depth = np.asarray(depth_m, dtype=float)
    rays = np.asarray(rays, dtype=float)
    mask = np.asarray(object_mask, dtype=bool)
    pose = np.asarray(cam_from_world, dtype=float)
    origin = np.asarray(support["origin"], dtype=float)
    normal = np.asarray(support["normal"], dtype=float)
    height = float(support["height_m"])
    plane_rmse = float(support["plane_rmse_m"])
    if (
        depth.ndim != 2
        or rays.shape != (*depth.shape, 3)
        or mask.shape != depth.shape
        or pose.shape != (3, 4)
        or origin.shape != (3,)
        or normal.shape != (3,)
        or not np.isfinite(depth).all()
        or not np.isfinite(rays).all()
        or not np.isfinite(pose).all()
        or not np.isfinite(origin).all()
        or not np.isfinite(normal).all()
        or not np.isfinite(height)
        or height <= 0
        or not np.isfinite(plane_rmse)
        or plane_rmse < 0
    ):
        raise DomainValidationError("metric object-depth evidence is malformed")
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-12:
        raise DomainValidationError("metric object-depth support normal is degenerate")
    normal = normal / normal_length
    # x_camera = R x_world + t, hence x_world = (x_camera - t) R.
    normal_in_camera = pose[:, :3] @ normal
    world_height = np.einsum(
        "ijk,k->ij", rays * depth[..., None] - pose[:, 3], normal_in_camera
    ) - float(origin @ normal)
    lower_tolerance = max(6 * plane_rmse, height * 0.05, 0.005)
    upper_tolerance = max(6 * plane_rmse, height * 0.15, 0.005)
    return (
        mask
        & (depth > 0)
        & (world_height >= -lower_tolerance)
        & (world_height <= height + upper_tolerance)
    )


def prepare_rgbd_frames(
    observations: ObservationSetManifest,
    camera_ref: ArtifactRef,
    artifacts: ArtifactStore,
    *,
    stride: int = 4,
    max_frames_per_stream: int | None = 8,
) -> ArtifactRef:
    """Publish real depth/RGB arrays and exact view bindings for downstream methods."""
    import numpy as np

    from physrealx.integrations.depth import CalibratedRGBDBundle, DepthFrame

    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or not 1 <= stride <= 16
        or (
            max_frames_per_stream is not None
            and (
                isinstance(max_frames_per_stream, bool)
                or not isinstance(max_frames_per_stream, int)
                or not 3 <= max_frames_per_stream <= _MAX_FRAMES_PER_STREAM
            )
        )
    ):
        raise DomainValidationError(
            "capture sampling requires integer stride 1..16 and 3..512 frames or None (all)"
        )
    camera = read_json(artifacts, camera_ref, "physrealx.camera-rig-calibration/1")
    source = camera.get("source_observations", {})
    if (
        source.get("resource_id") != observations.resource_id
        or source.get("version") != observations.version
    ):
        raise DomainValidationError("RGB-D observations differ from the metric camera source")
    if max_frames_per_stream is None:
        for stream in camera["streams"]:
            count = len(stream["components"][0]["images"])
            if count > _MAX_FRAMES_PER_STREAM:
                raise DomainValidationError(
                    f"RGB-D all-view stream {stream['stream_name']!r} has {count} frames; "
                    f"safety limit is {_MAX_FRAMES_PER_STREAM} per stream"
                )
    support = capture_support_region(camera)
    frames = []
    for stream in camera["streams"]:
        component = stream["components"][0]
        candidates = component["images"]
        count = len(candidates)
        if max_frames_per_stream is not None:
            count = min(count, max_frames_per_stream)
        indexes = np.linspace(0, len(candidates) - 1, count, dtype=int)
        images = [candidates[index] for index in indexes]
        calibration = {c["camera_id"]: c for c in component["cameras"]}
        depth_scale = camera["depth_calibrations"][stream["stream_name"]]["depth_scale_m"]
        for image, rgb, depth, mask in capture_frames(
            observations, artifacts, stream["stream_name"], images
        ):
            intrinsics = calibration[image["camera_id"]]
            h, w = depth.shape
            y, x = np.mgrid[0:h:stride, 0:w:stride]
            rays = camera_rays(intrinsics, np.column_stack([x.ravel(), y.ravel()])).reshape(
                *y.shape, 3
            )
            object_mask = (mask[y, x] > 0).astype(np.uint8)
            sampled_depth = depth[y, x].copy()
            measured = (object_mask > 0) & (sampled_depth > 0)
            accepted = _metric_object_depth_validity(
                sampled_depth.astype(float) * depth_scale,
                rays,
                object_mask,
                image["cam_from_world"],
                support,
            )
            sampled_depth[measured & ~accepted] = 0
            if object_mask.sum() < 100 or accepted.sum() < 100:
                raise MethodExecutionError(
                    f"registered view {image['name']} has insufficient valid object depth"
                )
            frames.append(
                DepthFrame(
                    frame_id=f"{stream['stream_name']}.{image['source_frame_index']}",
                    rgb=put_array(artifacts, rgb[::stride, ::stride].copy()),
                    depth=put_array(artifacts, sampled_depth),
                    depth_scale_m=depth_scale,
                    capture_id=f"{observations.resource_id}.{observations.version}.{stream['stream_name']}",
                    view={
                        "stream_name": stream["stream_name"],
                        "image_name": image["name"],
                        "source_frame_index": image["source_frame_index"],
                        "rays": put_array(artifacts, rays.astype(np.float32)).model_dump(
                            mode="json"
                        ),
                        "object_mask": put_array(artifacts, object_mask).model_dump(mode="json"),
                        "cam_from_world": image["cam_from_world"],
                        "source_rgb_artifact_id": image["source_artifact_id"],
                        "mask_source": "dataset_annotation",
                        "view_alignment": stream.get("view_alignment"),
                        "depth_validity": {
                            "method": "metric_support_envelope",
                            "measured_foreground_pixels": int(measured.sum()),
                            "accepted_foreground_pixels": int(accepted.sum()),
                            "rejected_foreground_pixels": int((measured & ~accepted).sum()),
                            "support_height_m": support["height_m"],
                            "support_plane_rmse_m": support["plane_rmse_m"],
                            "evidence_scope": camera.get("metric_evidence_scope"),
                        },
                    },
                )
            )
    bundle = CalibratedRGBDBundle(
        camera_rig=camera_ref, validation_scope="masked_observed_pixels", frames=tuple(frames)
    )
    return artifacts.put_json(bundle)
