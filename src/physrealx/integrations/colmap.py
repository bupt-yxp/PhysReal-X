"""COLMAP camera-recovery integration for immutable observation sets."""

from __future__ import annotations

import math
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, cast

from pydantic import Field, model_validator

from physrealx.contracts import (
    ArtifactStore,
    MethodExecutionOutput,
    MethodProbe,
)
from physrealx.digital_twin import (
    FiducialSpec,
    ObservationModality,
    ObservationSetManifest,
    ObservationStateRole,
    ObservationStream,
    ViewAlignmentMode,
    ViewAlignmentSpec,
)
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    JSONValue,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodMaturity,
    MethodRuntime,
    OperationOutput,
    OperationSpec,
    canonical_json_bytes,
)
from physrealx.errors import DomainValidationError, MethodExecutionError

_PYCOLMAP_VERSION = "4.1.1"
_OPENCV_VERSION = "4.10.0"
_CAMERA_SCHEMA = "physrealx.camera-rig-calibration/1"
_BUNDLE_SCHEMA = "physrealx.sparse-reconstruction-bundle/1"
_REPORT_SCHEMA = "physrealx.camera-recovery-report/1"
_FIDUCIAL_DICTIONARIES = {
    **{
        f"aruco_{bits}x{bits}_{size}": f"DICT_{bits}X{bits}_{size}"
        for bits in range(4, 8)
        for size in (50, 100, 250, 1000)
    },
    "aruco_original": "DICT_ARUCO_ORIGINAL",
    "apriltag_16h5": "DICT_APRILTAG_16H5",
    "apriltag_25h9": "DICT_APRILTAG_25H9",
    "apriltag_36h10": "DICT_APRILTAG_36H10",
    "apriltag_36h11": "DICT_APRILTAG_36H11",
}


COLMAP_DESCRIPTOR = MethodDescriptor(
    capability=MethodCapability.CAMERA_RECOVER,
    name="colmap",
    version="1.2.0",
    api_version="1",
    source_uri="https://github.com/colmap/colmap",
    source_revision=_PYCOLMAP_VERSION,
    license_spdx="BSD-3-Clause",
    maturity=MethodMaturity.STABLE,
    runtime=MethodRuntime(
        mode=MethodExecutionMode.IN_PROCESS,
        accelerator=MethodAccelerator.ANY,
        python=">=3.10,<3.13",
    ),
    input_contract="ObservationSetManifest/1",
    output_contract=("CameraRigCalibration/1+SparseReconstructionBundle/1+CameraRecoveryReport/1"),
)


class ColmapRunConfig(DomainModel):
    """Validated public configuration for one sparse reconstruction."""

    execution_device: Literal["auto", "cpu", "cuda"] = "auto"
    allow_cpu_fallback: bool = True
    matching: Literal["sequential", "exhaustive"] = "sequential"
    max_frames_per_stream: int = Field(default=32, ge=3, le=4096)
    max_num_features: int = Field(default=8192, ge=512, le=65536)
    sequential_overlap: int = Field(default=10, ge=1, le=100)
    num_threads: int = Field(default=1, ge=1, le=256)
    random_seed: int = Field(default=0, ge=0, le=2_147_483_647)
    min_registered_images: int = Field(default=3, ge=3)
    min_registration_rate: float = Field(default=0.5, gt=0.0, le=1.0)
    min_sparse_points: int = Field(default=20, ge=1)
    max_mean_reprojection_error_px: float = Field(default=4.0, gt=0.0, le=100.0)
    min_fiducial_frames: int = Field(default=3, ge=2)
    min_fiducial_detections: int = Field(default=3, ge=2)
    min_fiducial_tracks: int = Field(default=4, ge=3)
    max_fiducial_reprojection_error_px: float = Field(default=4.0, gt=0.0, le=100.0)
    min_cross_stream_fiducial_tracks: int = Field(default=6, ge=3)
    min_cross_stream_inlier_ratio: float = Field(default=0.75, gt=0.0, le=1.0)
    max_cross_stream_normalized_error: float = Field(default=0.05, gt=0.0, le=1.0)
    stream_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_thresholds(self) -> ColmapRunConfig:
        if self.min_registered_images > self.max_frames_per_stream:
            raise ValueError("min_registered_images cannot exceed max_frames_per_stream")
        if len(self.stream_names) != len(set(self.stream_names)):
            raise ValueError("stream_names must be unique")
        return self


@dataclass(frozen=True)
class _FiducialMarkerDetection:
    marker_id: int
    corners_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _FiducialFrameDetection:
    image_name: str
    markers: tuple[_FiducialMarkerDetection, ...]


@dataclass(frozen=True)
class _ComponentResult:
    component_id: int
    cameras: tuple[dict[str, Any], ...]
    images: tuple[dict[str, Any], ...]
    sparse_point_count: int
    observation_count: int
    mean_track_length: float
    mean_reprojection_error_px: float
    fiducial_tracks: tuple[dict[str, Any], ...]
    files: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True)
class _StreamResult:
    stream_name: str
    selected_image_count: int
    components: tuple[_ComponentResult, ...]
    execution_device: Literal["cpu", "cuda"]
    view_alignment: ViewAlignmentSpec
    fiducial_frame_count: int
    fiducial_detection_count: int

    @property
    def registered_image_count(self) -> int:
        return len(
            {
                cast(str, image["name"])
                for component in self.components
                for image in component.images
            }
        )

    @property
    def sparse_point_count(self) -> int:
        return sum(item.sparse_point_count for item in self.components)

    @property
    def observation_count(self) -> int:
        return sum(item.observation_count for item in self.components)


class _ColmapRunner(Protocol):
    def probe(self) -> MethodProbe: ...

    def run_stream(
        self,
        stream_name: str,
        image_directory: Path,
        selected_image_count: int,
        config: ColmapRunConfig,
        view_alignment: ViewAlignmentSpec,
        fiducials: tuple[_FiducialFrameDetection, ...],
    ) -> _StreamResult: ...

    def align_fiducial_streams(
        self,
        results: tuple[_StreamResult, ...],
        config: ColmapRunConfig,
    ) -> tuple[dict[str, Any], ...]: ...


class _FiducialDetector(Protocol):
    def probe(self) -> MethodProbe: ...

    def detect(
        self,
        image_directory: Path,
        specification: FiducialSpec,
    ) -> tuple[_FiducialFrameDetection, ...]: ...


def _load_pycolmap() -> ModuleType:
    return import_module("pycolmap")


def _load_cv2() -> ModuleType:
    return import_module("cv2")


def _finite_float(value: object, label: str) -> float:
    result = float(cast(Any, value))
    if not math.isfinite(result):
        raise MethodExecutionError(f"COLMAP returned non-finite {label}")
    return result


def _matrix_rows(value: object) -> list[list[float]]:
    raw = cast(Any, value).tolist()
    return [[_finite_float(item, "camera pose") for item in row] for row in raw]


class _OpenCvFiducialDetector:
    """OpenCV boundary for declared ArUco/AprilTag families."""

    def probe(self) -> MethodProbe:
        try:
            cv2 = _load_cv2()
            version = getattr(cv2, "__version__", None)
            aruco = getattr(cv2, "aruco", None)
            if version != _OPENCV_VERSION:
                return MethodProbe(
                    False,
                    f"OpenCV {_OPENCV_VERSION} is required; found {version!r}",
                )
            if aruco is None or not callable(getattr(aruco, "ArucoDetector", None)):
                return MethodProbe(False, "OpenCV ArUco detector API is unavailable")
            missing = tuple(
                family
                for family, constant in _FIDUCIAL_DICTIONARIES.items()
                if not hasattr(aruco, constant)
            )
            if missing:
                return MethodProbe(
                    False,
                    f"OpenCV does not provide declared fiducial families: {missing!r}",
                )
        except Exception as error:
            return MethodProbe(False, f"OpenCV fiducial probe failed: {error}")
        return MethodProbe(True, f"OpenCV {version} ArUco ready")

    def detect(
        self,
        image_directory: Path,
        specification: FiducialSpec,
    ) -> tuple[_FiducialFrameDetection, ...]:
        constant = _FIDUCIAL_DICTIONARIES.get(specification.family)
        if constant is None:
            raise DomainValidationError(
                f"unsupported fiducial family {specification.family!r}; "
                f"supported families: {sorted(_FIDUCIAL_DICTIONARIES)!r}"
            )
        cv2 = _load_cv2()
        api = cast(Any, cv2)
        parameters = api.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = api.aruco.CORNER_REFINE_SUBPIX
        detector = api.aruco.ArucoDetector(
            api.aruco.getPredefinedDictionary(getattr(api.aruco, constant)),
            parameters,
        )
        frames: list[_FiducialFrameDetection] = []
        for path in sorted(image_directory.iterdir(), key=lambda item: item.name):
            image = api.imread(str(path), api.IMREAD_COLOR)
            if image is None:
                raise MethodExecutionError(f"OpenCV could not decode marker frame {path.name!r}")
            corners, identifiers, _ = detector.detectMarkers(image)
            if identifiers is None:
                continue
            markers = tuple(
                _FiducialMarkerDetection(
                    marker_id=int(marker_id),
                    corners_xy=tuple(
                        (
                            _finite_float(point[0], "fiducial x"),
                            _finite_float(point[1], "fiducial y"),
                        )
                        for point in marker_corners.reshape(4, 2)
                    ),
                )
                for marker_id, marker_corners in sorted(
                    zip(identifiers.reshape(-1).tolist(), corners, strict=True),
                    key=lambda item: int(item[0]),
                )
            )
            frames.append(_FiducialFrameDetection(image_name=path.name, markers=markers))
        return tuple(frames)


def _triangulate_fiducial_tracks(
    api: Any,
    reconstruction: Any,
    detections: tuple[_FiducialFrameDetection, ...],
) -> tuple[dict[str, Any], ...]:
    numpy = import_module("numpy")
    np = cast(Any, numpy)
    images = {str(image.name): image for image in reconstruction.images.values()}
    grouped: dict[tuple[int, int], list[tuple[Any, Any, tuple[float, float]]]] = {}
    for frame in detections:
        image = images.get(frame.image_name)
        if image is None:
            continue
        camera = reconstruction.cameras[image.camera_id]
        for marker in frame.markers:
            for corner_index, xy in enumerate(marker.corners_xy):
                grouped.setdefault((marker.marker_id, corner_index), []).append((image, camera, xy))

    tracks: list[dict[str, Any]] = []
    for (marker_id, corner_index), observations in sorted(grouped.items()):
        if len(observations) < 2:
            continue
        poses = [np.asarray(image.cam_from_world().matrix()) for image, _, _ in observations]
        rays = np.asarray(
            [camera.cam_ray_from_img(xy) for _, camera, xy in observations],
            dtype=np.float64,
        )
        point = api.triangulate_multi_view_point(poses, rays)
        if point is None:
            continue
        xyz = np.asarray(point, dtype=np.float64).reshape(3)
        if not bool(np.isfinite(xyz).all()):
            continue
        residuals: list[float] = []
        valid = True
        for pose, (_, camera, xy) in zip(poses, observations, strict=True):
            camera_xyz = pose[:, :3] @ xyz + pose[:, 3]
            if camera_xyz[2] <= 1e-8:
                valid = False
                break
            projected = np.asarray(camera.img_from_cam(camera_xyz), dtype=np.float64)
            residuals.append(float(np.linalg.norm(projected - np.asarray(xy))))
        if not valid or not residuals:
            continue
        reprojection_error = math.sqrt(sum(value * value for value in residuals) / len(residuals))
        tracks.append(
            {
                "marker_id": marker_id,
                "corner_index": corner_index,
                "xyz": [_finite_float(value, "fiducial point") for value in xyz],
                "observation_count": len(observations),
                "reprojection_error_px": _finite_float(
                    reprojection_error,
                    "fiducial reprojection error",
                ),
                "observations": [
                    {"image_name": image.name, "xy": [xy[0], xy[1]]}
                    for image, _, xy in observations
                ],
            }
        )
    return tuple(tracks)


class _PycolmapRunner:
    """Thin runtime boundary around the pinned official PyCOLMAP API."""

    def probe(self) -> MethodProbe:
        try:
            pycolmap = _load_pycolmap()
            version = getattr(pycolmap, "__version__", None)
            required = (
                "extract_features",
                "match_sequential",
                "match_exhaustive",
                "incremental_mapping",
                "Reconstruction",
            )
            missing = tuple(
                name for name in required if not callable(getattr(pycolmap, name, None))
            )
            if version != _PYCOLMAP_VERSION:
                return MethodProbe(
                    False,
                    f"PyCOLMAP {_PYCOLMAP_VERSION} is required; found {version!r}",
                )
            if missing:
                return MethodProbe(False, f"PyCOLMAP API is incomplete: {', '.join(missing)}")
            options = cast(Any, pycolmap).IncrementalPipelineOptions()
            if not hasattr(options, "random_seed"):
                return MethodProbe(False, "PyCOLMAP mapper does not expose deterministic seeds")
        except Exception as error:
            return MethodProbe(False, f"PyCOLMAP import or API probe failed: {error}")
        accelerator = "CUDA" if bool(getattr(pycolmap, "has_cuda", False)) else "CPU"
        return MethodProbe(True, f"PyCOLMAP {version} ready ({accelerator})")

    @staticmethod
    def _device(pycolmap: ModuleType, config: ColmapRunConfig) -> Literal["cpu", "cuda"]:
        has_cuda = bool(getattr(pycolmap, "has_cuda", False))
        if config.execution_device == "cuda" and not has_cuda:
            if not config.allow_cpu_fallback:
                raise MethodExecutionError(
                    "COLMAP CUDA was requested but this PyCOLMAP build has no CUDA support"
                )
            return "cpu"
        if config.execution_device == "cpu":
            return "cpu"
        return "cuda" if has_cuda else "cpu"

    def run_stream(
        self,
        stream_name: str,
        image_directory: Path,
        selected_image_count: int,
        config: ColmapRunConfig,
        view_alignment: ViewAlignmentSpec,
        fiducials: tuple[_FiducialFrameDetection, ...],
    ) -> _StreamResult:
        pycolmap = _load_pycolmap()
        api = cast(Any, pycolmap)
        device_name = self._device(pycolmap, config)
        device = api.Device.cuda if device_name == "cuda" else api.Device.cpu
        root = image_directory.parent
        database = root / "database.db"
        mapped = root / "mapped"
        mapped.mkdir()

        reader_options = api.ImageReaderOptions()
        reader_options.camera_model = "SIMPLE_RADIAL"
        extraction_options = api.FeatureExtractionOptions()
        extraction_options.num_threads = config.num_threads
        extraction_options.sift.max_num_features = config.max_num_features
        matching_options = api.FeatureMatchingOptions()
        matching_options.num_threads = config.num_threads
        mapper_options = api.IncrementalPipelineOptions()
        mapper_options.num_threads = config.num_threads
        mapper_options.random_seed = config.random_seed
        mapper_options.mapper.num_threads = config.num_threads
        mapper_options.mapper.random_seed = config.random_seed
        mapper_options.triangulation.random_seed = config.random_seed

        try:
            api.extract_features(
                database,
                image_directory,
                camera_mode=api.CameraMode.SINGLE,
                reader_options=reader_options,
                extraction_options=extraction_options,
                device=device,
            )
            if config.matching == "sequential":
                pairing_options = api.SequentialPairingOptions()
                pairing_options.overlap = config.sequential_overlap
                pairing_options.num_threads = config.num_threads
                api.match_sequential(
                    database,
                    matching_options=matching_options,
                    pairing_options=pairing_options,
                    device=device,
                )
            else:
                api.match_exhaustive(
                    database,
                    matching_options=matching_options,
                    device=device,
                )
            reconstructions = api.incremental_mapping(
                database,
                image_directory,
                mapped,
                options=mapper_options,
            )
        except Exception as error:
            raise MethodExecutionError(
                f"COLMAP failed while reconstructing stream {stream_name!r}: {error}"
            ) from error
        if not reconstructions:
            raise MethodExecutionError(
                f"COLMAP produced no sparse reconstruction for stream {stream_name!r}"
            )

        components: list[_ComponentResult] = []
        for component_id, reconstruction in sorted(reconstructions.items()):
            text_root = root / f"component-{component_id}"
            text_root.mkdir()
            reconstruction.write_text(text_root)
            cameras = tuple(
                {
                    "camera_id": int(camera_id),
                    "model": str(camera.model_name),
                    "width": int(camera.width),
                    "height": int(camera.height),
                    "params": [_finite_float(item, "camera parameter") for item in camera.params],
                }
                for camera_id, camera in sorted(reconstruction.cameras.items())
            )
            images = tuple(
                {
                    "image_id": int(image_id),
                    "name": str(image.name),
                    "camera_id": int(image.camera_id),
                    "cam_from_world": _matrix_rows(image.cam_from_world().matrix()),
                }
                for image_id, image in sorted(reconstruction.images.items())
            )
            files = tuple(
                (path.name, path.read_bytes())
                for path in sorted(text_root.glob("*.txt"))
                if path.name
                in {"cameras.txt", "frames.txt", "images.txt", "points3D.txt", "rigs.txt"}
            )
            components.append(
                _ComponentResult(
                    component_id=int(component_id),
                    cameras=cameras,
                    images=images,
                    sparse_point_count=int(reconstruction.num_points3D()),
                    observation_count=int(reconstruction.compute_num_observations()),
                    mean_track_length=_finite_float(
                        reconstruction.compute_mean_track_length(), "mean track length"
                    ),
                    mean_reprojection_error_px=_finite_float(
                        reconstruction.compute_mean_reprojection_error(),
                        "mean reprojection error",
                    ),
                    fiducial_tracks=_triangulate_fiducial_tracks(
                        api,
                        reconstruction,
                        fiducials,
                    ),
                    files=files,
                )
            )
        return _StreamResult(
            stream_name=stream_name,
            selected_image_count=selected_image_count,
            components=tuple(components),
            execution_device=device_name,
            view_alignment=view_alignment,
            fiducial_frame_count=len(fiducials),
            fiducial_detection_count=sum(len(frame.markers) for frame in fiducials),
        )

    def align_fiducial_streams(
        self,
        results: tuple[_StreamResult, ...],
        config: ColmapRunConfig,
    ) -> tuple[dict[str, Any], ...]:
        pycolmap = cast(Any, _load_pycolmap())
        numpy = cast(Any, import_module("numpy"))
        grouped: dict[str, list[tuple[str, int, dict[tuple[int, int], list[float]]]]] = {}
        for result in results:
            specification = result.view_alignment.fiducial
            if specification is None or specification.layout_id is None:
                continue
            for component in result.components:
                tracks = {
                    (int(track["marker_id"]), int(track["corner_index"])): cast(
                        list[float], track["xyz"]
                    )
                    for track in component.fiducial_tracks
                }
                if tracks:
                    grouped.setdefault(specification.layout_id, []).append(
                        (result.stream_name, component.component_id, tracks)
                    )

        alignments: list[dict[str, Any]] = []
        for layout_id, entries in sorted(grouped.items()):
            entries.sort(key=lambda item: (item[0], item[1]))
            if len(entries) < 2:
                continue
            target_stream, target_component, target_tracks = entries[0]
            target_keys = set(target_tracks)
            for source_stream, source_component, source_tracks in entries[1:]:
                common = sorted(target_keys & set(source_tracks))
                if len(common) < config.min_cross_stream_fiducial_tracks:
                    raise MethodExecutionError(
                        f"fiducial layout {layout_id!r} has only {len(common)} common tracks "
                        f"between {source_stream!r} and {target_stream!r}"
                    )
                source = numpy.asarray([source_tracks[key] for key in common], dtype=numpy.float64)
                target = numpy.asarray([target_tracks[key] for key in common], dtype=numpy.float64)
                centered = target - target.mean(axis=0)
                span = float(numpy.linalg.norm(centered, axis=1).max())
                if not math.isfinite(span) or span <= 1e-8:
                    raise MethodExecutionError(
                        f"fiducial layout {layout_id!r} is geometrically degenerate"
                    )
                options = pycolmap.RANSACOptions()
                options.max_error = config.max_cross_stream_normalized_error * span
                options.min_inlier_ratio = config.min_cross_stream_inlier_ratio
                options.random_seed = config.random_seed
                options.num_threads = 1
                estimation = pycolmap.estimate_sim3d_robust(source, target, options)
                if estimation is None:
                    raise MethodExecutionError(
                        f"fiducial alignment failed between {source_stream!r} and {target_stream!r}"
                    )
                transform = estimation.get("tgt_from_src")
                if transform is None:
                    raise MethodExecutionError("PyCOLMAP returned no fiducial Sim(3) transform")
                matrix = numpy.asarray(transform.matrix(), dtype=numpy.float64)
                predicted = (matrix[:, :3] @ source.T).T + matrix[:, 3]
                residuals = numpy.linalg.norm(predicted - target, axis=1)
                threshold = config.max_cross_stream_normalized_error * span
                inlier_mask = numpy.asarray(estimation.get("inlier_mask"), dtype=numpy.bool_)
                if inlier_mask.shape != residuals.shape:
                    raise MethodExecutionError("PyCOLMAP returned an invalid fiducial inlier mask")
                inlier_mask &= residuals <= threshold
                inlier_count = int(inlier_mask.sum())
                inlier_ratio = inlier_count / len(common)
                normalized_error = (
                    float(numpy.sqrt(numpy.mean(numpy.square(residuals[inlier_mask])))) / span
                    if inlier_count
                    else math.inf
                )
                if (
                    inlier_count < config.min_cross_stream_fiducial_tracks
                    or inlier_ratio < config.min_cross_stream_inlier_ratio
                    or not math.isfinite(normalized_error)
                    or normalized_error > config.max_cross_stream_normalized_error
                ):
                    raise MethodExecutionError(
                        f"fiducial alignment between {source_stream!r} and "
                        f"{target_stream!r} failed quality gates"
                    )
                alignments.append(
                    {
                        "layout_id": layout_id,
                        "target_stream_name": target_stream,
                        "target_component_id": target_component,
                        "source_stream_name": source_stream,
                        "source_component_id": source_component,
                        "target_from_source": [
                            [_finite_float(value, "fiducial alignment") for value in row]
                            for row in matrix.tolist()
                        ],
                        "scale": _finite_float(transform.scale, "fiducial alignment scale"),
                        "common_track_count": len(common),
                        "inlier_count": inlier_count,
                        "inlier_ratio": inlier_ratio,
                        "normalized_rmse": normalized_error,
                    }
                )
        return tuple(alignments)


def _selected_indexes(frame_count: int, maximum: int) -> tuple[int, ...]:
    if frame_count <= maximum:
        return tuple(range(frame_count))
    denominator = maximum - 1
    return tuple(round(index * (frame_count - 1) / denominator) for index in range(maximum))


def _rgb_streams(
    manifest: ObservationSetManifest,
    config: ColmapRunConfig,
) -> tuple[ObservationStream, ...]:
    rgb_streams = tuple(
        stream for stream in manifest.streams if stream.modality is ObservationModality.RGB
    )
    if config.stream_names:
        by_name = {stream.name: stream for stream in rgb_streams}
        missing = tuple(name for name in config.stream_names if name not in by_name)
        if missing:
            raise DomainValidationError(f"COLMAP stream_names are not RGB streams: {missing!r}")
        candidates = tuple(by_name[name] for name in config.stream_names)
    else:
        candidates = tuple(
            stream for stream in rgb_streams if stream.role is ObservationStateRole.RECONSTRUCTION
        )
    if not candidates:
        raise DomainValidationError("COLMAP requires at least one reconstruction RGB stream")
    packed = tuple(
        stream.name for stream in candidates if len(stream.artifacts) != stream.frame_count
    )
    if packed:
        raise DomainValidationError(
            f"COLMAP requires one image artifact per RGB frame; packed streams: {packed!r}"
        )
    return tuple(sorted(candidates, key=lambda item: item.name))


def _fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _build_sparse_bundle(
    destination: Path,
    observation_artifact_id: str,
    streams: Sequence[_StreamResult],
) -> None:
    files: list[tuple[str, bytes]] = []
    components: list[dict[str, Any]] = []
    for stream in streams:
        for component in stream.components:
            prefix = f"streams/{stream.stream_name}/component-{component.component_id}"
            paths: list[str] = []
            for name, payload in component.files:
                path = f"{prefix}/{name}"
                files.append((path, payload))
                paths.append(path)
            components.append(
                {
                    "stream_name": stream.stream_name,
                    "component_id": component.component_id,
                    "files": paths,
                }
            )
    file_records: list[dict[str, Any]] = [
        {"path": path, "size_bytes": len(payload), "media_type": "text/plain"}
        for path, payload in sorted(files)
    ]
    manifest: dict[str, Any] = {
        "schema": _BUNDLE_SCHEMA,
        "status": "PASSED",
        "scale_status": "unresolved",
        "source_observation_artifact_id": observation_artifact_id,
        "components": components,
        "files": file_records,
    }
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        archive.writestr(
            _fixed_zip_info("bundle-manifest.json"),
            canonical_json_bytes(cast(JSONValue, manifest)),
        )
        for path, payload in sorted(files):
            archive.writestr(_fixed_zip_info(path), payload)


def _weighted_mean(
    values: Sequence[tuple[float, int]],
    *,
    label: str,
) -> float:
    weight = sum(item[1] for item in values)
    if weight <= 0:
        raise MethodExecutionError(f"COLMAP returned no samples for {label}")
    return sum(item[0] * item[1] for item in values) / weight


class ColmapMethodAdapter:
    """Registered ``camera.recover:colmap`` implementation."""

    descriptor = COLMAP_DESCRIPTOR

    def __init__(
        self,
        runner: _ColmapRunner | None = None,
        fiducial_detector: _FiducialDetector | None = None,
    ) -> None:
        self._runner = runner or _PycolmapRunner()
        self._fiducial_detector = fiducial_detector or _OpenCvFiducialDetector()

    def probe(self) -> MethodProbe:
        runner = self._runner.probe()
        if not runner.ready:
            return runner
        detector = self._fiducial_detector.probe()
        if not detector.ready:
            return detector
        return MethodProbe(True, f"{runner.detail}; {detector.detail}")

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.descriptor != self.descriptor:
            raise DomainValidationError("COLMAP operation uses a mismatched method descriptor")
        if len(spec.inputs) != 1 or spec.inputs[0].name != "observations":
            raise DomainValidationError("COLMAP requires exactly one observations resource")
        observations = spec.inputs[0].resource
        if observations is None or observations.artifact_id is None:
            raise DomainValidationError("COLMAP observations input must be a resolved resource")
        try:
            manifest = ObservationSetManifest.model_validate_json(
                artifacts.get_bytes(observations.artifact_id)
            )
            config = ColmapRunConfig.model_validate(dict(spec.method.config))
        except ValueError as error:
            raise DomainValidationError(
                "COLMAP input manifest or configuration is invalid"
            ) from error
        if (manifest.resource_id, manifest.version) != (
            observations.resource_id,
            observations.version,
        ):
            raise DomainValidationError("COLMAP observation manifest identity does not match input")

        streams = _rgb_streams(manifest, config)
        results: list[_StreamResult] = []
        source_bindings: dict[str, dict[str, tuple[int, ArtifactRef]]] = {}
        with tempfile.TemporaryDirectory(prefix="physrealx-colmap-") as temporary:
            root = Path(temporary)
            for stream_index, stream in enumerate(streams):
                indexes = _selected_indexes(stream.frame_count, config.max_frames_per_stream)
                stream_root = root / f"stream-{stream_index:03d}"
                image_root = stream_root / "images"
                image_root.mkdir(parents=True)
                bindings: dict[str, tuple[int, ArtifactRef]] = {}
                for output_index, source_index in enumerate(indexes):
                    artifact = stream.artifacts[source_index]
                    if artifact.media_type not in {"image/jpeg", "image/png"}:
                        raise DomainValidationError(
                            f"COLMAP stream {stream.name!r} contains unsupported media type "
                            f"{artifact.media_type!r}"
                        )
                    suffix = ".jpg" if artifact.media_type == "image/jpeg" else ".png"
                    name = f"{output_index:06d}{suffix}"
                    artifacts.materialize(artifact, image_root / name)
                    bindings[name] = (source_index, artifact)
                source_bindings[stream.name] = bindings
                view_alignment = stream.view_alignment or ViewAlignmentSpec()
                fiducials = (
                    self._fiducial_detector.detect(
                        image_root,
                        cast(FiducialSpec, view_alignment.fiducial),
                    )
                    if view_alignment.mode is ViewAlignmentMode.FIDUCIAL_ASSISTED
                    else ()
                )
                results.append(
                    self._runner.run_stream(
                        stream.name,
                        image_root,
                        len(indexes),
                        config,
                        view_alignment,
                        fiducials,
                    )
                )

            selected_total = sum(item.selected_image_count for item in results)
            fiducial_alignments = self._runner.align_fiducial_streams(tuple(results), config)
            registered_total = sum(item.registered_image_count for item in results)
            sparse_points = sum(item.sparse_point_count for item in results)
            observations_total = sum(item.observation_count for item in results)
            registration_rate = registered_total / selected_total
            mean_track_length = _weighted_mean(
                [
                    (component.mean_track_length, component.sparse_point_count)
                    for stream in results
                    for component in stream.components
                ],
                label="track length",
            )
            mean_reprojection_error = _weighted_mean(
                [
                    (component.mean_reprojection_error_px, component.observation_count)
                    for stream in results
                    for component in stream.components
                ],
                label="reprojection error",
            )
            stream_reports: list[dict[str, Any]] = []
            camera_streams: list[dict[str, Any]] = []
            fiducial_frame_total = 0
            fiducial_detection_total = 0
            fiducial_track_total = 0
            fiducial_track_observation_total = 0
            fiducial_error_weighted_sum = 0.0
            for result in results:
                registered = result.registered_image_count
                stream_rate = registered / result.selected_image_count
                stream_mean_track_length = _weighted_mean(
                    [
                        (component.mean_track_length, component.sparse_point_count)
                        for component in result.components
                    ],
                    label=f"track length for {result.stream_name}",
                )
                stream_mean_reprojection_error = _weighted_mean(
                    [
                        (component.mean_reprojection_error_px, component.observation_count)
                        for component in result.components
                    ],
                    label=f"reprojection error for {result.stream_name}",
                )
                fiducial_tracks = [
                    track for component in result.components for track in component.fiducial_tracks
                ]
                fiducial_track_observations = sum(
                    int(track["observation_count"]) for track in fiducial_tracks
                )
                fiducial_reprojection_error = (
                    sum(
                        float(track["reprojection_error_px"]) * int(track["observation_count"])
                        for track in fiducial_tracks
                    )
                    / fiducial_track_observations
                    if fiducial_track_observations
                    else None
                )
                gates = {
                    "registered_images": registered >= config.min_registered_images,
                    "registration_rate": stream_rate >= config.min_registration_rate,
                    "sparse_points": result.sparse_point_count >= config.min_sparse_points,
                }
                if result.view_alignment.mode is ViewAlignmentMode.FIDUCIAL_ASSISTED:
                    gates.update(
                        {
                            "fiducial_frames": (
                                result.fiducial_frame_count >= config.min_fiducial_frames
                            ),
                            "fiducial_detections": (
                                result.fiducial_detection_count >= config.min_fiducial_detections
                            ),
                            "fiducial_tracks": len(fiducial_tracks) >= config.min_fiducial_tracks,
                            "fiducial_reprojection_error": (
                                fiducial_reprojection_error is not None
                                and fiducial_reprojection_error
                                <= config.max_fiducial_reprojection_error_px
                            ),
                        }
                    )
                fiducial_frame_total += result.fiducial_frame_count
                fiducial_detection_total += result.fiducial_detection_count
                fiducial_track_total += len(fiducial_tracks)
                fiducial_track_observation_total += fiducial_track_observations
                if fiducial_reprojection_error is not None:
                    fiducial_error_weighted_sum += (
                        fiducial_reprojection_error * fiducial_track_observations
                    )
                stream_reports.append(
                    {
                        "stream_name": result.stream_name,
                        "view_alignment_mode": result.view_alignment.mode.value,
                        "fiducial_family": (
                            result.view_alignment.fiducial.family
                            if result.view_alignment.fiducial is not None
                            else None
                        ),
                        "selected_image_count": result.selected_image_count,
                        "registered_image_count": registered,
                        "registration_rate": stream_rate,
                        "component_count": len(result.components),
                        "sparse_point_count": result.sparse_point_count,
                        "observation_count": result.observation_count,
                        "mean_track_length": stream_mean_track_length,
                        "mean_reprojection_error_px": stream_mean_reprojection_error,
                        "fiducial_frame_count": result.fiducial_frame_count,
                        "fiducial_detection_count": result.fiducial_detection_count,
                        "fiducial_track_count": len(fiducial_tracks),
                        "fiducial_track_observation_count": fiducial_track_observations,
                        "fiducial_mean_reprojection_error_px": (fiducial_reprojection_error),
                        "gates": gates,
                    }
                )
                components: list[dict[str, Any]] = []
                for component in result.components:
                    images: list[dict[str, Any]] = []
                    for image in component.images:
                        name = cast(str, image["name"])
                        binding = source_bindings[result.stream_name].get(name)
                        if binding is None:
                            raise MethodExecutionError(
                                f"COLMAP returned unknown image name {name!r}"
                            )
                        source_index, source_artifact = binding
                        images.append(
                            {
                                **image,
                                "source_frame_index": source_index,
                                "source_artifact_id": source_artifact.artifact_id,
                            }
                        )
                    tracks: list[dict[str, Any]] = []
                    for track in component.fiducial_tracks:
                        observations_with_source: list[dict[str, Any]] = []
                        for observation in cast(
                            list[dict[str, Any]],
                            track["observations"],
                        ):
                            image_name = cast(str, observation["image_name"])
                            binding = source_bindings[result.stream_name].get(image_name)
                            if binding is None:
                                raise MethodExecutionError(
                                    f"fiducial track references unknown image {image_name!r}"
                                )
                            source_index, source_artifact = binding
                            observations_with_source.append(
                                {
                                    **observation,
                                    "source_frame_index": source_index,
                                    "source_artifact_id": source_artifact.artifact_id,
                                }
                            )
                        tracks.append({**track, "observations": observations_with_source})
                    components.append(
                        {
                            "component_id": component.component_id,
                            "cameras": list(component.cameras),
                            "images": images,
                            "sparse_point_count": component.sparse_point_count,
                            "observation_count": component.observation_count,
                            "mean_track_length": component.mean_track_length,
                            "mean_reprojection_error_px": (component.mean_reprojection_error_px),
                            "fiducial_tracks": tracks,
                        }
                    )
                camera_streams.append(
                    {
                        "stream_name": result.stream_name,
                        "execution_device": result.execution_device,
                        "view_alignment": result.view_alignment.model_dump(mode="json"),
                        "selected_image_count": result.selected_image_count,
                        "registered_image_count": registered,
                        "components": components,
                    }
                )

            gates = {
                "all_streams_reconstructed": all(item.components for item in results),
                "all_streams_registered": all(
                    item.registered_image_count >= config.min_registered_images for item in results
                ),
                "registration_rate": registration_rate >= config.min_registration_rate,
                "sparse_points": sparse_points >= config.min_sparse_points * len(results),
                "reprojection_error": (
                    mean_reprojection_error <= config.max_mean_reprojection_error_px
                ),
            }
            if not all(gates.values()) or any(
                not all(cast(Mapping[str, bool], item["gates"]).values()) for item in stream_reports
            ):
                failed = tuple(name for name, passed in gates.items() if not passed)
                raise MethodExecutionError(
                    f"COLMAP reconstruction failed quality gates: {failed or ('stream_gate',)!r}"
                )

            metrics: dict[str, float] = {
                "selected_image_count": float(selected_total),
                "registered_image_count": float(registered_total),
                "registration_rate": registration_rate,
                "component_count": float(sum(len(item.components) for item in results)),
                "sparse_point_count": float(sparse_points),
                "observation_count": float(observations_total),
                "mean_track_length": mean_track_length,
                "mean_reprojection_error_px": mean_reprojection_error,
                "fiducial_frame_count": float(fiducial_frame_total),
                "fiducial_detection_count": float(fiducial_detection_total),
                "fiducial_track_count": float(fiducial_track_total),
                "fiducial_track_observation_count": float(fiducial_track_observation_total),
                "fiducial_alignment_count": float(len(fiducial_alignments)),
            }
            if fiducial_track_observation_total:
                metrics["fiducial_mean_reprojection_error_px"] = (
                    fiducial_error_weighted_sum / fiducial_track_observation_total
                )
            method_identity: dict[str, Any] = {
                "name": self.descriptor.name,
                "version": self.descriptor.version,
                "source_revision": self.descriptor.source_revision,
                "runtime": "pycolmap",
                "runtime_version": _PYCOLMAP_VERSION,
            }
            camera_payload: dict[str, Any] = {
                "schema": _CAMERA_SCHEMA,
                "status": "PASSED",
                "scale_status": "unresolved",
                "unit": "arbitrary",
                "coordinate_frame": "colmap_world_unscaled",
                "source_observations": observations.model_dump(mode="json"),
                "method": method_identity,
                "streams": camera_streams,
                "fiducial_alignments": list(fiducial_alignments),
                "metrics": metrics,
            }
            report_payload: dict[str, Any] = {
                "schema": _REPORT_SCHEMA,
                "status": "PASSED",
                "scale_status": "unresolved",
                "source_observations": observations.model_dump(mode="json"),
                "method": method_identity,
                "configuration": config.model_dump(mode="json"),
                "streams": stream_reports,
                "fiducial_alignments": list(fiducial_alignments),
                "metrics": metrics,
                "gates": gates,
            }
            camera_ref = artifacts.put_json(camera_payload, media_type="application/json")
            report_ref = artifacts.put_json(report_payload, media_type="application/json")
            bundle_path = root / "sparse-reconstruction.zip"
            _build_sparse_bundle(bundle_path, observations.artifact_id, results)
            bundle_ref = artifacts.put_file(bundle_path, media_type="application/zip")

        by_contract = {
            "CameraRigCalibration/1": camera_ref,
            "SparseReconstructionBundle/1": bundle_ref,
            "CameraRecoveryReport/1": report_ref,
        }
        return MethodExecutionOutput(
            outputs=tuple(
                OperationOutput(
                    name=requirement.name,
                    contract=requirement.contract,
                    artifact=by_contract[requirement.contract],
                )
                for requirement in spec.outputs
            ),
            metrics=metrics,
        )


def create_colmap_adapter() -> ColmapMethodAdapter | None:
    """Return a registered adapter only when the pinned runtime passes its probe."""

    adapter = ColmapMethodAdapter()
    return adapter if adapter.probe().ready else None


__all__ = [
    "COLMAP_DESCRIPTOR",
    "ColmapMethodAdapter",
    "ColmapRunConfig",
    "create_colmap_adapter",
]
