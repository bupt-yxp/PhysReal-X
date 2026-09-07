from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from physrealx import platform as platform_module
from physrealx.contracts import MethodProbe
from physrealx.digital_twin import (
    FiducialSpec,
    ObservationModality,
    ObservationSetManifest,
    ObservationStateRole,
    ObservationStream,
    ViewAlignmentMode,
    ViewAlignmentSpec,
)
from physrealx.domain import MethodCapability
from physrealx.errors import MethodExecutionError
from physrealx.integrations.colmap import (
    ColmapMethodAdapter,
    ColmapRunConfig,
    _ComponentResult,
    _FiducialFrameDetection,
    _FiducialMarkerDetection,
    _OpenCvFiducialDetector,
    _PycolmapRunner,
    _StreamResult,
)
from physrealx.platform import PhysRealX


class _FakeColmapRunner:
    def __init__(self, registered_images: int = 3) -> None:
        self.registered_images = registered_images

    def probe(self) -> MethodProbe:
        return MethodProbe(True, "fake PyCOLMAP probe passed")

    def run_stream(
        self,
        stream_name: str,
        image_directory: Path,
        selected_image_count: int,
        config: ColmapRunConfig,
        view_alignment: ViewAlignmentSpec,
        fiducials: tuple[_FiducialFrameDetection, ...],
    ) -> _StreamResult:
        assert sorted(path.name for path in image_directory.iterdir()) == [
            "000000.png",
            "000001.png",
            "000002.png",
        ]
        assert config.random_seed == 7
        if view_alignment.mode is ViewAlignmentMode.FIDUCIAL_ASSISTED:
            assert len(fiducials) == 3
        else:
            assert fiducials == ()
        images = tuple(
            {
                "image_id": index + 1,
                "name": f"{index:06d}.png",
                "camera_id": 1,
                "cam_from_world": [
                    [1.0, 0.0, 0.0, float(index)],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            }
            for index in range(self.registered_images)
        )
        component = _ComponentResult(
            component_id=0,
            cameras=(
                {
                    "camera_id": 1,
                    "model": "SIMPLE_RADIAL",
                    "width": 16,
                    "height": 12,
                    "params": [12.0, 8.0, 6.0, 0.0],
                },
            ),
            images=images,
            sparse_point_count=25,
            observation_count=75,
            mean_track_length=3.0,
            mean_reprojection_error_px=0.5,
            fiducial_tracks=(
                tuple(
                    {
                        "marker_id": 2,
                        "corner_index": corner_index,
                        "xyz": [float(corner_index), 0.0, 1.0],
                        "observation_count": 3,
                        "reprojection_error_px": 0.25,
                        "observations": [
                            {"image_name": f"{index:06d}.png", "xy": [1.0, 2.0]}
                            for index in range(3)
                        ],
                    }
                    for corner_index in range(4)
                )
                if view_alignment.mode is ViewAlignmentMode.FIDUCIAL_ASSISTED
                else ()
            ),
            files=(("cameras.txt", b"camera fixture\n"), ("images.txt", b"image fixture\n")),
        )
        return _StreamResult(
            stream_name=stream_name,
            selected_image_count=selected_image_count,
            components=(component,),
            execution_device="cpu",
            view_alignment=view_alignment,
            fiducial_frame_count=len(fiducials),
            fiducial_detection_count=sum(len(frame.markers) for frame in fiducials),
        )

    def align_fiducial_streams(
        self,
        results: tuple[_StreamResult, ...],
        config: ColmapRunConfig,
    ) -> tuple[dict[str, object], ...]:
        assert results
        assert config.min_cross_stream_fiducial_tracks == 6
        return ()


class _FakeFiducialDetector:
    def probe(self) -> MethodProbe:
        return MethodProbe(True, "fake ArUco probe passed")

    def detect(
        self,
        image_directory: Path,
        specification: FiducialSpec,
    ) -> tuple[_FiducialFrameDetection, ...]:
        assert specification.family == "aruco_6x6_50"
        marker = _FiducialMarkerDetection(
            marker_id=2,
            corners_xy=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
        )
        return tuple(
            _FiducialFrameDetection(image_name=path.name, markers=(marker,))
            for path in sorted(image_directory.iterdir())
        )


class _FailIfCalledFiducialDetector:
    def probe(self) -> MethodProbe:
        return MethodProbe(True, "fake detector probe passed")

    def detect(
        self,
        image_directory: Path,
        specification: FiducialSpec,
    ) -> tuple[_FiducialFrameDetection, ...]:
        raise AssertionError("natural-feature streams must not invoke a fiducial detector")


def _observation_resource(
    platform: PhysRealX,
    *,
    view_alignment: ViewAlignmentSpec | None = None,
    role: ObservationStateRole = ObservationStateRole.RECONSTRUCTION,
):  # type: ignore[no-untyped-def]
    frames = tuple(
        platform.artifacts.put_bytes(f"png-{index}".encode(), media_type="image/png")
        for index in range(3)
    )
    manifest = ObservationSetManifest(
        resource_id="fixture.observations",
        version="1.0.0",
        display_name="COLMAP fixture",
        importer="fixture",
        importer_version="1.0.0",
        source_uri="file:///fixture",
        streams=(
            ObservationStream(
                name="closed.rgb",
                modality=ObservationModality.RGB,
                state="closed",
                role=role,
                encoding="png",
                frame_count=3,
                artifacts=frames,
                coordinate_frame="camera_unregistered",
                view_alignment=view_alignment
                or ViewAlignmentSpec(
                    mode=ViewAlignmentMode.FIDUCIAL_ASSISTED,
                    fiducial=FiducialSpec(family="aruco_6x6_50"),
                ),
            ),
        ),
    )
    return platform._register(
        manifest,
        manifest.reference(),
        "application/vnd.physrealx.observation-set+json",
    )


def _platform_with_fake_colmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    registered_images: int = 3,
) -> PhysRealX:
    monkeypatch.setattr(platform_module, "create_colmap_adapter", lambda: None)
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    platform.plugins.register_adapter(
        ColmapMethodAdapter(
            _FakeColmapRunner(registered_images),
            _FakeFiducialDetector(),
        )
    )
    return platform


def test_colmap_adapter_converts_observations_outputs_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = _platform_with_fake_colmap(tmp_path, monkeypatch)
    observations = _observation_resource(platform)

    operation = platform.assets.recover_cameras(
        observations,
        method="colmap",
        config={"max_frames_per_stream": 3, "random_seed": 7},
    )

    assert operation.metrics == {
        "selected_image_count": 3.0,
        "registered_image_count": 3.0,
        "registration_rate": 1.0,
        "component_count": 1.0,
        "sparse_point_count": 25.0,
        "observation_count": 75.0,
        "mean_track_length": 3.0,
        "mean_reprojection_error_px": 0.5,
        "fiducial_frame_count": 3.0,
        "fiducial_detection_count": 3.0,
        "fiducial_track_count": 4.0,
        "fiducial_track_observation_count": 12.0,
        "fiducial_alignment_count": 0.0,
        "fiducial_mean_reprojection_error_px": 0.25,
    }
    outputs = {item.name: item.artifact for item in operation.outputs}
    camera = json.loads(platform.artifacts.get_bytes(outputs["camera_rig_calibration"]))
    report = json.loads(platform.artifacts.get_bytes(outputs["camera_recovery_report"]))
    assert camera["schema"] == "physrealx.camera-rig-calibration/1"
    assert camera["scale_status"] == "unresolved"
    source_image = camera["streams"][0]["components"][0]["images"][0]
    assert source_image["source_frame_index"] == 0
    assert source_image["source_artifact_id"]
    assert camera["streams"][0]["view_alignment"]["mode"] == "fiducial_assisted"
    assert camera["streams"][0]["components"][0]["fiducial_tracks"][0]["observations"][0][
        "source_artifact_id"
    ]
    assert report["status"] == "PASSED"
    assert report["streams"][0]["mean_reprojection_error_px"] == 0.5
    assert all(report["gates"].values())

    archive = tmp_path / "sparse.zip"
    platform.artifacts.materialize(outputs["sparse_reconstruction_bundle"], archive)
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("bundle-manifest.json"))
        assert manifest["schema"] == "physrealx.sparse-reconstruction-bundle/1"
        assert manifest["scale_status"] == "unresolved"
        assert "streams/closed.rgb/component-0/cameras.txt" in bundle.namelist()


def test_colmap_adapter_rejects_reconstruction_below_quality_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = _platform_with_fake_colmap(
        tmp_path,
        monkeypatch,
        registered_images=2,
    )
    observations = _observation_resource(platform)

    with pytest.raises(MethodExecutionError, match="quality gates"):
        platform.assets.recover_cameras(
            observations,
            method="colmap",
            config={"max_frames_per_stream": 3, "random_seed": 7},
        )
    assert platform.operations.list() == []


def test_natural_feature_stream_bypasses_fiducial_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform_module, "create_colmap_adapter", lambda: None)
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    platform.plugins.register_adapter(
        ColmapMethodAdapter(
            _FakeColmapRunner(),
            _FailIfCalledFiducialDetector(),
        )
    )
    observations = _observation_resource(
        platform,
        view_alignment=ViewAlignmentSpec(mode=ViewAlignmentMode.NATURAL_FEATURES),
        role=ObservationStateRole.SEQUENCE,
    )

    operation = platform.assets.recover_cameras(
        observations,
        method="colmap",
        config={
            "max_frames_per_stream": 3,
            "random_seed": 7,
            "stream_names": ["closed.rgb"],
        },
    )

    assert operation.metrics["fiducial_detection_count"] == 0.0
    assert operation.metrics["fiducial_track_count"] == 0.0
    outputs = {item.name: item.artifact for item in operation.outputs}
    report = json.loads(platform.artifacts.get_bytes(outputs["camera_recovery_report"]))
    stream = report["streams"][0]
    assert stream["view_alignment_mode"] == "natural_features"
    assert stream["fiducial_family"] is None
    assert set(stream["gates"]) == {
        "registered_images",
        "registration_rate",
        "sparse_points",
    }


def test_platform_registers_colmap_only_after_dependency_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform_module, "create_colmap_adapter", lambda: None)
    unavailable = PhysRealX.open(tmp_path / "missing", discover_plugins=False)
    assert unavailable.methods.installations(MethodCapability.CAMERA_RECOVER) == ()

    adapter = ColmapMethodAdapter(_FakeColmapRunner(), _FakeFiducialDetector())
    monkeypatch.setattr(platform_module, "create_colmap_adapter", lambda: adapter)
    available = PhysRealX.open(tmp_path / "ready", discover_plugins=False)
    installation = available.methods.inspect(MethodCapability.CAMERA_RECOVER, "colmap")
    assert installation.executable
    assert installation.ready
    assert installation.detail == "fake PyCOLMAP probe passed; fake ArUco probe passed"


def test_colmap_config_rejects_impossible_threshold() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ColmapRunConfig(max_frames_per_stream=3, min_registered_images=4)


def test_opencv_detector_recovers_declared_aruco_family(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco"):
        pytest.skip("OpenCV ArUco module is unavailable")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 2, 256)
    image = cv2.copyMakeBorder(
        marker,
        64,
        64,
        64,
        64,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    assert cv2.imwrite(str(tmp_path / "000000.png"), image)

    detections = _OpenCvFiducialDetector().detect(
        tmp_path,
        FiducialSpec(family="aruco_6x6_50"),
    )

    assert len(detections) == 1
    assert detections[0].markers[0].marker_id == 2
    assert len(detections[0].markers[0].corners_xy) == 4


def test_pycolmap_aligns_streams_with_a_shared_fiducial_layout() -> None:
    pytest.importorskip("pycolmap")
    points = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
    )
    alignment = ViewAlignmentSpec(
        mode=ViewAlignmentMode.FIDUCIAL_ASSISTED,
        fiducial=FiducialSpec(
            family="aruco_6x6_50",
            layout_id="fixture.layout",
        ),
    )

    def stream(name: str, xyz: tuple[tuple[float, float, float], ...]) -> _StreamResult:
        tracks = tuple(
            {
                "marker_id": index // 4,
                "corner_index": index % 4,
                "xyz": list(point),
                "observation_count": 3,
                "reprojection_error_px": 0.1,
                "observations": [],
            }
            for index, point in enumerate(xyz)
        )
        component = _ComponentResult(
            component_id=0,
            cameras=(),
            images=(),
            sparse_point_count=20,
            observation_count=60,
            mean_track_length=3.0,
            mean_reprojection_error_px=0.1,
            fiducial_tracks=tracks,
            files=(),
        )
        return _StreamResult(name, 6, (component,), "cpu", alignment, 6, 6)

    source = tuple(tuple((value - 3.0) / 2.0 for value in point) for point in points)
    alignments = _PycolmapRunner().align_fiducial_streams(
        (stream("a.target.rgb", points), stream("b.source.rgb", tuple(source))),
        ColmapRunConfig(),
    )

    assert len(alignments) == 1
    assert alignments[0]["scale"] == pytest.approx(2.0)
    assert alignments[0]["inlier_ratio"] == 1.0
    assert alignments[0]["normalized_rmse"] == pytest.approx(0.0, abs=1e-12)
