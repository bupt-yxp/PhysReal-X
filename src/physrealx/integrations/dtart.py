"""DTArt reconstruction through the active interpreter and normalized project outputs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    DomainModel,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodMaturity,
    MethodRuntime,
    OperationSpec,
    canonical_json_bytes,
)
from physrealx.errors import MethodExecutionError
from physrealx.integrations._artifacts import input_artifact, publish, read_json
from physrealx.integrations.dtart_capture import prepare_dtart_capture
from physrealx.integrations.rgbd import MetricRGBDInput, load_rgbd
from physrealx.reconstruction import ReconstructedPart, ReconstructionPrediction


class DTArtRunConfig(DomainModel):
    # Joint type is an explicit modelling prior, not inferred from the reference USD.
    joint_types: tuple[Literal["slide", "hinge"], ...] = Field(min_length=1, max_length=16)
    state_streams: tuple[str, str]
    training_frames_per_state: int = Field(default=100, ge=2, le=2048, strict=True)
    iterations: int = Field(default=4000, ge=4000, le=100000)
    correspondence_neighbors: int = Field(default=30, ge=1, le=100)
    seed: int = Field(default=2026, ge=0, le=2**31 - 1)
    max_stage_seconds: int = Field(default=14400, ge=60, le=86400)


DTART_DESCRIPTOR = MethodDescriptor(
    capability=MethodCapability.GEOMETRY_RECONSTRUCT,
    name="dtart",
    version="1.0.0",
    api_version="1",
    source_uri="https://github.com/NVlabs/DigitalTwinArt",
    source_revision="1a48b402e4bf4bb7731296e8e230f0db3d86fe4f",
    license_spdx="LicenseRef-NVIDIA-Research",
    maturity=MethodMaturity.EXPERIMENTAL,
    runtime=MethodRuntime(
        mode=MethodExecutionMode.IN_PROCESS,
        accelerator=MethodAccelerator.CUDA,
        python=">=3.10,<3.13",
    ),
    input_contract="MetricMultiStateRGBDBundle/1",
    output_contract="VisualGeometryBundle/1+GeometryReconstructionReport/1",
)


def dtart_source_root() -> Path:
    module = Path(__file__).resolve()
    for path in (module.parents[3] / "algorithms/dtart", module.parents[1] / "_embedded/dtart"):
        if (path / "SOURCE.toml").is_file() and (path / "physrealx_run.py").is_file():
            return path
    raise FileNotFoundError("DTArt source snapshot is unavailable")


def _weights_path(root: Path) -> Path:
    path = root / "weights/outdoor_ds.ckpt"
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != (
        "21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec"
    ):
        raise MethodExecutionError("DTArt requires the verified LoFTR outdoor weights")
    return path


def normalized_dtart_prediction(
    result: Path, artifacts: ArtifactStore, config: DTArtRunConfig
) -> ReconstructionPrediction:
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    parts = {"0": ReconstructedPart()}
    for i, kind in enumerate(config.joint_types, 1):
        filename = "init_prismatic_motion.json" if kind == "slide" else "init_revolute_motion.json"
        rows = json.loads((result / filename).read_text())
        if len(rows) != len(config.joint_types):
            raise MethodExecutionError("DTArt motion count differs from requested part structure")
        motion = rows[i - 1]
        translation = np.asarray(motion["translation"], dtype=float)
        rotation = np.asarray(motion["rotation"], dtype=float)
        if (
            translation.shape != (3,)
            or rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
        ):
            raise MethodExecutionError("DTArt returned an invalid rigid motion")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1, atol=1e-5
        ):
            raise MethodExecutionError("DTArt motion rotation is not in SO(3)")
        if kind == "slide":
            span = float(np.linalg.norm(translation))
            axis, origin = translation / max(span, 1e-12), np.zeros(3)
        else:
            rotvec = Rotation.from_matrix(rotation).as_rotvec()
            span = float(np.linalg.norm(rotvec))
            axis = rotvec / max(span, 1e-12)
            # I-R is singular for every rotation; least squares yields the minimum-norm axis point.
            origin = np.linalg.lstsq(np.eye(3) - rotation, translation, rcond=1e-8)[0]
        if span <= 1e-8:
            raise MethodExecutionError(f"DTArt part {i} has no resolved motion")
        parts[str(i)] = ReconstructedPart(
            parent="0", kind=kind, axis=tuple(axis), origin=tuple(origin), span=span
        )
    states = {}
    for state, native in zip(config.state_streams, ("init", "last")):
        meshes = {}
        for identifier in parts:
            mesh = trimesh.load(
                result / f"{native}_part_{identifier}.obj", force="mesh", process=False
            )
            if (
                not isinstance(mesh, trimesh.Trimesh)
                or not len(mesh.faces)
                or not np.isfinite(mesh.vertices).all()
            ):
                raise MethodExecutionError(f"DTArt exported invalid {state}/{identifier} geometry")
            meshes[identifier] = artifacts.put_bytes(
                mesh.export(file_type="ply"), media_type="model/ply"
            )
        states[state] = meshes
    return ReconstructionPrediction(parts=parts, states=states, motion_pair=config.state_streams)


class DTArtMethodAdapter:
    descriptor = DTART_DESCRIPTOR

    def probe(self) -> MethodProbe:
        try:
            root = dtart_source_root()
            _weights_path(root)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import torch, kaolin, kornia, mesh_to_sdf, ruamel.yaml; "
                    "from mycuda import common, gridencoder; "
                    "import main; "
                    "assert torch.cuda.is_available(), 'CUDA unavailable'",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PYOPENGL_PLATFORM": "egl", "WANDB_MODE": "disabled"},
            )
            return MethodProbe(
                result.returncode == 0,
                "DTArt native runtime ready" if result.returncode == 0 else result.stderr[-2000:],
            )
        except (OSError, subprocess.SubprocessError, MethodExecutionError) as error:
            return MethodProbe(False, str(error))

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree

        config = DTArtRunConfig.model_validate(spec.method.config)
        source = MetricRGBDInput.model_validate(
            read_json(
                artifacts,
                input_artifact(spec, "source", artifacts),
                "physrealx.metric-multistate-rgbd-bundle/1",
            )
        )
        root = dtart_source_root()
        weights = _weights_path(root)
        with tempfile.TemporaryDirectory(prefix="physrealx-dtart-") as directory:
            work = Path(directory)
            prepared = prepare_dtart_capture(
                source,
                artifacts,
                work / "data",
                state_streams=config.state_streams,
                training_frames_per_state=config.training_frames_per_state,
            )
            cfg_path = work / "run.json"
            cfg_path.write_bytes(canonical_json_bytes(config.model_dump(mode="json")))
            log_path = work / "training.log"
            with log_path.open("w") as log:
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(root / "physrealx_run.py"),
                            "--data",
                            str(work / "data"),
                            "--output",
                            str(work / "runs"),
                            "--config",
                            str(cfg_path),
                            "--weights",
                            str(weights),
                        ],
                        cwd=root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={**os.environ, "PYOPENGL_PLATFORM": "egl", "WANDB_MODE": "disabled"},
                    )
                    try:
                        process.wait(timeout=config.max_stage_seconds)
                    except BaseException:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
                except subprocess.TimeoutExpired as error:
                    raise MethodExecutionError(
                        "DTArt exceeded its execution time budget"
                    ) from error
            if process.returncode:
                raise MethodExecutionError(
                    "DTArt training failed:\n" + log_path.read_text(errors="replace")[-4000:]
                )
            result = work / "runs/results" / f"step_{config.iterations:07d}"
            prediction = normalized_dtart_prediction(result, artifacts, config)
            _, _, views = load_rgbd(source, artifacts)
            validation_ids = set(prepared["validation_frame_ids"])
            evaluation = {}
            for state, references in prediction.states.items():
                mesh = trimesh.util.concatenate(
                    [
                        trimesh.load(
                            io.BytesIO(artifacts.get_bytes(ref)), file_type="ply", force="mesh"
                        )
                        for ref in references.values()
                    ]
                )
                points, _ = trimesh.sample.sample_surface(mesh, 100000, seed=config.seed)
                tree = cKDTree(points)
                distances = []
                for view in views:
                    if view.stream != state or view.frame_id not in validation_ids:
                        continue
                    mask = view.mask & (view.depth > 0)
                    xyz = (view.rays[mask] * view.depth[mask, None] - view.pose[:, 3]) @ view.pose[
                        :, :3
                    ]
                    if len(xyz):
                        distances.append(tree.query(xyz, workers=1)[0])
                if not distances:
                    raise MethodExecutionError("DTArt evaluation lacks valid independent views")
                values = np.concatenate(distances)
                evaluation[state] = {
                    "visible_surface_median_m": float(np.median(values)),
                    "visible_surface_p90_m": float(np.percentile(values, 90)),
                }
            metrics = {
                "validation_surface_median_m": max(
                    v["visible_surface_median_m"] for v in evaluation.values()
                ),
                "motion_group_count": float(len(prediction.parts)),
            }
            report: dict[str, Any] = {
                "schema": "physrealx.geometry-reconstruction-report/1",
                "method": self.descriptor.model_dump(mode="json"),
                "status": "MODEL_CONDITIONED",
                "unit": "m",
                **prepared,
                "source_camera_artifact_id": source.camera_rig.artifact_id,
                "source_depth_artifact_id": source.enhanced_depth.artifact_id,
                "prediction": prediction.model_dump(mode="json"),
                "evaluation_scope": "held_out_views",
                "joint_type_prior": list(config.joint_types),
                "motion_group_count": len(prediction.parts),
                "metrics": metrics,
                "evaluations": evaluation,
                "independent_physical_validation": False,
            }
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", canonical_json_bytes(report))
                archive.write(log_path, "training.log")
                archive.write(cfg_path, "run.json")
                archive.write(work / "data/scene_normalization.npz", "scene_normalization.npz")
                archive.write(work / "runs/ckpt/model_latest.pth", "model/checkpoint.pth")
                for path in result.iterdir():
                    if path.is_file() and path.suffix in (".obj", ".json", ".png"):
                        archive.write(path, "model/" + path.name)
            bundle = artifacts.put_bytes(buffer.getvalue(), media_type="application/zip")
            report["bundle"] = bundle.model_dump(mode="json")
            return publish(
                spec,
                artifacts,
                {"geometry_reconstruction_report": report},
                artifacts={"visual_geometry_bundle": bundle},
                metrics=metrics,
            )
