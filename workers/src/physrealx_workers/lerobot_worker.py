"""LeRobot v2.1 Episode packaging worker."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import zipfile
from pathlib import Path

from physrealx.external_workers import WorkerProducedFile, WorkerRequest, WorkerResponse, run_worker
from physrealx_workers.common import input_path, is_probe, report_probe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _handle(request: WorkerRequest, output_directory: Path) -> WorkerResponse:
    cache_directory = output_directory / ".cache"
    cache_directory.mkdir()
    os.environ.update(
        {
            "HF_HOME": str(cache_directory / "huggingface"),
            "HF_DATASETS_CACHE": str(cache_directory / "huggingface" / "datasets"),
            "XDG_CACHE_HOME": str(cache_directory),
            "TORCH_HOME": str(cache_directory / "torch"),
        }
    )
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = json.loads(input_path(request, "run").read_text(encoding="utf-8"))
    samples = source.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("LeRobot packer requires a StateTrajectory with non-empty samples")
    timestamps = np.asarray([sample["time"] for sample in samples], dtype=np.float64)
    if len(timestamps) > 1:
        intervals = np.diff(timestamps)
        if np.any(intervals <= 0):
            raise ValueError("StateTrajectory timestamps must be strictly increasing")
        fps = round(1.0 / float(np.median(intervals)))
        tolerance = max(1e-4, float(np.max(np.abs(intervals - 1.0 / fps))) + 1e-6)
    else:
        fps, tolerance = 1, 1e-4

    episode_directory = output_directory / "episode"
    dataset = LeRobotDataset.create(
        repo_id="physrealx/articulated-validation",
        fps=fps,
        root=episode_directory,
        robot_type="physrealx_articulated_asset",
        use_videos=False,
        tolerance_s=tolerance,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["joint_position", "joint_velocity"],
            },
            "observation.target": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["joint_target"],
            },
            "action": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["joint_control"],
            },
        },
    )
    task = "move articulated joint to target"
    for sample, timestamp in zip(samples, timestamps, strict=True):
        dataset.add_frame(
            {
                "observation.state": np.asarray(
                    [sample["position"], sample["velocity"]], dtype=np.float32
                ),
                "observation.target": np.asarray([sample["target"]], dtype=np.float32),
                "action": np.asarray([sample["control"]], dtype=np.float32),
            },
            task=task,
            timestamp=float(timestamp),
        )
    dataset.save_episode()

    # Reload through LeRobot itself before publication; this rejects malformed metadata/parquet.
    verified = LeRobotDataset(
        repo_id="physrealx/articulated-validation",
        root=episode_directory,
        download_videos=False,
    )
    if verified.num_episodes != 1 or verified.num_frames != len(samples):
        raise RuntimeError("LeRobot round-trip verification returned inconsistent counts")

    bundle = output_directory / "episode_bundle.zip"
    files = sorted(path for path in episode_directory.rglob("*") if path.is_file())
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(episode_directory))
    info = json.loads((episode_directory / "meta" / "info.json").read_text(encoding="utf-8"))
    manifest_payload = {
        "schema": "physrealx.lerobot-episode-manifest/1",
        "format": info["codebase_version"],
        "lerobot": importlib.metadata.version("lerobot"),
        "episodes": verified.num_episodes,
        "frames": verified.num_frames,
        "fps": verified.fps,
        "features": info["features"],
        "files": [
            {
                "path": path.relative_to(episode_directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "digest": _sha256(path),
            }
            for path in files
        ],
    }
    manifest = output_directory / "episode_manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.rmtree(cache_directory)
    produced = {
        "episode_bundle": WorkerProducedFile(
            name="episode_bundle",
            contract="EpisodeBundle",
            path=bundle.name,
            media_type="application/zip",
        ),
        "episode_manifest": WorkerProducedFile(
            name="episode_manifest",
            contract="EpisodeManifest",
            path=manifest.name,
            media_type="application/json",
        ),
    }
    return WorkerResponse(
        outputs=tuple(produced[item.name] for item in request.operation.outputs),
        metrics={"episodes": 1.0, "frames": float(len(samples)), "fps": float(fps)},
    )


def main() -> int:
    if is_probe():
        import lerobot
        import numpy
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        _ = lerobot.__path__, LeRobotDataset.create
        if numpy.__version__ != "1.26.4":
            raise RuntimeError("LeRobot worker requires NumPy 1.26.4")
        return report_probe(
            "lerobot_v2_1",
            {
                "lerobot": importlib.metadata.version("lerobot"),
                "numpy": numpy.__version__,
                "dataset_format": "v2.1",
            },
        )
    return run_worker(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
