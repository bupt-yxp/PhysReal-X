"""Python 3.8 bridge executed inside the official Ditto conda environment.

This module intentionally has no dependency on the PhysReal-X package.  The
platform worker invokes it with an immutable JSON request and reads one NPZ
result, keeping Ditto's historical CUDA stack out of the protocol runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

_UPSTREAM_REPOSITORY = "https://github.com/UT-Austin-RPL/Ditto"
_UPSTREAM_REVISION = "f69579f49bbfaadc665fee511ad960f5243e1c1a"
_ALLOWED_EXPERIMENTS = {"Ditto_s2m", "Ditto_syn"}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(repository, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repository), *list(arguments)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


def _attest(repository, revision, checkpoint, checkpoint_digest):
    if repository.is_symlink() or not repository.is_dir():
        raise RuntimeError("Ditto repository must be a real directory")
    if revision != _UPSTREAM_REVISION:
        raise RuntimeError("Ditto revision is not the adapter-pinned official revision")
    head = _git(repository, "rev-parse", "HEAD")
    if head != revision:
        raise RuntimeError("Ditto repository HEAD does not match the declared revision")
    origin = _git(repository, "remote", "get-url", "origin")
    normalized = origin[:-4] if origin.endswith(".git") else origin
    normalized = normalized.rstrip("/")
    if normalized not in {
        _UPSTREAM_REPOSITORY,
        "git@github.com:UT-Austin-RPL/Ditto",
    }:
        raise RuntimeError("Ditto origin is not the official UT-Austin-RPL repository")
    if _git(repository, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Ditto repository has modified tracked files")
    if _sha256(checkpoint) != checkpoint_digest:
        raise RuntimeError("Ditto checkpoint digest does not match its manifest")

    repository_string = str(repository)
    if repository_string not in sys.path:
        sys.path.insert(0, repository_string)
    torch = importlib.import_module("torch")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("official Ditto inference requires an available CUDA device")
    extensions = (
        "src.third_party.ConvONets.utils.libmcubes.mcubes",
        "src.third_party.ConvONets.utils.libmesh.triangle_hash",
        "src.third_party.ConvONets.utils.libmise.mise",
        "src.third_party.ConvONets.utils.libsimplify.simplify_mesh",
        "src.third_party.ConvONets.utils.libvoxelize.voxelize",
    )
    for module in extensions:
        importlib.import_module(module)
    pointnet = importlib.import_module("src.third_party.ConvONets.encoder.pointnetpp_attn")
    if not hasattr(pointnet, "PointNetPlusPlusAttnFusion"):
        raise RuntimeError("official Ditto PointNet++ encoder is unavailable")
    return {
        "repository": _UPSTREAM_REPOSITORY,
        "revision": revision,
        "checkpoint_digest": checkpoint_digest,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "convonets_extensions": "5/5",
        "pointnet_encoder": "PointNetPlusPlusAttnFusion",
    }


def _load_model(repository, experiment, checkpoint):
    hydra = importlib.import_module("hydra")
    compose = getattr(hydra, "compose", None)
    initialize_config_dir = getattr(hydra, "initialize_config_dir", None)
    if compose is None or initialize_config_dir is None:
        experimental = importlib.import_module("hydra.experimental")
        compose = experimental.compose
        initialize_config_dir = experimental.initialize_config_dir
    torch = importlib.import_module("torch")
    with initialize_config_dir(config_dir=str((repository / "configs").resolve())):
        runtime_config = compose(
            config_name="config",
            overrides=["experiment=" + experiment + ".yaml"],
        )
    model = hydra.utils.instantiate(runtime_config.model)
    value = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(value, dict) or "state_dict" not in value:
        raise RuntimeError("Ditto checkpoint does not contain a Lightning state_dict")
    model.load_state_dict(value["state_dict"], strict=True)
    return model.eval().cuda()


def _sample(points, count, seed):
    numpy = importlib.import_module("numpy")
    rng = numpy.random.default_rng(seed)
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    return numpy.asarray(points[indices], dtype=numpy.float32)


def _normalize_rows(values):
    numpy = importlib.import_module("numpy")
    return values / numpy.maximum(numpy.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)


def _infer(request):
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    repository = Path(request["repository"]).resolve(strict=True)
    checkpoint = Path(request["checkpoint"]).resolve(strict=True)
    revision = request["revision"]
    digest = request["checkpoint_digest"]
    experiment = request["experiment"]
    if experiment not in _ALLOWED_EXPERIMENTS:
        raise RuntimeError("unsupported Ditto checkpoint experiment")
    _attest(repository, revision, checkpoint, digest)
    with numpy.load(request["input_npz"], allow_pickle=False) as archive:
        if set(archive.files) != {"before", "after"}:
            raise RuntimeError("bridge input NPZ must contain exactly before and after")
        before = numpy.asarray(archive["before"], dtype=numpy.float64)
        after = numpy.asarray(archive["after"], dtype=numpy.float64)
    if before.ndim != 2 or after.ndim != 2 or before.shape[1:] != (3,) or after.shape[1:] != (3,):
        raise RuntimeError("bridge point clouds must have shape [N,3]")
    if not numpy.all(numpy.isfinite(before)) or not numpy.all(numpy.isfinite(after)):
        raise RuntimeError("bridge point clouds contain NaN or infinity")

    bounds_min = numpy.minimum(before.min(axis=0), after.min(axis=0))
    bounds_max = numpy.maximum(before.max(axis=0), after.max(axis=0))
    center = (bounds_min + bounds_max) / 2.0
    scale = float(numpy.max(bounds_max - bounds_min) * 1.1)
    if not math.isfinite(scale) or scale <= 1.0e-6:
        raise RuntimeError("Ditto normalization scale is degenerate")
    seed = int(request["seed"])
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    before_normalized = _sample((before - center) / scale, 8192, seed)
    after_normalized = _sample((after - center) / scale, 8192, seed + 1)
    device = torch.device("cuda:0")
    sample = {
        "pc_start": torch.from_numpy(before_normalized).unsqueeze(0).to(device),
        "pc_end": torch.from_numpy(after_normalized).unsqueeze(0).to(device),
    }
    model = _load_model(repository, experiment, checkpoint)
    generator_module = importlib.import_module(
        "src.third_party.ConvONets.conv_onet.generation_two_stage"
    )
    joint_module = importlib.import_module("src.utils.joint_estimation")
    generator = generator_module.Generator3D(
        model.model,
        device=device,
        threshold=0.4,
        seg_threshold=0.5,
        input_type="pointcloud",
        refinement_step=0,
        padding=0.1,
        resolution0=int(request["resolution"]),
    )
    with torch.no_grad():
        mesh_dict, mobile_points, latent, _ = generator.generate_mesh(sample)
        if int(mobile_points.shape[1]) == 0:
            raise RuntimeError("Ditto predicted an empty moving part")
        logits, revolute, prismatic = model.model.decode_joints(mobile_points, latent)
    probability = float(logits.sigmoid().mean().detach().cpu())
    moving_query = numpy.asarray(mobile_points[0].detach().cpu(), dtype=numpy.float64)
    if probability < 0.5:
        axes = _normalize_rows(
            numpy.asarray(revolute[0, :, :3].detach().cpu(), dtype=numpy.float64)
        )
        directions = _normalize_rows(
            numpy.asarray(revolute[0, :, 4:7].detach().cpu(), dtype=numpy.float64)
        )
        distances = numpy.asarray(revolute[0, :, 7].detach().cpu(), dtype=numpy.float64)
        pivots = moving_query + directions * distances[:, None]
        motions = numpy.asarray(revolute[0, :, 3].detach().cpu(), dtype=numpy.float64)
        axis, pivot, motion = joint_module.aggregate_dense_prediction_r(
            axes, pivots, motions, method="mean"
        )
        joint_type_code = 0
    else:
        axes = _normalize_rows(
            numpy.asarray(prismatic[0, :, :3].detach().cpu(), dtype=numpy.float64)
        )
        axis = numpy.mean(axes, axis=0)
        pivot = numpy.mean(numpy.asarray(mesh_dict[1].vertices, dtype=numpy.float64), axis=0)
        motion = float(numpy.mean(numpy.asarray(prismatic[0, :, 3].detach().cpu())))
        joint_type_code = 1
    static_vertices = numpy.asarray(mesh_dict[0].vertices, dtype=numpy.float64) * scale + center
    moving_vertices = numpy.asarray(mesh_dict[1].vertices, dtype=numpy.float64) * scale + center
    numpy.savez_compressed(
        request["output_npz"],
        joint_type=numpy.asarray([joint_type_code], dtype=numpy.int8),
        prismatic_probability=numpy.asarray([probability], dtype=numpy.float64),
        axis=numpy.asarray(axis, dtype=numpy.float64),
        pivot_m=numpy.asarray(pivot, dtype=numpy.float64) * scale + center,
        motion=numpy.asarray(
            [float(motion * scale if joint_type_code == 1 else motion)], dtype=numpy.float64
        ),
        axis_candidates=numpy.asarray(axes, dtype=numpy.float64),
        static_vertices_m=static_vertices,
        static_faces=numpy.asarray(mesh_dict[0].faces, dtype=numpy.int64),
        moving_vertices_m=moving_vertices,
        moving_faces=numpy.asarray(mesh_dict[1].faces, dtype=numpy.int64),
        normalization_center_m=numpy.asarray(center, dtype=numpy.float64),
        normalization_scale_m=numpy.asarray([scale], dtype=numpy.float64),
    )


def _probe(repository, checkpoint, digest, experiment):
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    versions = _attest(repository, _UPSTREAM_REVISION, checkpoint, digest)
    if experiment not in _ALLOWED_EXPERIMENTS:
        raise RuntimeError("unsupported Ditto checkpoint experiment")
    model = _load_model(repository, experiment, checkpoint)
    numpy.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    rng = numpy.random.default_rng(7)
    before = torch.from_numpy(rng.normal(size=(1, 1024, 3)).astype("float32")).cuda()
    after = before + torch.tensor([[[0.02, 0.0, 0.0]]], device="cuda")
    queries = before[:, :64, :]
    with torch.no_grad():
        outputs = model.model(before, after, queries, queries)
    if len(outputs) != 5 or any(not bool(torch.isfinite(item).all()) for item in outputs):
        raise RuntimeError("Ditto CUDA fixture produced invalid outputs")
    versions["small_cuda_fixture"] = "passed"
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(prog="physrealx-ditto-inference-bridge")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-digest")
    parser.add_argument("--experiment", default="Ditto_s2m")
    parser.add_argument("--request", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.probe:
            if (
                not arguments.repository
                or not arguments.checkpoint
                or not arguments.checkpoint_digest
            ):
                raise RuntimeError("probe requires repository, checkpoint and checkpoint digest")
            versions = _probe(
                arguments.repository.resolve(strict=True),
                arguments.checkpoint.resolve(strict=True),
                arguments.checkpoint_digest,
                arguments.experiment,
            )
            print(json.dumps({"ready": True, "versions": versions}, sort_keys=True))
            return 0
        if arguments.request is None:
            raise RuntimeError("execution requires --request")
        request = json.loads(arguments.request.resolve(strict=True).read_text(encoding="utf-8"))
        _infer(request)
        print(json.dumps({"ok": True}, sort_keys=True))
        return 0
    except Exception as error:  # boundary must turn every old-stack failure into one result
        print(
            json.dumps(
                {"ready": False, "error": str(error) or type(error).__name__},
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
