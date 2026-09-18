"""DINO-WM: frozen official DINOv2 patches and the upstream causal ViT predictor.

The platform supplies bounded episode windows, scalar conditioning, a small visual
decoder and artifact I/O. Predicted frames remain model evidence, never simulator
or independently measured physical evidence.
"""

from __future__ import annotations

import importlib
import io
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    Episode,
    EpisodeSummary,
    MethodCapability,
    OperationSpec,
    ResourceType,
    VersionedRef,
    canonical_record_id,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import input_artifact, publish, read_array, read_json
from physrealx.integrations.datasets import load_snapshot
from physrealx.integrations.learning_sources import algorithm_module
from physrealx.integrations.provenance import capture_runtime
from physrealx.methods import builtin_method
from physrealx.runtime import load_episode
from physrealx.training_lifecycle import (
    ModelKind,
    TrainingExecutionResult,
    TrainingLifecycle,
    WorldModelBundle,
    WorldModelTrainingPlan,
)

DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WM_REVISION = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
DINO_WEIGHTS_URL: Final = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
)
VISUAL_RESIDUAL_GAINS: Final = tuple(index / 8 for index in range(9))


def _quality_status(value: float, persistence: float, *, tolerance: float = 1e-8) -> str:
    """Classify a metric without turning floating-point ties into regressions."""
    if value < persistence - tolerance:
        return "BEATS_PERSISTENCE"
    if value <= persistence + tolerance:
        return "MATCHES_PERSISTENCE"
    return "BELOW_BASELINE"


class EncoderSpec(DomainModel):
    schema_id: Literal["physrealx.pretrained-encoder/1"] = Field(
        default="physrealx.pretrained-encoder/1", alias="schema"
    )
    source_uri: Literal[
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
    ] = DINO_WEIGHTS_URL
    source_revision: Literal["7764ea0f912e53c92e82eb78a2a1631e92725fc8"] = (
        "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    )
    architecture: Literal["dinov2_vits14"] = "dinov2_vits14"
    weights: ArtifactRef


def encoder_network() -> Any:
    package = algorithm_module("dinov2", package=True)
    module = importlib.import_module(package.__name__ + ".models.vision_transformer")
    return module.vit_small(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)


def import_dinov2_weights(path: Path, artifacts: ArtifactStore) -> ArtifactRef:
    """Import an explicitly obtained upstream checkpoint using weights-only decoding."""
    import torch
    from safetensors.torch import save

    if path.stat().st_size > 128 * 1024**2:
        raise DomainValidationError("DINOv2 ViT-S checkpoint exceeds its size bound")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    model = encoder_network()
    model.load_state_dict(tensors, strict=True)
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise DomainValidationError("DINOv2 checkpoint contains non-finite weights")
    weights = artifacts.put_bytes(save(tensors), media_type="application/vnd.safetensors")
    return artifacts.put_json(EncoderSpec(weights=weights).model_dump(mode="json", by_alias=True))


def prepare_dinov2_encoder(
    artifacts: ArtifactStore, cache: Path, *, weights_path: Path | None = None
) -> ArtifactRef:
    """Verify a cached encoder or import/download the official bounded checkpoint."""
    import tempfile
    import urllib.request

    if weights_path is None and cache.is_file():
        encoder = EncoderSpec.model_validate_json(cache.read_bytes())
        if artifacts.verify(encoder.weights):
            return artifacts.put_json(encoder)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".encoder-", dir=cache.parent) as directory:
        stage = Path(directory)
        if weights_path is None:
            weights_path = stage / "dinov2_vits14.pth"
            response = urllib.request.urlopen(DINO_WEIGHTS_URL, timeout=60)
            with response, weights_path.open("wb") as target:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 128 * 1024**2:
                        raise DomainValidationError("DINOv2 download exceeds checkpoint size bound")
                    target.write(chunk)
        reference = import_dinov2_weights(weights_path, artifacts)
        artifacts.materialize(reference, stage / "encoder.json")
        (stage / "encoder.json").replace(cache)
    return reference


class WorldModelConfig(DomainModel):
    image_signal: str = "camera.front.rgb"
    state_signals: tuple[str, ...] = ("joint.position", "joint.velocity")
    image_size: int = Field(default=56, ge=28, le=224)
    history: int = Field(default=3, ge=1, le=8)
    training_horizon: int = Field(default=4, ge=1, le=128)
    depth: int = Field(default=2, ge=1, le=12)
    steps: int = Field(default=400, ge=1, le=100000)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    state_loss_weight: float = Field(default=1.0, gt=0, le=100)
    seed: int = Field(default=2026, ge=0)
    device: Literal["cpu", "cuda"] = "cpu"
    threads: int = Field(default=4, ge=1, le=64)
    max_frames: int = Field(default=10000, ge=2, le=100000)
    evaluation_horizon: int = Field(default=8, ge=1, le=128)
    validation_interval: int = Field(default=100, ge=1, le=10000)

    @model_validator(mode="after")
    def validate_dimensions(self) -> WorldModelConfig:
        if self.image_size % 14:
            raise ValueError("DINOv2 image size must divide its 14-pixel patch size")
        if not self.state_signals or len(set(self.state_signals)) != len(self.state_signals):
            raise ValueError("world model requires unique state signals")
        return self


def _networks(config: WorldModelConfig, state_dim: int, action_dim: int) -> tuple[Any, Any]:
    from torch import nn

    # Give scalar controls their own learned representation instead of hiding
    # three scalar channels among 384 visual channels before layer normalization.
    backbone = algorithm_module("dino_wm").ViTPredictor(
        num_patches=(config.image_size // 14) ** 2,
        num_frames=config.history,
        dim=512,
        depth=config.depth,
        heads=4,
        dim_head=64,
        mlp_dim=512,
    )
    visual_delta = nn.Linear(512, 384)
    state_delta = nn.Sequential(
        nn.Linear(state_dim + action_dim, 128),
        nn.GELU(),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, state_dim),
    )
    for head in (visual_delta, state_delta[-1]):
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    predictor = nn.ModuleDict(
        {
            "conditioning": nn.Sequential(
                nn.Linear(state_dim + action_dim, 128), nn.GELU(), nn.Linear(128, 128)
            ),
            "backbone": backbone,
            "visual_delta": visual_delta,
            "state_delta": state_delta,
        }
    )
    decoder = nn.Sequential(nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 14 * 14 * 3))
    return predictor, decoder


def _rgb(
    episode: Episode,
    config: WorldModelConfig,
    store: ArtifactStore,
    *,
    start: int = 0,
    stop: int | None = None,
) -> Any:
    import numpy as np
    import torch
    from torch.nn.functional import interpolate

    images = []
    for step in episode.steps[start:stop]:
        value = step.observations.get(config.image_signal)
        if value is None or value.artifact is None:
            raise DomainValidationError("world model requires a published RGB image per step")
        image = read_array(store, value.artifact)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise DomainValidationError("world model expects uint8 RGB images")
        images.append(image)
    tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
    return interpolate(tensor, size=(config.image_size,) * 2, mode="bilinear", align_corners=False)


def _states(
    episode: Episode,
    config: WorldModelConfig,
    *,
    start: int = 0,
    stop: int | None = None,
) -> list[list[float]]:
    result = []
    for step in episode.steps[start:stop]:
        row: list[float] = []
        for name in config.state_signals:
            value = step.observations.get(name)
            if value is None or value.inline is None:
                raise DomainValidationError(f"world model requires inline state {name}")
            row.extend(float(item) for item in value.inline)
        result.append(row)
    return result


def _features(encoder: Any, rgb: Any, device: str, batch_size: int) -> Any:
    import torch

    rows = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    scale = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    with torch.inference_mode():
        for batch in rgb.split(batch_size):
            rows.append(
                encoder.forward_features((batch.to(device) - mean) / scale)[
                    "x_norm_patchtokens"
                ].cpu()
            )
    return torch.cat(rows)


def _predict(predictor: Any, features: Any, states: Any, actions: Any) -> tuple[Any, Any]:
    import torch

    condition = predictor["conditioning"](torch.cat((states, actions), dim=-1))
    condition = condition.unsqueeze(-2).expand(*features.shape[:-1], 128)
    tokens = torch.cat((features, condition), dim=-1)
    batch, frames, patches, width = tokens.shape
    hidden = predictor["backbone"](tokens.reshape(batch, frames * patches, width))
    hidden = hidden.reshape(batch, frames, patches, width)
    return (
        features + predictor["visual_delta"](hidden),
        states + predictor["state_delta"](torch.cat((states, actions), dim=-1)),
    )


def _decode(decoder: Any, features: Any, image_size: int) -> Any:
    batch = features.shape[0]
    grid = image_size // 14
    return (
        decoder(features)
        .reshape(batch, grid, grid, 14, 14, 3)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, 3, image_size, image_size)
    )


def _render_future(
    decoder: Any, features: Any, context_features: Any, context_rgb: Any, image_size: int
) -> Any:
    """Predict appearance changes while retaining observed static image detail."""
    batch, horizon = features.shape[:2]
    future = _decode(decoder, features.flatten(0, 1), image_size).reshape(
        batch, horizon, 3, image_size, image_size
    )
    initial = _decode(decoder, context_features, image_size)
    return context_rgb[:, None] + future - initial[:, None]


def _array(store: ArtifactStore, value: Any) -> ArtifactRef:
    import numpy as np

    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return store.put_bytes(stream.getvalue(), media_type="application/x-npy")


def _rollout(
    predictor: Any, features: Any, states: Any, history_actions: Any, actions: Any
) -> tuple[Any, Any]:
    """Differentiable batch rollout shared by training, validation and inference."""
    import torch

    predicted_features, predicted_states = [], []
    for action in actions.unbind(1):
        conditioned = torch.cat((history_actions, action[:, None]), dim=1)
        z, state = _predict(predictor, features, states, conditioned)
        next_z, next_state = z[:, -1], state[:, -1]
        predicted_features.append(next_z)
        predicted_states.append(next_state)
        features = torch.cat((features[:, 1:], next_z[:, None]), dim=1)
        states = torch.cat((states[:, 1:], next_state[:, None]), dim=1)
        history_actions = conditioned[:, 1:]
    return torch.stack(predicted_features, dim=1), torch.stack(predicted_states, dim=1)


def _evaluate(
    predictor: Any, decoder: Any, rows: list[dict[str, Any]], config: WorldModelConfig, scale: Any
) -> list[dict[str, Any]]:
    """Compare identical early/middle/late windows with temporal persistence."""
    import torch

    errors = []
    with torch.inference_mode():
        for row in rows:
            h = config.history
            horizon = min(config.evaluation_horizon, len(row["actions"]) - h + 1)
            last = len(row["actions"]) - h - horizon + 1
            for start in sorted({0, last // 2, last}):
                end = start + h
                z, state = _rollout(
                    predictor,
                    row["features"][None, start:end],
                    row["states"][None, start:end],
                    row["actions"][None, start : end - 1],
                    row["actions"][None, end - 1 : end - 1 + horizon],
                )
                z, state = z[0], state[0]
                gt_z, gt_state = (
                    row["features"][end : end + horizon],
                    row["states"][end : end + horizon],
                )
                errors.append(
                    {
                        "episode_id": row["id"],
                        "start_index": end - 1,
                        "horizon": horizon,
                        "latent_mse": float((z - gt_z).square().mean()),
                        "persistence_latent_mse": float(
                            (row["features"][end - 1] - gt_z).square().mean()
                        ),
                        "normalized_state_rmse": float((state - gt_state).square().mean().sqrt()),
                        "state_rmse_by_dimension": (
                            (state - gt_state).square().mean(0).sqrt() * scale
                        )
                        .cpu()
                        .tolist(),
                        "persistence_normalized_state_rmse": float(
                            (row["states"][end - 1] - gt_state).square().mean().sqrt()
                        ),
                        "rgb_mse": float(
                            (
                                _render_future(
                                    decoder,
                                    z[None],
                                    row["features"][None, end - 1],
                                    row["rgb"][None, end - 1],
                                    config.image_size,
                                )[0].clamp(0, 1)
                                - row["rgb"][end : end + horizon]
                            )
                            .square()
                            .mean()
                        ),
                        "persistence_rgb_mse": float(
                            (row["rgb"][end - 1] - row["rgb"][end : end + horizon]).square().mean()
                        ),
                    }
                )
    return errors


def _validation_selection(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Balance modalities by their validation persistence errors, never test outcomes."""
    if not rows:
        raise DomainValidationError("world model selection requires validation windows")
    ratios = {}
    for metric, power in (("latent_mse", 1), ("normalized_state_rmse", 2), ("rgb_mse", 1)):
        actual = sum(float(row[metric]) for row in rows) / len(rows)
        baseline = sum(float(row["persistence_" + metric]) for row in rows) / len(rows)
        if not all(math.isfinite(value) and value >= 0 for value in (actual, baseline)):
            raise MethodExecutionError(
                "world model validation errors must be finite and nonnegative"
            )
        ratios[metric + "_ratio"] = (actual / max(baseline, 1e-12)) ** power
    return {"score": max(ratios.values()), **ratios}


def _calibrate_visual_residual(
    predictor: Any,
    decoder: Any,
    rows: list[dict[str, Any]],
    config: WorldModelConfig,
    scale: Any,
) -> tuple[float, list[dict[str, Any]], list[dict[str, float]], dict[str, float]]:
    """Bound recurrent visual drift using validation data only.

    A zero gain is the temporal-persistence baseline. The fixed grid therefore
    cannot hide a residual that fails to generalize, while intermediate gains
    retain useful motion without compounding a one-step residual unchecked.
    """
    import torch

    layer = predictor["visual_delta"]
    weight = layer.weight.detach().clone()
    bias = layer.bias.detach().clone()
    best: tuple[float, float, list[dict[str, Any]], dict[str, float]] | None = None
    curve = []
    with torch.inference_mode():
        for gain in VISUAL_RESIDUAL_GAINS:
            layer.weight.copy_(weight * gain)
            layer.bias.copy_(bias * gain)
            errors = _evaluate(predictor, decoder, rows, config, scale)
            selection = _validation_selection(errors)
            visual_score = max(selection["latent_mse_ratio"], selection["rgb_mse_ratio"])
            curve.append({"gain": gain, "visual_score": visual_score, **selection})
            candidate = (visual_score, gain, errors, selection)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:  # pragma: no cover - fixed nonempty grid
            raise MethodExecutionError("world model residual calibration has no candidate")
        _, gain, errors, selection = best
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return gain, errors, curve, selection


def _apply_visual_residual_gain(predictor: Any, gain: float) -> None:
    """Bake a validation-selected gain into a visual residual head."""
    import torch

    with torch.no_grad():
        predictor["visual_delta"].weight.mul_(gain)
        predictor["visual_delta"].bias.mul_(gain)


class WorldModelTrainAdapter:
    descriptor = builtin_method(
        MethodCapability.WORLD_MODEL_TRAIN,
        "dino_wm",
        version="4.3.0",
        input_contract="DatasetSnapshot/1+PretrainedEncoder/1",
        output_contract="ModelBundle/1+TrainingReport/1",
        python=">=3.10,<3.13",
    ).model_copy(
        update={
            "source_uri": "https://github.com/gaoyuezhou/dino_wm",
            "source_revision": WM_REVISION,
            "license_spdx": "MIT",
        }
    )

    def probe(self) -> MethodProbe:
        from importlib.util import find_spec

        ready = all(
            find_spec(name) is not None
            for name in ("torch", "torchvision", "einops", "safetensors")
        )
        return MethodProbe(ready, "DINO-WM requires Torch, torchvision, einops and safetensors")

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import torch
        from safetensors.torch import load, save

        config = WorldModelConfig.model_validate(spec.method.config)
        started_at = datetime.now(timezone.utc)
        code, environment = capture_runtime(artifacts, algorithms=("dino_wm", "dinov2"))
        snapshot_ref = input_artifact(spec, "dataset", artifacts)
        snapshot, dataset, splits = load_snapshot(artifacts, snapshot_ref)
        encoder_ref = input_artifact(spec, "encoder", artifacts)
        encoder_spec = EncoderSpec.model_validate(
            read_json(artifacts, encoder_ref, "physrealx.pretrained-encoder/1")
        )
        frame_count = sum(len(episode.steps) for values in splits.values() for episode in values)
        if frame_count > config.max_frames:
            raise DomainValidationError(
                "world model dataset exceeds its explicit frame memory bound"
            )
        if config.device == "cuda" and not torch.cuda.is_available():
            raise DomainValidationError("world model requested CUDA without an available device")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(config.threads)
        try:
            with torch.random.fork_rng(
                devices=[] if config.device == "cpu" else [torch.cuda.current_device()]
            ):
                torch.manual_seed(config.seed)
                encoder = encoder_network().to(config.device).eval()
                encoder.load_state_dict(
                    load(artifacts.get_bytes(encoder_spec.weights)), strict=True
                )
                rows: dict[str, list[dict[str, Any]]] = {}
                for name, episodes in splits.items():
                    rows[name] = []
                    for episode in episodes:
                        if len(episode.steps) <= config.history:
                            raise DomainValidationError(
                                "episode is shorter than the world model context"
                            )
                        rgb = _rgb(episode, config, artifacts)
                        actions = [
                            step.action.inline
                            for step in episode.steps[:-1]
                            if step.action is not None and step.action.inline is not None
                        ]
                        if len(actions) != len(episode.steps) - 1:
                            raise DomainValidationError(
                                "world model requires complete inline actions"
                            )
                        intervals = [
                            right.timestamp_seconds - left.timestamp_seconds
                            for left, right in zip(episode.steps, episode.steps[1:])
                        ]
                        if max(intervals) - min(intervals) > 1e-8:
                            raise DomainValidationError(
                                "world model requires uniformly sampled episodes"
                            )
                        rows[name].append(
                            {
                                "id": episode.episode_id,
                                "rgb": rgb,
                                "features": _features(
                                    encoder, rgb, config.device, config.batch_size
                                ),
                                "states": torch.tensor(
                                    _states(episode, config), dtype=torch.float32
                                ),
                                "actions": torch.tensor(actions, dtype=torch.float32),
                                "interval": intervals[0],
                            }
                        )
                del encoder
                intervals = [row["interval"] for values in rows.values() for row in values]
                if max(intervals) - min(intervals) > 1e-8:
                    raise DomainValidationError("world model dataset mixes sampling frequencies")
                train_states = torch.cat([row["states"] for row in rows["train"]])
                train_actions = torch.cat([row["actions"] for row in rows["train"]])
                statistics = {
                    "state_mean": train_states.mean(0),
                    "state_scale": train_states.std(0).clamp_min(1e-4),
                    "action_mean": train_actions.mean(0),
                    "action_scale": train_actions.std(0).clamp_min(1e-4),
                }
                state_dim, action_dim = train_states.shape[-1], train_actions.shape[-1]
                for values in rows.values():
                    for row in values:
                        row["states"] = (row["states"] - statistics["state_mean"]) / statistics[
                            "state_scale"
                        ]
                        row["actions"] = (row["actions"] - statistics["action_mean"]) / statistics[
                            "action_scale"
                        ]
                        for name in ("states", "actions", "features", "rgb"):
                            row[name] = row[name].to(config.device)
                predictor, decoder = (
                    model.to(config.device) for model in _networks(config, state_dim, action_dim)
                )
                optimizer = torch.optim.AdamW(
                    [*predictor.parameters(), *decoder.parameters()], lr=config.learning_rate
                )
                windows = [
                    (row, index)
                    for row in rows["train"]
                    for index in range(
                        len(row["actions"]) - config.history - config.training_horizon + 2
                    )
                ]
                if not windows:
                    raise DomainValidationError("training split contains no valid context windows")
                losses = []
                validation_curve = []
                best_visual_score = best_state_score = float("inf")
                selected_visual_step = selected_state_step = 0
                selected_visual_gain = 0.0
                best_visual_weights: dict[str, Any] = {}
                best_state_weights: dict[str, Any] = {}
                best_residual_curve: list[dict[str, float]] = []
                scale = statistics["state_scale"].to(config.device)
                for iteration in range(config.steps):
                    chosen = [
                        windows[index]
                        for index in torch.randint(len(windows), (config.batch_size,)).tolist()
                    ]
                    history = config.history
                    horizon = config.training_horizon
                    z, s = (
                        torch.stack([row[name][index : index + history] for row, index in chosen])
                        for name in ("features", "states")
                    )
                    past = torch.stack(
                        [row["actions"][index : index + history - 1] for row, index in chosen]
                    )
                    future = torch.stack(
                        [
                            row["actions"][index + history - 1 : index + history - 1 + horizon]
                            for row, index in chosen
                        ]
                    )
                    target_z, target_s, target_rgb = (
                        torch.stack(
                            [
                                row[name][index + history : index + history + horizon]
                                for row, index in chosen
                            ]
                        )
                        for name in ("features", "states", "rgb")
                    )
                    predicted_z, predicted_s = _rollout(predictor, z, s, past, future)
                    context_rgb = torch.stack(
                        [row["rgb"][index + history - 1] for row, index in chosen]
                    )
                    decoded = _render_future(
                        decoder, predicted_z, z[:, -1], context_rgb, config.image_size
                    )
                    loss = (
                        (predicted_z - target_z).square().mean()
                        + config.state_loss_weight * (predicted_s - target_s).square().mean()
                        + (decoded - target_rgb).square().mean()
                    )
                    if not torch.isfinite(loss):
                        raise MethodExecutionError("DINO-WM training loss is non-finite")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [*predictor.parameters(), *decoder.parameters()],
                        1.0,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    if (
                        iteration + 1
                    ) % config.validation_interval == 0 or iteration + 1 == config.steps:
                        predictor.eval()
                        decoder.eval()
                        gain, _, residual_curve, selection = _calibrate_visual_residual(
                            predictor,
                            decoder,
                            rows["validation"],
                            config,
                            scale,
                        )
                        visual_score = max(
                            selection["latent_mse_ratio"], selection["rgb_mse_ratio"]
                        )
                        state_score = selection["normalized_state_rmse_ratio"]
                        validation_curve.append(
                            {
                                "step": iteration + 1,
                                "visual_residual_gain": gain,
                                "visual_score": visual_score,
                                "state_score": state_score,
                                **selection,
                            }
                        )
                        if visual_score < best_visual_score:
                            best_visual_score = visual_score
                            selected_visual_step = iteration + 1
                            selected_visual_gain = gain
                            best_residual_curve = residual_curve
                            best_visual_weights = {
                                prefix + key: value.detach().cpu().clone()
                                for prefix, model in (
                                    ("predictor.", predictor),
                                    ("decoder.", decoder),
                                )
                                for key, value in model.state_dict().items()
                                if prefix == "decoder." or not key.startswith("state_delta.")
                            }
                        if state_score < best_state_score:
                            best_state_score = state_score
                            selected_state_step = iteration + 1
                            best_state_weights = {
                                "predictor." + key: value.detach().cpu().clone()
                                for key, value in predictor.state_dict().items()
                                if key.startswith("state_delta.")
                            }
                        predictor.train()
                        decoder.train()
                if not best_visual_weights or not best_state_weights:
                    raise MethodExecutionError("DINO-WM has no finite validation checkpoints")
                best_weights = {**best_visual_weights, **best_state_weights}
                for prefix, model in (("predictor.", predictor), ("decoder.", decoder)):
                    model.load_state_dict(
                        {
                            key[len(prefix) :]: value
                            for key, value in best_weights.items()
                            if key.startswith(prefix)
                        }
                    )
                predictor.eval()
                decoder.eval()
                _apply_visual_residual_gain(predictor, selected_visual_gain)
                metrics: dict[str, float] = {
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                    "selected_visual_step": float(selected_visual_step),
                    "selected_state_step": float(selected_state_step),
                    "selected_visual_residual_gain": selected_visual_gain,
                    "validation_visual_selection_score": best_visual_score,
                    "validation_state_selection_score": best_state_score,
                }
                evaluations = {}
                for name in ("validation", "test"):
                    errors = _evaluate(predictor, decoder, rows[name], config, scale)
                    evaluations[name] = errors
                    for metric in (
                        "normalized_state_rmse",
                        "persistence_normalized_state_rmse",
                        "latent_mse",
                        "persistence_latent_mse",
                        "rgb_mse",
                        "persistence_rgb_mse",
                    ):
                        metrics[name + "_" + metric] = sum(row[metric] for row in errors) / len(
                            errors
                        )
                metrics["validation_selection_score"] = _validation_selection(
                    evaluations["validation"]
                )["score"]
                tensors = {
                    prefix + key: value.detach().cpu().contiguous()
                    for prefix, model in (("predictor.", predictor), ("decoder.", decoder))
                    for key, value in model.state_dict().items()
                }
                weights = artifacts.put_bytes(
                    save(tensors), media_type="application/vnd.safetensors"
                )
        finally:
            torch.set_num_threads(previous_threads)
        document = {
            "schema": "physrealx.dino-wm-checkpoint/3",
            "method": spec.method.model_dump(mode="json"),
            "snapshot": snapshot_ref.model_dump(mode="json"),
            "encoder": encoder_ref.model_dump(mode="json"),
            "weights": weights.model_dump(mode="json"),
            "configuration": config.model_dump(mode="json"),
            "statistics": {name: value.tolist() for name, value in statistics.items()},
            "state_dim": state_dim,
            "action_dim": action_dim,
            "interval_seconds": intervals[0],
            "observation_schema": [
                value.model_dump(mode="json") for value in dataset.observation_schema
            ],
            "action_schema": dataset.action_schema.model_dump(mode="json"),
            "evidence_scope": "learned_prediction_on_simulator_episode_split",
            "adaptations": [
                "learned state/action embeddings and separate residual visual/state heads",
                "state transitions use Markov proprioception and controls; "
                "visual drift cannot feed back into mechanics",
                "context-preserving residual RGB decoder trained with the inference rendering path",
                "bounded demo resolution and network size",
                "differentiable multi-step training using the inference rollout",
                "independent validation checkpoints for decoupled visual and state branches",
                "validation-only visual residual calibration bounds long-horizon drift",
            ],
        }
        report = {
            "schema": "physrealx.world-model-training-report/1",
            "execution_status": "COMPLETED",
            "metrics": metrics,
            "held_out": evaluations,
            "loss_curve": losses,
            "validation_curve": validation_curve,
            "visual_residual_calibration_curve": best_residual_curve,
            "state_quality_status": _quality_status(
                metrics["test_normalized_state_rmse"],
                metrics["test_persistence_normalized_state_rmse"],
            ),
            "latent_quality_status": _quality_status(
                sum(row["latent_mse"] for row in evaluations["test"]),
                sum(row["persistence_latent_mse"] for row in evaluations["test"]),
            ),
            "rgb_quality_status": _quality_status(
                metrics["test_rgb_mse"], metrics["test_persistence_rgb_mse"]
            ),
            "train_episode_ids": snapshot.split.train,
            "independent_physical_validation": False,
            "uncertainty_calibrated": False,
        }
        checkpoint = artifacts.put_json(document)
        diagnostics = artifacts.put_json(report)
        model = VersionedRef(
            resource_type=ResourceType.MODEL,
            resource_id=spec.resource_id,
            version=spec.version,
            artifact_id=checkpoint.artifact_id,
        )
        plan = WorldModelTrainingPlan(
            model_id=model.resource_id,
            version=model.version,
            dataset=snapshot,
            trainer=spec.method,
            rollout=WorldModelRolloutAdapter.descriptor.bind(),
            code=code,
            environment=environment,
            seed=config.seed,
        )
        result = TrainingExecutionResult(
            run_id=str(uuid.uuid4()),
            kind=ModelKind.WORLD_MODEL,
            model=model,
            dataset_snapshot_id=snapshot.record_id,
            trainer_record_id=canonical_record_id(spec.method),
            code_record_id=code.record_id,
            environment_record_id=environment.record_id,
            split_record_id=snapshot.split.record_id,
            seed=config.seed,
            checkpoint=weights,
            auxiliary_artifacts=(checkpoint, encoder_ref),
            evidence=(diagnostics,),
            metrics=metrics,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
        bundle = TrainingLifecycle.finalize_world_model(plan, result)
        return publish(
            spec, artifacts, {"model": bundle, "training_report": bundle.report}, metrics=metrics
        )


class WorldModelContext(DomainModel):
    schema_id: Literal["physrealx.world-model-context/1"] = Field(
        default="physrealx.world-model-context/1", alias="schema"
    )
    episode: EpisodeSummary
    start_index: int = Field(ge=0)


class WorldModelActions(DomainModel):
    schema_id: Literal["physrealx.world-model-actions/1"] = Field(
        default="physrealx.world-model-actions/1", alias="schema"
    )
    values: tuple[tuple[float, ...], ...] = Field(min_length=1, max_length=128)
    interval_seconds: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_actions(self) -> WorldModelActions:
        import math

        width = len(self.values[0])
        if not width or any(
            len(row) != width or not all(math.isfinite(value) for value in row)
            for row in self.values
        ):
            raise ValueError("world model actions must be finite nonempty equal-width vectors")
        return self


class WorldModelSignalEvaluation(DomainModel):
    unit: str
    frame: str
    column_start: int = Field(ge=0)
    column_stop: int = Field(gt=0)
    rmse: tuple[float, ...]
    persistence_rmse: tuple[float, ...]
    rmse_ratio: tuple[float | None, ...]
    status: tuple[Literal["BEATS_PERSISTENCE", "MATCHES_PERSISTENCE", "BELOW_BASELINE"], ...]

    @model_validator(mode="after")
    def validate_dimensions(self) -> WorldModelSignalEvaluation:
        width = self.column_stop - self.column_start
        if width <= 0 or any(
            len(values) != width
            for values in (self.rmse, self.persistence_rmse, self.rmse_ratio, self.status)
        ):
            raise ValueError("world model signal evaluation dimensions do not match")
        return self


class WorldModelRolloutEvaluation(DomainModel):
    schema_id: Literal["physrealx.world-model-rollout-evaluation/1"] = Field(
        default="physrealx.world-model-rollout-evaluation/1", alias="schema"
    )
    rollout: ArtifactRef
    reference_episode_id: str
    reference_run_id: str
    start_index: int = Field(ge=0)
    horizon: int = Field(gt=0)
    timestamp_seconds: tuple[float, ...]
    reference_states: ArtifactRef
    signals: dict[str, WorldModelSignalEvaluation]
    evidence_scope: Literal["episode_state_comparison"] = "episode_state_comparison"

    @model_validator(mode="after")
    def validate_evaluation(self) -> WorldModelRolloutEvaluation:
        if len(self.timestamp_seconds) != self.horizon or not self.signals:
            raise ValueError("world model rollout evaluation is incomplete")
        columns = sorted((row.column_start, row.column_stop) for row in self.signals.values())
        if columns[0][0] != 0 or any(
            left[1] != right[0] for left, right in zip(columns, columns[1:])
        ):
            raise ValueError("world model rollout evaluation columns are not contiguous")
        return self


def evaluate_world_model_rollout(
    artifacts: ArtifactStore,
    rollout_ref: ArtifactRef,
    reference: Episode,
    *,
    start_index: int,
) -> ArtifactRef:
    """Compare a rollout with an episode and its identical persistence baseline."""
    import numpy as np

    document = read_json(artifacts, rollout_ref, "physrealx.world-model-rollout/1")
    state_signals = tuple(document.get("state_signals", ()))
    timestamps = tuple(float(value) for value in document.get("timestamp_seconds", ()))
    predicted = read_array(artifacts, ArtifactRef.model_validate(document["states"]))
    if predicted.ndim != 2 or not state_signals or len(timestamps) != len(predicted):
        raise DomainValidationError("world model rollout state evidence is incomplete")
    stop = start_index + 1 + len(predicted)
    if start_index < 0 or stop > len(reference.steps):
        raise DomainValidationError("reference episode cannot cover the rollout horizon")
    truth_steps = reference.steps[start_index + 1 : stop]
    if any(
        abs(observed - expected.timestamp_seconds) > 1e-8
        for observed, expected in zip(timestamps, truth_steps)
    ):
        raise DomainValidationError("rollout and reference episode timestamps differ")

    truth_columns = []
    initial_columns = []
    signals = {}
    column = 0
    for name in state_signals:
        initial = reference.steps[start_index].observations.get(name)
        values = [step.observations.get(name) for step in truth_steps]
        if (
            initial is None
            or initial.inline is None
            or any(value is None or value.inline is None for value in values)
        ):
            raise DomainValidationError(f"reference episode lacks inline state {name}")
        rows = [value.inline for value in values if value is not None]
        truth = np.asarray(rows, dtype=np.float64)
        initial_array = np.asarray(initial.inline, dtype=np.float64)
        width = len(initial_array)
        if truth.shape != (len(predicted), width):
            raise DomainValidationError(f"reference state {name} changes dimensions")
        truth_columns.append(truth)
        initial_columns.append(initial_array)
        column += width
    reference_states = np.concatenate(truth_columns, axis=1)
    initial_state = np.concatenate(initial_columns)
    if predicted.shape != reference_states.shape or not np.isfinite(predicted).all():
        raise DomainValidationError("predicted and reference state dimensions differ")

    column = 0
    for name, truth in zip(state_signals, truth_columns):
        initial = reference.steps[start_index].observations[name]
        width = truth.shape[1]
        actual = np.sqrt(np.mean((predicted[:, column : column + width] - truth) ** 2, axis=0))
        baseline = np.sqrt(np.mean((initial_state[column : column + width] - truth) ** 2, axis=0))
        ratios: list[float | None] = []
        statuses = []
        for error, persistence in zip(actual, baseline):
            if persistence <= 1e-12:
                ratios.append(None)
                statuses.append("MATCHES_PERSISTENCE" if error <= 1e-12 else "BELOW_BASELINE")
            else:
                ratios.append(float(error / persistence))
                statuses.append(
                    "BEATS_PERSISTENCE"
                    if error < persistence
                    else "MATCHES_PERSISTENCE"
                    if error == persistence
                    else "BELOW_BASELINE"
                )
        signals[name] = WorldModelSignalEvaluation(
            unit=initial.unit,
            frame=initial.frame,
            column_start=column,
            column_stop=column + width,
            rmse=tuple(float(value) for value in actual),
            persistence_rmse=tuple(float(value) for value in baseline),
            rmse_ratio=tuple(ratios),
            status=tuple(statuses),
        )
        column += width
    return artifacts.put_json(
        WorldModelRolloutEvaluation(
            rollout=rollout_ref,
            reference_episode_id=reference.episode_id,
            reference_run_id=reference.run_id,
            start_index=start_index,
            horizon=len(predicted),
            timestamp_seconds=timestamps,
            reference_states=_array(artifacts, reference_states),
            signals=signals,
        ).model_dump(mode="json", by_alias=True)
    )


class WorldModelRolloutConfig(DomainModel):
    device: Literal["cpu", "cuda"] = "cpu"


class WorldModelRolloutAdapter:
    descriptor = builtin_method(
        MethodCapability.WORLD_MODEL_ROLLOUT,
        "dino_wm",
        version="4.0.0",
        input_contract="ModelBundle/1+ContextWindow/1+ActionSequence/1",
        output_contract="WorldModelRollout/1+UncertaintyReport/1",
        python=">=3.10,<3.13",
    ).model_copy(
        update={
            "source_uri": "https://github.com/gaoyuezhou/dino_wm",
            "source_revision": WM_REVISION,
            "license_spdx": "MIT",
        }
    )

    def probe(self) -> MethodProbe:
        return WorldModelTrainAdapter().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np
        import torch
        from safetensors.torch import load

        model_ref = input_artifact(spec, "model", artifacts)
        bundle = WorldModelBundle.model_validate(
            read_json(artifacts, model_ref, "physrealx.world-model-bundle/1")
        )
        if bundle.rollout.descriptor != self.descriptor or bundle.model.artifact_id is None:
            raise DomainValidationError(
                "world model rollout identity differs from its training bundle"
            )
        document = json.loads(artifacts.get_bytes(bundle.model.artifact_id))
        if (
            document.get("schema") != "physrealx.dino-wm-checkpoint/3"
            or ArtifactRef.model_validate(document["weights"]) != bundle.report.checkpoint
        ):
            raise DomainValidationError(
                "world model configuration differs from its pinned checkpoint"
            )
        config = WorldModelConfig.model_validate(document["configuration"])
        runtime = WorldModelRolloutConfig.model_validate(spec.method.config)
        context = WorldModelContext.model_validate(
            read_json(
                artifacts,
                input_artifact(spec, "context", artifacts),
                "physrealx.world-model-context/1",
            )
        )
        actions = WorldModelActions.model_validate(
            read_json(
                artifacts,
                input_artifact(spec, "actions", artifacts),
                "physrealx.world-model-actions/1",
            )
        )
        episode = load_episode(artifacts, context.episode)
        index, h = context.start_index, config.history
        if index < h - 1 or index >= len(episode.steps):
            raise DomainValidationError(
                "world model context index cannot supply the trained history"
            )
        if abs(actions.interval_seconds - document["interval_seconds"]) > 1e-8:
            raise DomainValidationError("rollout action frequency differs from training")
        from physrealx.domain import SignalSpec

        schemas = {
            row["name"]: SignalSpec.model_validate(row) for row in document["observation_schema"]
        }
        action_schema = SignalSpec.model_validate(document["action_schema"])
        for step in episode.steps[index - h + 1 : index + 1]:
            for name in (config.image_signal, *config.state_signals):
                if name not in step.observations:
                    raise DomainValidationError("context lacks a trained observation signal")
                step.observations[name].assert_matches(schemas[name])
        for step in episode.steps[index - h + 1 : index]:
            if step.action is None or step.action.inline is None:
                raise DomainValidationError("context lacks an action between historical frames")
            step.action.assert_matches(action_schema)
        for left, right in zip(
            episode.steps[index - h + 1 : index], episode.steps[index - h + 2 : index + 1]
        ):
            if (
                abs(right.timestamp_seconds - left.timestamp_seconds - actions.interval_seconds)
                > 1e-8
            ):
                raise DomainValidationError("context sampling frequency differs from training")
        array = np.asarray(actions.values, dtype=np.float32)
        if (
            array.shape != (len(actions.values), document["action_dim"])
            or not np.isfinite(array).all()
        ):
            raise DomainValidationError(
                "rollout actions differ from trained dimensions or are non-finite"
            )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(config.threads)
        try:
            predictor, decoder = (
                model.to(runtime.device).eval()
                for model in _networks(config, document["state_dim"], document["action_dim"])
            )
            weights = load(artifacts.get_bytes(ArtifactRef.model_validate(document["weights"])))
            for prefix, model in (("predictor.", predictor), ("decoder.", decoder)):
                model.load_state_dict(
                    {
                        key[len(prefix) :]: value
                        for key, value in weights.items()
                        if key.startswith(prefix)
                    },
                    strict=True,
                )
            encoder_spec = EncoderSpec.model_validate(
                read_json(
                    artifacts,
                    ArtifactRef.model_validate(document["encoder"]),
                    "physrealx.pretrained-encoder/1",
                )
            )
            encoder = encoder_network().to(runtime.device).eval()
            encoder.load_state_dict(load(artifacts.get_bytes(encoder_spec.weights)), strict=True)
            rgb = _rgb(episode, config, artifacts, start=index - h + 1, stop=index + 1)
            features = _features(encoder, rgb, runtime.device, config.batch_size).to(runtime.device)
            statistics = {
                name: torch.tensor(value, device=runtime.device)
                for name, value in document["statistics"].items()
            }
            states = torch.tensor(
                _states(episode, config, start=index - h + 1, stop=index + 1), device=runtime.device
            )
            states = (states - statistics["state_mean"]) / statistics["state_scale"]
            past = [
                step.action.inline
                for step in episode.steps[index - h + 1 : index]
                if step.action and step.action.inline is not None
            ]
            past_actions = torch.tensor(past, device=runtime.device).reshape(
                h - 1, document["action_dim"]
            )
            past_actions = (past_actions - statistics["action_mean"]) / statistics["action_scale"]
            future_actions = (
                torch.from_numpy(array).to(runtime.device) - statistics["action_mean"]
            ) / statistics["action_scale"]
            with torch.inference_mode():
                z, s = _rollout(
                    predictor,
                    features[None],
                    states[None],
                    past_actions[None],
                    future_actions[None],
                )
                z, s = z[0], s[0]
                raw_state = (s * statistics["state_scale"] + statistics["state_mean"]).cpu().numpy()
                predicted_rgb = (
                    (
                        _render_future(
                            decoder,
                            z[None],
                            features[-1:],
                            rgb[-1:].to(runtime.device),
                            config.image_size,
                        )[0].clamp(0, 1)
                        * 255
                    )
                    .round()
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                if not torch.isfinite(z).all() or not np.isfinite(raw_state).all():
                    raise MethodExecutionError("world model rollout diverged to non-finite values")
                distance = float((z - features[-1]).square().mean().sqrt())
        finally:
            torch.set_num_threads(previous_threads)
        rollout = {
            "schema": "physrealx.world-model-rollout/1",
            "model": model_ref.model_dump(mode="json"),
            "source": "world_model",
            "state_signals": config.state_signals,
            "states": _array(artifacts, raw_state).model_dump(mode="json"),
            "rgb": _array(artifacts, predicted_rgb).model_dump(mode="json"),
            "timestamp_seconds": [
                episode.steps[index].timestamp_seconds + (step + 1) * actions.interval_seconds
                for step in range(len(actions.values))
            ],
            "independent_physical_validation": False,
        }
        uncertainty = {
            "schema": "physrealx.uncertainty-report/1",
            "calibrated": False,
            "latent_change_rms": distance,
            "interpretation": "feature change diagnostic; uncalibrated",
        }
        return publish(
            spec,
            artifacts,
            {"rollout": rollout, "uncertainty": uncertainty},
            metrics={"horizon": float(len(actions.values)), "latent_change_rms": distance},
        )
