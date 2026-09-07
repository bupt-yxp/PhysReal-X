"""Train and execute the official LeRobot ACT network through the platform's learning ports."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore, ArtifactWriter, Frame
from physrealx.domain import (
    DatasetManifest,
    DataValue,
    DomainModel,
    Episode,
    JSONValue,
    MethodCapability,
    ModelManifest,
    TaskSpec,
    TrainingSpec,
)
from physrealx.errors import (
    DomainValidationError,
    InvalidStateTransitionError,
    MethodExecutionError,
)
from physrealx.integrations._artifacts import read_array
from physrealx.integrations.learning_sources import algorithm_module
from physrealx.integrations.provenance import capture_runtime
from physrealx.methods import builtin_method

SOURCE_REVISION = "b883328e6c95681ca90a18b102e4ae5e1f91e2bf"


class FeatureShape(DomainModel):
    shape: tuple[int, ...]


class ACTNetworkConfig(DomainModel):
    state_dim: int = Field(gt=0)
    action_dim: int = Field(gt=0)
    image_signals: tuple[str, ...] = Field(min_length=1)
    chunk_size: int = Field(default=8, ge=1, le=128)
    dim_model: int = Field(default=64, ge=16, le=1024)
    n_heads: int = Field(default=4, ge=1)
    dim_feedforward: int = Field(default=256, ge=32, le=8192)
    n_encoder_layers: int = Field(default=2, ge=1, le=12)
    n_decoder_layers: int = Field(default=1, ge=1, le=12)
    n_vae_encoder_layers: int = Field(default=1, ge=1, le=12)
    latent_dim: int = Field(default=16, ge=2, le=256)
    use_vae: Literal[True] = True
    pre_norm: bool = False
    dropout: float = Field(default=0.1, ge=0, lt=1)
    feedforward_activation: Literal["relu", "gelu"] = "relu"
    vision_backbone: Literal["resnet18"] = "resnet18"
    pretrained_backbone_weights: None = None
    replace_final_stride_with_dilation: Literal[False] = False

    @model_validator(mode="after")
    def validate_dimensions(self) -> ACTNetworkConfig:
        if self.dim_model % self.n_heads or self.dim_model % 4:
            raise ValueError(
                "ACT model dimension must divide attention heads and positional encoding"
            )
        if len(set(self.image_signals)) != len(self.image_signals):
            raise ValueError("ACT image signals must be unique")
        return self

    @property
    def robot_state_feature(self) -> FeatureShape:
        return FeatureShape(shape=(self.state_dim,))

    @property
    def action_feature(self) -> FeatureShape:
        return FeatureShape(shape=(self.action_dim,))

    @property
    def image_features(self) -> tuple[str, ...]:
        return self.image_signals

    @property
    def env_state_feature(self) -> None:
        return None


def network(config: ACTNetworkConfig) -> Any:
    """Load the same pinned source in editable checkouts and built wheels."""
    return algorithm_module("act").ACT(config)


class ACTTrainingConfig(DomainModel):
    observation_signals: tuple[str, ...] = ("joint.position", "joint.velocity")
    image_signals: tuple[str, ...] = ("camera.front.rgb",)
    steps: int = Field(default=300, ge=1, le=1000000)
    batch_size: int = Field(default=16, ge=1, le=512)
    chunk_size: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=0.0003, gt=0, le=1)
    kl_weight: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    seed: int = Field(default=2026, ge=0)
    device: Literal["cpu", "cuda"] = "cpu"
    threads: int = Field(default=4, ge=1, le=64)
    max_transitions: int = Field(default=10000, ge=1, le=1000000)


def _observation(
    values: Mapping[str, DataValue], config: ACTTrainingConfig, store: ArtifactStore
) -> tuple[list[float], list[Any]]:
    import numpy as np

    state: list[float] = []
    images = []
    for name in config.observation_signals:
        value = values.get(name)
        if value is None or value.inline is None:
            raise DomainValidationError(f"ACT requires inline state signal {name}")
        state.extend(float(v) for v in value.inline)
    for name in config.image_signals:
        value = values.get(name)
        if value is None or value.artifact is None:
            raise DomainValidationError(f"ACT requires an immutable RGB signal {name}")
        image = read_array(store, value.artifact)
        if (
            image.ndim != 3
            or image.shape[-1] != 3
            or image.dtype != np.uint8
            or min(image.shape[:2]) < 32
        ):
            raise DomainValidationError(
                "ACT expects RGB uint8 images of at least 32 pixels per side"
            )
        # ImageNet channel normalization used by the official visual ACT policy.
        image = (
            image.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        images.append(image.transpose(2, 0, 1))
    return state, images


class ACTTrainer:
    descriptor = builtin_method(
        MethodCapability.POLICY_TRAIN,
        "act",
        input_contract="TrainingSpec/1+DatasetManifest/1+Episode/1",
        output_contract="ModelManifest/1",
        python=">=3.10,<3.13",
    ).model_copy(
        update={
            "source_uri": "https://github.com/huggingface/lerobot",
            "source_revision": SOURCE_REVISION,
            "license_spdx": "Apache-2.0",
        }
    )

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def train(
        self,
        spec: TrainingSpec,
        dataset: DatasetManifest,
        episodes: Iterable[Episode],
        artifacts: ArtifactWriter,
    ) -> ModelManifest:
        import numpy as np
        import torch
        from safetensors.torch import save

        config = ACTTrainingConfig.model_validate(spec.trainer.config)
        started_at = datetime.now(timezone.utc)
        source, environment = capture_runtime(self.store, algorithms=("act",))
        if config.device == "cuda" and not torch.cuda.is_available():
            raise DomainValidationError(
                "CUDA ACT training requested without an available CUDA device"
            )
        states, images, actions, padding = [], [], [], []
        for episode in episodes:
            steps = [step for step in episode.steps if not step.is_last]
            if not steps:
                continue
            for index, step in enumerate(steps):
                state, rgb = _observation(step.observations, config, self.store)
                sequence = [
                    row.action.inline
                    for row in steps[index : index + config.chunk_size]
                    if row.action is not None and row.action.inline is not None
                ]
                if len(sequence) != min(config.chunk_size, len(steps) - index):
                    raise DomainValidationError("ACT requires complete inline action chunks")
                valid = len(sequence)
                sequence += [sequence[-1]] * (config.chunk_size - valid)
                states.append(state)
                images.append(rgb)
                actions.append(sequence)
                padding.append([False] * valid + [True] * (config.chunk_size - valid))
                if len(states) > config.max_transitions:
                    raise DomainValidationError(
                        "ACT training exceeds the explicit transition memory bound"
                    )
        if not states:
            raise DomainValidationError("ACT dataset has no transitions")
        state_array, action_array = np.asarray(states, np.float32), np.asarray(actions, np.float32)
        mean, scale = state_array.mean(axis=0), np.maximum(state_array.std(axis=0), 1e-4)
        valid_actions = action_array[~np.asarray(padding)]
        action_mean = valid_actions.mean(axis=0)
        action_scale = np.maximum(valid_actions.std(axis=0), 1e-4)
        architecture = ACTNetworkConfig(
            state_dim=state_array.shape[-1],
            action_dim=action_array.shape[-1],
            image_signals=config.image_signals,
            chunk_size=config.chunk_size,
        )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(config.threads)
        try:
            with torch.random.fork_rng(
                devices=[] if config.device == "cpu" else [torch.cuda.current_device()]
            ):
                torch.manual_seed(config.seed)
                model = network(architecture).to(config.device)
                model.train()
                x = torch.from_numpy((state_array - mean) / scale).to(config.device)
                y = torch.from_numpy((action_array - action_mean) / action_scale).to(config.device)
                cameras = [
                    torch.from_numpy(np.stack([row[i] for row in images])).to(config.device)
                    for i in range(len(config.image_signals))
                ]
                pad = torch.tensor(padding, dtype=torch.bool, device=config.device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=config.learning_rate, weight_decay=1e-4
                )
                losses = []
                for _ in range(config.steps):
                    indices = torch.randint(
                        len(states), (min(config.batch_size, len(states)),), device=config.device
                    )
                    batch = {
                        "observation.state": x[indices],
                        "observation.images": [image[indices] for image in cameras],
                        "action": y[indices],
                        "action_is_pad": pad[indices],
                    }
                    prediction, (mu, log_variance) = model(batch)
                    valid_mask = (~pad[indices]).unsqueeze(-1)
                    reconstruction = ((prediction - y[indices]).abs() * valid_mask).sum() / (
                        valid_mask.sum() * architecture.action_dim
                    )
                    kl = (
                        -0.5 * (1 + log_variance - mu.square() - log_variance.exp()).sum(-1)
                    ).mean()
                    loss = reconstruction + config.kl_weight * kl
                    if not torch.isfinite(loss):
                        raise MethodExecutionError("ACT training generated a non-finite loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                with torch.inference_mode():
                    predicted, _ = model(
                        {
                            "observation.state": x[: config.batch_size],
                            "observation.images": [image[: config.batch_size] for image in cameras],
                        }
                    )
                    training_mae = float(
                        (
                            predicted[:, 0] * torch.tensor(action_scale, device=config.device)
                            + torch.tensor(action_mean, device=config.device)
                            - torch.tensor(
                                action_array[: config.batch_size, 0], device=config.device
                            )
                        )
                        .abs()
                        .mean()
                    )
                weights = artifacts.put_bytes(
                    save(
                        {
                            name: value.detach().cpu().contiguous()
                            for name, value in model.state_dict().items()
                        }
                    ),
                    media_type="application/vnd.safetensors",
                )
        finally:
            torch.set_num_threads(previous_threads)
        selected_names = (*config.observation_signals, *config.image_signals)
        schema = {signal.name: signal for signal in dataset.observation_schema}
        if len(set(selected_names)) != len(selected_names) or any(
            name not in schema for name in selected_names
        ):
            raise DomainValidationError(
                "ACT training signals must uniquely match the dataset schema"
            )
        return ModelManifest(
            resource_id=spec.resource_id,
            version=spec.version,
            trainer=spec.trainer,
            policy=ACTPolicy.descriptor.bind(
                checkpoint=weights, config={"n_action_steps": architecture.chunk_size}
            ),
            dataset=spec.dataset,
            weights=weights,
            observation_signals=tuple(schema[name] for name in selected_names),
            action_signal=dataset.action_schema,
            training_config={
                "configuration": config.model_dump(mode="json"),
                "architecture": architecture.model_dump(mode="json"),
                "state_mean": mean.tolist(),
                "state_scale": scale.tolist(),
                "action_mean": action_mean.tolist(),
                "action_scale": action_scale.tolist(),
                "source_revision": SOURCE_REVISION,
                "loss_curve": losses,
                "training_scope": "from_scratch_on_selected_episode_split",
                "code": source.model_dump(mode="json"),
                "environment": environment.model_dump(mode="json"),
                "started_at": started_at.isoformat(),
            },
            metrics={
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "training_action_mae": training_mae,
                "transitions": float(len(states)),
            },
            created_at=datetime.now(timezone.utc),
        )


class ACTPolicyConfig(DomainModel):
    model_artifact_id: str
    device: Literal["cpu", "cuda"] = "cpu"
    n_action_steps: int | None = Field(default=None, ge=1)


class ACTPolicy:
    descriptor = builtin_method(
        MethodCapability.POLICY_INFERENCE,
        "act",
        input_contract="Frame/1+TaskSpec/1+ModelManifest/1",
        output_contract="DataValue/1",
        python=">=3.10,<3.13",
    ).model_copy(
        update={
            "source_uri": "https://github.com/huggingface/lerobot",
            "source_revision": SOURCE_REVISION,
            "license_spdx": "Apache-2.0",
        }
    )

    def __init__(self, config: Mapping[str, JSONValue], store: ArtifactStore) -> None:
        import torch
        from safetensors.torch import load

        self.store = store
        self.config = ACTPolicyConfig.model_validate(config)
        manifest = ModelManifest.model_validate_json(store.get_bytes(self.config.model_artifact_id))
        if manifest.policy.descriptor != self.descriptor:
            raise DomainValidationError(
                "ACT policy requires a checkpoint with matching source identity"
            )
        self.training = ACTTrainingConfig.model_validate(manifest.training_config["configuration"])
        self.architecture = ACTNetworkConfig.model_validate(
            manifest.training_config["architecture"]
        )
        self.n_action_steps = self.config.n_action_steps or self.architecture.chunk_size
        if self.n_action_steps > self.architecture.chunk_size:
            raise DomainValidationError(
                "ACT execution chunk exceeds the trained prediction horizon"
            )
        self.model = network(self.architecture).to(self.config.device)
        self.model.load_state_dict(load(store.get_bytes(manifest.weights)), strict=True)
        self.model.eval()
        self.statistics = {
            name: torch.tensor(
                manifest.training_config[name], dtype=torch.float32, device=self.config.device
            )
            for name in ("state_mean", "state_scale", "action_mean", "action_scale")
        }
        self.action_signal = manifest.action_signal
        self._commands: deque[Any] = deque()
        self._closed = False

    def reset(self, seed: int) -> None:
        del seed
        if self._closed:
            raise InvalidStateTransitionError("ACT policy is closed")
        self._commands.clear()

    def act(self, frame: Frame, task: TaskSpec, step_index: int) -> DataValue:
        import numpy as np
        import torch

        del step_index
        if self._closed:
            raise InvalidStateTransitionError("ACT policy is closed")
        if task.action.signal != self.action_signal:
            raise DomainValidationError("ACT action schema differs from its training data")
        if not self._commands:
            state, images = _observation(frame.observations, self.training, self.store)
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(self.training.threads)
            try:
                with torch.inference_mode():
                    vector = torch.tensor([state], dtype=torch.float32, device=self.config.device)
                    batch = {
                        "observation.state": (vector - self.statistics["state_mean"])
                        / self.statistics["state_scale"],
                        "observation.images": [
                            torch.from_numpy(image[None]).to(self.config.device) for image in images
                        ],
                    }
                    prediction, _ = self.model(batch)
                    commands = (
                        (
                            prediction[0] * self.statistics["action_scale"]
                            + self.statistics["action_mean"]
                        )
                        .cpu()
                        .numpy()
                    )
            finally:
                torch.set_num_threads(previous_threads)
            if not np.isfinite(commands).all():
                raise MethodExecutionError("ACT produced non-finite commands")
            self._commands.extend(commands[: self.n_action_steps])
        bounded = np.clip(self._commands.popleft(), task.action.lower, task.action.upper)
        return DataValue(
            dtype=task.action.signal.dtype,
            shape=task.action.signal.shape,
            unit=task.action.signal.unit,
            frame=task.action.signal.frame,
            inline=tuple(float(v) for v in bounded),
        )

    def close(self) -> None:
        self._closed = True
        self._commands.clear()
