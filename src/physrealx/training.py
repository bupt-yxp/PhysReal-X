"""Optional PyTorch/cu121 behavior-cloning adapter.

Torch and safetensors are imported lazily so the platform core remains usable on
machines that only validate or generate data.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from typing import Any

from physrealx.contracts import API_VERSION, ArtifactReader, ArtifactWriter, Frame
from physrealx.domain import (
    DatasetManifest,
    DataValue,
    Episode,
    JSONValue,
    ModelManifest,
    TaskSpec,
    TrainingSpec,
)
from physrealx.errors import DomainValidationError, InvalidStateTransitionError


def _torch_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors import torch as safetensors_torch
    except ImportError as error:
        raise DomainValidationError(
            "Torch training requires the PhysReal-X cu121 environment and safetensors"
        ) from error
    return torch, torch.nn, safetensors_torch


def _reject_unknown(
    config: Mapping[str, JSONValue],
    allowed: frozenset[str],
    *,
    label: str,
) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise DomainValidationError(f"unknown {label} config fields: {sorted(unknown)!r}")


def _string(config: Mapping[str, JSONValue], name: str, default: str) -> str:
    value = config.get(name, default)
    if not isinstance(value, str) or not value:
        raise DomainValidationError(f"training config {name!r} must be a non-empty string")
    return value


def _integer(config: Mapping[str, JSONValue], name: str, default: int, *, minimum: int = 1) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DomainValidationError(f"training config {name!r} must be >= {minimum}")
    return value


def _float(
    config: Mapping[str, JSONValue], name: str, default: float, *, minimum: float = 0.0
) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainValidationError(f"training config {name!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= minimum:
        raise DomainValidationError(f"training config {name!r} must be > {minimum}")
    return result


def _device(torch: Any, requested: str) -> Any:
    if requested not in {"auto", "cpu", "cuda"}:
        raise DomainValidationError("device must be one of: auto, cpu, cuda")
    cuda_ready = bool(torch.cuda.is_available()) and torch.version.cuda == "12.1"
    if requested == "cuda" and not cuda_ready:
        raise DomainValidationError(
            "CUDA training requires an available PyTorch build compiled for CUDA 12.1"
        )
    selected = "cuda" if requested == "cuda" or (requested == "auto" and cuda_ready) else "cpu"
    return torch.device(selected)


def _network(nn: Any, input_dim: int, hidden_dim: int, output_dim: int) -> Any:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TorchBehaviorCloningTrainer:
    name = "torch_bc"
    plugin_version = "1.0.0"
    api_version = API_VERSION

    def train(
        self,
        spec: TrainingSpec,
        dataset: DatasetManifest,
        episodes: Iterable[Episode],
        artifacts: ArtifactWriter,
    ) -> ModelManifest:
        torch, nn, safetensors_torch = _torch_modules()
        config = spec.config
        _reject_unknown(
            config,
            frozenset(
                {
                    "observation_signals",
                    "seed",
                    "epochs",
                    "hidden_dim",
                    "learning_rate",
                    "device",
                    "max_transitions",
                }
            ),
            label="torch_bc trainer",
        )
        max_transitions = _integer(config, "max_transitions", 1_000_000)
        requested_signals = config.get(
            "observation_signals",
            [item.name for item in dataset.observation_schema],
        )
        if (
            not isinstance(requested_signals, list)
            or not requested_signals
            or not all(isinstance(item, str) for item in requested_signals)
        ):
            raise DomainValidationError("observation_signals must be a non-empty string list")
        observation_signals = tuple(item for item in requested_signals if isinstance(item, str))
        schema_by_name = {item.name: item for item in dataset.observation_schema}
        if set(observation_signals) - set(schema_by_name):
            raise DomainValidationError("training requests observations outside the dataset schema")

        features: list[list[float]] = []
        targets: list[list[float]] = []
        for episode in episodes:
            for step in episode.steps:
                if step.is_last:
                    continue
                row: list[float] = []
                for signal_name in observation_signals:
                    value = step.observations.get(signal_name)
                    if value is None or value.inline is None:
                        raise DomainValidationError(
                            f"Torch BC requires inline observation {signal_name!r}"
                        )
                    row.extend(float(item) for item in value.inline)
                if step.action is None or step.action.inline is None:
                    raise DomainValidationError("Torch BC requires inline actions")
                features.append(row)
                targets.append([float(item) for item in step.action.inline])
                if len(features) > max_transitions:
                    raise DomainValidationError(
                        "Torch BC reference trainer exceeded max_transitions; "
                        "use a streaming production trainer"
                    )
        if not features:
            raise DomainValidationError("dataset contains no trainable transitions")

        seed = _integer(config, "seed", 17, minimum=0)
        epochs = _integer(config, "epochs", 120)
        hidden_dim = _integer(config, "hidden_dim", 32)
        learning_rate = _float(config, "learning_rate", 0.01)
        selected_device = _device(torch, _string(config, "device", "auto"))
        if selected_device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(seed)
        if selected_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        inputs = torch.tensor(features, dtype=torch.float32, device=selected_device)
        expected = torch.tensor(targets, dtype=torch.float32, device=selected_device)
        model = _network(nn, inputs.shape[1], hidden_dim, expected.shape[1]).to(selected_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        final_loss = float("nan")
        model.train()
        previous_determinism = bool(torch.are_deterministic_algorithms_enabled())
        torch.use_deterministic_algorithms(True)
        try:
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                predicted = model(inputs)
                loss = loss_fn(predicted, expected)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu().item())
        finally:
            torch.use_deterministic_algorithms(previous_determinism)
        if not math.isfinite(final_loss):
            raise DomainValidationError("training produced a non-finite loss")

        cpu_state = {
            name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
        }
        weights = artifacts.put_bytes(
            safetensors_torch.save(cpu_state),
            media_type="application/vnd.safetensors",
        )
        input_dim = int(inputs.shape[1])
        output_dim = int(expected.shape[1])
        return ModelManifest(
            resource_id=spec.resource_id,
            version=spec.version,
            trainer=self.name,
            trainer_version=self.plugin_version,
            policy=TorchBehaviorCloningPolicy.name,
            policy_version=TorchBehaviorCloningPolicy.plugin_version,
            dataset=spec.dataset,
            weights=weights,
            observation_signals=tuple(schema_by_name[name] for name in observation_signals),
            action_signal=dataset.action_schema,
            training_config={
                "observation_signals": list(observation_signals),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "seed": seed,
                "max_transitions": max_transitions,
                "trained_device": selected_device.type,
            },
            metrics={"training_mse": final_loss, "transition_count": float(len(features))},
            created_at=dataset.created_at,
        )


class TorchBehaviorCloningPolicy:
    name = "torch_bc"
    plugin_version = "1.0.0"
    api_version = API_VERSION

    def __init__(
        self,
        config: Mapping[str, JSONValue],
        artifacts: ArtifactReader,
    ) -> None:
        torch, nn, safetensors_torch = _torch_modules()
        _reject_unknown(
            config,
            frozenset({"model_digest", "device"}),
            label="torch_bc policy",
        )
        digest = _string(config, "model_digest", "")
        manifest = ModelManifest.model_validate_json(artifacts.get_bytes(digest))
        training = manifest.training_config
        input_dim = _json_int(training, "input_dim")
        hidden_dim = _json_int(training, "hidden_dim")
        output_dim = _json_int(training, "output_dim")
        self._device = _device(torch, _string(config, "device", "auto"))
        self._torch = torch
        self._model = _network(nn, input_dim, hidden_dim, output_dim).to(self._device)
        state = safetensors_torch.load(artifacts.get_bytes(manifest.weights))
        self._model.load_state_dict(state, strict=True)
        self._model.eval()
        self._signals = tuple(item.name for item in manifest.observation_signals)
        self._closed = False

    def reset(self, seed: int) -> None:
        del seed
        if self._closed:
            raise InvalidStateTransitionError("policy is closed")

    def act(self, frame: Frame, task: TaskSpec, step_index: int) -> DataValue:
        del step_index
        if self._closed:
            raise InvalidStateTransitionError("policy is closed")
        values: list[float] = []
        timestamp: float | None = None
        for name in self._signals:
            signal = frame.observations.get(name)
            if signal is None or signal.inline is None:
                raise DomainValidationError(f"Torch BC policy requires inline signal {name!r}")
            values.extend(float(item) for item in signal.inline)
            timestamp = signal.timestamp_seconds if timestamp is None else timestamp
        with self._torch.inference_mode():
            tensor = self._torch.tensor([values], dtype=self._torch.float32, device=self._device)
            predicted = self._model(tensor)[0].detach().cpu().tolist()
        bounded = tuple(
            min(upper, max(lower, float(value)))
            for value, lower, upper in zip(
                predicted, task.action.lower, task.action.upper, strict=True
            )
        )
        return DataValue(
            dtype=task.action.signal.dtype,
            shape=task.action.signal.shape,
            unit=task.action.signal.unit,
            frame=task.action.signal.frame,
            timestamp_seconds=timestamp,
            inline=bounded,
        )

    def close(self) -> None:
        self._closed = True


def _json_int(mapping: Mapping[str, JSONValue], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DomainValidationError(f"model training_config {name!r} is invalid")
    return value


__all__ = ["TorchBehaviorCloningPolicy", "TorchBehaviorCloningTrainer"]
