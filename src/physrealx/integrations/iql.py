"""Offline Implicit Q-Learning for continuous PhysReal-X action spaces."""

from __future__ import annotations

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
from physrealx.errors import DomainValidationError, InvalidStateTransitionError
from physrealx.integrations.provenance import capture_runtime
from physrealx.methods import builtin_method

SOURCE_REVISION = "09d700248117881a75cb21f0adb95c6c8a694cb2"
SOURCE_URI = "https://github.com/ikostrikov/implicit_q_learning"


class IQLTrainingConfig(DomainModel):
    observation_signals: tuple[str, ...] = ("joint.position", "joint.velocity")
    steps: int = Field(default=500, ge=1, le=1_000_000)
    batch_size: int = Field(default=128, ge=1, le=4096)
    hidden_dim: int = Field(default=128, ge=16, le=2048)
    learning_rate: float = Field(default=3e-4, gt=0, le=1)
    discount: float = Field(default=0.99, ge=0, le=1)
    expectile: float = Field(default=0.7, gt=0, lt=1)
    temperature: float = Field(default=3.0, gt=0, le=100)
    max_weight: float = Field(default=100.0, ge=1, le=1e6)
    target_update_rate: float = Field(default=0.005, gt=0, le=1)
    seed: int = Field(default=2026, ge=0)
    device: Literal["cpu", "cuda"] = "cpu"
    threads: int = Field(default=4, ge=1, le=64)
    max_transitions: int = Field(default=100_000, ge=1, le=1_000_000)


class IQLArchitecture(DomainModel):
    state_dim: int = Field(gt=0)
    action_dim: int = Field(gt=0)
    hidden_dim: int = Field(ge=16)


class IQLStatistics(DomainModel):
    state_mean: tuple[float, ...]
    state_scale: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_scale: tuple[float, ...]

    @model_validator(mode="after")
    def validate_dimensions(self) -> IQLStatistics:
        if not self.state_mean or len(self.state_mean) != len(self.state_scale):
            raise ValueError("IQL state statistics have inconsistent dimensions")
        if not self.action_mean or len(self.action_mean) != len(self.action_scale):
            raise ValueError("IQL action statistics have inconsistent dimensions")
        if any(value <= 0 for value in (*self.state_scale, *self.action_scale)):
            raise ValueError("IQL normalization scales must be positive")
        return self


def _network(nn: Any, input_dim: int, hidden_dim: int, output_dim: int) -> Any:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _state(values: Mapping[str, DataValue], signals: tuple[str, ...]) -> list[float]:
    result: list[float] = []
    for name in signals:
        value = values.get(name)
        if value is None or value.inline is None:
            raise DomainValidationError(f"IQL requires inline observation {name!r}")
        result.extend(float(item) for item in value.inline)
    return result


class IQLTrainer:
    descriptor = builtin_method(
        MethodCapability.POLICY_TRAIN,
        "iql",
        input_contract="TrainingSpec/1+DatasetManifest/1+Episode/1",
        output_contract="ModelManifest/1",
        python=">=3.10,<3.13",
    ).model_copy(
        update={
            "source_uri": SOURCE_URI,
            "source_revision": SOURCE_REVISION,
            "license_spdx": "MIT",
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

        config = IQLTrainingConfig.model_validate(spec.trainer.config)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise DomainValidationError("CUDA IQL training requested without an available device")
        schema = {signal.name: signal for signal in dataset.observation_schema}
        if len(set(config.observation_signals)) != len(config.observation_signals) or any(
            name not in schema for name in config.observation_signals
        ):
            raise DomainValidationError("IQL signals must uniquely match the dataset schema")

        states: list[list[float]] = []
        next_states: list[list[float]] = []
        actions: list[list[float]] = []
        rewards: list[float] = []
        discounts: list[float] = []
        for episode in episodes:
            for index, step in enumerate(episode.steps[:-1]):
                if step.action is None or step.action.inline is None:
                    raise DomainValidationError("IQL requires inline continuous actions")
                if step.reward is None or step.discount is None:
                    raise DomainValidationError(
                        "IQL requires reward and discount on every transition"
                    )
                states.append(_state(step.observations, config.observation_signals))
                next_states.append(
                    _state(episode.steps[index + 1].observations, config.observation_signals)
                )
                actions.append([float(item) for item in step.action.inline])
                rewards.append(float(step.reward))
                discounts.append(float(step.discount))
                if len(states) > config.max_transitions:
                    raise DomainValidationError("IQL training exceeds max_transitions")
        if not states:
            raise DomainValidationError("IQL dataset has no transitions")

        state_array = np.asarray(states, dtype=np.float32)
        next_state_array = np.asarray(next_states, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.float32)
        state_mean = state_array.mean(axis=0)
        state_scale = np.maximum(state_array.std(axis=0), 1e-4)
        action_mean = action_array.mean(axis=0)
        action_scale = np.maximum(action_array.std(axis=0), 1e-4)
        architecture = IQLArchitecture(
            state_dim=state_array.shape[1],
            action_dim=action_array.shape[1],
            hidden_dim=config.hidden_dim,
        )
        statistics = IQLStatistics(
            state_mean=tuple(float(value) for value in state_mean),
            state_scale=tuple(float(value) for value in state_scale),
            action_mean=tuple(float(value) for value in action_mean),
            action_scale=tuple(float(value) for value in action_scale),
        )
        source, environment = capture_runtime(self.store)
        started_at = datetime.now(timezone.utc)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(config.threads)
        try:
            with torch.random.fork_rng(
                devices=[] if config.device == "cpu" else [torch.cuda.current_device()]
            ):
                torch.manual_seed(config.seed)
                device = torch.device(config.device)
                state = torch.from_numpy((state_array - state_mean) / state_scale).to(device)
                next_state = torch.from_numpy((next_state_array - state_mean) / state_scale).to(
                    device
                )
                action = torch.from_numpy((action_array - action_mean) / action_scale).to(device)
                reward = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
                discount = torch.tensor(discounts, dtype=torch.float32, device=device).unsqueeze(-1)
                actor = _network(
                    torch.nn,
                    architecture.state_dim,
                    architecture.hidden_dim,
                    architecture.action_dim,
                ).to(device)
                value = _network(torch.nn, architecture.state_dim, architecture.hidden_dim, 1).to(
                    device
                )
                q1 = _network(
                    torch.nn,
                    architecture.state_dim + architecture.action_dim,
                    architecture.hidden_dim,
                    1,
                ).to(device)
                q2 = _network(
                    torch.nn,
                    architecture.state_dim + architecture.action_dim,
                    architecture.hidden_dim,
                    1,
                ).to(device)
                target_q1 = _network(
                    torch.nn,
                    architecture.state_dim + architecture.action_dim,
                    architecture.hidden_dim,
                    1,
                ).to(device)
                target_q2 = _network(
                    torch.nn,
                    architecture.state_dim + architecture.action_dim,
                    architecture.hidden_dim,
                    1,
                ).to(device)
                target_q1.load_state_dict(q1.state_dict())
                target_q2.load_state_dict(q2.state_dict())
                target_q1.requires_grad_(False)
                target_q2.requires_grad_(False)
                actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate)
                value_optimizer = torch.optim.AdamW(value.parameters(), lr=config.learning_rate)
                q_optimizer = torch.optim.AdamW(
                    [*q1.parameters(), *q2.parameters()], lr=config.learning_rate
                )
                actor_losses: list[float] = []
                value_losses: list[float] = []
                q_losses: list[float] = []
                count = len(states)
                for _ in range(config.steps):
                    indices = torch.randint(count, (min(config.batch_size, count),), device=device)
                    s, ns, a = state[indices], next_state[indices], action[indices]
                    r, d = reward[indices], discount[indices]
                    sa = torch.cat((s, a), dim=-1)

                    with torch.no_grad():
                        target_q = torch.minimum(target_q1(sa), target_q2(sa))
                    value_error = target_q - value(s)
                    expectile_weight = torch.where(
                        value_error > 0, config.expectile, 1 - config.expectile
                    )
                    value_loss = (expectile_weight * value_error.square()).mean()
                    value_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        value.parameters(), 10.0, error_if_nonfinite=True
                    )
                    value_optimizer.step()

                    with torch.no_grad():
                        q_target = r + config.discount * d * value(ns)
                    q1_prediction, q2_prediction = q1(sa), q2(sa)
                    q_loss = torch.nn.functional.mse_loss(
                        q1_prediction, q_target
                    ) + torch.nn.functional.mse_loss(q2_prediction, q_target)
                    q_optimizer.zero_grad(set_to_none=True)
                    q_loss.backward()  # type: ignore[no-untyped-call]
                    torch.nn.utils.clip_grad_norm_(
                        [*q1.parameters(), *q2.parameters()], 10.0, error_if_nonfinite=True
                    )
                    q_optimizer.step()

                    with torch.no_grad():
                        advantage = torch.minimum(target_q1(sa), target_q2(sa)) - value(s)
                        actor_weight = torch.exp(config.temperature * advantage).clamp(
                            max=config.max_weight
                        )
                    actor_error = (actor(s) - a).square().mean(dim=-1, keepdim=True)
                    actor_loss = (actor_weight * actor_error).mean()
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), 10.0, error_if_nonfinite=True
                    )
                    actor_optimizer.step()
                    with torch.no_grad():
                        for target, current in ((target_q1, q1), (target_q2, q2)):
                            for target_parameter, parameter in zip(
                                target.parameters(), current.parameters()
                            ):
                                target_parameter.lerp_(parameter, config.target_update_rate)
                    losses = (actor_loss, value_loss, q_loss)
                    if not all(bool(torch.isfinite(item).item()) for item in losses):
                        raise DomainValidationError("IQL training produced a non-finite loss")
                    actor_losses.append(float(actor_loss.detach().cpu().item()))
                    value_losses.append(float(value_loss.detach().cpu().item()))
                    q_losses.append(float(q_loss.detach().cpu().item()))

                actor.eval()
                with torch.inference_mode():
                    normalized_prediction = actor(state)
                    predicted = normalized_prediction * torch.tensor(
                        action_scale, device=device
                    ) + torch.tensor(action_mean, device=device)
                    action_mae = float(
                        (predicted - torch.from_numpy(action_array).to(device))
                        .abs()
                        .mean()
                        .cpu()
                        .item()
                    )
                    state_action = torch.cat((state, action), dim=-1)
                    final_advantage = torch.minimum(
                        target_q1(state_action), target_q2(state_action)
                    ) - value(state)
                    mean_advantage = float(final_advantage.mean().cpu().item())
                tensors = {}
                for prefix, module in (("actor", actor), ("value", value), ("q1", q1), ("q2", q2)):
                    tensors.update(
                        {
                            f"{prefix}.{name}": tensor.detach().cpu().contiguous()
                            for name, tensor in module.state_dict().items()
                        }
                    )
                weights = artifacts.put_bytes(
                    save(tensors), media_type="application/vnd.safetensors"
                )
        finally:
            torch.set_num_threads(previous_threads)

        return ModelManifest(
            resource_id=spec.resource_id,
            version=spec.version,
            trainer=spec.trainer,
            policy=IQLPolicy.descriptor.bind(checkpoint=weights),
            dataset=spec.dataset,
            weights=weights,
            observation_signals=tuple(schema[name] for name in config.observation_signals),
            action_signal=dataset.action_schema,
            training_config={
                "configuration": config.model_dump(mode="json"),
                "architecture": architecture.model_dump(mode="json"),
                **statistics.model_dump(mode="json"),
                "loss_curve": actor_losses,
                "value_loss_curve": value_losses,
                "q_loss_curve": q_losses,
                "source_revision": SOURCE_REVISION,
                "adaptation": "PyTorch continuous-action IQL over PhysReal-X Episode transitions",
                "code": source.model_dump(mode="json"),
                "environment": environment.model_dump(mode="json"),
                "started_at": started_at.isoformat(),
            },
            metrics={
                "initial_loss": actor_losses[0],
                "final_loss": actor_losses[-1],
                "training_action_mae": action_mae,
                "mean_advantage": mean_advantage,
                "transitions": float(len(states)),
                "final_value_loss": value_losses[-1],
                "final_q_loss": q_losses[-1],
            },
            created_at=datetime.now(timezone.utc),
        )


class IQLPolicyConfig(DomainModel):
    model_artifact_id: str
    device: Literal["cpu", "cuda"] = "cpu"


class IQLPolicy:
    descriptor = builtin_method(
        MethodCapability.POLICY_INFERENCE,
        "iql",
        input_contract="Frame/1+TaskSpec/1+ModelManifest/1",
        output_contract="DataValue/1",
        python=">=3.10,<3.13",
    ).model_copy(
        update={
            "source_uri": SOURCE_URI,
            "source_revision": SOURCE_REVISION,
            "license_spdx": "MIT",
        }
    )

    def __init__(self, config: Mapping[str, JSONValue], store: ArtifactStore) -> None:
        import torch
        from safetensors.torch import load

        self.config = IQLPolicyConfig.model_validate(config)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise DomainValidationError("CUDA IQL inference requested without an available device")
        manifest = ModelManifest.model_validate_json(store.get_bytes(self.config.model_artifact_id))
        if manifest.policy.descriptor != self.descriptor:
            raise DomainValidationError("IQL policy source identity does not match the model")
        architecture = IQLArchitecture.model_validate(manifest.training_config["architecture"])
        statistics = IQLStatistics.model_validate(
            {
                name: manifest.training_config[name]
                for name in ("state_mean", "state_scale", "action_mean", "action_scale")
            }
        )
        if (
            len(statistics.state_mean) != architecture.state_dim
            or len(statistics.action_mean) != architecture.action_dim
        ):
            raise DomainValidationError("IQL statistics differ from the saved architecture")
        self._signals = tuple(item.name for item in manifest.observation_signals)
        self._action_signal = manifest.action_signal
        self._torch = torch
        self._device = torch.device(self.config.device)
        self._actor = _network(
            torch.nn,
            architecture.state_dim,
            architecture.hidden_dim,
            architecture.action_dim,
        ).to(self._device)
        checkpoint = load(store.get_bytes(manifest.weights))
        actor_state = {
            name[len("actor.") :]: tensor
            for name, tensor in checkpoint.items()
            if name.startswith("actor.")
        }
        self._actor.load_state_dict(actor_state, strict=True)
        self._actor.eval()
        self._statistics = {
            name: torch.tensor(getattr(statistics, name), dtype=torch.float32, device=self._device)
            for name in ("state_mean", "state_scale", "action_mean", "action_scale")
        }
        self._closed = False

    def reset(self, seed: int) -> None:
        del seed
        if self._closed:
            raise InvalidStateTransitionError("IQL policy is closed")

    def act(self, frame: Frame, task: TaskSpec, step_index: int) -> DataValue:
        del step_index
        if self._closed:
            raise InvalidStateTransitionError("IQL policy is closed")
        values = _state(frame.observations, self._signals)
        timestamp = next(
            (
                frame.observations[name].timestamp_seconds
                for name in self._signals
                if frame.observations[name].timestamp_seconds is not None
            ),
            None,
        )
        with self._torch.inference_mode():
            state = self._torch.tensor([values], dtype=self._torch.float32, device=self._device)
            state = (state - self._statistics["state_mean"]) / self._statistics["state_scale"]
            predicted = (
                self._actor(state)[0] * self._statistics["action_scale"]
                + self._statistics["action_mean"]
            ).cpu()
        bounded = tuple(
            min(upper, max(lower, float(value)))
            for value, lower, upper in zip(predicted, task.action.lower, task.action.upper)
        )
        return DataValue(
            dtype=self._action_signal.dtype,
            shape=self._action_signal.shape,
            unit=self._action_signal.unit,
            frame=self._action_signal.frame,
            timestamp_seconds=timestamp,
            inline=bounded,
        )

    def close(self) -> None:
        self._closed = True


__all__ = ["IQLPolicy", "IQLStatistics", "IQLTrainer", "IQLTrainingConfig"]
