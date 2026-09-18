from __future__ import annotations

import pytest

from physrealx.errors import MethodExecutionError
from physrealx.integrations.world_model import (
    VISUAL_RESIDUAL_GAINS,
    WorldModelConfig,
    _apply_visual_residual_gain,
    _calibrate_visual_residual,
    _quality_status,
    _validation_selection,
)


def test_validation_selection_balances_modalities_and_preserves_unit_scaling():
    state_focused = {
        "latent_mse": 2.0,
        "persistence_latent_mse": 1.0,
        "normalized_state_rmse": 0.1,
        "persistence_normalized_state_rmse": 1.0,
        "rgb_mse": 0.0001,
        "persistence_rgb_mse": 0.0002,
    }
    balanced = {**state_focused, "latent_mse": 0.8, "normalized_state_rmse": 0.6}
    assert _validation_selection([balanced])["score"] < 1
    assert _validation_selection([state_focused])["score"] > 1
    scaled = {
        key: value * (1000 if "latent" in key else 0.01 if "rgb" in key else 1)
        for key, value in balanced.items()
    }
    assert _validation_selection([scaled]) == pytest.approx(_validation_selection([balanced]))
    with pytest.raises(MethodExecutionError, match="finite"):
        _validation_selection([{**balanced, "latent_mse": float("nan")}])


def test_training_can_cover_the_requested_five_second_rollout():
    config = WorldModelConfig(training_horizon=100, evaluation_horizon=100)
    assert config.training_horizon == config.evaluation_horizon
    with pytest.raises(ValueError):
        WorldModelConfig(training_horizon=129)


def test_quality_status_distinguishes_improvement_tie_and_regression():
    assert _quality_status(0.5, 1.0) == "BEATS_PERSISTENCE"
    assert _quality_status(1.0, 1.0) == "MATCHES_PERSISTENCE"
    assert _quality_status(1.0 - 1e-10, 1.0) == "MATCHES_PERSISTENCE"
    assert _quality_status(1.1, 1.0) == "BELOW_BASELINE"


def test_visual_residual_gain_is_selected_only_from_validation(monkeypatch):
    torch = pytest.importorskip("torch")
    from physrealx.integrations import world_model as module

    predictor = torch.nn.ModuleDict({"visual_delta": torch.nn.Linear(1, 1)})
    predictor["visual_delta"].weight.data.fill_(2)
    predictor["visual_delta"].bias.data.fill_(1)

    def evaluate(predictor, decoder, rows, config, scale):
        del decoder, rows, config, scale
        gain = float(predictor["visual_delta"].weight.item() / 2)
        error = 0.2 + (gain - 0.375) ** 2
        return [
            {
                "latent_mse": error,
                "persistence_latent_mse": 1.0,
                "normalized_state_rmse": 0.1,
                "persistence_normalized_state_rmse": 1.0,
                "rgb_mse": error,
                "persistence_rgb_mse": 1.0,
            }
        ]

    monkeypatch.setattr(module, "_evaluate", evaluate)
    gain, errors, curve, selection = _calibrate_visual_residual(
        predictor,
        object(),
        [{}],
        WorldModelConfig(),
        object(),
    )
    assert gain == 0.375
    assert len(curve) == len(VISUAL_RESIDUAL_GAINS)
    assert selection["score"] == pytest.approx(0.2)
    assert errors[0]["latent_mse"] == pytest.approx(0.2)
    assert predictor["visual_delta"].weight.item() == pytest.approx(2)
    assert predictor["visual_delta"].bias.item() == pytest.approx(1)
    _apply_visual_residual_gain(predictor, gain)
    assert predictor["visual_delta"].weight.item() == pytest.approx(0.75)
    assert predictor["visual_delta"].bias.item() == pytest.approx(0.375)
