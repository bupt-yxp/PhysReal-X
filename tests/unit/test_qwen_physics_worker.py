import pytest

from physrealx.integrations._qwen_physics_worker import _inference_memory_budget

GIB = 1024**3


def test_auto_placement_reserves_runtime_and_offload_space():
    available = {0: 14 * GIB, 1: 8 * GIB, "cpu": 100 * GIB}
    budget = _inference_memory_budget(available)
    assert available[0] - budget[0] >= 4 * GIB
    assert available[1] - budget[1] >= 2 * GIB
    assert budget["cpu"] == 90 * GIB
    assert available == {0: 14 * GIB, 1: 8 * GIB, "cpu": 100 * GIB}


def test_busy_gpu_is_skipped_and_next_gpu_receives_primary_reserve():
    budget = _inference_memory_budget({0: GIB, 1: 8 * GIB, "cpu": 32 * GIB})
    assert 0 not in budget
    assert budget[1] == 4 * GIB
    assert "cpu" in budget


@pytest.mark.parametrize("gpu", [{}, {0: GIB, 1: 2 * GIB}])
def test_cpu_fallback_keeps_host_workspace(gpu):
    assert _inference_memory_budget({**gpu, "cpu": 16 * GIB}) == {"cpu": 14 * GIB}


def test_exhausted_memory_is_rejected_before_model_loading():
    with pytest.raises(MemoryError, match="workspace"):
        _inference_memory_budget({0: GIB, "cpu": GIB})
