from __future__ import annotations

import pytest


@pytest.mark.gpu
@pytest.mark.integration
def test_pytorch_cu121_executes_small_matmul() -> None:
    """Validate the pinned CUDA runtime without placing meaningful load on a busy GPU."""
    torch = pytest.importorskip("torch", reason="PyTorch is installed by environment.yml")

    assert torch.version.cuda == "12.1", (
        f"Expected a cu121 PyTorch build, received torch.version.cuda={torch.version.cuda!r}"
    )
    if not torch.cuda.is_available():
        pytest.skip("A CUDA-capable GPU is not available to this process")

    device = torch.device("cuda", 0)
    left_cpu = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 256
    right_cpu = torch.eye(16, dtype=torch.float32)

    with torch.inference_mode():
        actual = (left_cpu.to(device) @ right_cpu.to(device)).cpu()

    assert actual.isfinite().all()
    torch.testing.assert_close(actual, left_cpu, rtol=1e-5, atol=1e-6)
