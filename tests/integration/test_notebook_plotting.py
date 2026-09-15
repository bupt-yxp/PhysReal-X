"""Exercise the actual inline backend, independently of pytest's plotting state."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_inline_backend_creates_and_encodes_a_figure() -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("matplotlib_inline")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib
import numpy as np
matplotlib.use('module://matplotlib_inline.backend_inline')
from matplotlib import pyplot as plt
from examples._display import show_rgbd, show_image_strip
figure, axis = plt.subplots()
axis.plot([0, 1], [0, 1])
image = io.BytesIO()
figure.savefig(image, format='png')
assert image.getvalue().startswith(b'\\x89PNG\\r\\n\\x1a\\n')
assert len(image.getvalue()) > 1000
with TemporaryDirectory() as directory:
    destination = Path(directory)
    before = plt.get_fignums()
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    depth = np.ones((8, 12), dtype=float)
    depth[0, :2] = [0, np.nan]
    show_rgbd(rgb, depth, destination / 'depth.png', mask=np.ones((8, 12), dtype=np.uint8))
    show_image_strip({'closed': rgb, 'open': rgb}, destination / 'motion.png')
    assert plt.get_fignums() == before
    assert all((destination / name).stat().st_size > 1000 for name in ('depth.png', 'motion.png'))
plt.close(figure)
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
