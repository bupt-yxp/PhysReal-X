from __future__ import annotations

import pytest

from physrealx.errors import DomainValidationError
from physrealx.integrations.sensors import EventAccumulator, project_depth

np = pytest.importorskip("numpy")


def test_metric_projection_roundtrips_pixels_and_preserves_missing_depth():
    depth = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    camera = np.array([[2, 0, 1], [0, 3, 0.5], [0, 0, 1]])
    points = project_depth(depth, camera)
    valid = depth > 0
    uvz = points[valid] @ camera.T
    y, x = np.indices(depth.shape)
    np.testing.assert_allclose(uvz[:, :2] / uvz[:, 2:], np.c_[x[valid], y[valid]], atol=1e-6)
    np.testing.assert_array_equal(points[~valid], 0)


@pytest.mark.parametrize("threshold", [0, -0.1, float("nan"), float("inf")])
def test_events_reject_invalid_thresholds(threshold):
    with pytest.raises(DomainValidationError):
        EventAccumulator(threshold)


def test_events_accumulate_subthreshold_changes_and_reset_without_a_false_burst():
    events = EventAccumulator(0.2)

    def frame(level):
        return np.full((2, 3, 3), level, np.uint8)

    assert not events.update(frame(100)).any()
    assert not events.update(frame(110)).any()
    positive = events.update(frame(130))
    assert np.all(positive[..., 0] == 1) and not positive[..., 1].any()
    negative = events.update(frame(80))
    assert np.all(negative[..., 1] == 2) and not negative[..., 0].any()
    with pytest.raises(DomainValidationError, match="resolution"):
        events.update(np.ones((3, 3, 3), np.uint8))
    events.reset()
    assert not events.update(frame(255)).any()


@pytest.mark.parametrize("depth", [np.array([1]), np.array([[-1]]), np.array([[np.nan]])])
def test_projection_rejects_invalid_depth(depth):
    with pytest.raises(DomainValidationError):
        project_depth(depth, np.eye(3))
