"""Tests for Kalman filter + BYTE tracker association."""

import pytest

from src.core.tracking.byte_tracker import BYTETracker
from src.core.tracking.kalman_filter import KalmanFilterXYAH
from src.core.types import Detection


def _kf():
    return KalmanFilterXYAH()


class TestKalman:
    def test_bbox_roundtrip(self):
        _kf()
        mean = KalmanFilterXYAH.from_bbox((0, 0, 40, 100))
        x1, y1, x2, y2 = KalmanFilterXYAH.to_bbox(mean)
        assert (x1, y1, x2, y2) == pytest.approx((0.0, 0.0, 40.0, 100.0))

    def test_constant_velocity_prediction(self):
        """A box moving right by 10px/frame should be predicted accordingly."""
        kf = _kf()
        mean = None
        for frame in range(10):
            meas = KalmanFilterXYAH.from_bbox((10 * frame, 0, 40 + 10 * frame, 100))
            if mean is None:
                mean, cov = kf.initiate(meas)
            else:
                pred_mean, pred_cov = kf.predict(mean, cov)
                mean, cov = kf.update(pred_mean, pred_cov, meas)

        pred_mean, _ = kf.predict(mean, cov)
        x1, _, x2, _ = KalmanFilterXYAH.to_bbox(pred_mean)
        # True next box is x∈[100,140]. Kalman velocity estimate lags the true
        # velocity (standard for BYTE noise settings) — accept ±18 px.
        assert x1 == pytest.approx(100, abs=18)
        assert x2 == pytest.approx(140, abs=18)

    def test_update_pulls_toward_measurement(self):
        kf = _kf()
        mean, cov = kf.initiate(KalmanFilterXYAH.from_bbox((0, 0, 50, 100)))
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, KalmanFilterXYAH.from_bbox((30, 30, 80, 130)))
        cx, cy = float(mean[0]), float(mean[1])
        # Posterior must move decisively toward the measurement (cx: 25→55).
        assert cx > 35 and cx < 60
        assert cy > 55 and cy < 82


def _det(x: float, conf: float = 0.9) -> Detection:
    return Detection(bbox=(x, 100.0, x + 40.0, 200.0), confidence=conf)


class TestByteTracker:
    def test_single_object_keeps_id(self):
        trk = BYTETracker("cam_001", n_init=1)
        ids_per_frame = []
        for f in range(20):
            dets = [_det(10.0 + 2.0 * f)]
            out = trk.update(dets, timestamp=f / 25.0)
            ids_per_frame.append([t["track_id"] for t in out])

        confirmed_ids = {i for ids in ids_per_frame[5:] for i in ids}
        assert len(confirmed_ids) == 1, f"expected stable ID, got {ids_per_frame}"

    def test_two_objects_distinct_ids(self):
        trk = BYTETracker("cam_001", n_init=1)
        for f in range(25):
            trk.update([_det(50.0), _det(500.0)], timestamp=f / 25.0)
        # Same overlapping positions on the final frame -> both stay matched.
        final_ids = {
            t["track_id"] for t in trk.update([_det(52.0), _det(502.0)], timestamp=26 / 25.0)
        }
        assert len(final_ids) == 2

    def test_low_conf_rescue_no_new_track(self):
        """Occlusion dip to low confidence should NOT spawn a new track."""
        trk = BYTETracker("cam_001", n_init=1, new_track_thresh=0.6)
        seen_ids = set()
        for f in range(15):
            conf = 0.9 if f != 8 else 0.3  # one low-conf frame
            out = trk.update([_det(60.0 + f, conf)], timestamp=f / 25.0)
            seen_ids |= {t["track_id"] for t in out}
        assert len(seen_ids) == 1

    def test_disappearance_then_reentry_new_id_without_features(self):
        trk = BYTETracker("cam_001", n_init=1, max_age=5)
        first = None
        for f in range(10):
            out = trk.update([_det(100.0)], timestamp=f / 25.0)
            if out:
                first = out[0]["track_id"]
        # long gap > max_age
        for f in range(40):
            trk.update([], timestamp=(10 + f) / 25.0)
        # Re-entry: new tentative track confirms on its second match.
        out = trk.update([_det(100.0)], timestamp=60 / 25.0)
        out2 = trk.update([_det(100.0)], timestamp=61 / 25.0)
        assert out2 and out2[0]["track_id"] != first
