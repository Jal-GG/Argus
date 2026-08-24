"""Tests for SQLite analytics persistence (AnalyticsDB + AnalyticsWriter)."""

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.path.database import AnalyticsDB, AnalyticsWriter


@pytest.fixture()
def db(tmp_path):
    return AnalyticsDB(tmp_path / "analytics.db")


class TestAnalyticsDB:
    def test_schema_created_and_counts(self, db):
        assert db.counts() == {"persons": 0, "positions": 0, "events": 0}

    def test_position_roundtrip(self, db):
        writer = AnalyticsWriter(db, flush_interval_s=60)  # manual flush
        for i in range(10):
            writer.add_position(7, ts=100.0 + i, x=float(i), y=2.0, camera_id="c1")
        assert writer.flush() is True
        assert db.counts()["positions"] == 10
        assert db.counts()["persons"] == 1

        rows = db.positions_in_range(since=0)
        assert len(rows) == 10
        assert rows[3]["x"] == pytest.approx(3.0)

    def test_time_and_person_filters(self, db):
        w = AnalyticsWriter(db, flush_interval_s=60)
        for i in range(6):
            w.add_position(1, ts=i * 10, x=i, y=i, camera_id="c")
            w.add_position(2, ts=i * 10, x=-i, y=-i, camera_id="c")
        w.flush()

        recent = db.positions_in_range(since=25)
        assert {r["global_id"] for r in recent} == {1, 2}
        only_p2 = db.positions_in_range(since=0, global_id=2)
        assert {r["global_id"] for r in only_p2} == {2}

    def test_events_query_with_type_filter(self, db):
        db.insert_event(
            {
                "global_id": 5,
                "rule_type": "restricted_zone",
                "zone_name": "Vault",
                "timestamp": 50.0,
                "reason": "r1",
            }
        )
        db.insert_event(
            {
                "global_id": 5,
                "rule_type": "loitering",
                "zone_name": "Lobby",
                "timestamp": 60.0,
                "reason": "r2",
            }
        )

        all_events = db.events_in_range(since=0)
        assert len(all_events) == 2
        restricted = db.events_in_range(since=0, event_type="restricted_zone")
        assert len(restricted) == 1 and restricted[0]["event_type"] == "restricted_zone"

    def test_upsert_updates_last_seen(self, db):
        db.upsert_persons(
            [{"global_id": 9, "first_seen": 10.0, "last_seen": 20.0, "total_distance": 1.0}]
        )
        db.upsert_persons(
            [{"global_id": 9, "first_seen": 5.0, "last_seen": 99.0, "total_distance": 7.0}]
        )
        summary = db.person_summaries()
        assert len(summary) == 1
        assert summary[0]["last_seen"] == pytest.approx(99.0)
        assert summary[0]["total_distance"] == pytest.approx(7.0)


class TestAnalyticsWriter:
    def test_auto_flush_on_interval(self, db):
        writer = AnalyticsWriter(db, flush_interval_s=0.05)
        writer.add_position(3, ts=time.time(), x=1, y=1, camera_id="c")
        time.sleep(0.06)
        assert writer.maybe_flush() is True
        assert db.counts()["positions"] == 1
        # Second immediate call: nothing buffered -> no-op.
        assert writer.maybe_flush() is False

    def test_max_buffer_forces_flush(self, db):
        writer = AnalyticsWriter(db, flush_interval_s=999, max_buffer=5)
        for i in range(5):
            writer.add_position(1, ts=i, x=i, y=0, camera_id="c")
        assert db.counts()["positions"] >= 5  # flushed on buffer-full

    def test_close_flushes_remaining(self, db):
        writer = AnalyticsWriter(db, flush_interval_s=999)
        writer.add_position(4, ts=1.0, x=0, y=0, camera_id="c")
        writer.close()
        assert db.counts()["positions"] == 1
