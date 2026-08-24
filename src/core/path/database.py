"""SQLite persistence for trajectories, zone visits, and events.

SQLAlchemy 2.0 ORM with SQLite (WAL) by default. The pipeline writes in
batches via :class:`AnalyticsWriter` so per-frame cost stays negligible;
the API queries read straight from the same file.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ...utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class PersonRow(Base):
    __tablename__ = "persons"

    global_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_seen: Mapped[float] = mapped_column(Float, index=True)
    last_seen: Mapped[float] = mapped_column(Float, index=True)
    total_distance: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="active")


class PositionRow(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_id: Mapped[int] = mapped_column(ForeignKey("persons.global_id"), index=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    camera_id: Mapped[str] = mapped_column(String(50))
    floor_id: Mapped[str] = mapped_column(String(50), default="floor_0")


class ZoneVisitRow(Base):
    __tablename__ = "zone_visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_id: Mapped[int] = mapped_column(ForeignKey("persons.global_id"), index=True)
    zone_name: Mapped[str] = mapped_column(String(100))
    enter_time: Mapped[float] = mapped_column(Float)
    exit_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    dwell_time: Mapped[float] = mapped_column(Float, default=0.0)


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.global_id"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    zone_name: Mapped[str] = mapped_column(String(100), default="")
    ts: Mapped[float] = mapped_column(Float, index=True)
    reason: Mapped[str] = mapped_column(String, default="")


class AnalyticsDB:
    """Owns the engine/session factory and all query helpers."""

    def __init__(self, db_path: str | Path = "data/output/analytics.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False, "timeout": 10},
        )
        # WAL lets API readers work while the pipeline writes.
        from sqlalchemy import event

        @event.listens_for(self.engine, "connect")
        def _set_wal(dbapi_conn, _record):  # pragma: no cover - driver hook
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        logger.info("Analytics DB ready at %s", self.db_path)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert_persons(self, rows: Iterable[dict[str, Any]]) -> None:
        with Session(self.engine) as session:
            for r in rows:
                existing = session.get(PersonRow, r["global_id"])
                if existing is None:
                    session.add(PersonRow(**r))
                else:
                    existing.last_seen = max(existing.last_seen, r["last_seen"])
                    existing.total_distance = max(
                        existing.total_distance, r.get("total_distance", 0.0)
                    )
                    existing.status = r.get("status", existing.status)
            session.commit()

    def insert_positions(self, rows: Sequence[dict[str, Any]]) -> int:
        if not rows:
            return 0
        with Session(self.engine) as session:
            session.add_all([PositionRow(**r) for r in rows])
            session.commit()
        return len(rows)

    def insert_zone_visit(
        self,
        global_id: int,
        zone_name: str,
        enter_time: float,
        exit_time: float | None,
        dwell_time: float,
    ) -> None:
        with Session(self.engine) as session:
            session.add(
                ZoneVisitRow(
                    global_id=global_id,
                    zone_name=zone_name,
                    enter_time=enter_time,
                    exit_time=exit_time,
                    dwell_time=dwell_time,
                )
            )
            session.commit()

    def insert_event(self, event_dict: dict[str, Any]) -> None:
        with Session(self.engine) as session:
            session.add(
                EventRow(
                    global_id=event_dict.get("global_id"),
                    event_type=event_dict["rule_type"],
                    zone_name=event_dict.get("zone_name", ""),
                    ts=float(event_dict.get("timestamp", time.time())),
                    reason=event_dict.get("reason", ""),
                )
            )
            session.commit()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def positions_in_range(
        self,
        since: float,
        until: float | None = None,
        global_id: int | None = None,
        limit: int = 50000,
    ) -> list[dict[str, Any]]:
        stmt = select(PositionRow).where(PositionRow.ts >= since).limit(limit)
        if until is not None:
            stmt = stmt.where(PositionRow.ts <= until)
        if global_id is not None:
            stmt = stmt.where(PositionRow.global_id == global_id)
        stmt = stmt.order_by(PositionRow.ts)
        with Session(self.engine) as session:
            return [
                {"global_id": p.global_id, "ts": p.ts, "x": p.x, "y": p.y, "camera_id": p.camera_id}
                for p in session.scalars(stmt)
            ]

    def events_in_range(
        self,
        since: float,
        until: float | None = None,
        event_type: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        stmt = select(EventRow).where(EventRow.ts >= since).limit(limit)
        if until is not None:
            stmt = stmt.where(EventRow.ts <= until)
        if event_type:
            stmt = stmt.where(EventRow.event_type == event_type)
        stmt = stmt.order_by(EventRow.ts.desc())
        with Session(self.engine) as session:
            return [
                {
                    "id": e.id,
                    "global_id": e.global_id,
                    "event_type": e.event_type,
                    "zone_name": e.zone_name,
                    "ts": e.ts,
                    "reason": e.reason,
                }
                for e in session.scalars(stmt)
            ]

    def heatmap_points(
        self, since: float, until: float | None = None, limit: int = 200000
    ) -> list[tuple[float, float]]:
        """Raw (x, y) samples for server-side heatmap aggregation."""
        stmt = select(PositionRow.x, PositionRow.y).where(PositionRow.ts >= since).limit(limit)
        if until is not None:
            stmt = stmt.where(PositionRow.ts <= until)
        with Session(self.engine) as session:
            return [(x, y) for x, y in session.execute(stmt)]

    def person_summaries(self, limit: int = 200) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(PersonRow).order_by(PersonRow.last_seen.desc()).limit(limit)
            ).all()
            return [
                {
                    "global_id": p.global_id,
                    "first_seen": p.first_seen,
                    "last_seen": p.last_seen,
                    "total_distance": round(p.total_distance, 2),
                    "status": p.status,
                }
                for p in rows
            ]

    def counts(self) -> dict[str, int]:
        with Session(self.engine) as session:
            return {
                "persons": session.scalar(select(func.count()).select_from(PersonRow)) or 0,
                "positions": session.scalar(select(func.count()).select_from(PositionRow)) or 0,
                "events": session.scalar(select(func.count()).select_from(EventRow)) or 0,
            }


class AnalyticsWriter:
    """Buffers pipeline output and flushes to SQLite on an interval."""

    def __init__(
        self, db: AnalyticsDB, flush_interval_s: float = 2.0, max_buffer: int = 2000
    ) -> None:
        self.db = db
        self.flush_interval_s = float(flush_interval_s)
        self.max_buffer = int(max_buffer)
        self._position_buffer: list[dict[str, Any]] = []
        self._last_flush = time.time()
        self.flushes = 0

    def add_position(
        self,
        global_id: int,
        ts: float,
        x: float,
        y: float,
        camera_id: str,
        floor_id: str = "floor_0",
    ) -> None:
        self._position_buffer.append(
            {
                "global_id": global_id,
                "ts": float(ts),
                "x": float(x),
                "y": float(y),
                "camera_id": str(camera_id),
                "floor_id": str(floor_id),
            }
        )
        if len(self._position_buffer) >= self.max_buffer:
            self.flush()

    def maybe_flush(self) -> bool:
        if time.time() - self._last_flush >= self.flush_interval_s and self._position_buffer:
            return self.flush()
        return False

    def flush(self) -> bool:
        buffer, self._position_buffer = self._position_buffer, []
        self._last_flush = time.time()
        if not buffer:
            return False
        try:
            gids = {r["global_id"] for r in buffer}
            now = max(r["ts"] for r in buffer)
            earliest = min(r["ts"] for r in buffer)
            self.db.insert_positions(buffer)
            self.db.upsert_persons(
                [
                    {
                        "global_id": gid,
                        "first_seen": earliest,
                        "last_seen": now,
                        "total_distance": 0.0,
                        "status": "active",
                    }
                    for gid in gids
                ]
            )
            self.flushes += 1
            return True
        except Exception as exc:  # never kill tracking over DB hiccups
            logger.error("Analytics flush failed: %s", exc)
            return False

    def close(self) -> None:
        self.flush()
