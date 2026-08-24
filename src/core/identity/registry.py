"""Identity registry: names, roles, and face templates.

Storage layout (all local, no cloud):
    data/identities/registry.json   — metadata (name, roles, created_at)
    data/identities/templates.npz   — float32 embedding matrix

Privacy: biometric templates are stored on-device only. Enrollment is an
explicit operator action; deletion removes templates immediately. Operators
are responsible for GDPR/BIPA compliance before enabling recognition.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Identity:
    name: str
    roles: list[str]
    template_ids: list[int]  # rows into the template matrix
    created_at: float


class IdentityRegistry:
    def __init__(
        self, store_dir: str | Path = "data/identities", match_distance_threshold: float = 0.363
    ) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.store_dir / "registry.json"
        self.templates_path = self.store_dir / "templates.npz"
        self.threshold = float(match_distance_threshold)

        self.identities: dict[str, Identity] = {}
        self._templates = np.zeros((0, 128), dtype=np.float32)
        self._next_template_id = 0
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self.templates_path.exists():
            with np.load(self.templates_path) as data:
                self._templates = data["templates"].astype(np.float32)
                self._next_template_id = int(data["next_id"])
        if self.meta_path.exists():
            raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.identities = {
                name: Identity(
                    name=name,
                    roles=entry["roles"],
                    template_ids=entry["template_ids"],
                    created_at=entry["created_at"],
                )
                for name, entry in raw.items()
            }
        logger.info(
            "Identity registry: %d identities, %d templates",
            len(self.identities),
            len(self._templates),
        )

    def save(self) -> None:
        np.savez_compressed(
            self.templates_path,
            templates=self._templates,
            next_id=np.int64(self._next_template_id),
        )
        meta = {
            name: {
                "roles": ident.roles,
                "template_ids": ident.template_ids,
                "created_at": ident.created_at,
            }
            for name, ident in self.identities.items()
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Enrollment
    # ------------------------------------------------------------------ #
    def enroll(
        self, name: str, embeddings: list[np.ndarray], roles: list[str] | None = None
    ) -> Identity:
        """Add or update a person with one-or-more face embeddings."""
        name = name.strip()
        if not name:
            raise ValueError("Name required")
        if not embeddings:
            raise ValueError("At least one face embedding required")

        clean = []
        for emb in embeddings:
            v = np.asarray(emb, dtype=np.float32).flatten()
            norm = np.linalg.norm(v)
            if norm > 0:
                clean.append(v / norm)
        if not clean:
            raise ValueError("All embeddings were degenerate")

        ident = self.identities.get(name) or Identity(
            name=name, roles=[], template_ids=[], created_at=time.time()
        )
        ident.roles = sorted(set((roles or []) + ident.roles))

        new_rows = []
        for v in clean:
            tid = self._next_template_id
            self._next_template_id += 1
            ident.template_ids.append(tid)
            new_rows.append(v)

        self._templates = (
            np.vstack([self._templates, np.stack(new_rows)])
            if len(self._templates)
            else np.stack(new_rows)
        )
        self.identities[name] = ident
        self.save()
        logger.info(
            "Enrolled '%s' (%d templates total, roles=%s)",
            name,
            len(ident.template_ids),
            ident.roles,
        )
        return ident

    def remove(self, name: str) -> bool:
        ident = self.identities.pop(name, None)
        if ident is None:
            return False
        # Template ids are row indices; compact the matrix and remap survivors.
        drop = set(ident.template_ids)
        keep_rows = [i for i in range(len(self._templates)) if i not in drop]
        remap = {old: new for new, old in enumerate(keep_rows)}
        self._templates = (
            self._templates[keep_rows]
            if keep_rows
            else np.zeros((0, self._templates.shape[1]), dtype=np.float32)
        )
        for other in self.identities.values():
            other.template_ids = [remap[t] for t in other.template_ids if t in remap]
        self.save()
        logger.info("Removed identity '%s'", name)
        return True

    # ------------------------------------------------------------------ #
    # Matching
    # ------------------------------------------------------------------ #
    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Best identity above threshold; ``(None, distance)`` otherwise."""
        v = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        if len(self._templates) == 0:
            return None, 1.0

        distances = 1.0 - (self._templates @ v)
        best_row = int(np.argmin(distances))
        best_dist = float(distances[best_row])
        if best_dist > self.threshold:
            return None, best_dist

        for ident in self.identities.values():
            if best_row in ident.template_ids:
                return ident.name, best_dist
        return None, best_dist  # pragma: no cover - bookkeeping invariant

    # ------------------------------------------------------------------ #
    def get(self, name: str) -> Identity | None:
        return self.identities.get(name)

    def roles_of(self, name: str) -> list[str]:
        ident = self.identities.get(name)
        return list(ident.roles) if ident else []

    def list_identities(self) -> list[dict]:
        return [
            {
                "name": i.name,
                "roles": i.roles,
                "num_templates": len(i.template_ids),
                "created_at": i.created_at,
            }
            for i in self.identities.values()
        ]
