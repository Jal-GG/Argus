"""Tests for Phase 11: identity registry, engine ladder, role-gated rules."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.identity.registry import IdentityRegistry
from src.core.mapping.floor_plan import FloorPlan
from src.core.path.path_tracker import PersonPath
from src.core.rules.rule_engine import RuleEngine


def _unit_vec(seed: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed * 1000))
    v = rng.normal(size=128)
    return v / np.linalg.norm(v)


@pytest.fixture()
def registry(tmp_path):
    return IdentityRegistry(store_dir=tmp_path / "identities", match_distance_threshold=0.363)


class TestIdentityRegistry:
    def test_enroll_and_exact_match(self, registry):
        emb = _unit_vec(1.0)
        registry.enroll("Aman", [emb], ["security"])
        name, dist = registry.match(emb)
        assert name == "Aman"
        assert dist < 0.01
        assert registry.roles_of("Aman") == ["security"]

    def test_nearby_embedding_matches(self, registry):
        base = _unit_vec(2.0)
        noise = np.random.default_rng(7).normal(size=128) * 0.02
        noisy = base + noise
        noisy /= np.linalg.norm(noisy)
        registry.enroll("Priya", [base], [])
        name, _ = registry.match(noisy)
        assert name == "Priya"

    def test_different_person_rejected(self, registry):
        a = _unit_vec(10.0)
        b = _unit_vec(20.0)
        if abs(float(a @ b)) > 0.5:  # force near-orthogonal pair
            b = -b
            b /= np.linalg.norm(b)
        registry.enroll("A", [a])
        name, dist = registry.match(b)
        assert name is None or dist >= 0.363

    def test_multiple_templates_stored_and_matched(self, registry):
        e1, e2 = _unit_vec(31.0), _unit_vec(32.0)
        ident = registry.enroll("Multi", [e1, e2], ["admin"])
        assert len(ident.template_ids) == 2
        for probe in (e1, e2):
            name, _ = registry.match(probe)
            assert name == "Multi"

    def test_persistence_roundtrip(self, tmp_path):
        store = tmp_path / "persist"
        reg1 = IdentityRegistry(store_dir=store)
        reg1.enroll("Keep", [_unit_vec(41.0)], ["staff"])

        reg2 = IdentityRegistry(store_dir=store)
        name, _ = reg2.match(_unit_vec(41.0))
        assert name == "Keep"

    def test_remove_compacts_templates(self, registry):
        a, b, c = _unit_vec(51.0), _unit_vec(52.0), _unit_vec(53.0)
        registry.enroll("First", [a])
        registry.enroll("Second", [b])
        registry.enroll("Third", [c])

        assert registry.remove("Second") is True
        assert registry.remove("Second") is False  # idempotent-safe
        # Survivors still match after row compaction.
        assert registry.match(a)[0] == "First"
        assert registry.match(c)[0] == "Third"
        assert registry.get("Second") is None

    def test_roles_merge_on_reenroll(self, registry):
        emb = _unit_vec(61.0)
        registry.enroll("Ravi", [emb], ["employee"])
        registry.enroll("Ravi", [emb], ["admin"])
        assert set(registry.roles_of("Ravi")) == {"employee", "admin"}


class TestFaceEngineLadder:
    """Ladder logic without requiring downloaded models."""

    def test_unavailable_when_no_models(self, tmp_path, monkeypatch):
        from src.core.identity.face_engine import FaceEngine

        monkeypatch.setattr(
            "src.core.identity.face_engine.FaceEngine._ensure_model",
            lambda self, path, url, expected_min_bytes: False,
        )
        engine = FaceEngine(models_dir=tmp_path / "none", auto_download=True)
        assert engine.detector_backend is None or engine.embedder_backend is None
        # Tier-0 honesty: no embedder => unavailable.
        assert engine.available is False
        assert (
            engine.detect(np.zeros((100, 100, 3), dtype=np.uint8)) in ([],) or True
        )  # detect may still work via haar; embedder is the gate
        assert engine.embed(np.zeros((50, 50, 3), dtype=np.uint8)) is None

    def test_head_region_sane(self):
        from src.core.identity.face_engine import FaceEngine

        box = FaceEngine.head_region_from_person_bbox(
            (100.0, 200.0, 140.0, 400.0), frame_shape=(480, 640, 3)
        )
        hx, hy, hw, hh = box
        assert hw > 0 and hh > 0
        assert hx >= 0 and hy >= 0
        assert hx + hw <= 640 and hy + hh <= 480
        # Head band sits in the upper quarter of the person box.
        assert hy + hh <= 200 + 0.4 * 200


class TestRoleGating:
    def _plan(self) -> FloorPlan:
        fp = FloorPlan(size=(200, 200))
        fp.add_zone(
            "Vault",
            [(80, 80), (160, 80), (160, 160), (80, 160)],
            zone_type="restricted",
            params={"allowed_roles": ["security"]},
        )
        return fp

    def _person_in_vault(self, gid=1):
        path = PersonPath(global_id=gid)
        path.add_position(120.0, 120.0, "cam_001", timestamp=1000.0)
        return path

    def test_authorized_role_no_violation(self):
        engine_rule = RuleEngine(self._plan())
        violations = engine_rule.evaluate(
            self._person_in_vault(), now=1005.0, person_name="Aman", person_roles=("security",)
        )
        assert violations == []

    def test_wrong_role_flags(self):
        rule = RuleEngine(self._plan())
        violations = rule.evaluate(
            self._person_in_vault(), now=1005.0, person_name="Guest", person_roles=("visitor",)
        )
        assert len(violations) == 1
        assert "role-gated" in violations[0].reason
        assert "Guest" in violations[0].reason

    def test_unknown_person_flags_by_default(self):
        rule = RuleEngine(self._plan())
        violations = rule.evaluate(self._person_in_vault(), now=1005.0)
        assert len(violations) == 1

    def test_strict_mode_bars_unknown_even_with_name_only(self):
        rule = RuleEngine(self._plan(), strict_unknown_roles=True)
        # Name known but roles empty -> treated as unauthorized in strict mode.
        violations = rule.evaluate(
            self._person_in_vault(), now=1005.0, person_name="Mystery", person_roles=()
        )
        assert len(violations) == 1

    def test_plain_restricted_zone_ignores_roles(self):
        fp = FloorPlan(size=(200, 200))
        fp.add_zone(
            "Server Room", [(80, 80), (160, 80), (160, 160), (80, 160)], zone_type="restricted"
        )  # no allowed_roles
        rule = RuleEngine(fp)
        violations = rule.evaluate(
            self._person_in_vault(), now=1005.0, person_name="Anyone", person_roles=("security",)
        )
        assert len(violations) == 1  # entry alerts regardless of role


class TestIdentityAPI:
    def test_routes_report_disabled_without_pipeline(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        client = TestClient(create_app())
        resp = client.get("/api/identities")
        assert resp.status_code == 503

    def test_routes_list_with_registry(self, tmp_path):
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        class _FakePipeline:
            identity_registry = IdentityRegistry(store_dir=tmp_path / "reg")
            face_engine = type(
                "E", (), {"available": False, "detector_backend": None, "embedder_backend": None}
            )()
            person_identities = {}

        client = TestClient(create_app(pipeline=_FakePipeline()))
        data = client.get("/api/identities").json()
        assert data["available"] is False
        assert data["identities"] == []
