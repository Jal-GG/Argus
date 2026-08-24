"""Tests for config loading (env interpolation, zones normalization)."""

import textwrap

from src.utils.config import _interpolate, load_zones_config


def test_env_interpolation(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_USER", "admin")
    yaml_text = """
    cameras:
      - id: cam_001
        source: "rtsp://${TEST_USER}:${MISSING_VAR:-fallback}@host/stream"
        enabled: true
    """
    from src.utils.config import load_yaml

    tmp_file = tmp_path / "cam.yaml"
    tmp_file.write_text(textwrap.dedent(yaml_text), encoding="utf-8")

    data = load_yaml(tmp_file)
    src = data["cameras"][0]["source"]
    assert "admin" in src and "fallback" in src


def test_interpolate_defaults():
    assert _interpolate("${NOPE_XYZ:-dflt}") == "dflt"
    assert _interpolate("plain") == "plain"


def test_load_zones_normalization(tmp_path):
    zones_yaml = textwrap.dedent(
        """
        zones:
          floor_1:
            floor_plan: "data/floor_plans/floor_1.png"
            scale: 0.05
            zones:
              - name: Vault
                type: restricted
                polygon: [[10, 10], [50, 10], [50, 40]]
                rules:
                  - type: restricted_access
                    allowed_roles: [security]
                  - type: loitering_detection
                    max_dwell_time: 120
              - name: Broken
                polygon: []
        """
    )
    path = tmp_path / "zones.yaml"
    path.write_text(zones_yaml, encoding="utf-8")

    zones = load_zones_config(path)
    assert len(zones) == 1  # broken zone dropped
    z = zones[0]
    assert z["name"] == "Vault"
    assert z["rule"] == "restricted_zone"  # primary rule wins
    assert z["params"]["allowed_roles"] == ["security"]
    assert z["params"]["max_dwell_time"] == 120  # secondary rule merged
    assert len(z["polygon"]) == 3
    assert z["scale"] == 0.05
