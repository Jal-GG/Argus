"""YAML configuration loading with env-var interpolation.

Values in YAML files may reference environment variables using
``${VAR}`` or ``${VAR:-default}`` syntax, e.g.::

    source: "rtsp://${CAM_USER:-admin}:${CAM_PASS:-password}@192.168.1.10/stream"
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .logger import get_logger

logger = get_logger(__name__)

_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name = match.group("name")
            default = match.group("default")
            return os.getenv(name, default if default is not None else "")

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(item) for item in value]
    return value


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML file and interpolate environment references."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _interpolate(data)


def load_cameras_config(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load cameras.yaml and return only enabled camera entries."""
    data = load_yaml(path)
    cameras = data.get("cameras", [])
    enabled = [cam for cam in cameras if cam.get("enabled", True)]
    skipped = len(cameras) - len(enabled)
    if skipped:
        logger.info("Skipped %d disabled camera(s)", skipped)
    return enabled


# Map zones.yaml rule verbs onto RuleEngine rule types.
_RULE_ALIASES = {
    "required_visit": "required_zone",
    "restricted_access": "restricted_zone",
    "loitering_detection": "loitering",
}


def load_zones_config(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load zones.yaml into a normalized zone list.

    Expected schema (nested by floor)::

        zones:
          floor_0:
            floor_plan: data/floor_plans/floor_0.png
            scale: 0.05
            zones:
              - name: Server Room
                type: restricted
                polygon: [[650,100], [750,100], ...]
                rules:
                  - type: restricted_access
                    allowed_roles: [security]
    """
    data = load_yaml(path)
    raw_zones: list[dict[str, Any]] = []

    floors = data.get("zones", {}) or {}
    if isinstance(floors, list):  # tolerate flat schema too
        floors = {"floor_0": {"zones": floors}}

    for floor_id, floor_block in floors.items():
        if isinstance(floor_block, str) or floor_block is None:
            continue
        for entry in floor_block.get("zones", []) or []:
            polygon = entry.get("polygon") or entry.get("points")
            if not polygon or len(polygon) < 3:
                logger.warning("Zone '%s' has invalid/missing polygon; skipping", entry.get("name"))
                continue

            # A zone may declare multiple rules; keep them all.
            rules_out: list[dict[str, Any]] = []
            for rule in entry.get("rules", []) or []:
                if not isinstance(rule, dict):
                    continue
                rtype = _RULE_ALIASES.get(str(rule.get("type")), rule.get("type"))
                params = {k: v for k, v in rule.items() if k not in ("type", "description")}
                rules_out.append({"type": rtype, "params": params})

            # Primary rule = highest priority present, else implied by type.
            primary = None
            for wanted in ("restricted_zone", "loitering", "required_zone"):
                primary = next((r for r in rules_out if r["type"] == wanted), None)
                if primary is not None:
                    break
            implicit = {"restricted": "restricted_zone", "required": "required_zone"}
            primary_type = (
                primary["type"] if primary else implicit.get(str(entry.get("type", "custom")))
            )

            params = {}
            for r in rules_out:
                params.update(r["params"])

            raw_zones.append(
                {
                    "name": str(entry["name"]),
                    "type": str(entry.get("type", "custom")),
                    "floor_id": str(floor_id),
                    "polygon": [(float(x), float(y)) for x, y in polygon],
                    "rule": primary_type,
                    "params": params,
                    "all_rules": rules_out,
                    "enabled": bool(entry.get("enabled", True)),
                    "color": tuple(entry["color"]) if entry.get("color") else None,
                    "scale": float(floor_block.get("scale", 1.0)),
                    "floor_plan": floor_block.get("floor_plan"),
                }
            )
    return [z for z in raw_zones if z["enabled"]]


def project_root() -> Path:
    """Return the repository root inferred from this file's location."""
    return Path(__file__).resolve().parents[2]
