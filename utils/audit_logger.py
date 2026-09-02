"""Structured audit logging for crisis-response workflow events."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

_LOGGER_NAME = "project.audit"
_LOG_FILE = Path("logs") / "crisis_audit.log"


def _logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def audit_event(event: str, details: Optional[Mapping[str, Any]] = None) -> None:
    """Append a JSON-structured event to the crisis audit log."""
    payload = {"event": event, "details": dict(details or {})}
    _logger().info(json.dumps(payload, default=str, sort_keys=True))


def summarize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, log-safe summary of the current workflow state."""
    return {
        "zones": len(state.get("zone_map") or {}),
        "affected_zones": len(state.get("most_affected_zones") or []),
        "people_detected": sum((state.get("people_counts") or {}).values()),
        "rescue_plan_zones": len(state.get("rescue_plan") or {}),
        "routes": len(state.get("route_plan") or []),
        "resource_approved": state.get("resource_approved"),
        "route_approved": state.get("route_approved"),
    }
