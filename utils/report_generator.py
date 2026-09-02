"""Generate a concise final report from the master crisis-response state."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def build_report(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build structured summary data without mutating LangGraph state."""
    zone_map = state.get("zone_map") or {}
    people_counts = state.get("people_counts") or {}
    rescue_plan = state.get("rescue_plan") or {}
    routes = state.get("route_plan") or []
    dispatch_result = state.get("dispatch_result") or {}

    critical_zones = [
        zone_id for zone_id, data in zone_map.items()
        if _severity_score(data.get("severity", 0)) >= 0.7
    ]

    resources = {"ambulances": 0, "boats": 0, "rescue_teams": 0}
    for allocation in rescue_plan.values():
        if isinstance(allocation, Mapping):
            for key in resources:
                resources[key] += _safe_int(allocation.get(key, 0))

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "zones_analyzed": len(zone_map),
        "critical_zones": critical_zones,
        "people_detected": sum(_safe_int(v) for v in people_counts.values()),
        "resources": resources,
        "routes_planned": len(routes),
        "resource_approved": bool(state.get("resource_approved")),
        "route_approved": bool(state.get("route_approved")),
        "dispatch_summary": dispatch_result.get("summary", state.get("dispatch_message", "")),
    }


def render_text_report(report: Mapping[str, Any]) -> str:
    """Render the structured report as operator-friendly plain text."""
    resources = report.get("resources") or {}
    critical = report.get("critical_zones") or []
    critical_text = ", ".join(critical) if critical else "None"

    return "\n".join([
        "CRISIS RESPONSE REPORT",
        "=" * 24,
        f"Generated: {report.get('generated_at', '')}",
        f"Zones analyzed: {report.get('zones_analyzed', 0)}",
        f"Critical zones: {critical_text}",
        f"People detected: {report.get('people_detected', 0)}",
        f"Ambulances: {resources.get('ambulances', 0)}",
        f"Boats: {resources.get('boats', 0)}",
        f"Rescue teams: {resources.get('rescue_teams', 0)}",
        f"Routes planned: {report.get('routes_planned', 0)}",
        f"Resource approval: {'APPROVED' if report.get('resource_approved') else 'NOT APPROVED'}",
        f"Route approval: {'APPROVED' if report.get('route_approved') else 'NOT APPROVED'}",
        f"Dispatch: {report.get('dispatch_summary', '')}",
    ])


def save_report(state: Mapping[str, Any], output_dir: str = "reports") -> tuple[dict[str, Any], str]:
    """Build and save the final response report; return data and file path."""
    report = build_report(state)
    path = Path(output_dir) / "crisis_response_report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_text_report(report), encoding="utf-8")
    return report, str(path)


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _severity_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    labels = {"critical": 1.0, "high": 0.8, "moderate": 0.5, "low": 0.2}
    return labels.get(str(value).strip().lower(), 0.0)
