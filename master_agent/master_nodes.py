"""
master_nodes.py
---------------
All LangGraph node functions for the master crisis-response pipeline.

Each function:
  • Receives the current MasterState
  • Does one well-defined job
  • Returns a dict of updated state fields

Pipeline order:
  vision → store_zone → drone_analysis → drone_decision → drone_dispatch
    → drone_vision → update_people → rescue_decision → admin_resource
    → route_planner → admin_route → communication → END

FIX (2026-03): _to_python() recursively converts all numpy scalar/array types
to native Python before any state is returned, preventing the LangGraph
MemorySaver msgpack serialization error:
  "TypeError: Type is not msgpack serializable: numpy.float64"
"""

import numpy as np

from agents.vision_agent.vision_agent               import analyze_image
from agents.resource_agent.drone_analysis           import get_most_affected_zones
from agents.drone_agent.drone_nodes                 import drone_decision_node, drone_dispatch_node
from agents.drone_agent.drone_vision                import drone_vision_node
from agents.resource_agent.rescue_decision_llm      import allocate_rescue_resources_llm
from agents.route_agent.route_agent                 import plan_all_routes, print_routes
from agents.communication_agent.communication_agent import dispatch_all

from db.update_from_vision  import update_zones_from_vision
from db.update_people_count import update_people_count
from utils.admin_interface  import admin_approval
from generate_route_map     import generate_route_map


# ── Default image metadata (Mumbai area — matches Streamlit defaults) ─────────

_DEFAULT_IMAGE_META = {
    "center_lat":  19.062061,
    "center_lon":  72.863542,
    "coverage_km": 1.6,
    "width_px":    1024,
    "height_px":   522,
}

_DEFAULT_BASE_LOCATIONS = {
    "ambulance":   {"name": "Hospital",      "lat": 19.06546856543151,  "lon": 72.86100899070198},
    "rescue_team": {"name": "Rescue Center", "lat": 19.06847079812735,  "lon": 72.85793995490616},
    "boat":        {"name": "Boat Depot",    "lat": 19.063380373548366, "lon": 72.85538649195271},
}


# ── Numpy → native Python converter ──────────────────────────────────────────

def _to_python(obj):
    """
    Recursively convert numpy scalar/array types to native Python so that
    LangGraph's MemorySaver (msgpack serializer) can persist them.

    Handles:
      numpy.integer  → int
      numpy.floating → float
      numpy.ndarray  → list  (or omitted if it's the flood mask, handled separately)
      tuple          → tuple (preserves structure)
      list           → list
      dict           → dict
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_python(v) for v in obj)
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    return obj


def _clean_routes(routes: list) -> list:
    """
    Convert route plans to fully native-Python dicts.
    Also fixes ETA when it is polluted by the 1e9 penalty weight on blocked roads:
    if eta_minutes > distance_km * 200, recalculate from distance at 30 km/h.
    """
    cleaned = []
    for r in routes:
        r2 = _to_python(r)
        # Fix unreasonable ETA caused by penalty roads
        dist   = r2.get("distance_km", 0.0)
        eta    = r2.get("eta_minutes", 0.0)
        if dist > 0 and eta > dist * 200:
            # Estimate ETA: 30 km/h average for emergency vehicle in flooded area
            r2["eta_minutes"] = round(dist / 30.0 * 60.0, 1)
            r2["eta_note"]    = "ETA estimated at 30 km/h (flood route)"
        cleaned.append(r2)
    return cleaned


# ── Vision Node ───────────────────────────────────────────────────────────────

def vision_node(state):
    """
    Run the Vision Agent on the satellite image.

    Reads : state["satellite_image"]  — file path to the image
            state["image_meta"]       — optional GPS metadata dict

    Writes: state["zone_map"]    — 100-zone severity/flood/damage map
            state["image_meta"]  — GPS metadata (defaults if not supplied)
            state["flood_mask"]  — raw float flood probability array (H×W)
    """
    print("\n[MASTER] ── Vision Agent ─────────────────────────────────")

    result     = analyze_image(state["satellite_image"])
    flood_mask = result.get("flood_prob_map")
    height, width = flood_mask.shape[:2]

    image_meta = state.get("image_meta") or _DEFAULT_IMAGE_META.copy()

    if not state.get("image_meta"):
        print(f"[VISION] No image_meta in state — using defaults: {image_meta}")

    image_meta.update({
        "width_px":  int(width),
        "height_px": int(height),
    })

    zone_map     = result.get("zone_map", {})
    zone_count   = len(zone_map)
    flood_zones  = sum(1 for z in zone_map.values() if z.get("flood_score",  0) >= 0.45)
    damage_zones = sum(1 for z in zone_map.values() if z.get("damage_score", 0) >= 0.45)

    print(f"[VISION] Complete — {zone_count} zones analysed  "
          f"| flood zones: {flood_zones}  | damage zones: {damage_zones}")

    return {
        "zone_map":   _to_python(zone_map),
        "flood_mask": flood_mask,           # kept as ndarray for route masking
        "image_meta": _to_python(image_meta),
    }


# ── Store Zone Node ───────────────────────────────────────────────────────────

def store_zone_node(state):
    """Persist vision results to the SQLite database."""
    print("\n[DB] ── Storing zone data ───────────────────────────────")
    update_zones_from_vision(state["zone_map"])
    print("[DB] Zone data written to crisis.db")
    return {}


# ── Drone Analysis Node ───────────────────────────────────────────────────────

def drone_analysis_node(state):
    """Query DB for the top-N most affected zones."""
    print("\n[RESOURCE AGENT] ── Drone Analysis ─────────────────────")
    affected_zones = get_most_affected_zones(top_n=5)
    print(f"[RESOURCE AGENT] Top affected zones: {affected_zones}")
    return {"most_affected_zones": affected_zones}


# ── Update People Node ────────────────────────────────────────────────────────

def update_people_node(state):
    """Write drone-detected people counts to the DB."""
    print("\n[DB] ── Updating people counts ──────────────────────────")
    update_people_count(state["people_counts"])
    total = sum(state["people_counts"].values()) if state.get("people_counts") else 0
    print(f"[DB] People counts saved — total detected: {total}")
    return {}


# ── Rescue Decision Node ──────────────────────────────────────────────────────

def rescue_decision_node(state):
    """Ask the Gemini LLM to allocate rescue resources across zones."""
    print("\n[LLM] ── Rescue Resource Allocation ─────────────────────")

    rescue_plan = allocate_rescue_resources_llm(
        zone_map      = state["zone_map"],
        people_counts = state.get("people_counts", {}),
        zones         = state.get("most_affected_zones", []),
    )

    print("\n[LLM] Proposed rescue plan:")
    for zone, plan in rescue_plan.items():
        resources = ", ".join(f"{v}× {k}" for k, v in plan.items() if v)
        print(f"  {zone}  →  {resources}")

    return {"rescue_plan": _to_python(rescue_plan)}


# ── Admin Resource Approval Node ──────────────────────────────────────────────

def admin_resource_node(state):
    """
    Ask the admin to approve or reject the rescue plan.
    In Streamlit mode this node is interrupted BEFORE execution;
    the UI injects {resource_approved} via graph.update_state().
    """
    print("\n[ADMIN] ── Resource Allocation Approval ─────────────────")
    approved = admin_approval("Approve rescue resource allocation?")
    print("[ADMIN] Resources " + ("APPROVED ✓" if approved else "REJECTED ✗"))
    return {"resource_approved": bool(approved)}


def resource_approval_router(state):
    """Conditional edge: approved → route_planner, rejected → rescue_decision."""
    return "approved" if state.get("resource_approved") else "rejected"


# ── Route Planner Node ────────────────────────────────────────────────────────

def route_planner_node(state):
    """
    Call plan_all_routes() and auto-generate the HTML route map.

    Reads : state["rescue_plan"]     — zone → resource allocations
            state["image_meta"]      — GPS coverage for geo-transform
            state["flood_mask"]      — optional flood probability array
            state["base_locations"]  — optional override for resource origins

    Writes: state["route_plan"]      — list of route dicts (all native Python)
            state["route_map_path"]  — absolute path to the generated HTML map
    """
    print("\n[ROUTE PLANNER] ── Planning Routes ──────────────────────")

    resource_assignments = state.get("rescue_plan", {})
    if not resource_assignments:
        print("[ROUTE PLANNER] WARNING: rescue_plan is empty — no routes to plan.")
        return {"route_plan": [], "route_map_path": None}

    image_meta     = state.get("image_meta")     or _DEFAULT_IMAGE_META
    base_locations = state.get("base_locations") or _DEFAULT_BASE_LOCATIONS
    flood_mask     = state.get("flood_mask")

    blocked_masks = {}
    if flood_mask is not None:
        blocked_masks["flood"] = flood_mask

    routes = plan_all_routes(
        image_meta           = image_meta,
        resource_assignments = resource_assignments,
        base_locations       = base_locations,
        blocked_masks        = blocked_masks if blocked_masks else None,
        use_real_osm         = True,
        flood_threshold      = 0.45,
    )

    # ── Fix numpy types + unreasonable ETAs BEFORE saving to LangGraph state ──
    routes = _clean_routes(routes)

    print_routes(routes)

    # Auto-generate HTML route map
    print("\n[ROUTE MAP] Generating HTML map ...")
    try:
        map_path = generate_route_map(
            route_plans    = routes,
            image_meta     = image_meta,
            base_locations = base_locations,
            zone_map       = state.get("zone_map"),
        )
        print(f"[ROUTE MAP] Saved → {map_path}")
    except Exception as e:
        print(f"[ROUTE MAP] WARNING: Could not generate map: {e}")
        map_path = None

    return {
        "route_plan":     routes,
        "route_map_path": map_path,
    }


# ── Admin Route Approval Node ─────────────────────────────────────────────────

def admin_route_node(state):
    """
    Ask the operator to approve the route plan.
    In Streamlit mode this node is interrupted BEFORE execution;
    the UI injects {route_approved} via graph.update_state().
    """
    print("\n[ADMIN] ── Route Plan Approval ──────────────────────────")

    map_path = state.get("route_map_path")
    if map_path:
        print(f"[ADMIN] Route map: {map_path}")

    approved = admin_approval("Approve rescue routes?")
    print("[ADMIN] Routes " + ("APPROVED ✓" if approved else "REJECTED ✗ — re-running planner"))
    return {"route_approved": bool(approved)}


def route_approval_router(state):
    """Conditional edge: approved → communication, rejected → route_planner."""
    return "approved" if state.get("route_approved") else "rejected"


# ── Communication Node ────────────────────────────────────────────────────────

def communication_node(state):
    """Generate dispatch instructions and send via SMS / audio."""
    print("\n[COMMUNICATION AGENT] ── Dispatch Pipeline ──────────────")

    route_plan      = state.get("route_plan")      or []
    zone_map        = state.get("zone_map")         or {}
    people_counts   = state.get("people_counts")    or {}
    field_reports   = state.get("field_reports")    or []
    dispatch_config = state.get("dispatch_config")  or {}

    if not route_plan:
        print("[COMMUNICATION AGENT] WARNING: No route_plan — nothing to dispatch.")
        return {
            "dispatch_result":  {
                "instructions": {}, "sms_results": [],
                "audio_files": [], "summary": "No routes planned.",
            },
            "dispatch_message": "No routes planned.",
        }

    zone_metadata = {}
    for zone_id, data in zone_map.items():
        raw_severity = data.get("severity", 0)
        if isinstance(raw_severity, float):
            severity_label = (
                "Critical" if raw_severity >= 0.7 else
                "Moderate" if raw_severity >= 0.4 else
                "Low"
            )
        else:
            severity_label = str(raw_severity)
        zone_metadata[zone_id] = {
            "severity":     severity_label,
            "victim_count": people_counts.get(zone_id, 0),
        }

    result = dispatch_all(
        route_plans     = route_plan,
        zone_metadata   = zone_metadata,
        field_reports   = field_reports,
        dispatch_config = dispatch_config,
    )

    n_instructions = len(result.get("instructions", {}))
    n_sms          = len(result.get("sms_results",  []))
    print(f"[COMMUNICATION AGENT] Complete — "
          f"{n_instructions} instruction(s)  |  {n_sms} SMS(es) sent")

    return {
        "dispatch_result":  _to_python(result),
        "dispatch_message": result.get("summary", ""),
    }


# ── Re-exports ────────────────────────────────────────────────────────────────

__all__ = [
    "vision_node",
    "store_zone_node",
    "drone_analysis_node",
    "drone_decision_node",
    "drone_dispatch_node",
    "drone_vision_node",
    "update_people_node",
    "rescue_decision_node",
    "admin_resource_node",
    "resource_approval_router",
    "route_planner_node",
    "admin_route_node",
    "route_approval_router",
    "communication_node",
]