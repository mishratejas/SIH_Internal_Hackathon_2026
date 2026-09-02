"""Validation helpers for the crisis-response pipeline.

Keeps malformed caller input and incomplete intermediate state from reaching
expensive vision, routing, and dispatch stages.
"""

from typing import Any, Mapping


class ValidationError(ValueError):
    """Raised when crisis pipeline input is invalid."""


def validate_initial_state(state: Mapping[str, Any]) -> None:
    """Validate the minimum state required before the vision node runs."""
    image_path = state.get("satellite_image")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValidationError("satellite_image must be a non-empty file path")

    image_meta = state.get("image_meta")
    if image_meta is not None:
        if not isinstance(image_meta, Mapping):
            raise ValidationError("image_meta must be a mapping when provided")
        for key in ("center_lat", "center_lon"):
            if key in image_meta and not isinstance(image_meta[key], (int, float)):
                raise ValidationError(f"image_meta[{key!r}] must be numeric")

    base_locations = state.get("base_locations")
    if base_locations is not None and not isinstance(base_locations, Mapping):
        raise ValidationError("base_locations must be a mapping when provided")


def validate_zone_map(zone_map: Mapping[str, Any]) -> None:
    """Validate the shape and score ranges of vision-generated zones."""
    if not isinstance(zone_map, Mapping):
        raise ValidationError("zone_map must be a mapping")

    for zone_id, zone in zone_map.items():
        if not isinstance(zone_id, str) or not zone_id:
            raise ValidationError("zone identifiers must be non-empty strings")
        if not isinstance(zone, Mapping):
            raise ValidationError(f"Zone {zone_id} must contain a mapping")

        for score_name in ("flood_score", "damage_score"):
            if score_name in zone:
                score = zone[score_name]
                if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                    raise ValidationError(
                        f"{zone_id}.{score_name} must be a number between 0 and 1"
                    )


def validate_people_counts(people_counts: Mapping[str, Any]) -> None:
    """Ensure drone victim counts are non-negative integers."""
    if not isinstance(people_counts, Mapping):
        raise ValidationError("people_counts must be a mapping")

    for zone_id, count in people_counts.items():
        if not isinstance(zone_id, str):
            raise ValidationError("people_counts zone identifiers must be strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationError(f"people_counts[{zone_id!r}] must be a non-negative integer")
