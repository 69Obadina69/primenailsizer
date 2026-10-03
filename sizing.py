"""
Size recommendation engine. Deliberately decoupled from the measurement
engine (spec section 27): this module only maps mm -> size label given a
configurable profile, and never produces a measurement itself.
"""
from __future__ import annotations
import json
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "sizing_profiles.json")


def load_profiles(path: str = DATA_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def recommend_size(width_mm: float, profile: dict) -> dict | None:
    """profile is a single profile dict like {"sizes": {"XS": {...}, ...}}.
    Returns the matching size entry, or None if out of all configured
    ranges (caller should surface this rather than guessing).
    """
    sizes = profile["sizes"]
    for label, bounds in sizes.items():
        if bounds["min_width_mm"] <= width_mm < bounds["max_width_mm"]:
            return {"label": label, **bounds}

    # Outside all ranges: clamp to nearest with a flag so the caller knows
    # this wasn't a confident in-range match.
    labels_sorted = sorted(sizes.items(), key=lambda kv: kv[1]["min_width_mm"])
    if width_mm < labels_sorted[0][1]["min_width_mm"]:
        label, bounds = labels_sorted[0]
    else:
        label, bounds = labels_sorted[-1]
    return {"label": label, **bounds, "out_of_range": True}
