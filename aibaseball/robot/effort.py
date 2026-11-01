"""Delivery batter strength (spec value, the user's). 11-01 20:0x user: back to human level -> 1.0 (1.5 was 10-26 .. 11-01)."""
from __future__ import annotations

import os

AIB_DELIVERY_EFFORT = os.environ.get("AIB_DELIVERY_EFFORT", "1.0")


def is_delivery_effort() -> bool:
    """True when AIB_EFFORT_SCALE (default 1.0, as in make_aib1_cfg) is the delivery strength."""
    return float(os.environ.get("AIB_EFFORT_SCALE", "1.0")) == float(AIB_DELIVERY_EFFORT)
