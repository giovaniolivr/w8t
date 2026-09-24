"""Suggest a new period when the observed trend contradicts the goal of an open-ended period.

Periods are the user's regime boundaries. For an *indefinite* period (no planned end), a
statistically clear reversal - the trend detector declares a direction, which requires the 95%
interval to exclude zero and a practically relevant pace - that goes against the goal is a
natural moment to ask whether a new phase started. It is only a suggestion; the user decides.

A plateau during loss/gain ("estável") is *not* a trigger: stalls are normal inside a cut or
bulk and are already reported as plateaus.
"""

from __future__ import annotations

from w8t.core.trend import Trend, TrendDirection
from w8t.data.models import GoalDirection, Period

_OBSERVED_GOAL = {
    TrendDirection.DOWN: GoalDirection.LOSS,
    TrendDirection.UP: GoalDirection.GAIN,
}


def suggested_goal(period: Period | None, trend: Trend | None) -> GoalDirection | None:
    """New goal to suggest, or None when there's nothing to suggest."""
    if period is None or trend is None or period.end_date is not None:
        return None
    observed = _OBSERVED_GOAL.get(trend.direction)
    if observed is None or observed == period.goal_direction:
        return None
    return observed
