"""W8T palette for charts - mirrors .streamlit/config.toml (green / dark gray / black).

Convention: real measurements are neutral gray; derived values (moving averages, and later
trend/forecast) are green, so "what was measured" and "what was computed" never look alike.
"""

GREEN = "#22C55E"
GREEN_LIGHT = "#86EFAC"
GRAY = "#9CA3AF"
GRAY_DARK = "#374151"

MEASUREMENT = GRAY
MOVING_AVG_SHORT = GREEN
MOVING_AVG_LONG = GREEN_LIGHT
TARGET = GRAY
