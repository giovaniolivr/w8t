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

# Pattern annotations. Amber is the one deliberate off-palette color: an anomaly must stand out
# from both the gray measurements and the green derived lines.
ANOMALY = "#F59E0B"
PLATEAU_FILL = "rgba(156, 163, 175, 0.12)"

# Forecasts are derived values (green family) but must never look like a measured line:
# dashed mean + translucent interval band.
FORECAST = GREEN
FORECAST_BAND = "rgba(34, 197, 94, 0.15)"

# Kalman smoothed trend: a derived state (green), drawn with its uncertainty band.
TREND_STATE = GREEN
TREND_STATE_BAND = "rgba(34, 197, 94, 0.12)"

TEXT = "#E5E7EB"
SURFACE = "#1F2322"

# Passed to st.plotly_chart(config=...): scroll-to-zoom, no Plotly logo, clean export.
PLOTLY_CONFIG = {
    "displaylogo": False,
    # No wheel zoom: with it, the wheel over a chart zooms instead of scrolling the page, and the
    # charts fill most of the window. Drag-to-zoom, the 7d/1m/... buttons and double-click reset
    # remain.
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "w8t", "scale": 2},
}


def style_figure(fig, *, height: int = 440, time_axis: bool = True, range_slider: bool = True):
    """Shared look & interactions: unified hover with a crosshair, range buttons, a thin range
    slider, smooth transitions, transparent backgrounds that sit on the card CSS."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": TEXT, "size": 13},
        margin={"l": 10, "r": 10, "t": 56, "b": 10},
        hovermode="x unified",
        hoverlabel={"bgcolor": SURFACE, "bordercolor": GREEN, "font": {"color": TEXT}},
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right", "bgcolor": "rgba(0,0,0,0)"},
        transition={"duration": 450, "easing": "cubic-in-out"},
        yaxis={"gridcolor": "rgba(255,255,255,0.06)", "zeroline": False, "title": "kg"},
    )
    xaxis = {
        "gridcolor": "rgba(255,255,255,0.04)",
        "showspikes": True,
        "spikemode": "across",
        "spikesnap": "cursor",
        "spikecolor": GRAY,
        "spikethickness": 1,
        "spikedash": "dot",
    }
    if time_axis:
        # Brazilian dates; plotly's default English month names ("Aug 23") clashed with the UI.
        xaxis["hoverformat"] = "%d/%m/%Y"
        xaxis["tickformatstops"] = [
            {"dtickrange": [None, 86_400_000 * 40], "value": "%d/%m"},  # up to ~monthly ticks
            {"dtickrange": [86_400_000 * 40, None], "value": "%m/%Y"},
        ]
        xaxis["rangeselector"] = {
            "buttons": [
                {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
                {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                {"step": "all", "label": "tudo"},
            ],
            "bgcolor": SURFACE,
            "activecolor": GREEN,
            "bordercolor": "rgba(255,255,255,0.08)",
            "borderwidth": 1,
            "font": {"color": TEXT},
            "x": 0,
            "y": 1.12,
        }
        if range_slider:
            xaxis["rangeslider"] = {
                "visible": True,
                "thickness": 0.07,
                "bgcolor": "rgba(255,255,255,0.03)",
            }
    fig.update_xaxes(**xaxis)
    return fig
