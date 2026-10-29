"""
src/visualizations.py
---------------------
FlightWise AI: Aircraft Delay Risk Prediction & Operational Intelligence

PURPOSE:
Interactive Plotly visualizations tailored with a modern aviation dark navy theme:
- Delay trends over time
- Delays by airline
- Delays by departure hour
- Top airports by delay percentage
- Delay cause breakdown
- Interactive route network map
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Aviation Dark Navy Palette Tokens
THEME_BG = "#0A1128"
CARD_BG = "#101F42"
TEXT_COLOR = "#FFFFFF"
MUTED_TEXT = "#94A3B8"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
CYAN_ACCENT = "#00D2D3"
BLUE_ACCENT = "#2E86DE"
AMBER_ACCENT = "#FECA57"
RED_ACCENT = "#FF6B6B"
GREEN_ACCENT = "#1DD1A1"

def _apply_dark_theme(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply unified aviation dark navy styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "y": 0.94,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": TEXT_COLOR, "family": "Inter, sans-serif"}
        },
        height=height,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "Inter, sans-serif", "size": 12},
        margin=dict(l=40, r=30, t=55, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(color=MUTED_TEXT),
            title_font=dict(color=TEXT_COLOR, size=13),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(color=MUTED_TEXT),
            title_font=dict(color=TEXT_COLOR, size=13),
        ),
        legend=dict(
            bgcolor="rgba(16, 31, 66, 0.7)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color=TEXT_COLOR)
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )
    return fig

def plot_delay_trends(trends_df: pd.DataFrame) -> go.Figure:
    """
    Line and Area chart tracking daily delay percentage and volume over time.
    """
    if trends_df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Delay Trends Over Time (No Data)")

    fig = go.Figure()

    # Flight volume bar
    fig.add_trace(go.Bar(
        x=trends_df["date"],
        y=trends_df["total_flights"],
        name="Scheduled Flights",
        marker_color="rgba(46, 134, 222, 0.35)",
        yaxis="y2",
        hovertemplate="<b>Date:</b> %{x}<br><b>Total Flights:</b> %{y}<extra></extra>"
    ))

    # Delay rate line
    fig.add_trace(go.Scatter(
        x=trends_df["date"],
        y=trends_df["delay_rate"],
        name="Delay Rate (%)",
        mode="lines+markers",
        line=dict(color=CYAN_ACCENT, width=3),
        marker=dict(size=6, color="#FFFFFF", line=dict(color=CYAN_ACCENT, width=2)),
        hovertemplate="<b>Date:</b> %{x}<br><b>Delay Rate:</b> %{y:.1f}%<extra></extra>"
    ))

    _apply_dark_theme(fig, "Daily Delay Rate (%) & Scheduled Flight Volume", height=380)

    # Dual axis configuration
    fig.update_layout(
        yaxis=dict(
            title="Delay Percentage (%)",
            ticksuffix="%",
            side="left"
        ),
        yaxis2=dict(
            title="Total Daily Flights",
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color=MUTED_TEXT)
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_airline_delays(airline_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart comparing airlines by delay rate.
    """
    if airline_df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Airline Delay Comparison (No Data)")

    # Sort ascending for natural top-to-bottom bar order
    df_sorted = airline_df.sort_values(by="delay_rate", ascending=True)

    fig = go.Figure(go.Bar(
        y=df_sorted["airline"],
        x=df_sorted["delay_rate"],
        orientation="h",
        marker=dict(
            color=df_sorted["delay_rate"],
            colorscale=[
                [0.0, GREEN_ACCENT],
                [0.5, AMBER_ACCENT],
                [1.0, RED_ACCENT]
            ],
            showscale=False
        ),
        text=[f"{v:.1f}% ({d} delayed)" for v, d in zip(df_sorted["delay_rate"], df_sorted["delayed_flights"])],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="#000000", size=11, family="Inter, sans-serif"),
        customdata=df_sorted[["total_flights", "avg_delay_min"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Delay Rate: <b>%{x:.1f}%</b><br>"
            "Total Flights: %{customdata[0]}<br>"
            "Avg Delay: %{customdata[1]:.1f} min<extra></extra>"
        )
    ))

    _apply_dark_theme(fig, "Airline Delay Rates (%)", height=380)
    fig.update_layout(
        xaxis=dict(title="Delayed Flights (%)", ticksuffix="%"),
        yaxis=dict(title="")
    )
    return fig

def plot_hourly_delays(hourly_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing flight delays distributed across 24 departure hours.
    """
    if hourly_df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Delays by Departure Hour (No Data)")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=hourly_df["hour_label"],
        y=hourly_df["delay_rate"],
        name="Delay Rate (%)",
        marker=dict(
            color=hourly_df["delay_rate"],
            colorscale=[
                [0.0, BLUE_ACCENT],
                [0.6, AMBER_ACCENT],
                [1.0, RED_ACCENT]
            ],
            showscale=False
        ),
        hovertemplate=(
            "<b>Time Window:</b> %{x}<br>"
            "<b>Delay Rate:</b> %{y:.1f}%<br>"
            "<extra></extra>"
        )
    ))

    _apply_dark_theme(fig, "Delay Rate by Departure Hour (Peak Analysis)", height=380)
    fig.update_layout(
        xaxis=dict(title="Scheduled Departure Hour (24-Hour)", tickangle=-45),
        yaxis=dict(title="Delay Rate (%)", ticksuffix="%")
    )
    return fig

def plot_top_airports(airport_df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """
    Vertical bar chart for top airports with highest delay percentages.
    """
    if airport_df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Top Airports by Delay (No Data)")

    top_airports = airport_df.head(top_n)

    fig = go.Figure(go.Bar(
        x=top_airports["airport"],
        y=top_airports["delay_rate"],
        marker_color=AMBER_ACCENT,
        text=[f"{val:.1f}%" for val in top_airports["delay_rate"]],
        textposition="outside",
        customdata=top_airports[["airport_city", "total_flights", "avg_delay_min"]],
        hovertemplate=(
            "<b>%{x}</b> (%{customdata[0]})<br>"
            "Delay Rate: <b>%{y:.1f}%</b><br>"
            "Total Departures: %{customdata[1]}<br>"
            "Average Delay: %{customdata[2]:.1f} min<extra></extra>"
        )
    ))

    _apply_dark_theme(fig, f"Top {top_n} Origin Airports by Delay Rate (%)", height=380)
    fig.update_layout(
        xaxis=dict(title="Airport IATA Code"),
        yaxis=dict(title="Delay Rate (%)", ticksuffix="%", range=[0, max(top_airports["delay_rate"]) * 1.25])
    )
    return fig

def plot_delay_reasons_pie(reasons_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of primary delay reasons.
    """
    if reasons_df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Delay Reasons Breakdown (No Data)")

    palette = [CYAN_ACCENT, BLUE_ACCENT, AMBER_ACCENT, RED_ACCENT, "#9B59B6"]

    fig = go.Figure(go.Pie(
        labels=reasons_df["delay_reason"],
        values=reasons_df["count"],
        hole=0.55,
        marker=dict(colors=palette),
        textinfo="percent+label",
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>Delayed Flights: %{value}<br>Share: %{percent}<extra></extra>"
    ))

    _apply_dark_theme(fig, "Primary Delay Reasons Breakdown", height=350)
    fig.update_layout(showlegend=False)
    return fig

def plot_route_network_map(route_df: pd.DataFrame, airports_dict: dict) -> go.Figure:
    """
    Geographic route network map with connecting flight corridors and airport hubs.
    """
    fig = go.Figure()

    if route_df.empty:
        return _apply_dark_theme(fig, "Route Network (No Data)")

    # Draw flight paths as curved/straight lines between origins and destinations
    for _, row in route_df.iterrows():
        # Line color based on route delay rate
        rate = row["delay_rate"]
        if rate > 25:
            line_color = "rgba(255, 107, 107, 0.75)"
            line_width = 2.5
        elif rate > 18:
            line_color = "rgba(254, 202, 87, 0.65)"
            line_width = 1.8
        else:
            line_color = "rgba(0, 210, 211, 0.55)"
            line_width = 1.2

        fig.add_trace(go.Scattergeo(
            locationmode="USA-states",
            lon=[row["origin_lon"], row["dest_lon"]],
            lat=[row["origin_lat"], row["dest_lat"]],
            mode="lines",
            line=dict(width=line_width, color=line_color),
            hoverinfo="text",
            text=(
                f"<b>Route: {row['route']}</b><br>"
                f"Delay Rate: {row['delay_rate']:.1f}%<br>"
                f"Total Flights: {row['total_flights']}<br>"
                f"Avg Delay: {row['avg_delay_min']:.1f} min<br>"
                f"Distance: {row['distance_km']} km"
            ),
            showlegend=False
        ))

    # Airport Hub nodes
    hub_codes = list(airports_dict.keys())
    hub_lats = [airports_dict[c]["lat"] for c in hub_codes]
    hub_lons = [airports_dict[c]["lon"] for c in hub_codes]
    hub_names = [f"<b>{c}</b> - {airports_dict[c]['name']}<br>{airports_dict[c]['city']}" for c in hub_codes]

    fig.add_trace(go.Scattergeo(
        locationmode="USA-states",
        lon=hub_lons,
        lat=hub_lats,
        mode="markers+text",
        text=hub_codes,
        textposition="top center",
        textfont=dict(color="#FFFFFF", size=11, family="Inter, sans-serif"),
        marker=dict(
            size=10,
            color=CYAN_ACCENT,
            line=dict(width=2, color="#FFFFFF")
        ),
        hoverinfo="text",
        hovertext=hub_names,
        name="Hub Airports"
    ))

    fig.update_layout(
        title={
            "text": "<b>Commercial Aviation Flight Corridors & Delay Risk Network</b>",
            "x": 0.02,
            "y": 0.95,
            "xanchor": "left",
            "font": {"size": 16, "color": TEXT_COLOR}
        },
        height=540,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="#0A1128",
            subunitcolor="rgba(255, 255, 255, 0.15)",
            countrycolor="rgba(255, 255, 255, 0.25)",
            countrywidth=1,
            subunitwidth=0.8,
            bgcolor=CARD_BG,
            lakecolor="#070E20",
            showlakes=True
        ),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig
