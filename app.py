
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_flight_data
from src.generate_data import AIRLINES, AIRPORTS, WEATHER_CONDITIONS
from src.prediction_engine import predict_flight_risk


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FlightWise AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1228;
        color: #ffffff;
    }

    [data-testid="stSidebar"] {
        background-color: #080e20;
    }

    [data-testid="stMetric"] {
        background-color: #122143;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #26395d;
    }

    [data-testid="stMetricLabel"] {
        color: #9fb6d9;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    .hero {
        background: linear-gradient(110deg, #10203e, #1d3976);
        border: 1px solid #176c94;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #00d2d3;
        margin-bottom: 5px;
    }

    .notice {
        background-color: #332b22;
        border: 1px solid #99723b;
        color: #ffd17a;
        border-radius: 8px;
        padding: 10px;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner="Loading flight dataset...")
def get_data():
    return load_flight_data("data/synthetic_flights.csv")


try:
    df = get_data()
except Exception as error:
    st.error(f"Could not load the flight dataset: {error}")
    st.info(
        "Check that data/synthetic_flights.csv exists and contains "
        "the required columns."
    )
    st.stop()


if df.empty:
    st.warning("The flight dataset is empty.")
    st.stop()


df["flight_date"] = pd.to_datetime(
    df["flight_date"], errors="coerce"
)

df["delay_minutes"] = pd.to_numeric(
    df["delay_minutes"], errors="coerce"
).fillna(0)

df["departure_hour"] = pd.to_numeric(
    df["departure_hour"], errors="coerce"
).fillna(0).astype(int)

df["is_delayed"] = (
    df["delay_minutes"] >= 15
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_high_risk_watchlist(data):
    """Return flights with the largest recorded delays."""

    if data.empty:
        return pd.DataFrame()

    watchlist = data.copy()

    watchlist["risk_category"] = watchlist[
        "delay_minutes"
    ].apply(
        lambda delay: (
            "Critical"
            if delay >= 70
            else "High"
            if delay >= 35
            else "Moderate"
            if delay >= 15
            else "Low"
        )
    )

    watchlist = watchlist[
        watchlist["delay_minutes"] >= 15
    ].sort_values(
        "delay_minutes", ascending=False
    )

    columns = [
        "flight_number",
        "airline",
        "origin",
        "destination",
        "flight_date",
        "departure_hour",
        "delay_minutes",
        "risk_category",
        "weather_condition",
    ]

    columns = [
        col for col in columns
        if col in watchlist.columns
    ]

    return watchlist[columns].head(20)


def style_chart(fig):
    """Apply a consistent dark theme to Plotly charts."""

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#122143",
        plot_bgcolor="#122143",
        font=dict(color="#dce8ff"),
        margin=dict(l=25, r=25, t=55, b=25),
        legend=dict(orientation="h"),
    )

    return fig


def get_route_table(data):
    """Summarize flights and delays by route."""

    if data.empty:
        return pd.DataFrame()

    routes = (
        data.groupby(
            ["origin", "destination"],
            dropna=False
        )
        .agg(
            total_flights=("flight_id", "count"),
            delayed_flights=("is_delayed", "sum"),
            avg_delay_min=("delay_minutes", "mean"),
            distance_km=("distance_km", "mean"),
        )
        .reset_index()
    )

    routes["delay_rate"] = (
        routes["delayed_flights"]
        / routes["total_flights"]
        * 100
    ).round(1)

    routes["avg_delay_min"] = (
        routes["avg_delay_min"].round(1)
    )

    routes["distance_km"] = (
        routes["distance_km"].round(1)
    )

    return routes.sort_values(
        "delay_rate", ascending=False
    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("✈️ FlightWise AI")
st.sidebar.caption("Aircraft Delay Risk & Operational Analytics")
st.sidebar.divider()

nav_selection = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Operations Dashboard",
        "🗺️ Route Analytics",
        "🔮 Flight Delay Predictor",
        "ℹ️ Project Architecture",
    ],
)

st.sidebar.divider()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.subheader("🔎 Filter Controls")

selected_airlines = st.sidebar.multiselect(
    "Airlines",
    options=sorted(df["airline"].dropna().unique()),
)

selected_origins = st.sidebar.multiselect(
    "Origin Hubs",
    options=sorted(df["origin"].dropna().unique()),
)

selected_weather = st.sidebar.multiselect(
    "Weather Condition",
    options=sorted(
        df["weather_condition"].dropna().unique()
    ),
)

df_filtered = df.copy()

if selected_airlines:
    df_filtered = df_filtered[
        df_filtered["airline"].isin(selected_airlines)
    ]

if selected_origins:
    df_filtered = df_filtered[
        df_filtered["origin"].isin(selected_origins)
    ]

if selected_weather:
    df_filtered = df_filtered[
        df_filtered["weather_condition"].isin(
            selected_weather
        )
    ]

st.sidebar.caption(
    f"Showing {len(df_filtered):,} of {len(df):,} flights"
)

if st.sidebar.button("Reset Filters", use_container_width=True):
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Phase 2 Prototype • ML-Enabled")
st.sidebar.caption("Academic Data Science Project")


# =========================================================
# PAGE 1: OPERATIONS DASHBOARD
# =========================================================

if nav_selection == "📊 Operations Dashboard":

    st.markdown(
        """
        <div class="hero">
            <h1>FlightWise AI</h1>
            <p>
                Aircraft Delay Risk Prediction & Operational
                Fleet Analytics
            </p>
            <div class="notice">
                ⚠️ Academic prototype using synthetic flight data.
                Results do not represent actual airline operations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_filtered.empty:
        st.warning("No flights match the selected filters.")
        st.stop()

    total_flights = len(df_filtered)

    delayed_flights = int(
        df_filtered["is_delayed"].sum()
    )

    delay_percentage = (
        delayed_flights / total_flights * 100
    )

    average_delay = df_filtered[
        "delay_minutes"
    ].mean()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "TOTAL FLIGHTS MONITORED",
        f"{total_flights:,}",
    )

    col2.metric(
        "DELAYED DEPARTURES",
        f"{delayed_flights:,}",
    )

    col3.metric(
        "FLEET DELAY PERCENTAGE",
        f"{delay_percentage:.1f}%",
    )

    col4.metric(
        "AVERAGE FLIGHT DELAY",
        f"{average_delay:.1f} min",
    )

    st.divider()

    # DAILY DELAY TREND

    st.subheader(
        "📈 Daily Delay Rate and Scheduled Flight Volume"
    )

    daily = (
        df_filtered.groupby("flight_date")
        .agg(
            total_flights=("flight_id", "count"),
            delayed_flights=("is_delayed", "sum"),
        )
        .reset_index()
    )

    daily["delay_rate"] = (
        daily["delayed_flights"]
        / daily["total_flights"]
        * 100
    )

    fig_daily = go.Figure()

    fig_daily.add_trace(
        go.Bar(
            x=daily["flight_date"],
            y=daily["total_flights"],
            name="Scheduled Flights",
            opacity=0.55,
        )
    )

    fig_daily.add_trace(
        go.Scatter(
            x=daily["flight_date"],
            y=daily["delay_rate"],
            name="Delay Rate (%)",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#00d2d3", width=3),
        )
    )

    fig_daily.update_layout(
        template="plotly_dark",
        paper_bgcolor="#122143",
        plot_bgcolor="#122143",
        xaxis_title="Flight Date",
        yaxis_title="Total Flights",
        yaxis2=dict(
            title="Delay Rate (%)",
            overlaying="y",
            side="right",
            rangemode="tozero",
        ),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True,
    )

    # AIRLINE AND DEPARTURE HOUR CHARTS

    left, right = st.columns(2)

    with left:
        st.subheader("Airline Delay Rates (%)")

        airline_summary = (
            df_filtered.groupby("airline")
            .agg(
                total=("flight_id", "count"),
                delayed=("is_delayed", "sum"),
            )
            .reset_index()
        )

        airline_summary["delay_rate"] = (
            airline_summary["delayed"]
            / airline_summary["total"]
            * 100
        )

        airline_summary = airline_summary.sort_values(
            "delay_rate"
        )

        fig_airline = px.bar(
            airline_summary,
            x="delay_rate",
            y="airline",
            orientation="h",
            labels={
                "delay_rate": "Delayed Flights (%)",
                "airline": "Airline",
            },
            text=airline_summary["delay_rate"].round(1),
        )

        fig_airline.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
        )

        st.plotly_chart(
            style_chart(fig_airline),
            use_container_width=True,
        )

    with right:
        st.subheader("Delay Rate by Departure Hour")

        hour_summary = (
            df_filtered.groupby("departure_hour")
            .agg(
                total=("flight_id", "count"),
                delayed=("is_delayed", "sum"),
            )
            .reset_index()
        )

        hour_summary["delay_rate"] = (
            hour_summary["delayed"]
            / hour_summary["total"]
            * 100
        )

        fig_hour = px.bar(
            hour_summary,
            x="departure_hour",
            y="delay_rate",
            labels={
                "departure_hour": "Departure Hour (24h)",
                "delay_rate": "Delay Rate (%)",
            },
        )

        st.plotly_chart(
            style_chart(fig_hour),
            use_container_width=True,
        )

    # ORIGIN AIRPORT AND DELAY REASONS

    left, right = st.columns(2)

    with left:
        st.subheader("Top Origin Airports by Delay Rate")

        airport_summary = (
            df_filtered.groupby("origin")
            .agg(
                total=("flight_id", "count"),
                delayed=("is_delayed", "sum"),
            )
            .reset_index()
        )

        airport_summary["delay_rate"] = (
            airport_summary["delayed"]
            / airport_summary["total"]
            * 100
        )

        airport_summary = (
            airport_summary.sort_values(
                "delay_rate", ascending=False
            ).head(8)
        )

        fig_airport = px.bar(
            airport_summary.sort_values("delay_rate"),
            x="delay_rate",
            y="origin",
            orientation="h",
            labels={
                "origin": "Airport IATA Code",
                "delay_rate": "Delay Rate (%)",
            },
        )

        st.plotly_chart(
            style_chart(fig_airport),
            use_container_width=True,
        )

    with right:
        st.subheader("Primary Delay Reasons")

        if "delay_reason" in df_filtered.columns:
            reasons = (
                df_filtered.loc[
                    df_filtered["is_delayed"],
                    "delay_reason",
                ]
                .fillna("Unknown")
                .replace("", "Unknown")
                .value_counts()
                .reset_index()
            )

            if not reasons.empty:
                reasons.columns = ["reason", "count"]

                fig_reasons = px.pie(
                    reasons,
                    names="reason",
                    values="count",
                    hole=0.48,
                )

                st.plotly_chart(
                    style_chart(fig_reasons),
                    use_container_width=True,
                )
            else:
                st.info("No delayed flights in the filtered data.")
        else:
            st.info("Delay reason information is unavailable.")

    # RECENT FLIGHT LOG

    st.divider()
    st.subheader("📋 Recent Flight Activity Log")

    display_cols = [
        "flight_number",
        "airline",
        "origin",
        "destination",
        "flight_date",
        "departure_time",
        "weather_condition",
        "delay_minutes",
        "is_delayed",
        "delay_reason",
    ]

    available_cols = [
        col for col in display_cols
        if col in df_filtered.columns
    ]

    st.dataframe(
        df_filtered[available_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )

    # HIGH-RISK WATCHLIST

    st.divider()
    st.subheader("✈️ High-Risk Flight Watchlist")

    watchlist = get_high_risk_watchlist(df_filtered)

    if watchlist.empty:
        st.info(
            "No flights with recorded delays of 15 minutes "
            "or more were found."
        )
    else:
        st.caption(
            "This watchlist uses recorded delays in the dataset. "
            "It is not a future prediction."
        )

        st.dataframe(
            watchlist,
            use_container_width=True,
            hide_index=True,
        )

    # DOWNLOAD FILTERED DATA

    st.divider()

    st.download_button(
        "⬇️ Download Filtered Flight Data",
        data=df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="flightwise_filtered_data.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 2: ROUTE ANALYTICS
# =========================================================

elif nav_selection == "🗺️ Route Analytics":

    st.title("🗺️ Route Analytics")

    st.caption(
        "Analyze flight routes, route distance, and recorded delays."
    )

    if df_filtered.empty:
        st.warning("No flights match the selected filters.")
        st.stop()

    routes_df = get_route_table(df_filtered)

    st.subheader("Route Performance Summary")

    st.dataframe(
        routes_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("✈️ Airport Network Map")

    map_data = df_filtered.dropna(
        subset=[
            "origin_lat",
            "origin_lon",
            "dest_lat",
            "dest_lon",
        ]
    )

    if map_data.empty:
        st.info("Airport coordinate data is unavailable.")
    else:
        # Plot each flight route as a line between airports.
        fig_map = go.Figure()

        shown_routes = map_data.head(150)

        for _, row in shown_routes.iterrows():
            fig_map.add_trace(
                go.Scattergeo(
                    lon=[row["origin_lon"], row["dest_lon"]],
                    lat=[row["origin_lat"], row["dest_lat"]],
                    mode="lines",
                    line=dict(
                        width=1,
                        color="#00d2d3",
                    ),
                    opacity=0.35,
                    showlegend=False,
                    hovertemplate=(
                        f'{row["origin"]} → {row["destination"]}'
                        "<extra></extra>"
                    ),
                )
            )

        airports_origin = map_data[
            ["origin", "origin_city", "origin_lat", "origin_lon"]
        ].rename(
            columns={
                "origin": "airport",
                "origin_city": "city",
                "origin_lat": "lat",
                "origin_lon": "lon",
            }
        )

        airports_dest = map_data[
            ["destination", "destination_city", "dest_lat", "dest_lon"]
        ].rename(
            columns={
                "destination": "airport",
                "destination_city": "city",
                "dest_lat": "lat",
                "dest_lon": "lon",
            }
        )

        airport_points = pd.concat(
            [airports_origin, airports_dest],
            ignore_index=True,
        ).drop_duplicates(subset=["airport"])

        fig_map.add_trace(
            go.Scattergeo(
                lon=airport_points["lon"],
                lat=airport_points["lat"],
                text=airport_points["airport"],
                customdata=airport_points[["city"]],
                mode="markers+text",
                textposition="top center",
                marker=dict(
                    size=7,
                    color="#ffb84d",
                ),
                name="Airports",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "%{customdata[0]}<extra></extra>"
                ),
            )
        )

        fig_map.update_layout(
            template="plotly_dark",
            paper_bgcolor="#122143",
            geo=dict(
                bgcolor="#122143",
                showland=True,
                landcolor="#1b2d4c",
                showocean=True,
                oceancolor="#0b1228",
                showcountries=True,
            ),
            margin=dict(l=10, r=10, t=20, b=10),
            height=600,
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True,
        )

        st.caption(
            "Map displays up to 150 routes from the selected records."
        )

    st.subheader("Route Delay Rate Comparison")

    if not routes_df.empty:
        top_routes = routes_df.head(15).copy()

        top_routes["route"] = (
            top_routes["origin"]
            + " → "
            + top_routes["destination"]
        )

        fig_routes = px.bar(
            top_routes.sort_values("delay_rate"),
            x="delay_rate",
            y="route",
            orientation="h",
            labels={
                "delay_rate": "Delay Rate (%)",
                "route": "Flight Route",
            },
        )

        st.plotly_chart(
            style_chart(fig_routes),
            use_container_width=True,
        )


# =========================================================
# PAGE 3: FLIGHT DELAY PREDICTOR
# =========================================================

elif nav_selection == "🔮 Flight Delay Predictor":

    st.title("🔮 Flight Delay Risk Predictor")

    st.write(
        "Enter planned flight and weather details to estimate "
        "the flight's delay risk."
    )

    st.warning(
        "Predictions are estimates from your current model and "
        "are not guarantees of actual flight delays."
    )

    airline_options = sorted(
        AIRLINES.keys()
    )

    airport_options = sorted(
        AIRPORTS.keys()
    )

    weather_options = sorted(
        WEATHER_CONDITIONS.keys()
    )

    if not airline_options or len(airport_options) < 2:
        st.error(
            "Airline or airport options are missing. "
            "Check src/generate_data.py."
        )
        st.stop()

    with st.form("flight_prediction_form"):

        left, right = st.columns(2)

        with left:
            st.subheader("✈️ Flight Itinerary")

            airline = st.selectbox(
                "Operating Airline",
                airline_options,
            )

            origin = st.selectbox(
                "Origin Airport",
                airport_options,
            )

            possible_destinations = [
                airport for airport in airport_options
                if airport != origin
            ]

            destination = st.selectbox(
                "Destination Airport",
                possible_destinations,
            )

            departure_hour = st.slider(
                "Scheduled Departure Hour (24h)",
                min_value=0,
                max_value=23,
                value=17,
            )

        with right:
            st.subheader("🌧️ Weather Conditions")

            weather_condition = st.selectbox(
                "Weather Condition",
                weather_options,
            )

            wind_speed_kmh = st.slider(
                "Surface Wind Speed (km/h)",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
            )

            visibility_km = st.slider(
                "Visibility (km)",
                min_value=0.1,
                max_value=20.0,
                value=10.0,
                step=0.5,
            )

        submitted = st.form_submit_button(
            "⚡ Assess Flight Delay Risk",
            use_container_width=True,
        )

    if submitted:

        try:
            result = predict_flight_risk(
                airline=airline,
                origin=origin,
                destination=destination,
                departure_hour=departure_hour,
                weather_condition=weather_condition,
                wind_speed_kmh=wind_speed_kmh,
                visibility_km=visibility_km,
            )

            if result.get("error"):
                st.error(result["error"])
                st.stop()

            st.divider()
            st.subheader("📊 Operational Risk Assessment")

            risk_score = result.get("risk_score", 0)
            risk_level = result.get("risk_level", "Unknown")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Delay Probability / Risk Score",
                f"{risk_score:.1f}%",
            )

            col2.metric(
                "Risk Category",
                risk_level,
            )

            col3.metric(
                "Estimated Delay Range",
                result.get("estimated_delay", "Unavailable"),
            )

            st.progress(
                max(0.0, min(1.0, float(risk_score) / 100))
            )

            if risk_score >= 75:
                st.error("🔴 Critical delay risk")
            elif risk_score >= 50:
                st.warning("🟠 High delay risk")
            elif risk_score >= 25:
                st.info("🟡 Moderate delay risk")
            else:
                st.success("🟢 Low delay risk")

            distance = result.get("distance_km")

            if distance is not None:
                st.metric(
                    "Great-Circle Distance",
                    f"{distance:,.1f} km",
                )

            st.subheader("⚠️ Key Risk Drivers Identified")

            risk_factors = result.get("risk_factors", [])

            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("No specific risk factors were returned.")

            st.subheader("💡 Operational Recommendation")

            st.info(
                result.get(
                    "recommendation",
                    "Continue routine operational monitoring.",
                )
            )

            st.caption(
                "Model engine: "
                + str(
                    result.get(
                        "model_version",
                        "Current configured prediction engine",
                    )
                )
            )

        except Exception as error:
            st.error(
                f"Prediction could not be completed: {error}"
            )


# =========================================================
# PAGE 4: PROJECT ARCHITECTURE
# =========================================================

elif nav_selection == "ℹ️ Project Architecture":

    st.title("ℹ️ FlightWise AI — Project Architecture")

    st.write(
        "FlightWise AI combines flight data processing, "
        "machine learning, analytics, and a Streamlit dashboard."
    )

    st.subheader("System Workflow")

    st.markdown(
        """
        ```text
        Synthetic Flight Dataset (CSV)
                    |
                    v
          Data Loading & Validation
                    |
                    v
            Data Preprocessing
                    |
          +---------+----------+
          |                    |
          v                    v
        Analytics          Trained ML Model
          |                    |
          v                    v
        Charts & KPIs      Delay Risk Prediction
          |                    |
          +---------+----------+
                    |
                    v
          Streamlit User Interface
                    |
                    v
        Dashboard / Route Analytics /
             Flight Predictor
        ```
        """
    )

    st.subheader("Technology Stack")

    stack = pd.DataFrame(
        {
            "Component": [
                "Programming Language",
                "Dashboard",
                "Data Processing",
                "Machine Learning",
                "Visualization",
                "Dataset",
                "Model Storage",
            ],
            "Technology": [
                "Python",
                "Streamlit",
                "Pandas",
                "Scikit-learn",
                "Plotly",
                "synthetic_flights.csv",
                "Joblib",
            ],
        }
    )

    st.dataframe(
        stack,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Project Limitations")

    st.markdown(
        """
        - The current flight dataset is synthetic.
        - Model results depend on the quality of the training data.
        - Predictions are estimates and may not reflect real flight conditions.
        - Live weather feeds and real-time airline data are not connected.
        - A RAG-based aviation document assistant can be added in a future phase.
        """
    )

    st.subheader("Current Dataset Summary")

    st.write(f"**Total records:** {len(df):,}")

    st.write(
        "**Airlines:** "
        + str(df["airline"].nunique())
    )

    st.write(
        "**Origin airports:** "
        + str(df["origin"].nunique())
    )

    st.write(
        "**Destination airports:** "
        + str(df["destination"].nunique())
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "FlightWise AI | Academic Prototype | "
    "Synthetic data demonstration — not for operational flight decisions"
)
