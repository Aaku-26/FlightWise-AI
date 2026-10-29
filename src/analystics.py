
import pandas as pd

def get_high_risk_watchlist(df):
    """Return flights with the highest recorded delays."""

    required = [
        "flight_number",
        "airline",
        "origin",
        "destination",
        "departure_time",
        "delay_minutes",
        "weather_condition"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame()

    watchlist = df.copy()

    # Use recorded delay as a simple historical risk indicator.
    # This is not a model-predicted risk score.
    watchlist["Risk Level"] = watchlist["delay_minutes"].apply(
        lambda delay: (
            "Critical" if delay >= 70 else
            "High" if delay >= 35 else
            "Moderate" if delay >= 15 else
            "Low"
        )
    )

    watchlist = watchlist.sort_values(
        "delay_minutes", ascending=False
    ).head(10)

    watchlist = watchlist.rename(columns={
        "flight_number": "Flight Number",
        "airline": "Airline",
        "origin": "Origin",
        "destination": "Destination",
        "departure_time": "Departure Time",
        "delay_minutes": "Recorded Delay (min)",
        "weather_condition": "Weather"
    })

    return watchlist[
        [
            "Flight Number", "Airline", "Origin", "Destination",
            "Departure Time", "Recorded Delay (min)",
            "Risk Level", "Weather"
        ]
    ]
