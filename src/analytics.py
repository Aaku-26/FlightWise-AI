"""
src/analytics.py
----------------
FlightWise AI: Aircraft Delay Risk Prediction & Operational Intelligence

PURPOSE:
Provides statistical calculations, KPI summaries, and aggregations for:
- Operational KPIs (Total, Delayed, Delay %, Avg Delay)
- Time-series delay trends
- Airline operational performance
- Hourly departure delay profiles
- Airport delay rankings
- Route-level analytics
"""

import pandas as pd
import numpy as np

def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Compute operational Key Performance Indicators (KPIs).
    
    Returns:
        dict: High-level metrics for dashboard cards.
    """
    total_flights = len(df)
    if total_flights == 0:
        return {
            "total_flights": 0,
            "delayed_flights": 0,
            "on_time_flights": 0,
            "delay_percentage": 0.0,
            "avg_delay_minutes": 0.0,
            "avg_delay_delayed_only": 0.0,
            "max_delay_minutes": 0,
        }

    delayed_mask = df["is_delayed"] == True
    delayed_flights = int(delayed_mask.sum())
    on_time_flights = total_flights - delayed_flights
    delay_percentage = round((delayed_flights / total_flights) * 100.0, 1)
    avg_delay_minutes = round(float(df["delay_minutes"].mean()), 1)
    
    # Average delay among delayed flights only
    delayed_subset = df[delayed_mask]
    avg_delay_delayed_only = round(float(delayed_subset["delay_minutes"].mean()), 1) if not delayed_subset.empty else 0.0
    max_delay = int(df["delay_minutes"].max())

    return {
        "total_flights": total_flights,
        "delayed_flights": delayed_flights,
        "on_time_flights": on_time_flights,
        "delay_percentage": delay_percentage,
        "avg_delay_minutes": avg_delay_minutes,
        "avg_delay_delayed_only": avg_delay_delayed_only,
        "max_delay_minutes": max_delay,
    }

def get_delay_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate flight delays by date to track performance over time.
    """
    if df.empty:
        return pd.DataFrame(columns=["flight_date", "total_flights", "delayed_flights", "delay_rate", "avg_delay_min"])

    # Safely convert to date whether datetime or string
    flight_dates = pd.to_datetime(df["flight_date"]).dt.date
    grouped = df.groupby(flight_dates).agg(
        total_flights=("flight_id", "count"),
        delayed_flights=("is_delayed", lambda x: int(x.sum())),
        avg_delay_min=("delay_minutes", "mean")
    ).reset_index()

    grouped["delay_rate"] = ((grouped["delayed_flights"] / grouped["total_flights"]) * 100.0).round(1)
    grouped["avg_delay_min"] = grouped["avg_delay_min"].round(1)
    grouped = grouped.rename(columns={"flight_date": "date"})
    return grouped.sort_values(by="date")

def get_airline_delays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare delay performance across different airlines.
    """
    if df.empty:
        return pd.DataFrame(columns=["airline", "airline_code", "total_flights", "delayed_flights", "delay_rate", "avg_delay_min"])

    grouped = df.groupby(["airline", "airline_code"]).agg(
        total_flights=("flight_id", "count"),
        delayed_flights=("is_delayed", lambda x: int(x.sum())),
        avg_delay_min=("delay_minutes", "mean")
    ).reset_index()

    grouped["delay_rate"] = ((grouped["delayed_flights"] / grouped["total_flights"]) * 100.0).round(1)
    grouped["avg_delay_min"] = grouped["avg_delay_min"].round(1)
    return grouped.sort_values(by="delay_rate", ascending=False)

def get_hourly_delays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate delays by scheduled hour of departure (0-23).
    """
    if df.empty:
        return pd.DataFrame(columns=["departure_hour", "total_flights", "delayed_flights", "delay_rate", "avg_delay_min"])

    grouped = df.groupby("departure_hour").agg(
        total_flights=("flight_id", "count"),
        delayed_flights=("is_delayed", lambda x: int(x.sum())),
        avg_delay_min=("delay_minutes", "mean")
    ).reset_index()

    grouped["delay_rate"] = ((grouped["delayed_flights"] / grouped["total_flights"]) * 100.0).round(1)
    grouped["avg_delay_min"] = grouped["avg_delay_min"].round(1)
    
    # Label hours nicely (e.g. 0 -> "00:00", 14 -> "14:00")
    grouped["hour_label"] = grouped["departure_hour"].apply(lambda h: f"{h:02d}:00")
    return grouped.sort_values(by="departure_hour")

def get_airport_delays(df: pd.DataFrame, airport_col: str = "origin") -> pd.DataFrame:
    """
    Rank airports by delay percentage.
    """
    if df.empty:
        return pd.DataFrame(columns=["airport", "airport_city", "total_flights", "delayed_flights", "delay_rate", "avg_delay_min"])

    city_col = "origin_city" if airport_col == "origin" else "destination_city"
    grouped = df.groupby([airport_col, city_col]).agg(
        total_flights=("flight_id", "count"),
        delayed_flights=("is_delayed", lambda x: int(x.sum())),
        avg_delay_min=("delay_minutes", "mean")
    ).reset_index()

    grouped["delay_rate"] = ((grouped["delayed_flights"] / grouped["total_flights"]) * 100.0).round(1)
    grouped["avg_delay_min"] = grouped["avg_delay_min"].round(1)
    grouped = grouped.rename(columns={airport_col: "airport", city_col: "airport_city"})
    return grouped.sort_values(by="delay_rate", ascending=False)

def get_route_analytics(df: pd.DataFrame, min_flights: int = 5) -> pd.DataFrame:
    """
    Aggregate performance at the flight route level.
    """
    if df.empty:
        return pd.DataFrame(columns=["route", "origin", "destination", "total_flights", "delayed_flights", "delay_rate", "avg_delay_min", "distance_km"])

    grouped = df.groupby(["route", "origin", "destination", "origin_lat", "origin_lon", "dest_lat", "dest_lon"]).agg(
        total_flights=("flight_id", "count"),
        delayed_flights=("is_delayed", lambda x: int(x.sum())),
        avg_delay_min=("delay_minutes", "mean"),
        distance_km=("distance_km", "first")
    ).reset_index()

    # Filter for meaningful sample size
    grouped = grouped[grouped["total_flights"] >= min_flights]
    grouped["delay_rate"] = ((grouped["delayed_flights"] / grouped["total_flights"]) * 100.0).round(1)
    grouped["avg_delay_min"] = grouped["avg_delay_min"].round(1)
    return grouped.sort_values(by="delay_rate", ascending=False)

def get_delay_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate counts of delay causes for delayed flights.
    """
    delayed = df[df["is_delayed"] == True]
    if delayed.empty:
        return pd.DataFrame(columns=["delay_reason", "count", "percentage"])

    counts = delayed["delay_reason"].value_counts().reset_index()
    counts.columns = ["delay_reason", "count"]
    counts["percentage"] = ((counts["count"] / counts["count"].sum()) * 100.0).round(1)
    return counts
