"""
src/data_loader.py
------------------
FlightWise AI: Aircraft Delay Risk Prediction & Operational Intelligence

PURPOSE:
Loads, validates, and cleans the aviation dataset.
Provides caching and robust fallback generation if the dataset file is absent.
"""

import os
import pandas as pd
import streamlit as st
from datetime import datetime

# Expected columns for Phase 1 dataset
REQUIRED_COLUMNS = [
    "flight_id", "flight_number", "airline", "airline_code",
    "origin", "origin_city", "origin_lat", "origin_lon",
    "destination", "destination_city", "dest_lat", "dest_lon",
    "route", "flight_date", "departure_hour", "departure_time",
    "scheduled_duration_min", "distance_km", "weather_condition",
    "visibility_km", "wind_speed_kmh", "delay_minutes",
    "is_delayed", "delay_reason", "data_source"
]

def _read_and_process_csv(filepath: str) -> pd.DataFrame:
    """Read CSV from disk and cast data types safely."""
    df = pd.read_csv(filepath)
    
    # Verify required schema
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    # Standardize data types
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df["departure_hour"] = df["departure_hour"].astype(int)
    df["delay_minutes"] = df["delay_minutes"].astype(int)
    df["is_delayed"] = df["is_delayed"].astype(bool)
    df["distance_km"] = df["distance_km"].astype(float)
    df["visibility_km"] = df["visibility_km"].astype(float)
    df["wind_speed_kmh"] = df["wind_speed_kmh"].astype(float)
    
    return df

@st.cache_data(show_spinner=False)
def load_flight_data(filepath: str = "data/synthetic_flights.csv") -> pd.DataFrame:
    """
    Load aviation dataset with Streamlit caching.
    Automatically generates data if the CSV file does not exist.
    
    Parameters:
        filepath (str): Path to the flights CSV file.
        
    Returns:
        pd.DataFrame: Cleaned and validated flights DataFrame.
    """
    if not os.path.exists(filepath):
        # Graceful fallback: generate synthetic data on demand
        from src.generate_data import save_synthetic_data
        save_synthetic_data(output_path=filepath)
        
    return _read_and_process_csv(filepath)

def load_flight_data_raw(filepath: str = "data/synthetic_flights.csv") -> pd.DataFrame:
    """
    Load dataset without Streamlit cache (useful for automated testing or CLI scripts).
    """
    if not os.path.exists(filepath):
        from src.generate_data import save_synthetic_data
        save_synthetic_data(output_path=filepath)
        
    return _read_and_process_csv(filepath)

def get_unique_filter_options(df: pd.DataFrame) -> dict:
    """
    Extract unique options for sidebar dropdowns and selectors.
    """
    dates = pd.to_datetime(df["flight_date"])
    return {
        "airlines": sorted(df["airline"].unique().tolist()),
        "airports": sorted(list(set(df["origin"].unique().tolist() + df["destination"].unique().tolist()))),
        "weather_conditions": sorted(df["weather_condition"].unique().tolist()),
        "min_date": dates.min().date(),
        "max_date": dates.max().date(),
    }

def filter_flight_data(
    df: pd.DataFrame,
    selected_airlines: list = None,
    selected_origins: list = None,
    selected_destinations: list = None,
    selected_weather: list = None,
    start_date = None,
    end_date = None,
) -> pd.DataFrame:
    """
    Filter dataset based on user selections from the dashboard sidebar.
    """
    filtered = df.copy()

    if selected_airlines:
        filtered = filtered[filtered["airline"].isin(selected_airlines)]

    if selected_origins:
        filtered = filtered[filtered["origin"].isin(selected_origins)]

    if selected_destinations:
        filtered = filtered[filtered["destination"].isin(selected_destinations)]

    if selected_weather:
        filtered = filtered[filtered["weather_condition"].isin(selected_weather)]

    if start_date or end_date:
        flight_dates = pd.to_datetime(filtered["flight_date"]).dt.date
        if start_date:
            filtered = filtered[flight_dates >= start_date]
        if end_date:
            filtered = filtered[flight_dates <= end_date]

    return filtered
