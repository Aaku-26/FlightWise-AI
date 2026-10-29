"""
src/generate_data.py
--------------------
FlightWise AI: Aircraft Delay Risk Prediction & Operational Intelligence

PURPOSE:
Generates a realistic, clearly labeled synthetic dataset of flight records
for Phase 1 development and exploratory analytics.

DISCLAIMER:
This dataset is purely synthetic mock data generated for educational and
demonstration purposes. It does not contain actual FAA/airline proprietary data.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# List of major commercial airlines
AIRLINES = {
    "Delta Air Lines": {"code": "DL", "base_delay_rate": 0.16},
    "American Airlines": {"code": "AA", "base_delay_rate": 0.22},
    "United Airlines": {"code": "UA", "base_delay_rate": 0.20},
    "Southwest Airlines": {"code": "WN", "base_delay_rate": 0.23},
    "JetBlue Airways": {"code": "B6", "base_delay_rate": 0.26},
    "Alaska Airlines": {"code": "AS", "base_delay_rate": 0.14},
    "SkyWest Airlines": {"code": "OO", "base_delay_rate": 0.19},
}

# Major hub airports with realistic coordinates
AIRPORTS = {
    "ATL": {"name": "Hartsfield-Jackson Atlanta", "city": "Atlanta, GA", "lat": 33.6407, "lon": -84.4277, "congestion": 1.2},
    "ORD": {"name": "Chicago O'Hare", "city": "Chicago, IL", "lat": 41.9742, "lon": -87.9073, "congestion": 1.35},
    "DFW": {"name": "Dallas/Fort Worth", "city": "Dallas, TX", "lat": 32.8998, "lon": -97.0403, "congestion": 1.15},
    "DEN": {"name": "Denver International", "city": "Denver, CO", "lat": 39.8561, "lon": -104.6737, "congestion": 1.25},
    "LAX": {"name": "Los Angeles International", "city": "Los Angeles, CA", "lat": 33.9416, "lon": -118.4085, "congestion": 1.20},
    "JFK": {"name": "John F. Kennedy International", "city": "New York, NY", "lat": 40.6413, "lon": -73.7781, "congestion": 1.40},
    "SFO": {"name": "San Francisco International", "city": "San Francisco, CA", "lat": 37.6213, "lon": -122.3790, "congestion": 1.30},
    "SEA": {"name": "Seattle-Tacoma", "city": "Seattle, WA", "lat": 47.4502, "lon": -122.3088, "congestion": 1.10},
    "MIA": {"name": "Miami International", "city": "Miami, FL", "lat": 25.7959, "lon": -80.2870, "congestion": 1.15},
    "BOS": {"name": "Boston Logan", "city": "Boston, MA", "lat": 42.3656, "lon": -71.0096, "congestion": 1.25},
}

# Weather scenarios and impact weights
WEATHER_CONDITIONS = {
    "Clear": {"prob": 0.48, "delay_factor": 0.0, "min_vis": 9.0, "max_vis": 10.0, "wind_range": (5, 20)},
    "Cloudy": {"prob": 0.22, "delay_factor": 0.1, "min_vis": 7.0, "max_vis": 10.0, "wind_range": (10, 30)},
    "Rain": {"prob": 0.15, "delay_factor": 0.45, "min_vis": 3.0, "max_vis": 7.0, "wind_range": (20, 45)},
    "Fog": {"prob": 0.06, "delay_factor": 0.65, "min_vis": 0.5, "max_vis": 2.5, "wind_range": (5, 18)},
    "Thunderstorm": {"prob": 0.05, "delay_factor": 0.85, "min_vis": 1.0, "max_vis": 5.0, "wind_range": (35, 75)},
    "Snow": {"prob": 0.04, "delay_factor": 0.80, "min_vis": 0.8, "max_vis": 4.0, "wind_range": (20, 55)},
}

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine formula to approximate distance between airports in kilometers."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return round(float(r * c), 1)

def generate_flight_data(num_records: int = 3500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic flight records with realistic aviation patterns.
    
    Parameters:
        num_records (int): Total number of flight rows to generate.
        random_seed (int): Seed for reproducible data generation.
        
    Returns:
        pd.DataFrame: A populated pandas DataFrame.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    records = []
    base_date = datetime.now() - timedelta(days=45)
    airport_codes = list(AIRPORTS.keys())
    airline_names = list(AIRLINES.keys())
    weather_types = list(WEATHER_CONDITIONS.keys())
    raw_weather_weights = np.array([WEATHER_CONDITIONS[w]["prob"] for w in weather_types], dtype=float)
    weather_weights = raw_weather_weights / np.sum(raw_weather_weights)

    # Hour probability distribution: peak morning (7-9) and evening (16-19)
    raw_hour_weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.03,  # 00:00 - 05:00
        0.06, 0.08, 0.08, 0.07, 0.05, 0.05,  # 06:00 - 11:00
        0.05, 0.05, 0.05, 0.06, 0.08, 0.08,  # 12:00 - 17:00
        0.07, 0.06, 0.04, 0.03, 0.02, 0.01   # 18:00 - 23:00
    ], dtype=float)
    hour_weights = raw_hour_weights / np.sum(raw_hour_weights)

    for i in range(num_records):
        flight_id = f"FW-{10000 + i}"
        airline = random.choice(airline_names)
        airline_code = AIRLINES[airline]["code"]
        flight_number = f"{airline_code}-{random.randint(100, 999)}"

        # Pick distinct origin and destination
        origin = random.choice(airport_codes)
        dest_candidates = [a for a in airport_codes if a != origin]
        destination = random.choice(dest_candidates)

        orig_data = AIRPORTS[origin]
        dest_data = AIRPORTS[destination]
        distance = calculate_distance(orig_data["lat"], orig_data["lon"], dest_data["lat"], dest_data["lon"])

        # Flight duration: approx 80 km/h taxi + 750 km/h cruising speed + 30 min buffer
        scheduled_duration = int(35 + (distance / 750.0) * 60)

        # Scheduled date and departure hour
        day_offset = random.randint(0, 44)
        flight_date = base_date + timedelta(days=day_offset)
        dep_hour = int(np.random.choice(range(24), p=hour_weights))
        dep_minute = random.choice([0, 10, 15, 20, 25, 30, 40, 45, 50, 55])
        departure_time = f"{dep_hour:02d}:{dep_minute:02d}"

        # Weather condition
        weather = np.random.choice(weather_types, p=weather_weights)
        w_info = WEATHER_CONDITIONS[weather]
        visibility = round(random.uniform(w_info["min_vis"], w_info["max_vis"]), 1)
        wind_speed = round(random.uniform(w_info["wind_range"][0], w_info["wind_range"][1]), 1)

        # Realistic delay probability calculation
        base_rate = AIRLINES[airline]["base_delay_rate"]
        congestion_mult = (orig_data["congestion"] + dest_data["congestion"]) / 2.0
        time_mult = 1.35 if (15 <= dep_hour <= 20) else (1.15 if (7 <= dep_hour <= 10) else 0.85)
        weather_boost = w_info["delay_factor"]

        delay_probability = min(0.92, (base_rate * congestion_mult * time_mult) + weather_boost * 0.45)
        is_delayed_bool = (random.random() < delay_probability)

        if is_delayed_bool:
            # Right-skewed delay distribution (most are 15-45 min, some 60-180 min)
            skewed_delay = np.random.exponential(scale=35.0) + 15.0
            delay_minutes = int(min(360, skewed_delay))
            
            # Attribute primary reason
            if weather in ["Thunderstorm", "Snow", "Fog"] and random.random() < 0.70:
                delay_reason = "Weather Delay"
            elif congestion_mult > 1.25 and random.random() < 0.55:
                delay_reason = "Air Traffic Control"
            elif dep_hour >= 17 and random.random() < 0.60:
                delay_reason = "Late Aircraft Inbound"
            else:
                delay_reason = "Carrier / Maintenance"
        else:
            # Flight is on-time or early (-15 to 14 min)
            delay_minutes = int(random.choice([-10, -5, -3, 0, 0, 2, 5, 8, 12]))
            delay_reason = "On-Time / Early"

        # Official FAA standard: delayed if delay >= 15 minutes
        official_delayed = (delay_minutes >= 15)

        records.append({
            "flight_id": flight_id,
            "flight_number": flight_number,
            "airline": airline,
            "airline_code": airline_code,
            "origin": origin,
            "origin_city": orig_data["city"],
            "origin_lat": orig_data["lat"],
            "origin_lon": orig_data["lon"],
            "destination": destination,
            "destination_city": dest_data["city"],
            "dest_lat": dest_data["lat"],
            "dest_lon": dest_data["lon"],
            "route": f"{origin} ➔ {destination}",
            "flight_date": flight_date.strftime("%Y-%m-%d"),
            "departure_hour": dep_hour,
            "departure_time": departure_time,
            "scheduled_duration_min": scheduled_duration,
            "distance_km": distance,
            "weather_condition": weather,
            "visibility_km": visibility,
            "wind_speed_kmh": wind_speed,
            "delay_minutes": delay_minutes,
            "is_delayed": official_delayed,
            "delay_reason": delay_reason,
            "data_source": "SYNTHETIC_SAMPLE_DATA"
        })

    df = pd.DataFrame(records)
    # Sort chronologically
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df = df.sort_values(by=["flight_date", "departure_hour"]).reset_index(drop=True)
    df["flight_date"] = df["flight_date"].dt.strftime("%Y-%m-%d")
    return df

def save_synthetic_data(output_path: str = "data/synthetic_flights.csv", num_records: int = 3500) -> str:
    """
    Generate and save synthetic flights data to a CSV file.
    
    Parameters:
        output_path (str): Relative or absolute path for the output CSV.
        num_records (int): Number of synthetic rows.
        
    Returns:
        str: Absolute path of saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df = generate_flight_data(num_records=num_records)
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(df)} synthetic flight records saved to: {output_path}")
    return os.path.abspath(output_path)

if __name__ == "__main__":
    save_synthetic_data()
