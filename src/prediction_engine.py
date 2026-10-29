"""
src/prediction_engine.py
------------------------
FlightWise AI: Aircraft Delay Risk Prediction & Operational Intelligence

PURPOSE:
Phase 1 Delay Risk Assessment Engine.

NOTE ON MACHINE LEARNING ARCHITECTURE:
In Phase 1, this module uses a domain-grounded aviation heuristic risk scoring model
to calculate delay probability, risk levels, and contributing factors based on
historical airline performance, airport congestion indices, time-of-day curves,
and meteorological severity.

In Phase 2, this module will seamlessly load a trained Scikit-Learn model
(e.g., RandomForestClassifier, GradientBoostingClassifier, or XGBoost)
via the provided `MLModelWrapper` interface.
"""

import os
from typing import Dict, Any, List
from src.generate_data import AIRLINES, AIRPORTS, WEATHER_CONDITIONS, calculate_distance

class DelayRiskEngine:
    """
    Flight Delay Risk Assessment Engine for operational intelligence.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.ml_model = None
        self.is_ml_active = False

        # Attempt to load trained scikit-learn model if available (Phase 2 hook)
        if model_path and os.path.exists(model_path):
            self._load_scikit_model(model_path)

    def _load_scikit_model(self, path: str):
        """Phase 2 hook to load trained scikit-learn model pipeline."""
        try:
            import joblib
            self.ml_model = joblib.load(path)
            self.is_ml_active = True
            print(f"[FlightWise AI] Loaded Scikit-Learn Model from {path}")
        except Exception as e:
            print(f"[FlightWise AI] Could not load ML model: {e}. Falling back to Phase 1 Risk Engine.")
            self.is_ml_active = False

    def predict(
        self,
        airline: str,
        origin: str,
        destination: str,
        departure_hour: int,
        weather_condition: str,
        wind_speed_kmh: float = 15.0,
        visibility_km: float = 10.0
    ) -> Dict[str, Any]:
        """
        Assess aircraft delay risk based on operational parameters.

        Parameters:
            airline (str): Full airline name (e.g. 'Delta Air Lines')
            origin (str): Origin airport IATA code (e.g. 'JFK')
            destination (str): Destination airport IATA code (e.g. 'LAX')
            departure_hour (int): Scheduled hour of departure (0-23)
            weather_condition (str): 'Clear', 'Cloudy', 'Rain', 'Fog', 'Thunderstorm', 'Snow'
            wind_speed_kmh (float): Wind speed in km/h
            visibility_km (float): Surface visibility in km

        Returns:
            dict containing risk_score, risk_level, estimated_delay, risk_factors, and recommendations.
        """
        # If origin == destination, guard check
        if origin == destination:
            return {
                "error": "Origin and destination airports must be different.",
                "risk_score": 0.0,
                "risk_level": "Invalid",
            }
            
        # Phase 2: Use trained ML model when available
        if self.ml_model is not None and self.is_ml_active:
            import pandas as pd

            features = pd.DataFrame([{
                "airline": airline,
                "airline_code": AIRLINES.get(
                    airline, {}
                ).get("code", "Unknown"),
                "origin": origin,
                "destination": destination,
                "weather_condition": weather_condition,
                "departure_hour": departure_hour,
                "scheduled_duration_min": 120,
                "distance_km": calculate_distance(
                    AIRPORTS.get(origin, {}).get("lat", 0),
                    AIRPORTS.get(origin, {}).get("lon", 0),
                    AIRPORTS.get(destination, {}).get("lat", 0),
                    AIRPORTS.get(destination, {}).get("lon", 0),
                ),
                "visibility_km": visibility_km,
                "wind_speed_kmh": wind_speed_kmh,
            }])

            probability = float(
                self.ml_model.predict_proba(features)[0][1]
            )
            delay_probability = round(probability * 100, 1)

            if delay_probability < 25:
                risk_level = "Low Risk"
                risk_color = "#1DD1A1"
                badge = "🟢"
                estimated_delay = "0 - 15 min"
            elif delay_probability < 50:
                risk_level = "Moderate Risk"
                risk_color = "#FECA57"
                badge = "🟡"
                estimated_delay = "15 - 35 min"
            elif delay_probability < 75:
                risk_level = "High Risk"
                risk_color = "#FF9F43"
                badge = "🟠"
                estimated_delay = "35 - 70 min"
            else:
                risk_level = "Critical Delay Risk"
                risk_color = "#FF6B6B"
                badge = "🔴"
                estimated_delay = "70+ min"

            risk_factors = []
            if weather_condition in ["Thunderstorm", "Snow", "Fog", "Rain"]:
                risk_factors.append(
                    f"Weather condition: {weather_condition}"
                )
            if wind_speed_kmh > 40:
                risk_factors.append("High wind speed")
            if visibility_km < 4:
                risk_factors.append("Low visibility")
            if 15 <= departure_hour <= 20:
                risk_factors.append("Departure during evening peak hours")
            if not risk_factors:
                risk_factors.append(
                    "No major risk factors identified by these basic checks."
                )

            if delay_probability >= 50:
                recommendation = (
                    "Monitor weather and airport operations; "
                    "prepare for a possible delay."
                )
            elif delay_probability >= 25:
                recommendation = (
                    "Monitor the flight status and local weather conditions."
                )
            else:
                recommendation = (
                    "Current model estimate indicates relatively low delay risk."
                )

            return {
                "risk_score": delay_probability,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "badge": badge,
                "estimated_delay": estimated_delay,
                "risk_factors": risk_factors,
                "recommendation": recommendation,
                "distance_km": float(features["distance_km"].iloc[0]),
                "model_version": "Phase 2 - Random Forest Classifier",
            }


        # 1. Base Airline Performance Factor
        airline_data = AIRLINES.get(airline, {"base_delay_rate": 0.20})
        airline_score = airline_data["base_delay_rate"] * 100.0  # e.g. 16% to 26%

        # 2. Airport Congestion Factor (Origin & Destination)
        orig_data = AIRPORTS.get(origin, {"congestion": 1.15, "lat": 0, "lon": 0, "city": origin})
        dest_data = AIRPORTS.get(destination, {"congestion": 1.15, "lat": 0, "lon": 0, "city": destination})
        avg_congestion = (orig_data["congestion"] + dest_data["congestion"]) / 2.0
        congestion_penalty = (avg_congestion - 1.0) * 35.0  # -0 to +15%

        # 3. Time-of-Day Traffic Surge Factor
        # Aviation statistics: Early morning (05:00-08:00) has low delays; evening (16:00-20:00) compounds delays
        if 6 <= departure_hour <= 9:
            time_factor = 4.0   # Morning ramp up
        elif 10 <= departure_hour <= 14:
            time_factor = 8.0   # Midday steady
        elif 15 <= departure_hour <= 20:
            time_factor = 22.0  # Evening peak congestion
        elif 21 <= departure_hour <= 23:
            time_factor = 14.0  # Late inbound compound delays
        else:
            time_factor = -5.0  # Overnight low traffic

        # 4. Weather & Meteorological Impact
        weather_info = WEATHER_CONDITIONS.get(weather_condition, {"delay_factor": 0.1})
        weather_penalty = weather_info["delay_factor"] * 45.0  # up to +38%

        # Adverse wind and low visibility adjustments
        wind_penalty = max(0.0, (wind_speed_kmh - 30.0) * 0.4)
        vis_penalty = max(0.0, (5.0 - visibility_km) * 2.5) if visibility_km < 5.0 else 0.0

        # Calculate Raw Delay Probability (%)
        raw_score = (
            (airline_score * 0.35) +
            (congestion_penalty * 0.8) +
            time_factor +
            weather_penalty +
            wind_penalty +
            vis_penalty
        )

        # Bound risk score between 5% and 95%
        delay_probability = float(min(95.0, max(5.0, round(raw_score, 1))))

        # Classify Risk Level
        if delay_probability < 25.0:
            risk_level = "Low Risk"
            risk_color = "#1DD1A1"  # Emerald Green
            est_delay_min = "0 - 15 min"
            badge = "🟢"
        elif delay_probability < 50.0:
            risk_level = "Moderate Risk"
            risk_color = "#FECA57"  # Amber
            est_delay_min = "15 - 35 min"
            badge = "🟡"
        elif delay_probability < 75.0:
            risk_level = "High Risk"
            risk_color = "#FF9F43"  # Orange
            est_delay_min = "35 - 70 min"
            badge = "🟠"
        else:
            risk_level = "Critical Delay Risk"
            risk_color = "#FF6B6B"  # Crimson
            est_delay_min = "70+ min"
            badge = "🔴"

        # Identify Key Contributing Risk Factors
        risk_factors: List[str] = []

        if weather_condition in ["Thunderstorm", "Snow", "Fog"]:
            risk_factors.append(f"Adverse weather condition ({weather_condition}) creates severe gate/runway holds.")
        elif weather_condition == "Rain":
            risk_factors.append("Precipitation may slow apron operations and reduce braking action.")

        if visibility_km < 4.0:
            risk_factors.append(f"Low surface visibility ({visibility_km} km) triggers Instrument Flight Rules (IFR).")

        if wind_speed_kmh > 40.0:
            risk_factors.append(f"Elevated wind speeds ({wind_speed_kmh} km/h) may force single-runway operations.")

        if 15 <= departure_hour <= 20:
            risk_factors.append(f"Scheduled during evening peak air traffic window ({departure_hour:02d}:00).")

        if orig_data["congestion"] >= 1.3:
            risk_factors.append(f"High origin hub traffic congestion at {origin} ({orig_data['city']}).")

        if dest_data["congestion"] >= 1.3:
            risk_factors.append(f"High arrival traffic congestion at destination {destination} ({dest_data['city']}).")

        if airline_data["base_delay_rate"] >= 0.22:
            risk_factors.append(f"Airline historical turnaround delay baseline: {airline_data['base_delay_rate']*100:.0f}%.")

        if not risk_factors:
            risk_factors.append("Normal operational conditions with optimal flight corridor clearance.")

        # Actionable Operational Guidance
        if delay_probability >= 50.0:
            recommendation = (
                "Monitor ATC ground stops and runway capacity. Prepare inbound aircraft turnaround buffers "
                "and alert passengers of potential departure schedule revisions."
            )
        elif delay_probability >= 25.0:
            recommendation = (
                "Standard operational watch. Verify gate availability at destination hub and monitor "
                "local weather updates 2 hours prior to pushback."
            )
        else:
            recommendation = (
                "Favorable flight envelope. Expected on-time departure with routine pushback clearance."
            )

        # Distance calculation
        distance_km = calculate_distance(orig_data["lat"], orig_data["lon"], dest_data["lat"], dest_data["lon"])

        return {
            "risk_score": delay_probability,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "badge": badge,
            "estimated_delay": est_delay_min,
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "distance_km": distance_km,
            "model_version": "Phase 1 - Aviation Heuristic Engine (Scikit-Learn Ready)"
        }

# Global singleton engine instance
default_engine = DelayRiskEngine(
    model_path="models/flight_delay_model.joblib"
)

def predict_flight_risk(
    airline: str,
    origin: str,
    destination: str,
    departure_hour: int,
    weather_condition: str,
    wind_speed_kmh: float = 15.0,
    visibility_km: float = 10.0
) -> Dict[str, Any]:
    """Helper entry point for Streamlit UI."""
    return default_engine.predict(
        airline=airline,
        origin=origin,
        destination=destination,
        departure_hour=departure_hour,
        weather_condition=weather_condition,
        wind_speed_kmh=wind_speed_kmh,
        visibility_km=visibility_km
    )
