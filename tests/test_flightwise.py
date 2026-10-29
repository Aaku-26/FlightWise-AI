"""
tests/test_flightwise.py
------------------------
Automated test suite for FlightWise AI Phase 1 modules.

Runs with standard Python unittest without needing extra test dependencies.
"""

import os
import unittest
import pandas as pd
import plotly.graph_objects as go

from src.generate_data import generate_flight_data, calculate_distance, AIRPORTS
from src.data_loader import load_flight_data_raw, get_unique_filter_options, filter_flight_data, REQUIRED_COLUMNS
from src.analytics import (
    calculate_kpis,
    get_delay_trends,
    get_airline_delays,
    get_hourly_delays,
    get_airport_delays,
    get_route_analytics,
    get_delay_reasons
)
from src.visualizations import (
    plot_delay_trends,
    plot_airline_delays,
    plot_hourly_delays,
    plot_top_airports,
    plot_delay_reasons_pie,
    plot_route_network_map
)
from src.prediction_engine import predict_flight_risk, DelayRiskEngine

class TestFlightWiseAI(unittest.TestCase):
    """Test suite for data generation, loading, analytics, and prediction."""

    @classmethod
    def setUpClass(cls):
        """Generate a small test dataset."""
        cls.df = generate_flight_data(num_records=100, random_seed=999)

    def test_01_synthetic_data_generation(self):
        """Verify synthetic data generator outputs required shape and columns."""
        self.assertEqual(len(self.df), 100)
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, self.df.columns, f"Missing required column: {col}")
        self.assertTrue((self.df["data_source"] == "SYNTHETIC_SAMPLE_DATA").all())

    def test_02_distance_calculation(self):
        """Verify haversine distance calculation between JFK and LAX."""
        dist = calculate_distance(40.6413, -73.7781, 33.9416, -118.4085)
        # Approximate JFK to LAX distance is ~3970-3990 km
        self.assertGreater(dist, 3800)
        self.assertLess(dist, 4100)

    def test_03_data_loader(self):
        """Verify data loading, schema integrity, and filter extraction."""
        opts = get_unique_filter_options(self.df)
        self.assertIn("airlines", opts)
        self.assertIn("airports", opts)
        self.assertIn("weather_conditions", opts)
        self.assertGreater(len(opts["airlines"]), 0)

        # Test filter function
        filtered = filter_flight_data(self.df, selected_airlines=[opts["airlines"][0]])
        self.assertTrue((filtered["airline"] == opts["airlines"][0]).all())

    def test_04_analytics_kpi_calculation(self):
        """Verify KPI metric calculations."""
        kpis = calculate_kpis(self.df)
        self.assertEqual(kpis["total_flights"], 100)
        self.assertEqual(kpis["delayed_flights"] + kpis["on_time_flights"], 100)
        self.assertGreaterEqual(kpis["delay_percentage"], 0.0)
        self.assertLessEqual(kpis["delay_percentage"], 100.0)

    def test_05_analytics_aggregations(self):
        """Verify trend, airline, hourly, airport, and route aggregations."""
        trends = get_delay_trends(self.df)
        self.assertFalse(trends.empty)
        self.assertIn("delay_rate", trends.columns)

        airlines = get_airline_delays(self.df)
        self.assertFalse(airlines.empty)

        hourly = get_hourly_delays(self.df)
        self.assertFalse(hourly.empty)

        airports = get_airport_delays(self.df)
        self.assertFalse(airports.empty)

        routes = get_route_analytics(self.df, min_flights=1)
        self.assertFalse(routes.empty)

        reasons = get_delay_reasons(self.df)
        # Should return a dataframe
        self.assertIsInstance(reasons, pd.DataFrame)

    def test_06_visualizations_render_figures(self):
        """Ensure all Plotly visualization functions return valid go.Figure objects."""
        trends = get_delay_trends(self.df)
        fig1 = plot_delay_trends(trends)
        self.assertIsInstance(fig1, go.Figure)

        airlines = get_airline_delays(self.df)
        fig2 = plot_airline_delays(airlines)
        self.assertIsInstance(fig2, go.Figure)

        hourly = get_hourly_delays(self.df)
        fig3 = plot_hourly_delays(hourly)
        self.assertIsInstance(fig3, go.Figure)

        airports = get_airport_delays(self.df)
        fig4 = plot_top_airports(airports)
        self.assertIsInstance(fig4, go.Figure)

        reasons = get_delay_reasons(self.df)
        fig5 = plot_delay_reasons_pie(reasons)
        self.assertIsInstance(fig5, go.Figure)

        routes = get_route_analytics(self.df, min_flights=1)
        fig6 = plot_route_network_map(routes, AIRPORTS)
        self.assertIsInstance(fig6, go.Figure)

    def test_07_prediction_engine(self):
        """Test delay risk predictor under various conditions."""
        # Test baseline prediction
        res = predict_flight_risk(
            airline="Delta Air Lines",
            origin="JFK",
            destination="LAX",
            departure_hour=18,
            weather_condition="Clear",
            wind_speed_kmh=15.0,
            visibility_km=10.0
        )
        self.assertIn("risk_score", res)
        self.assertIn("risk_level", res)
        self.assertGreaterEqual(res["risk_score"], 5.0)
        self.assertLessEqual(res["risk_score"], 95.0)
        self.assertIsInstance(res["risk_factors"], list)

        # Severe weather should increase delay probability compared to clear weather
        res_severe = predict_flight_risk(
            airline="Delta Air Lines",
            origin="JFK",
            destination="LAX",
            departure_hour=18,
            weather_condition="Thunderstorm",
            wind_speed_kmh=55.0,
            visibility_km=2.0
        )
        self.assertGreater(res_severe["risk_score"], res["risk_score"])

        # Edge case: Same origin and destination
        res_same = predict_flight_risk(
            airline="Delta Air Lines",
            origin="JFK",
            destination="JFK",
            departure_hour=10,
            weather_condition="Clear"
        )
        self.assertIn("error", res_same)

    def test_08_streamlit_app_navigation_and_prediction(self):
        """End-to-end integration test of Streamlit app pages and prediction form."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Dashboard threw exception: {at.exception}")

        # Navigate to Route Analytics
        at.sidebar.radio[0].set_value("🗺️ Route Analytics").run()
        self.assertEqual(len(at.exception), 0, f"Route Analytics threw exception: {at.exception}")

        # Navigate to Flight Delay Predictor and click assess
        at.sidebar.radio[0].set_value("🔮 Flight Delay Predictor").run()
        self.assertEqual(len(at.exception), 0, f"Predictor page threw exception: {at.exception}")
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0, f"Prediction button execution threw exception: {at.exception}")

        # Navigate to Project Architecture
        at.sidebar.radio[0].set_value("ℹ️ Project Architecture").run()
        self.assertEqual(len(at.exception), 0, f"Architecture page threw exception: {at.exception}")

if __name__ == "__main__":
    unittest.main()
