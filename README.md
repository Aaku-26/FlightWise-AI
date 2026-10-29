# ✈️ FlightWise AI: Aircraft Delay Risk Prediction & Operational Intelligence
> **Final-Year Data Science Project — Phase 1: Operational Analytics & Baseline Risk Intelligence**

FlightWise AI is an aviation operations dashboard and flight delay risk prediction platform designed for airline dispatchers, ground operations controllers, and passengers.

---

## 📌 Phase 1 Highlights & Objectives

- **Operations Command Center**: Modern dark navy & cyan dashboard with real-time operational KPI metrics (Total Flights, Delayed Departures, Fleet Delay Percentage, and Average Delay Duration).
- **Comprehensive Visual Analytics**:
  - 📈 **Daily Delay Trends & Flight Volume**: Interactive dual-axis trend analysis.
  - 🛫 **Airline Delay Benchmarks**: Comparative ranking across major commercial carriers.
  - ⏰ **Departure Hour Spikes**: 24-hour diurnal delay profile identifying morning and evening bottlenecks.
  - 🏢 **Airport Delay Rankings**: Top origin hubs exhibiting elevated departure delays.
  - 🍩 **Delay Reason Breakdown**: Proportion of delays attributed to Weather, Air Traffic Control (ATC), Carrier Turnaround, and Late Inbound Aircraft.
- **Route Corridor Analytics**: Interactive geographic map (`Plotly Scattergeo`) visualizing domestic flight corridors and corridor-level delay propensities.
- **Flight Delay Risk Predictor**: Baseline operational risk engine scoring delay probabilities, delay duration ranges, and explainable risk factors based on airline performance, airport congestion, peak hours, and weather conditions.
- **Academic Disclaimer**: Clearly labeled synthetic sample dataset (3,500 records) designed for educational development and evaluation. Does not use proprietary or paid external APIs.

---

## 🏗️ Project Architecture & Directory Layout

```text
Mini project/
│
├── .streamlit/
│   └── config.toml             # Aviation dark navy theme configuration
│
├── data/
│   └── synthetic_flights.csv   # Labeled synthetic sample dataset (3,500 records)
│
├── src/
│   ├── __init__.py             # Python package marker
│   ├── generate_data.py        # Reproducible synthetic dataset generator
│   ├── data_loader.py          # Data validation, caching, and filtering
│   ├── analytics.py            # KPI metrics, aggregations, and statistics
│   ├── visualizations.py       # Plotly aviation visualizations (dark theme)
│   └── prediction_engine.py    # Delay risk scoring engine (Scikit-Learn ready)
│
├── tests/
│   ├── __init__.py
│   └── test_flightwise.py      # Automated unit & integration test suite (8 tests)
│
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation & run guide
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.9 or higher (tested on Python 3.11)
- `pip` package manager

### 2. Install Dependencies
Open your terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-generate Synthetic Dataset
The dataset `data/synthetic_flights.csv` is already provided. If you ever want to re-generate it with a fresh random seed:
```bash
python src/generate_data.py
```

### 4. Run Automated Tests
FlightWise AI includes an automated test suite verifying data integrity, KPI calculations, visualizations, and multi-page UI flows:
```bash
python -m unittest tests/test_flightwise.py
```
*Expected result: `Ran 8 tests ... OK`*

### 5. Launch the Streamlit Dashboard
Start the web dashboard by running:
```bash
streamlit run app.py
```
Once launched, open your web browser at:
👉 **`http://localhost:8501`**

---

## 🔮 Machine Learning & Future Phase Roadmap

| Phase | Milestone | Status | Details |
| :--- | :--- | :---: | :--- |
| **Phase 1** | **Operational Intelligence & Heuristic Scorer** | ✅ Complete | Clean modular architecture, dark navy UI, Plotly charts, corridor map, synthetic dataset, heuristic risk engine. |
| **Phase 2** | **Scikit-Learn ML Model Training** | ✅ Complete  | Trained the flight delay prediction model using Scikit-Learn and integrated it into prediction_engine.py. |
| **Phase 3** | **RAG Aviation Operations Assistant** | ✅ Complete |Implemented aviation document retrieval and question-answering functionality using the project's RAG-related modules and aviation data chunks. |
| **Phase 4** | **Explainable AI (SHAP)** | ✅ Complete |Integrated SHAP-based feature explanations and a feature-importance chart into the flight delay prediction dashboard. |

---

## 📋 Synthetic Data Attributes

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `flight_id` | String | Unique synthetic identifier (e.g. `FW-10042`) |
| `flight_number` | String | Airline flight code (e.g. `DL-482`) |
| `airline` | String | Operating carrier name (Delta, American, United, etc.) |
| `origin` / `destination` | String | IATA 3-letter airport codes (JFK, LAX, ORD, ATL, etc.) |
| `flight_date` | Date | Scheduled departure date |
| `departure_hour` | Integer | Scheduled departure hour (0 - 23) |
| `weather_condition` | String | Surface meteorology (Clear, Rain, Fog, Snow, etc.) |
| `delay_minutes` | Integer | Actual departure delay (minutes, negative = early) |
| `is_delayed` | Boolean | `True` if `delay_minutes >= 15` (FAA official standard) |
| `delay_reason` | String | Primary cause (Weather, ATC, Turnaround, Carrier) |
| `data_source` | String | Labeled `SYNTHETIC_SAMPLE_DATA` |

---

## 🛡️ License & Academic Integrity
This project is developed for educational purposes as part of a final-year Data Science degree curriculum. All sample data is explicitly marked as synthetic demonstration data.
