# weather-prediction

- Simple weather prediction app based on historical data

## w-prediction.go

```bash
Loading and aggregating weather data...
Loaded 16802 daily records (1979-01-01 to 2024-12-31)
Training models...

Prediction for Kavadarci, 2026-03-20:
  Temperature: 9.2°C (feels like 8.0°C)
  Min: 4.0°C, Max: 14.2°C
  Humidity: 66%, Wind: 2.0 m/s, Pressure: 1016 hPa
  Conditions: Clouds (overcast clouds)

Server running on http://localhost:8080
  curl localhost:8080                  # text output
  curl localhost:8080/predict           # JSON output
  curl localhost:8080/predict/tomorrow   # JSON output
```
