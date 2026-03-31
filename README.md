# weather-prediction

- Simple weather prediction app based on historical data

## w-prediction.go

```bash
Loading and aggregating weather data...
Loaded 16802 daily records (1979-01-01 to 2024-12-31)
Training models...

Prediction for Kavadarci, 2026-04-01:
  Temperature: 11.7°C (feels like 10.6°C)
  Min: 6.4°C, Max: 16.8°C
  Humidity: 65%, Wind: 2.0 m/s, Pressure: 1015 hPa
  Conditions: Clouds (overcast clouds)

Server running on http://localhost:8080
  curl localhost:8080                          # text output
  curl localhost:8080/predict                   # JSON output (tomorrow)
  curl localhost:8080/predict/tomorrow           # JSON output (tomorrow)
  curl localhost:8080/predict/2026-05-15         # JSON output (specific date)
```
