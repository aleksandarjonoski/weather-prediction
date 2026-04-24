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

### How it works

`w-prediction.go` predicts weather for Kavadarci using linear regression trained on ~46 years of historical data (1979–2024) from OpenWeather.

### Data pipeline

1. **Load CSV** - reads `latest_dataset_open_weather_kav_1979-2024.csv` containing hourly observations (temp, dew point, feels-like, pressure, humidity, wind speed, cloud cover, weather condition).
2. **Aggregate to daily** - groups hourly rows by date, computing daily averages for numeric fields, tracking min/max temperature, and picking the dominant weather condition per day. This produces ~16,800 daily records.

### Feature engineering

Each day is represented by 8 features:

- `sin(2π · day_of_year / 365)` and `cos(2π · day_of_year / 365)` - cyclical encoding of seasonality
- Temperature, dew point, pressure, humidity, wind speed, cloud cover from the previous day

All features and targets are z-score normalized (zero mean, unit variance) before training.

### Model training

Seven separate linear regression models are trained - one for each prediction target: avg temp, min temp, max temp, feels-like, humidity, pressure, and wind speed.

Training uses [Gorgonia](https://github.com/gorgonia/gorgonia) (a Go ML library) to build a computation graph:

- Forward pass: `pred = X @ w` (features matrix times weight vector, with a bias column of ones appended)
- Loss: mean squared error
- Optimization: vanilla gradient descent (learning rate 0.01, 200 epochs) with automatic differentiation via Gorgonia

### Weather condition prediction

Weather conditions (Clear, Clouds, Rain, etc.) are predicted separately using a frequency-based lookup: for a given month and temperature bucket (rounded to nearest 5°C), the historically most common condition is returned.

### Prediction

- **Tomorrow**: builds a "proxy" for today by averaging all historical records that share the same day-of-year, extracts features, and runs each model.
- **Arbitrary date** (`/predict/YYYY-MM-DD`): same approach but uses the day before the target date as the proxy.

### HTTP API

Starts a server on `:8080` with these endpoints:

| Endpoint                  | Response                          |
| ------------------------- | --------------------------------- |
| `GET /`                   | Plain text forecast for tomorrow  |
| `GET /predict`            | JSON forecast for tomorrow        |
| `GET /predict/tomorrow`   | JSON forecast for tomorrow        |
| `GET /predict/YYYY-MM-DD` | JSON forecast for a specific date |

### Training set

Data split 67/33 chronologically - first 67% of daily records (roughly 1979 through mid-2009) used to fit the regression weights. The plot overlays predicted vs actual daily temperature across the fitted range. Tight alignment along the diagonal confirms the model captures the dominant seasonal cycle, but scatter around the line reveals residual noise the linear model cannot explain - cold snaps, heat waves, multi-day anomalies.

Training error is the **optimistic** bound: if the model does poorly here, it will do worse on new data.

![image](https://github.com/user-attachments/assets/caf52962-4d57-4ecb-be1b-85afad21bdb6)

### Test set (33%)

Remaining 33% (roughly mid-2009 through 2024) held out during training, used only for evaluation. The plot shows predictions on unseen days - measures how well the model generalizes beyond data it memorized during fitting.

If test error ≈ training error, model is not overfitting - it learned generalizable patterns, not noise. If test error is much worse, model memorized training quirks. Here both track closely because seasonality is a stable, repeating signal; the learned weights transfer cleanly from past to future.

![image](https://github.com/user-attachments/assets/108ddccd-9f85-4a65-aee9-6793e359937d)

### Limitations

- **Linear regression is too simple for weather** - weather is a nonlinear, chaotic system. A linear model can capture broad seasonal trends but misses complex interactions between variables (e.g., how humidity and pressure jointly affect temperature differently in summer vs winter).
- **No real-time input** - predictions are based entirely on historical averages for the day-of-year, not on actual current conditions. A forecast for tomorrow ignores what the weather is actually doing today.
- **Single-station data** - the model is trained on one location (Kavadarci) and cannot generalize to other cities. Weather is also influenced by surrounding geography and regional patterns that a single point cannot capture.
- **No spatial features** - real weather forecasting uses atmospheric data across a grid of locations. This model has no concept of approaching fronts, pressure gradients, or wind direction changes.
- **Weather conditions are just frequency lookups** - the "Clouds" or "Rain" prediction is the historically most common condition for that month and temperature range, not a modeled output. It cannot predict unusual events.
- **Models are independent** - each target variable (temp, humidity, wind, etc.) is predicted by a separate model with no shared state, so the predictions can be physically inconsistent (e.g., predicting high temperature with high humidity and low dew point simultaneously).
- **No uncertainty estimates** - the model outputs a single point prediction with no confidence interval, so there is no way to know how reliable the forecast is for a given day.
