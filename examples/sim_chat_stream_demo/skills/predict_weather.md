# Skill: Predict Weather

## Name
`predict_weather`

## Description
Returns a short-term weather forecast for a given location.

## Inputs
- `location` (string, required): City or location name  
- `days_ahead` (integer, optional, default=1): Number of days to forecast (max 7)  
- `units` (string, optional): `metric` or `imperial`

## Output
```json
{
  "location": "Seattle, WA",
  "forecast": [
    {
      "date": "2026-01-22",
      "condition": "Rain",
      "temp_min": 4,
      "temp_max": 8,
      "precip_prob": 0.65
    }
  ],
  "units": "metric"
}
```

## Example Call
```json
{
  "skill": "predict_weather",
  "arguments": {
    "location": "Seattle, WA",
    "days_ahead": 1
  }
}
```

## Notes
- Forecasts are probabilistic
- Accuracy depends on the external weather provider
