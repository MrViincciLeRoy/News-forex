name: Test HF Method 3 - Time Series Forecasting

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method3_forecasting.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-forecasting:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install pandas numpy yfinance
        pip install chronos-forecasting || echo "Chronos not available, using fallback"
    
    - name: Test forecaster
      run: |
        python hf_method3_forecasting.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-forecast-results
        path: hf_forecast_results.json
        retention-days: 30
