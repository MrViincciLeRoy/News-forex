name: Test HF Method 7 - Anomaly Detection

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method7_anomaly.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-anomaly:
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
        pip install pyod || echo "PyOD not available, using statistical methods"
    
    - name: Test anomaly detector
      run: |
        python hf_method7_anomaly.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-anomaly-results
        path: hf_anomaly_detection_results.json
        retention-days: 30
