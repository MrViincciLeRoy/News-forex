name: Test HF Analytics Master

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_analytics_master.py'
      - 'hf_method*.py'
  pull_request:
    branches: [ main, master ]
  schedule:
    - cron: '0 12 * * 1'

jobs:
  test-master:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install core dependencies
      run: |
        pip install pandas numpy
    
    - name: Install HF dependencies
      run: |
        pip install transformers torch sentence-transformers yfinance
        pip install chronos-forecasting || echo "Chronos optional"
        pip install pyod scikit-learn || echo "PyOD/sklearn optional"
    
    - name: Test master orchestrator
      run: |
        python hf_analytics_master.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-master-results
        path: |
          hf_analytics_output/*.json
          hf_analytics_output/*.txt
        retention-days: 60
    
    - name: Display summary
      if: success()
      run: |
        echo "Master analytics test completed"
        ls -lh hf_analytics_output/
