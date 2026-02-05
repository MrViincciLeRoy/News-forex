name: Test HF Method 8 - Multi-Modal Analysis

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method8_multimodal.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-multimodal:
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
        pip install transformers torch pandas numpy yfinance
    
    - name: Test multimodal analyzer
      run: |
        python hf_method8_multimodal.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-multimodal-results
        path: hf_multimodal_results.json
        retention-days: 30
