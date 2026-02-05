name: Test HF Method 10 - Causal Explanations

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method10_causal.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-causal:
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
        pip install transformers torch pandas
    
    - name: Test causal explainer
      run: |
        python hf_method10_causal.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-causal-results
        path: hf_causal_explanation_report.txt
        retention-days: 30
