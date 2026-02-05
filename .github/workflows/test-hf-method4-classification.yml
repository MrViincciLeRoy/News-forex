name: Test HF Method 4 - Event Classification

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method4_classification.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-classification:
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
        pip install transformers torch pandas numpy
    
    - name: Test event classifier
      run: |
        python hf_method4_classification.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-classification-results
        path: hf_event_classification_results.json
        retention-days: 30
