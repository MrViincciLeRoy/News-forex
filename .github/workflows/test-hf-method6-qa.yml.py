name: Test HF Method 6 - Market QA System

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method6_qa.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-qa:
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
    
    - name: Test QA system
      run: |
        python hf_method6_qa.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-qa-results
        path: hf_market_qa_results.json
        retention-days: 30
