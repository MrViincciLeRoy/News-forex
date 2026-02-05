name: Test HF Method 1 - Sentiment Analysis

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method1_sentiment.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-sentiment:
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
    
    - name: Test sentiment analyzer
      run: |
        python hf_method1_sentiment.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-sentiment-results
        path: hf_sentiment_results.json
        retention-days: 30
