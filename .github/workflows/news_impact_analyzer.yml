name: Test News Impact Analyzer

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'news_impact_analyzer.py'
      - 'symbol_indicators.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-news-impact:
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
        pip install -r requirements.txt
    
    - name: Test news impact analyzer
      env:
        ALPHA_VANTAGE_API_KEY: ${{ secrets.ALPHA_VANTAGE_API_KEY }}
        FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
      run: |
        python news_impact_analyzer.py
