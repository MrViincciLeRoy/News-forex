name: Generate Economic Calendar

on:
  schedule:
    - cron: '0 0 * * 0'
  workflow_dispatch:

jobs:
  generate:
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
        pip install pandas numpy aiohttp
    
    - name: Run calendar generator
      env:
        ALPHA_VANTAGE_API_KEY_1: ${{ secrets.ALPHA_VANTAGE_API_KEY_1 }}
        ALPHA_VANTAGE_API_KEY_2: ${{ secrets.ALPHA_VANTAGE_API_KEY_2 }}
        ALPHA_VANTAGE_API_KEY_3: ${{ secrets.ALPHA_VANTAGE_API_KEY_3 }}
        ALPHA_VANTAGE_API_KEY_4: ${{ secrets.ALPHA_VANTAGE_API_KEY_4 }}
        ALPHA_VANTAGE_API_KEY_5: ${{ secrets.ALPHA_VANTAGE_API_KEY_5 }}
        ALPHA_VANTAGE_API_KEY_6: ${{ secrets.ALPHA_VANTAGE_API_KEY_6 }}
        FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
      run: |
        python economic_calendar_generator.py
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v4
      with:
        name: economic-calendar-data
        path: |
          gold_calendar_*.json
          gold_calendar_*.csv
        retention-days: 90
