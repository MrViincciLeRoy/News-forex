name: Test HF Method 9 - Zero-Shot Categorization

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method9_zeroshot.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-zeroshot:
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
    
    - name: Test zero-shot categorizer
      run: |
        python hf_method9_zeroshot.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-zeroshot-results
        path: hf_zeroshot_categorization_results.json
        retention-days: 30
