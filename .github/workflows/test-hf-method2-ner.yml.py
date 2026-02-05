name: Test HF Method 2 - Named Entity Recognition

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method2_ner.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-ner:
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
    
    - name: Test entity extractor
      run: |
        python hf_method2_ner.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-ner-results
        path: hf_entity_extraction_results.json
        retention-days: 30
