name: Test HF Method 5 - Correlation Discovery

on:
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths:
      - 'hf_method5_embeddings.py'
  pull_request:
    branches: [ main, master ]

jobs:
  test-embeddings:
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
        pip install sentence-transformers pandas numpy scikit-learn
    
    - name: Test correlation discovery
      run: |
        python hf_method5_embeddings.py
    
    - name: Upload results
      if: success()
      uses: actions/upload-artifact@v4
      with:
        name: hf-correlation-results
        path: hf_correlation_discovery_results.json
        retention-days: 30
