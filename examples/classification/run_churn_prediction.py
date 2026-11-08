"""Enterprise AutoML 1.0.0 — Example: Churn Prediction (Classification)"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from enterprise_automl import main

if __name__ == '__main__':
    sys.argv = [
        'enterprise_automl.py', 'train',
        '--data', 'practice.csv',
        '--task', 'classification',
        '--target', 'churn',
        '--mode', 'hybrid',
        '--models', 'logistic,xgboost_classifier'
    ]
    main()

