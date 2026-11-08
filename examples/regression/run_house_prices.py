"""Enterprise AutoML 1.0.0 — Example: Real Estate Valuation (Regression)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from enterprise_automl import main

if __name__ == '__main__':
    sys.argv = [
        'enterprise_automl.py', 'train',
        '--data', 'practice.csv',
        '--task', 'regression',
        '--target', 'price',
        '--mode', 'hybrid',
        '--models', 'linear,ridge,xgboost_regressor'
    ]
    main()

