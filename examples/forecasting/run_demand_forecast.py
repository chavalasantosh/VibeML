"""Enterprise AutoML 1.0.0 — Example: Demand Trajectory (Time-Series Forecasting)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from enterprise_automl import main

if __name__ == '__main__':
    sys.argv = [
        'enterprise_automl.py', 'forecast',
        '--data', 'practice.csv',
        '--target', 'demand',
        '--time-col', 'timestamp',
        '--horizon', '7'
    ]
    main()

