"""Enterprise AutoML 1.0.0 — Example: Fraud & Outlier Detection (Anomaly Detection)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from enterprise_automl import main

if __name__ == '__main__':
    sys.argv = [
        'enterprise_automl.py', 'anomaly',
        '--data', 'practice.csv',
        '--models', 'isolation_forest,one_class_svm',
        '--contamination', '0.05'
    ]
    main()

