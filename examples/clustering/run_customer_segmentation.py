"""Enterprise AutoML 1.0.0 — Example: Customer Segmentation (Clustering)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from enterprise_automl import main

if __name__ == '__main__':
    sys.argv = [
        'enterprise_automl.py', 'cluster',
        '--data', 'practice.csv',
        '--models', 'kmeans,gaussian_mixture,dbscan',
        '--k-min', '2',
        '--k-max', '5'
    ]
    main()

