"""Enterprise AutoML 1.0.0 — Test Suite: Universal Optimization Gate"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enterprise_automl import test_optimize_cmd

def test_optimization():
    args = argparse.Namespace(command='test-optimize')
    test_optimize_cmd(args)

if __name__ == '__main__':
    test_optimization()

