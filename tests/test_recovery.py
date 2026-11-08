"""Enterprise AutoML 1.0.0 — Test Suite: End-to-End Lifecycle & Recovery"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enterprise_automl import test_e2e_cmd

def test_recovery():
    args = argparse.Namespace(command='test-e2e')
    test_e2e_cmd(args)

if __name__ == '__main__':
    test_recovery()

