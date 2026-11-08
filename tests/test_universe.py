"""Enterprise AutoML 1.0.0 — Test Suite: Experiment Universe Gate"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enterprise_automl import test_universe_cmd

def test_universe():
    args = argparse.Namespace(command='test-universe')
    test_universe_cmd(args)

if __name__ == '__main__':
    test_universe()

