"""Enterprise AutoML 1.0.0 — Test Suite: Model Compatibility Gate"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enterprise_automl import test_compat_cmd

def test_all_registered_models():
    args = argparse.Namespace(command='test-compat')
    test_compat_cmd(args)

if __name__ == '__main__':
    test_all_registered_models()

