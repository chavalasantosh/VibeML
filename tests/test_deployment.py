"""Enterprise AutoML 1.0.0 — Test Suite: Deployment Acceptance Gate"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enterprise_automl import test_deploy_cmd

def test_deployment():
    args = argparse.Namespace(command='test-deploy')
    test_deploy_cmd(args)

if __name__ == '__main__':
    test_deployment()

