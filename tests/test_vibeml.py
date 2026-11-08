"""VibeML 1.0.0 — Automated Test Suite & Release Certification Gate.

Executes all 5 verification test gates across the single-file VibeML engine:
1. test_universe: Experiment Universe & Tamper-Evident Ledger Gate (6 checks)
2. test_optimize: Universal Normalization, Quantization & Regularization Gate (6 checks)
3. test_compat:   37 Registered ML Model Compatibility Gate (37 models)
4. test_e2e:      End-to-End Lifecycle Suite across 5 Workloads (5 workloads)
5. test_deploy:   Production FastAPI & Deployment Acceptance Gate (7 checks)
"""
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 stream output across all operating systems
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vibeml import (
    test_universe_cmd,
    test_optimize_cmd,
    test_compat_cmd,
    test_e2e_cmd,
    test_deploy_cmd,
)

def run_all_gates():
    print("=" * 80)
    print(" VIBEML 1.0.0 — COMPREHENSIVE AUTOMATED VERIFICATION SUITE")
    print("=" * 80)

    try:
        print("\n>>> [1/5] Running Experiment Universe & Traceability Gate...")
        test_universe_cmd(argparse.Namespace(command='test-universe'))

        print("\n>>> [2/5] Running Universal Optimization Gate...")
        test_optimize_cmd(argparse.Namespace(command='test-optimize'))

        print("\n>>> [3/5] Running 37-Model Compatibility Gate...")
        test_compat_cmd(argparse.Namespace(command='test-compat'))

        print("\n>>> [4/5] Running End-to-End 5-Workload Lifecycle Gate...")
        test_e2e_cmd(argparse.Namespace(command='test-e2e'))

        print("\n>>> [5/5] Running Production Deployment Acceptance Gate...")
        test_deploy_cmd(argparse.Namespace(command='test-deploy'))

        print("\n" + "=" * 80)
        print(" >> ALL VIBEML 1.0.0 VERIFICATION GATES PASSED SUCCESSFULLY!")
        print("=" * 80)
    except Exception as exc:
        import traceback
        print("\n" + "!" * 80)
        print(f"VERIFICATION FAILURE: {exc}")
        traceback.print_exc()
        print("!" * 80)
        sys.exit(1)

if __name__ == '__main__':
    run_all_gates()
