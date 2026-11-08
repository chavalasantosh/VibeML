# Contributing to VibeML

## Core Repository Principles

1. **`main == PROD`**:
   - The `main` branch is production-ready code only.
   - Every commit on `main` must pass all verification test suites (`test-universe`, `test-optimize`, `test-compat`, `test-e2e`, `test-deploy`).
   - Never push scratch files, temporary datasets, local training runs, model binaries (`.joblib`), or cache files to `main`.

2. **Single-File Runtime Architecture**:
   - The core runtime engine is completely self-contained in `enterprise_automl.py`.
   - Modifying runtime logic must preserve zero mandatory external file requirements for `python enterprise_automl.py`.

3. **Cryptographic Traceability**:
   - Every change affecting experiment execution must maintain tamper-evident logging via `EventLedger` (SHA-256 hash chaining).

4. **Zero-Leakage Guarantee**:
   - All feature transformations, scaling, imputation, and quantization must be strictly fitted on training splits only. Holdout test sets must remain completely isolated.

## Pre-Push Verification Checklist

Before pushing to `main`, run:
```bash
python enterprise_automl.py test-universe
python enterprise_automl.py test-optimize
python enterprise_automl.py test-compat
python enterprise_automl.py test-e2e
python enterprise_automl.py test-deploy
```
All suites must report 100% green.
