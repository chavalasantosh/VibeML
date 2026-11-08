# VibeML — Enterprise AutoML & MLOps Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-brightgreen.svg)](pyproject.toml)
[![Release](https://img.shields.io/badge/Release-1.0.0--alpha.1-orange.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-5%2F5%20Suites%20Passing-success.svg)](release-evidence/test-results.json)

> **One source application. Five supported ML workloads. Every configured experiment accounted for. Every decision traceable. Reproducible end-to-end results.**

**VibeML** is a single-file, production-grade autonomous machine learning and local MLOps engine. It automates dataset profiling, leakage-free feature engineering, universal optimization (normalization, quantization, regularization), Cartesian experiment universe execution, cryptographic audit logging, champion selection, and one-click production deployment packaging (FastAPI microservice, Docker, Compose, and monitoring).

---

## Key Capabilities

- **Single Master Application**: Zero external application files needed at runtime — runs out of the box via `python enterprise_automl.py`.
- **37 Registered ML Models across 5 Workloads**:
  - **Classification** (13 models): Logistic, SGD, GaussianNB, KNN, SVC, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost.
  - **Regression** (13 models): Linear, Ridge, Lasso, ElasticNet, KNN, SVR, DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost, LightGBM, CatBoost.
  - **Clustering** (4 models): KMeans, MiniBatchKMeans, DBSCAN, GaussianMixture.
  - **Anomaly Detection** (3 models): IsolationForest, LocalOutlierFactor, OneClassSVM.
  - **Time-Series Forecasting** (4 models + recursive tournament): Naive, SeasonalNaive, ARIMA, ETS.
- **Genuine Cartesian Experiment Universe**: Every compatible transformation combination is dynamically compiled into an isolated Scikit-learn pipeline and executed.
- **Tamper-Evident Cryptographic Ledger**: Every experiment decision is chained using SHA-256 (`prev_hash` $\to$ `event_hash`) to guarantee an unalterable audit trail.
- **Full Production Deployment Bundle**: Generates FastAPI service (`serve_api.py`), `Dockerfile`, `docker-compose.yml`, `MODEL_CARD.md`, `client_example.py`, and `run_monitoring.py` (PSI drift detection).

---

## Quick Start

### 1. Interactive Workbench (Zero Arguments)
```bash
python enterprise_automl.py
```

### 2. Autonomous Project-to-Production Orchestrator
```bash
python enterprise_automl.py run --data practice.csv
```

### 3. Exhaustive Candidate Universe Training
```bash
python enterprise_automl.py train --data practice.csv --target price --mode exhaustive
```

### 4. Interactive Replay & Traceability Audit
```bash
python enterprise_automl.py replay --run-id <RUN_ID>
```

### 5. Automated Multi-Step Forecasting
```bash
python enterprise_automl.py forecast --data sales.csv --target demand --time-col timestamp --horizon 7
```

---

## Repository Structure

```
VibeML/
│
├── enterprise_automl.py             # Single Master Runtime Application (8,355 lines)
├── README.md                        # Documentation & Quick Start
├── LICENSE                          # Apache-2.0 Open Source License
├── pyproject.toml                   # Standard PEP 517/621 packaging metadata
├── requirements.txt                 # Core dependencies
├── requirements-lock.txt            # Frozen, deterministic dependencies
├── CHANGELOG.md                     # Semantic version history and audit resolutions
├── RELEASE_NOTES.md                 # Official release notes
├── SECURITY.md                      # Security model and disclosure policy
├── CONTRIBUTING.md                  # Development guidelines (main == PROD)
│
├── examples/                        # Standalone runnable workload examples
│   ├── classification/run_churn_prediction.py
│   ├── regression/run_house_prices.py
│   ├── clustering/run_customer_segmentation.py
│   ├── anomaly/run_fraud_detection.py
│   └── forecasting/run_demand_forecast.py
│
├── tests/                           # Modular test entry points
│   ├── test_models.py               # 37-model compatibility suite
│   ├── test_universe.py             # 6-gate experiment universe suite
│   ├── test_optimization.py         # 6-gate universal optimization suite
│   ├── test_recovery.py             # 5-workload lifecycle recovery suite
│   ├── test_deployment.py           # 7-check deployment acceptance suite
│   └── test_security.py             # Cryptographic ledger & tamper detection test
│
└── release-evidence/                # Immutable release audit artifacts
    ├── test-results.json            # Machine-readable test pass records
    ├── compatibility-matrix.csv     # Model taxonomy, family & scaling requirements
    └── artifact-checksums.sha256    # Cryptographic SHA-256 fingerprints of all files
```

---

## Verification Test Gates

Run the automated verification gates:
```bash
python enterprise_automl.py test-universe   # 6/6 Universe & Traceability Gate
python enterprise_automl.py test-optimize   # 6/6 Universal Optimization Gate
python enterprise_automl.py test-compat     # 37/37 Model Compatibility Gate
python enterprise_automl.py test-e2e        # 5/5 End-to-End Production Lifecycle Gate
python enterprise_automl.py test-deploy     # 7/7 Deployment Acceptance Gate
```

---

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
