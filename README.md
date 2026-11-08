# VibeML — Autonomous Machine Learning & MLOps Engine

[![CI](https://github.com/chavalasantosh/VibeML/actions/workflows/ci.yml/badge.svg)](https://github.com/chavalasantosh/VibeML/actions/workflows/ci.yml)
[![Security Audit](https://github.com/chavalasantosh/VibeML/actions/workflows/security.yml/badge.svg)](https://github.com/chavalasantosh/VibeML/actions/workflows/security.yml)
[![Production Release](https://github.com/chavalasantosh/VibeML/actions/workflows/release.yml/badge.svg)](https://github.com/chavalasantosh/VibeML/actions/workflows/release.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-1.0.0-orange.svg)](https://github.com/chavalasantosh/VibeML/releases)

> **One source application. Five supported ML workloads. Every configured experiment accounted for. Every decision traceable. Reproducible end-to-end results.**

**VibeML** is a single-file, production-grade autonomous machine learning and local MLOps engine. It automates dataset profiling, leakage-free feature engineering, universal optimization (normalization, quantization, regularization), Cartesian experiment universe execution, cryptographic audit logging, champion selection, and one-click production deployment packaging (FastAPI microservice, Docker, Compose, and monitoring).

---

## Key Capabilities

- **Single Master Application (`vibeml.py`)**: Zero external application files needed at runtime — runs out of the box via `python vibeml.py`.
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

## Repository Structure (Minimalist Production Clean)

```
VibeML/
├── .github/
│   └── workflows/
│       ├── ci.yml            # Multi-OS matrix tests & cleanliness check
│       ├── security.yml      # Bandit SAST & pip-audit vulnerability scanner
│       └── release.yml       # Automated Python packaging & Docker build
├── tests/
│   └── test_vibeml.py        # Complete 5-gate automated verification suite
├── .gitignore                # Production ignore rules (zero trash / cache / dumps)
├── Dockerfile                # Hardened non-root Python 3.11-slim container
├── LICENSE                   # Apache 2.0 Open-Source License
├── pyproject.toml            # PEP 517 build specification & CLI entrypoint
├── README.md                 # Documentation and quickstart guide
├── requirements.txt          # Production runtime dependencies
└── vibeml.py                 # Self-contained Master Engine (8,350+ LOC)
```

---

## Installation

### From Source
```bash
git clone https://github.com/chavalasantosh/VibeML.git
cd VibeML
pip install -r requirements.txt
```

### Docker
```bash
docker build -t vibeml .
docker run --rm vibeml --help
```

---

## Workload Quickstart

### 1. Interactive Workbench (Zero Arguments)
```bash
python vibeml.py
```

### 2. Autonomous Project-to-Production Orchestrator
Auto-detects data schema, selects task, evaluates universe, trains champion, and outputs FastAPI & Docker deliverables:
```bash
python vibeml.py run --data dataset.csv
```

### 3. Classification & Regression Training
```bash
# Classification with exhaustive candidate universe evaluation
python vibeml.py train --data churn.csv --target churn --task classification --mode exhaustive

# Regression with hybrid candidate screening
python vibeml.py train --data housing.csv --target price --task regression --mode hybrid
```

### 4. Unsupervised Clustering
```bash
python vibeml.py cluster --data customers.csv --k-min 2 --k-max 6
```

### 5. Anomaly & Outlier Detection
```bash
python vibeml.py anomaly --data transactions.csv --contamination 0.05
```

### 6. Automated Time-Series Forecasting
```bash
python vibeml.py forecast --data sales.csv --target demand --time-col timestamp --horizon 7
```

### 7. Interactive Replay & Traceability Audit
```bash
python vibeml.py replay --run-id <RUN_ID>
```

---

## Verification Test Gates

Run the automated verification suite:
```bash
python tests/test_vibeml.py
```
Or individual verification gates directly:
```bash
python vibeml.py test-universe   # 6/6 Universe & Traceability Gate
python vibeml.py test-optimize   # 6/6 Universal Optimization Gate
python vibeml.py test-compat     # 37/37 Model Compatibility Gate
python vibeml.py test-e2e        # 5/5 End-to-End Production Lifecycle Gate
python vibeml.py test-deploy     # 7/7 Deployment Acceptance Gate
```

---

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
