# Security Policy — Enterprise AutoML

## Supported Versions
Only the latest release channel receives security updates:
| Version | Supported |
| :--- | :---: |
| 1.0.0-alpha.1 | Yes |
| < 1.0.0 | No |

## Security Architecture & Threat Model

1. **Artifact Deserialization Safety**:
   - Model pipelines are serialized using joblib with atomic temporary writes.
   - Enterprise AutoML refuses to deserialize models over unauthenticated networks.
   - Remote model ingestion is strictly subject to SHA-256 integrity verification (`model_sha256`).

2. **Tamper-Evident Cryptographic Ledger**:
   - Every candidate outcome, hyperparameter set, and stage status is recorded in `events.jsonl` using SHA-256 cryptographic hash chaining (`prev_hash` -> `event_hash`).
   - Tampered logs are automatically detected via `EventLedger.verify_integrity()`.

3. **Inference Boundary Hardening**:
   - Generated FastAPI services reject unknown features or unexpected types with HTTP 422 Unprocessable Entity.
   - Input batch size is bounded (max 100,000 rows) to prevent out-of-memory DoS vectors.

4. **Data Isolation & Leakage Prevention**:
   - Train, validation, and test partitions are strictly isolated.
   - The test partition is evaluated exactly once on the winning candidate and never used to influence selection.

## Reporting Vulnerabilities
To report a security vulnerability, please submit an encrypted report to security@enterprise-mlops.local.

