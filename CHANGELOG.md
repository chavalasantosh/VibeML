# Changelog — Enterprise AutoML

All notable changes to Enterprise AutoML are documented in this file.

## [1.0.0-alpha.1] - 2026-11-08

### Major Milestone: First Official 1.0 Release Channel (Alpha 1)
- **Genuine Experiment Universe Execution**: Connected `ExperimentUniverseBuilder` to actual fitted pipelines via `build_candidate_pipeline()` iterating over concrete candidate configurations.
- **Cryptographic Event Ledger**: Implemented SHA-256 hash chaining (`prev_hash` -> `event_hash`) in `EventLedger` with automatic tamper detection.
- **Persistent Candidates Leaderboard**: Added `candidates_leaderboard.csv` tracking all candidate metrics, latencies, and artifact footprints.
- **Candidate Disk Serialization**: Individual candidate pipelines saved under `runs/<task>/<run_id>/candidates/<cand_id>/`.
- **Downstream Contract Preservation**: Best candidate per model family promoted to standard model paths (`models/<task>/<run_id>/<model_name>/`).
- **Complete Verification**:
  - `test-universe`: 6/6 gates passed.
  - `test-optimize`: 6/6 gates passed.
  - `test-compat`: 37/37 registered models verified.
  - `test-e2e`: 5/5 workload lifecycles passed.
  - `test-deploy`: 7/7 deployment acceptance checks passed.

## [5.7.0] - 2026-11-08
- Initial Cartesian experiment universe generator and explanation engine.

## [5.6.0] - 2026-11-08
- Universal optimization registries (normalization, quantization, regularization).

## [5.5.0] - 2026-11-08
- Automated deployment gate and FastAPI microservice generator.

