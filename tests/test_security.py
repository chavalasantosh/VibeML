"""Enterprise AutoML 1.0.0 — Test Suite: Security & Tamper Detection"""
import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enterprise_automl import EventLedger

def test_tamper_detection():
    temp_dir = Path(tempfile.mkdtemp(prefix='sec_test_'))
    ledger = EventLedger(temp_dir)
    ledger.log_event('run_started', details='Init')
    ledger.log_event('candidate_completed', candidate_id='EXP-001', metrics={'auc': 0.95})
    assert ledger.verify_integrity()['verified'] is True

    # Tamper with log
    with open(ledger.events_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    bad = json.loads(lines[1])
    bad['metrics']['auc'] = 0.999
    lines[1] = json.dumps(bad) + '\n'
    with open(ledger.events_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    assert ledger.verify_integrity()['verified'] is False
    print("[PASS] Security: SHA-256 tamper-evident integrity verified.")

if __name__ == '__main__':
    test_tamper_detection()

