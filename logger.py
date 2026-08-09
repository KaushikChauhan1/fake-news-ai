"""
logger.py — Structured JSONL logging for the verification pipeline.

Logs every request and its outcome to logs/verification.jsonl.
Non-blocking: log failures never propagate to the caller.
"""

import json
import os
import time
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_FILE = _LOG_DIR / "verification.jsonl"


def _ensure_dir() -> bool:
    try:
        _LOG_DIR.mkdir(exist_ok=True)
        return True
    except Exception:
        return False


def log_request(
    input_text: str,
    claim_detected: str,
    verdict: str,
    confidence: float,
    sources: list,
    processing_time: float,
    errors: list = None,
) -> None:
    """
    Write a single log entry to verification.jsonl.
    Never raises — silently swallows any I/O failure.
    """
    try:
        if not _ensure_dir():
            return
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input":           input_text[:300],
            "claim_detected":  claim_detected[:300],
            "verdict":         verdict,
            "confidence":      confidence,
            "sources":         sources[:3],
            "processing_time": round(processing_time, 3),
            "errors":          errors or [],
        }
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never crash the main app
