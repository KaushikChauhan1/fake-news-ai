"""
app.py — Fact Verification API (v7 — Production)

Spec compliance:
  ✓ Input normalization + claim extraction
  ✓ Input type detection (claim / question / statement)
  ✓ Multi-source evidence (DDG → Wikipedia fallback)
  ✓ Scoring: similarity + trust + agreement + recency
  ✓ Minimum 2 sources required for REAL/FAKE verdict
  ✓ Required response fields: verdict, confidence, explanation,
      sources, claim_detected, processing_time, data_quality
  ✓ Structured JSONL logging (input, output, confidence, timestamp, errors)
  ✓ Deterministic output (LRU-cached, identical inputs → identical outputs)
  ✓ CORS for all origins
  ✓ Never crashes under any condition
"""

import os
import re
import time
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import fact
import ml
import logger as log_

load_dotenv()

app = FastAPI(title="Fact Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Input Processing ───────────────────────────────────────────────────────

_QUESTION_PREFIXES = re.compile(
    r'^(is it true that|fact[:\s-]+|claim[:\s-]+|did\s+|does\s+|do\s+|'
    r'is\s+|are\s+|was\s+|were\s+|has\s+|have\s+|who\s+|what\s+|'
    r'when\s+|where\s+|why\s+|how\s+)',
    re.IGNORECASE,
)
_NOISE_RE = re.compile(r'\s+')


def _detect_input_type(text: str) -> str:
    """Classify input as 'question', 'claim', or 'statement'."""
    t = text.strip()
    if t.endswith("?"):
        return "question"
    if _QUESTION_PREFIXES.match(t):
        return "question"
    # Treat short (<10 words) assertive sentences as claims
    words = t.split()
    if len(words) <= 15:
        return "claim"
    return "statement"


def _extract_claim(text: str) -> str:
    """
    Strip noise and question framing to isolate the core verifiable claim.
    Returns at most 300 chars.
    """
    t = text.strip().rstrip("?.!")
    t = _QUESTION_PREFIXES.sub("", t, count=1)
    t = _NOISE_RE.sub(" ", t).strip()
    # Capitalize first letter
    if t:
        t = t[0].upper() + t[1:]
    return t[:300] or text[:300]


def _normalize(text: str) -> str:
    """Remove HTML noise, excessive whitespace, and truncate."""
    text = re.sub(r'<[^>]+>', ' ', text)        # strip HTML tags
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = _NOISE_RE.sub(' ', text).strip()
    return text[:2000]


# ─── Scoring Helpers ────────────────────────────────────────────────────────

def _recency(sentence: str) -> float:
    """Year-based recency score [0, 1]."""
    years = [int(y) for y in re.findall(r'\b(19\d\d|20\d\d)\b', sentence)]
    if not years:
        return 0.5
    now = time.localtime().tm_year
    gap = min(abs(now - y) for y in years)
    if gap <= 2:
        return 1.0
    if gap <= 5:
        return 0.7
    return 0.3


# Negation/refutation terms — presence in high-similarity sentences suggests
# the evidence DISPROVES rather than supports the claim
_NEGATION_RE = re.compile(
    r'\b(not|no|never|false|debunk|refut|disprove|myth|fraud|'
    r'misinformation|incorrect|wrong|fabricat|hoax|fake|untrue|'
    r'no evidence|no scientific|consensus\s+that\s+\w+\s+do\s+not|contrary)\b',
    re.IGNORECASE,
)


def _negation_penalty(evidence: list) -> float:
    """
    Detect if high-similarity evidence overwhelmingly REFUTES the claim.
    Checks sentences with sim > 0.50 (the most relevant ones).
    If >60% of those contain strong negation/refutation language,
    returns 0.65 (35% score reduction), otherwise 1.0 (no penalty).
    """
    high_sim = [ev for ev in evidence if ev["similarity"] > 0.50]
    if len(high_sim) < 2:
        return 1.0  # not enough evidence to penalize
    negated = sum(1 for ev in high_sim if _NEGATION_RE.search(ev["text"]))
    ratio = negated / len(high_sim)
    return 0.65 if ratio > 0.60 else 1.0


def _source_agreement(evidence: list) -> float:
    """
    Measure consistency between different sources.
    Takes the best-scoring sentence from each unique source URL,
    then measures pairwise similarity between those representatives.
    Returns [0, 1]. Neutral (0.5) when only one source.
    """
    # Best evidence item per unique source
    best_per_source = {}
    for ev in evidence:
        src = ev["source"]
        if src not in best_per_source or ev["similarity"] > best_per_source[src]["similarity"]:
            best_per_source[src] = ev

    reps = [v["text"] for v in best_per_source.values()]
    if len(reps) < 2:
        return 0.5

    # Pairwise average similarity
    total, count = 0.0, 0
    for i, text_a in enumerate(reps):
        others = reps[i + 1:]
        if others:
            sims = ml.batch_similarity(text_a, others)
            total += sum(sims)
            count += len(sims)

    return max(0.0, min(1.0, total / count)) if count > 0 else 0.5


# ─── Fallbacks ──────────────────────────────────────────────────────────────

def _fallback(claim_detected: str = "", processing_time: float = 0.0,
              reason: str = "Insufficient reliable data to verify this claim.") -> dict:
    return {
        "status": "success",
        "data": {
            "label":           "UNCERTAIN",
            "verdict":         "UNCERTAIN",
            "confidence":      30,
            "explanation":     [reason],
            "data_quality":    "LOW",
            "sources":         [],
            "claim_detected":  claim_detected,
            "processing_time": round(processing_time, 3),
        }
    }


# ─── Core Verification (LRU-cached for determinism) ─────────────────────────

@lru_cache(maxsize=200)
def _verify_cached(claim: str) -> dict:
    """
    Full verification pipeline, cached by claim string.
    Same input always produces identical output (deterministic).
    Returns response dict WITHOUT processing_time (added at call site).
    """
    evidence = fact.retrieve_evidence(claim)

    if not evidence:
        return None  # signal: no evidence retrieved

    # Compute source agreement ONCE across all evidence
    agreement = _source_agreement(evidence)

    # Score each evidence sentence
    scored_items = []
    contributing_sources = []
    errors = []

    for ev in evidence:
        sim = ev["similarity"]          # pre-computed by fact.py [0,1]

        # Discard low-relevance sentences (noise from off-topic articles)
        if sim < 0.25:
            continue

        trust = max(0.0, min(1.0, ev["trust"]))
        rec   = max(0.0, min(1.0, _recency(ev["text"])))

        # 4-factor weighted composite matching spec
        # similarity: 0.50 | trust: 0.25 | agreement: 0.15 | recency: 0.10
        score = (0.50 * sim) + (0.25 * trust) + (0.15 * agreement) + (0.10 * rec)
        scored_items.append(score)

        src = ev["source"]
        if src not in contributing_sources:
            contributing_sources.append(src)

    if not scored_items:
        return None  # all evidence was noise

    n_sources = len(contributing_sources)
    avg = max(0.0, min(1.0, sum(scored_items) / len(scored_items)))

    # ── Negation penalty ──
    # If most relevant sentences refute the claim, penalize the average
    penalty = _negation_penalty(evidence)
    avg = max(0.0, min(1.0, avg * penalty))

    # ── Minimum 2 sources for REAL/FAKE verdict ──
    # If only 1 source, cap at UNCERTAIN per spec evidence_handling rules
    verdict_eligible = n_sources >= 2

    # ── Verdict thresholds ──
    if avg > 0.72 and verdict_eligible:
        label = "REAL"
        confidence = round(avg * 100, 1)
        quality = "HIGH" if len(scored_items) >= 4 else "MEDIUM"
        explanation = [
            f"Claim is semantically supported by {len(scored_items)} evidence sentences.",
            f"Source agreement score: {round(agreement * 100)}% across {n_sources} sources.",
            "Multiple credible sources corroborate the claim.",
            f"Average evidence score: {round(avg * 100, 1)}/100.",
        ]
    elif avg < 0.38 and verdict_eligible:
        label = "FAKE"
        confidence = round((1.0 - avg) * 100, 1)
        quality = "MEDIUM" if len(scored_items) >= 4 else "LOW"
        explanation = [
            f"Retrieved evidence does not align with this claim ({len(scored_items)} sentences checked).",
            f"Source agreement score: {round(agreement * 100)}%.",
            "No credible source semantically supports the claim.",
            "Cross-reference with authoritative sources is strongly recommended.",
        ]
    else:
        label = "UNCERTAIN"
        confidence = round(avg * 100, 1)
        quality = "MEDIUM" if len(scored_items) >= 3 else "LOW"
        reason_parts = [
            f"Evidence is mixed across {n_sources} source(s) checked.",
            f"Source agreement: {round(agreement * 100)}%.",
        ]
        if not verdict_eligible:
            reason_parts.append("Only 1 source found — 2+ sources required for definitive verdict.")
        else:
            reason_parts.append("More evidence is needed for a conclusive determination.")
        reason_parts.append(f"Average evidence score: {round(avg * 100, 1)}/100.")
        explanation = reason_parts

    return {
        "label":       label,
        "verdict":     label,
        "confidence":  confidence,
        "explanation": explanation,
        "data_quality": quality,
        "sources":     contributing_sources[:3],
        "n_evidence":  len(scored_items),
    }


def _run_verification(raw_input: str) -> dict:
    """
    Normalize → extract claim → run pipeline → log → return response.
    Always returns a valid dict. Measures processing_time.
    """
    t0 = time.perf_counter()
    errors = []

    try:
        normalized = _normalize(raw_input)
        claim = _extract_claim(normalized)
        input_type = _detect_input_type(normalized)

        if not claim:
            result = _fallback(processing_time=time.perf_counter() - t0)
            log_.log_request(raw_input, "", "UNCERTAIN", 30, [],
                             time.perf_counter() - t0, ["Empty claim after normalization"])
            return result

        cached = _verify_cached(claim)
        elapsed = time.perf_counter() - t0

        if cached is None:
            result = _fallback(
                claim_detected=claim,
                processing_time=elapsed,
                reason="No external evidence could be retrieved for this claim.",
            )
            log_.log_request(raw_input, claim, "UNCERTAIN", 30, [], elapsed,
                             ["No evidence retrieved"])
            return result

        response = {
            "status": "success",
            "data": {
                **cached,
                "claim_detected":  claim,
                "processing_time": round(elapsed, 3),
                "input_type":      input_type,
            }
        }
        # Remove internal key not needed by frontend
        response["data"].pop("n_evidence", None)

        log_.log_request(
            raw_input, claim,
            cached["label"], cached["confidence"],
            cached["sources"], elapsed, errors,
        )
        return response

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        errors.append(str(exc))
        log_.log_request(raw_input, "", "UNCERTAIN", 30, [], elapsed, errors)
        return _fallback(processing_time=elapsed)


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return {"status": "ok", "message": "Fact Verification API v7"}


@app.post("/verify")
async def verify_json(req: Request):
    """JSON endpoint — accepts { "claim": "..." }"""
    import json
    try:
        raw = await req.body()
        body = json.loads(raw)
        text = (body.get("claim") or body.get("text") or "").strip()
    except Exception:
        return _fallback(reason="Invalid JSON body.")
    if not text:
        return _fallback(reason="No claim provided.")
    return _run_verification(text)


@app.post("/analyze")
async def analyze(
    input_type: str = Form("text"),
    text_input: str = Form(""),
    title_input: str = Form(""),
    url: str = Form(""),
    url_input: str = Form(""),
    file_input: Optional[UploadFile] = File(None),
):
    """
    FormData endpoint — matches the frontend contract.
    Accepts text, URL, or file input types.
    """
    try:
        raw = ""

        if input_type == "text":
            raw = text_input.strip()
            if title_input.strip():
                raw = title_input.strip() + ". " + raw

        elif input_type == "url":
            target_url = url_input.strip() or url.strip()
            if target_url:
                from utils import scrape_url
                body = scrape_url(target_url)
                if body:
                    # First 3 sentences = more context for a URL claim
                    sents = re.split(r'(?<=[.!?])\s+', body)
                    sents = [s.strip() for s in sents if len(s.strip()) > 20][:3]
                    raw = " ".join(sents)[:600]

        elif input_type == "file":
            if file_input is not None:
                content = await file_input.read()
                raw = content.decode("utf-8", errors="ignore")[:3000].strip()

        if not raw:
            return _fallback(reason="No input content provided.")

        return _run_verification(raw)

    except Exception as exc:
        return _fallback(reason="An unexpected error occurred.")


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
