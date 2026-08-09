"""
ml.py — ML module. Single model, single responsibility.

Loads all-MiniLM-L6-v2 once on first use (lazy).
Provides batch_similarity(query, docs) → list[float] for efficient inference.
Falls back to word-overlap (Jaccard) if model fails to load.
"""

from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer, util

# Enforce CPU + single thread before any model loads
torch.set_num_threads(1)

_model: Optional[SentenceTransformer] = None
_model_failed = False


def _get_model() -> Optional[SentenceTransformer]:
    global _model, _model_failed
    if _model_failed:
        return None
    if _model is not None:
        return _model
    try:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return _model
    except Exception:
        _model_failed = True
        return None


def _jaccard(a: str, b: str) -> float:
    """Word-overlap fallback. Returns [0, 1]."""
    wa = set(a.lower().split())
    wb = set(b.lower().split()[:100])
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def batch_similarity(query: str, docs: List[str]) -> List[float]:
    """
    Compute cosine similarity between one query and N documents.
    Returns list of floats, each clamped to [0, 1].
    Encodes query ONCE, encodes all docs in a single batch → fast.
    Falls back to Jaccard on failure.
    """
    if not query or not docs:
        return [0.0] * len(docs)

    model = _get_model()
    if model is None:
        return [_jaccard(query, d) for d in docs]

    try:
        # Truncate each doc to 256 chars to keep encoding fast
        truncated = [d[:256] for d in docs]
        q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        d_embs = model.encode(truncated, convert_to_tensor=True,
                              normalize_embeddings=True, batch_size=32)
        scores = util.cos_sim(q_emb, d_embs)[0].tolist()
        return [max(0.0, min(1.0, s)) for s in scores]
    except Exception:
        return [_jaccard(query, d) for d in docs]


def similarity(query: str, doc: str) -> float:
    """Single-pair convenience wrapper. Returns [0, 1]."""
    if not query or not doc:
        return 0.0
    return batch_similarity(query, [doc])[0]
