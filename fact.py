"""
fact.py — Evidence retrieval and ranking module.

Pipeline:
  1. Search (DDG → Wikipedia API fallback)
  2. Scrape (cached, timeout-safe via utils.scrape_url)
  3. Sentence extraction (split, filter, truncate)
  4. Rank by semantic similarity (ml.batch_similarity — single batch)
  5. Return top-K evidence with trust scores

Constraints:
  - Max 3 URLs searched
  - Max 20 sentences extracted per article
  - Max 5 sentences returned per article (highest similarity)
  - Max 9 total evidence sentences returned
  - Each sentence truncated to 300 chars
"""

import re
import urllib.parse
from functools import lru_cache
from typing import Dict, List, Tuple

import requests
from utils import scrape_url
import ml


# ─── Domain Trust ───────────────────────────────────────────────────────────

TRUST_SCORES = {
    "reuters.com": 1.0,  "apnews.com": 1.0,
    "bbc.com": 1.0,      "bbc.co.uk": 1.0,
    "wikipedia.org": 0.9, "en.wikipedia.org": 0.9,
    "nytimes.com": 0.85, "washingtonpost.com": 0.85,
    "theguardian.com": 0.85, "npr.org": 0.85,
    "bloomberg.com": 0.8, "wsj.com": 0.8,
    "snopes.com": 0.9,   "politifact.com": 0.9,
    "medium.com": 0.4,   "blogspot.com": 0.35,
    "wordpress.com": 0.35,
}


def _trust_for_url(url: str) -> float:
    """Return trust score [0.0, 1.0] for a URL's domain. Unknown → 0.5."""
    try:
        domain = urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
        for key, score in TRUST_SCORES.items():
            if domain == key or domain.endswith("." + key):
                return score
    except Exception:
        pass
    return 0.5


# ─── Sentence Extraction ───────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _extract_sentences(text: str, max_count: int = 20) -> List[str]:
    """Split text into sentences. Keep those > 40 chars, truncate to 300."""
    parts = _SENTENCE_RE.split(text)
    out = []
    for s in parts:
        s = s.strip()
        if len(s) > 40:
            out.append(s[:300])
            if len(out) >= max_count:
                break
    return out


# ─── Search ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=200)
def _search(query: str) -> Tuple[str, ...]:
    """
    Search DDG (max 3 results). Falls back to Wikipedia API if DDG fails
    or returns nothing (rate-limited). Returns tuple of URLs.
    """
    # DuckDuckGo — try new package name (ddgs), fall back to old (duckduckgo_search)
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=3))
        if hits:
            urls = [h["href"] for h in hits if h.get("href")]
            if urls:
                return tuple(urls[:3])
    except Exception:
        pass

    # Wikipedia fallback
    try:
        api_url = (
            "https://en.wikipedia.org/w/api.php"
            "?action=query&list=search"
            f"&srsearch={urllib.parse.quote(query)}"
            "&srlimit=3&format=json"
        )
        resp = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if resp.status_code == 200:
            search_hits = resp.json().get("query", {}).get("search", [])
            links = [
                f"https://en.wikipedia.org/wiki/{urllib.parse.quote(h['title'])}"
                for h in search_hits
            ]
            if links:
                return tuple(links[:3])
    except Exception:
        pass

    return ()


# ─── Main retrieval ─────────────────────────────────────────────────────────

def retrieve_evidence(claim: str) -> List[Dict]:
    """
    Returns list of evidence dicts sorted by relevance:
      { "text": str, "source": str, "trust": float, "similarity": float }

    The similarity is pre-computed here so app.py doesn't need to call
    the model again — no duplicated work.
    """
    urls = _search(claim)
    if not urls:
        return []

    all_sentences: List[Dict] = []   # { text, source, trust }

    for url in urls[:3]:
        body = scrape_url(url)
        if not body:
            continue
        sentences = _extract_sentences(body)
        if not sentences:
            continue
        trust = _trust_for_url(url)

        # Batch-score all sentences for this article in ONE call
        scores = ml.batch_similarity(claim, sentences)

        # Pair and keep top 5 by similarity
        paired = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)
        for sim, sent in paired[:5]:
            all_sentences.append({
                "text":       sent,
                "source":     url,
                "trust":      trust,
                "similarity": sim,
            })

    # Sort all evidence across articles by similarity, return top 9
    all_sentences.sort(key=lambda x: x["similarity"], reverse=True)
    return all_sentences[:9]
