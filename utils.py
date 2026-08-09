"""
utils.py — I/O utilities for the fact verification pipeline.

scrape_url: fetches a URL, strips HTML, returns plain text.
  - 5-second timeout on all HTTP requests
  - 2 retry attempts
  - Returns empty string on any failure (never throws)
  - Results cached via @lru_cache (URL → body text)
"""

import os
import re
from functools import lru_cache

import requests
from bs4 import BeautifulSoup


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 5        # seconds per request
_MAX_RETRIES = 2
_MAX_PARAGRAPHS = 25  # limit DOM traversal


@lru_cache(maxsize=200)
def scrape_url(url: str) -> str:
    """
    Fetch a URL and return extracted paragraph text.
    Never raises. Returns "" on any failure.
    Cached by URL string.
    """
    for _ in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.content, "html.parser")
            # Strip non-content tags
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            paragraphs = [
                p.get_text(separator=" ", strip=True)
                for p in soup.find_all("p")[:_MAX_PARAGRAPHS]
            ]
            body = " ".join(p for p in paragraphs if len(p) > 20)
            if body:
                return body
        except Exception:
            continue
    return ""


def extract_text(file_path: str) -> str:
    """Read a .txt file and return its contents. Returns '' on failure."""
    if not file_path or not os.path.isfile(file_path):
        return ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""
