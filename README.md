# AI-Powered Truth Verification Engine 🚀

A production-ready, explainable AI-powered misinformation verification engine using multi-signal reasoning and real-time evidence analysis. 

Built to elevate fake news detection from simple heuristic parsing to a highly robust, defensible, and intelligent architecture capable of being demonstrated to startup judges and investors.

## 🧠 Architectural Overview

This system fundamentally rejects the "black box" approach to Fake News detection. Instead of using a single ML model to hallucinate a "FAKE" or "REAL" label, it implements a **Multi-Signal Reality Engine**.

The final verdict and confidence score are driven by the variance and consensus between the following intelligence signals:

### 1. Linguistic Tone Signal (`transformers`)
We utilize a localized HuggingFace `distilbert` pipeline to analyze the input for sensational, manipulative, or emotionally volatile language. This provides a `language_tone_signal` rather than a ground-truth declaration.

### 2. Live Fact Verification (`sentence-transformers` + `GNews API`)
The system intelligently parses the input text for hard claims (numbers, entities, verbs) and runs a semantic search against live news feeds. It uses the `paraphrase-MiniLM-L3-v2` SentenceTransformer to compute cosine similarity against live evidence, outputting a highly accurate Fact Score.

### 3. Domain Credibility Engine
Strictly parses the domain of provided URLs to score inherent source credibility, punishing known satire/misinformation networks and rewarding established journalistic outlets.

### 4. Bias & Consistency Analysis
Penalizes emotional biases and calculates headline-to-body semantic mismatch to detect clickbait framing.

## ⚡ Production Hardening

- **Global Model Loading:** Heavy NLP models are loaded strictly once at server startup, eliminating request-flow bottlenecks and drastically improving latency.
- **Fail-Safe Degradation:** If external APIs (like GNews) are down or rate-limited, the system refuses to fake data. It degrades gracefully, capping its own confidence and flagging the data quality prominently.
- **Rate Limiting:** Protects the backend with an in-memory IP rate limiter (5 requests / minute).
- **TTL Caching:** Implements a time-to-live and size-bounded cache to prevent memory leaks while returning instant results for redundant queries.

## 🛠 Tech Stack

- **Backend:** Flask, Python 3.9+
- **Machine Learning:** PyTorch, Transformers, Sentence-Transformers
- **Frontend:** Vanilla JS, CSS3 (Glassmorphism, CSS Animations)
- **Deployment:** Render (with custom cold-start UI mitigation)

## 🚀 Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your GNews API Key in a `.env` file:
```env
GNEWS_API_KEY=your_key_here
```

3. Run the Flask server:
```bash
python3 app.py
```

4. Open `http://127.0.0.1:8080` in your browser.

> Note: The first time you execute a query, the backend will download the `distilbert` and `paraphrase-MiniLM` model weights. Subsequent runs will be extremely fast.
