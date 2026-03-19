# 🔍 VeriVaani — Automated Fact-Checker for Vernacular News

> **Intel Unnati GenAI for GenZ** — Problem Statement: Automated Fact-Checker for Vernacular News

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20(Free)-green)](https://console.groq.com)
[![ScaleDown](https://img.shields.io/badge/RAG-ScaleDown-orange)](https://scaledown.ai)
[![Cost](https://img.shields.io/badge/Cost-%240.00-brightgreen)](https://console.groq.com)

---

## 📌 Problem Statement

Misinformation spreads rapidly on social media in India. Human fact-checkers cannot keep up with the volume. Most AI fact-checking tools only work in English — leaving 900M+ vernacular internet users unprotected.

**VeriVaani** solves this with a scalable, multilingual, AI-powered fact-checking pipeline that works in Hindi, Tamil, Telugu, Bengali, Marathi, and all Indian languages.

---

## 🏗️ Architecture

```
Raw Social Media Post (any Indian language)
        │
        ▼
[1] 🌐 Language Detection      (Groq — llama-3.3-70b)
        │
        ▼
[2] 🔄 Translation to English  (Groq — llama-3.3-70b)
        │
        ▼
[3] ✂️  Claim Extraction        (Groq) ← PIPELINE OPTIMIZATION ★
        │   Strips emojis, hashtags, greetings, emotional fluff
        │   Reduces tokens by ~70-80% before RAG
        ▼
[4] 🗜️  ScaleDown RAG Compress  (ScaleDown API)
        │   Dumps entire verified facts DB as context
        │   Returns only facts relevant to the claim
        │   ~94% average token reduction
        ▼
[5] ⚖️  Claim Verification      (Groq — llama-3.3-70b)
        │   Verifies claim against compressed relevant facts
        ▼
   Verdict: ✅ TRUE / ❌ FALSE / ⚠️ MISLEADING / ❓ UNVERIFIABLE
```

### Required Technique: Pipeline Optimization
The claim extraction step (Stage 3) is the **required pipeline optimization technique**. It strips all conversational fluff from social media posts before sending to the LLM, reducing token usage by ~70-80% and enabling high throughput processing.

Combined with ScaleDown's RAG compression (Stage 4), the system achieves **~94% average token savings** on the verification step.

---

## 📊 Live Results

| Language | Post | Verdict | Tokens Saved |
|----------|------|---------|-------------|
| Hindi | भारत की GDP 20% बढ़ी है | ❌ FALSE | 97.4% |
| English | Chandrayaan-3 landed on Mars | ❌ FALSE | 89.1% |
| Tamil | இந்தியா மக்கள் தொகை 200 கோடி | ❌ FALSE | 91.7% |
| English | UPI crossed 10B transactions in 2023 | ✅ TRUE | 98.1% |
| English | Good morning! Have a great day! | ❓ NO CLAIM | — |

**Average token saving: 94.1% | Total cost: $0.00**

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Groq (llama-3.3-70b) | Free, fast inference |
| RAG | ScaleDown API | Vector-DB-free fact retrieval |
| Backend | FastAPI + Python | REST API server |
| Languages | Auto-detect | All 22 Indian languages |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Piyush-Mishra-03/verivaani.git
cd verivaani
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get free API keys
- **Groq** (free): [console.groq.com](https://console.groq.com) → Create API Key
- **ScaleDown**: Your existing key

### 4. Add your keys
Open `pipeline_groq.py` and update lines 20-21:
```python
GROQ_API_KEY      = "gsk_your_groq_key_here"
SCALEDOWN_API_KEY = "your_scaledown_key_here"
```

### 5. Run the demo
```bash
python pipeline_groq.py
```

### 6. Run the API server
```bash
uvicorn api:app --reload --port 8000
```
Then open **http://localhost:8000/docs** to test via browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/check` | Check a single post |
| POST | `/check/batch` | Check multiple posts (high throughput) |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive Swagger UI |

### Example Request
```bash
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "Chandrayaan-3 has landed on Mars!", "source": "Twitter"}'
```

### Example Response
```json
{
  "verdict": "FALSE",
  "confidence": 1.0,
  "language": "English",
  "extracted_claim": "Chandrayaan-3 has landed on Mars.",
  "explanation": "Chandrayaan-3 landed on the Moon, not Mars, according to ISRO.",
  "original_tokens": 579,
  "compressed_tokens": 63,
  "compression_saving_pct": 89.1,
  "processing_time_ms": 1477
}
```

---

## 📁 Project Structure

```
verivaani/
├── pipeline_groq.py     # Core fact-checking pipeline
├── api.py               # FastAPI REST server
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🔑 Key Innovations

1. **No Vector Database** — ScaleDown replaces Pinecone/ChromaDB with a single API call
2. **Pipeline Optimization** — Claim extraction reduces tokens ~80% before verification
3. **Truly Multilingual** — Works with all 22 official Indian languages + dialects
4. **Zero Cost** — Groq's free tier handles all LLM inference
5. **Smart Skipping** — Posts with no factual claim are detected and skipped instantly

---

## 👥 Team
Intel Unnati GenAI for GenZ Participants

## Changelog
- v1.0: Core pipeline with ScaleDown RAG + Groq
---

