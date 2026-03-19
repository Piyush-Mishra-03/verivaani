"""
VeriVaani — FastAPI Server
Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import time
import uuid
import sys, os

# Import pipeline
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import Post, process_post

app = FastAPI(
    title="VeriVaani — Vernacular Fact-Checker",
    description="Automated misinformation detection for Indian vernacular social media. Powered by Groq + ScaleDown RAG.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ──────────────────────────────────────────────────
class CheckRequest(BaseModel):
    text: str
    source: Optional[str] = "api"

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Chandrayaan-3 has landed on Mars! India conquers space!",
                "source": "Twitter"
            }
        }

class BatchCheckRequest(BaseModel):
    posts: list[CheckRequest]

    class Config:
        json_schema_extra = {
            "example": {
                "posts": [
                    {"text": "भारत की GDP 20% बढ़ी है इस साल!", "source": "WhatsApp"},
                    {"text": "UPI processed 10 billion transactions in 2023", "source": "Telegram"}
                ]
            }
        }

class CheckResponse(BaseModel):
    post_id: str
    language: str
    extracted_claim: str
    verdict: str
    confidence: float
    explanation: str
    original_tokens: int
    compressed_tokens: int
    compression_saving_pct: float
    processing_time_ms: float

# ── Landing page ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VeriVaani API</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #0a0a14; color: #e8e8f8; 
                   display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
            .card { background: #16161f; border: 1px solid #2a2a3a; border-radius: 16px; 
                    padding: 40px; max-width: 500px; text-align: center; }
            h1 { font-size: 2.5rem; margin: 0 0 8px; }
            .badge { background: rgba(108,99,255,0.2); border: 1px solid #6c63ff; color: #6c63ff;
                     padding: 4px 12px; border-radius: 20px; font-size: 12px; display: inline-block; margin-bottom: 20px; }
            p { color: #8888aa; line-height: 1.6; }
            .btn { display: inline-block; background: #6c63ff; color: white; padding: 12px 28px;
                   border-radius: 10px; text-decoration: none; font-weight: bold; margin: 8px; }
            .btn2 { background: transparent; border: 1px solid #2a2a3a; color: #8888aa; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="badge">Intel Unnati GenAI for GenZ</div>
            <h1>🔍 VeriVaani</h1>
            <p>Automated fact-checker for Indian vernacular news.<br>
               Powered by <strong>Groq</strong> + <strong>ScaleDown RAG</strong></p>
            <br>
            <a href="/docs" class="btn">📖 API Docs</a>
            <a href="/health" class="btn btn2">❤️ Health</a>
        </div>
    </body>
    </html>
    """

# Supports Hindi, Tamil, Telugu, Bengali, Marathi, English
# ── Single post check ──────────────────────────────────────────────────────────
@app.post("/check", response_model=CheckResponse, summary="Check a single post for misinformation")
async def check_single(req: CheckRequest):
    """
    Check a single social media post or news headline for misinformation.

    Supports all Indian languages — Hindi, Tamil, Telugu, Bengali, Marathi, and more.

    Returns verdict: **TRUE**, **FALSE**, **MISLEADING**, or **UNVERIFIABLE**
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long. Max 2000 characters.")

    post = Post(id=str(uuid.uuid4()), text=req.text, source=req.source or "api")
    try:
        result = process_post(post)
        return CheckResponse(
            post_id=result.post_id,
            language=result.language,
            extracted_claim=result.extracted_claim,
            verdict=result.verdict,
            confidence=result.confidence,
            explanation=result.explanation,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_saving_pct=result.compression_saving_pct,
            processing_time_ms=result.processing_time_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Batch check ────────────────────────────────────────────────────────────────
@app.post("/check/batch", summary="Check multiple posts (high throughput)")
async def check_batch(req: BatchCheckRequest):
    """
    Check multiple posts at once.

    Posts are processed sequentially with caching — duplicate posts are returned instantly.
    Max 20 posts per batch.
    """
    if len(req.posts) > 20:
        raise HTTPException(status_code=400, detail="Max 20 posts per batch")
    if len(req.posts) == 0:
        raise HTTPException(status_code=400, detail="At least 1 post required")

    start = time.time()
    results = []

    for p in req.posts:
        post = Post(id=str(uuid.uuid4()), text=p.text, source=p.source or "api")
        try:
            r = process_post(post)
            results.append({
                "post_id": r.post_id,
                "language": r.language,
                "extracted_claim": r.extracted_claim,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "explanation": r.explanation,
                "compression_saving_pct": r.compression_saving_pct,
                "processing_time_ms": r.processing_time_ms
            })
        except Exception as e:
            results.append({
                "post_id": str(uuid.uuid4()),
                "error": str(e),
                "original_text": p.text[:100]
            })

    total_ms = (time.time() - start) * 1000
    verdicts = [r.get("verdict") for r in results if "verdict" in r]

    return {
        "total_posts": len(results),
        "total_time_ms": round(total_ms, 1),
        "verdict_summary": {
            "TRUE":          verdicts.count("TRUE"),
            "FALSE":         verdicts.count("FALSE"),
            "MISLEADING":    verdicts.count("MISLEADING"),
            "UNVERIFIABLE":  verdicts.count("UNVERIFIABLE"),
        },
        "results": results
    }

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "healthy",
        "service": "VeriVaani Fact-Checker",
        "version": "1.0.0",
        "timestamp": time.time()
    }