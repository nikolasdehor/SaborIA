"""
SaborAI API — FastAPI application.

Endpoints:
  POST /ingest/file   — upload a menu PDF or TXT
  POST /ingest/text   — ingest raw text
  POST /query         — query the multi-agent system
  POST /evaluate      — run the eval suite
  GET  /health        — health check
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.supervisor import SupervisorAgent
from evals.runner import run_evals
from ingestion.pipeline import ingest_file, ingest_text

logging.basicConfig(level="INFO", format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SaborAI",
    description="Multi-agent RAG system for restaurant menu analysis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

supervisor = SupervisorAgent()

# ── Models ────────────────────────────────────────────────────────────────────


class TextIngestRequest(BaseModel):
    menu_name: str
    text: str


class QueryRequest(BaseModel):
    query: str
    menu_name: str | None = None


class EvalRequest(BaseModel):
    suite: str = "default"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0", "app": "SaborAI"}


@app.post("/ingest/file", summary="Upload a menu PDF or TXT file")
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    menu_name: str = Form(...),
):
    allowed = {".pdf", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"File type '{ext}' not supported. Use PDF or TXT.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = ingest_file(tmp_path, menu_name)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"status": "ingested", **result}


@app.post("/ingest/text", summary="Ingest raw menu text")
def ingest_text_endpoint(req: TextIngestRequest):
    result = ingest_text(req.text, req.menu_name)
    return {"status": "ingested", **result}


@app.post("/query", summary="Query the multi-agent system")
def query_endpoint(req: QueryRequest):
    try:
        result = supervisor.run(req.query, req.menu_name)
        return result
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(500, str(exc)) from exc


@app.post("/evaluate", summary="Run the eval framework")
def evaluate_endpoint(req: EvalRequest):
    try:
        report = run_evals(suite=req.suite)
        return report
    except Exception as exc:
        logger.exception("Eval failed")
        raise HTTPException(500, str(exc)) from exc
