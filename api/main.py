"""
SaborAI API — FastAPI application.

Endpoints:
  POST /ingest/file   — upload a menu PDF or TXT
  POST /ingest/text   — ingest raw text
  POST /query         — query the multi-agent system (async, parallel agents)
  POST /query/stream  — query with Server-Sent Events streaming
  POST /evaluate      — run the eval suite
  GET  /health        — health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.supervisor import SupervisorAgent
from api.middleware import RequestTrackingMiddleware, configure_logging
from evals.runner import run_evals
from ingestion.pipeline import ingest_file, ingest_text

configure_logging(level="INFO", structured=True)
logger = logging.getLogger(__name__)

APP_VERSION = "0.2.0"

app = FastAPI(
    title="SaborAI",
    description="Multi-agent RAG system for restaurant menu analysis",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestTrackingMiddleware)

supervisor = SupervisorAgent()

# ── Models ────────────────────────────────────────────────────────────────────


class TextIngestRequest(BaseModel):
    menu_name: str
    text: str


class QueryRequest(BaseModel):
    query: str
    menu_name: Optional[str] = None


class EvalRequest(BaseModel):
    suite: str = "default"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION, "app": "SaborAI"}


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


@app.post("/query", summary="Query the multi-agent system (async, parallel agents)")
async def query_endpoint(req: QueryRequest):
    try:
        result = await supervisor.arun(req.query, req.menu_name)
        return result
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(500, str(exc)) from exc


@app.post("/query/stream", summary="Query with Server-Sent Events streaming")
async def query_stream_endpoint(req: QueryRequest):
    """Stream intermediate results as SSE events.

    Event types:
      - ``routing``   — agent selection result
      - ``agent``     — individual agent output (one per agent)
      - ``response``  — final consolidated response
      - ``done``      — end-of-stream marker
    """

    async def _event_generator():
        try:
            # 1. Route
            selected = await supervisor._aroute(req.query)
            yield _sse("routing", {"agents": selected})

            # 2. Execute agents in parallel, yield each as it completes
            async def _invoke(name: str):
                try:
                    result = await supervisor.agents[name].arun(req.query, req.menu_name)
                    return name, result
                except Exception as exc:
                    return name, f"[Erro no agente {name}: {exc}]"

            tasks = {
                asyncio.ensure_future(_invoke(n)): n for n in selected
            }
            agent_outputs: dict[str, str] = {}
            for coro in asyncio.as_completed(tasks):
                name, output = await coro
                agent_outputs[name] = output
                yield _sse("agent", {"agent": name, "output": output})

            # 3. Consolidate
            final = await supervisor._aconsolidate(req.query, agent_outputs)
            yield _sse("response", {
                "query": req.query,
                "agents_used": selected,
                "response": final,
            })

            yield _sse("done", {})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/evaluate", summary="Run the eval framework")
def evaluate_endpoint(req: EvalRequest):
    try:
        report = run_evals(suite=req.suite)
        return report
    except Exception as exc:
        logger.exception("Eval failed")
        raise HTTPException(500, str(exc)) from exc
