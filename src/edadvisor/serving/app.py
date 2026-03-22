from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from edadvisor.config import settings
from edadvisor.generation.chain import RAGChain
from edadvisor.ingestion.store import get_embedder, load_index
from edadvisor.serving.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceRef,
)

_boot = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("EdAdvisor API starting…")
    try:
        embedder = get_embedder()
        store = load_index(settings.vector_store_path, embedder)
        app.state.chain = RAGChain(store)
        logger.info("RAG chain ready")
    except Exception as e:
        logger.warning(f"index not available: {e}  — running in degraded mode")
        app.state.chain = None
    yield
    logger.info("shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="EdAdvisor GenAI",
        description="Domain-specific RAG for international student counselling",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


@app.post("/v1/chat", response_model=QueryResponse)
def chat(body: QueryRequest, request: Request) -> QueryResponse:
    if request.app.state.chain is None:
        raise HTTPException(503, "knowledge base not loaded — run `make ingest` first")

    result = request.app.state.chain.query(
        question=body.question,
        session_id=body.session_id,
    )

    return QueryResponse(
        answer=result.answer,
        sources=[SourceRef(**s) for s in result.sources],
        confidence=result.confidence,
        escalated=result.escalated,
        session_id=result.session_id,
        prompt_version=result.prompt_version,
    )


@app.get("/v1/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = request.app.state.chain is not None
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        index_loaded=loaded,
        uptime_seconds=round(time.monotonic() - _boot, 1),
    )


@app.delete("/v1/session/{session_id}")
def clear_session(session_id: str, request: Request) -> dict:
    if request.app.state.chain:
        request.app.state.chain._memory.clear(session_id)
    return {"cleared": session_id}
