"""
FastAPI Backend — Exposes the AI agent via a REST API.

Endpoints:
  POST /ask     — Query the AI agent
  GET  /health  — Health check
"""

import os
import re
import uuid
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging & Azure Monitor (Bonus)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Automatically configure Azure Monitor if connection string is provided
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor()
        logger.info("Azure Monitor (Application Insights) configured successfully.")
    except ImportError:
        logger.warning("azure-monitor-opentelemetry not installed. Azure Monitor is disabled.")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Agent RAG API",
    description="RAG-powered AI agent for answering questions about internal company documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation memory (auto-generated if omitted)"
    )

class QueryResponse(BaseModel):
    answer: str
    source: List[str]
    session_id: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_sources(intermediate_steps) -> List[str]:
    """Extract source document filenames from LangChain intermediate steps."""
    sources: set[str] = set()
    for step in intermediate_steps:
        step_str = str(step)
        # Match file paths like documents/hr_policy.txt or documents\hr_policy.txt
        found = re.findall(r"documents[/\\][\w\-.]+", step_str)
        sources.update(found)
    return sorted(sources) if sources else ["llm_direct"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """Query the AI agent. It will decide whether to search documents or answer directly."""
    # Lazy import to avoid loading the vectorstore before the app is ready
    from agent import build_agent

    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"[{session_id}] Query: {request.query}")

    try:
        agent_executor = build_agent(session_id)
        result = agent_executor.invoke({"input": request.query})

        sources = _extract_sources(result.get("intermediate_steps", []))
        answer = result["output"]
        logger.info(f"[{session_id}] Sources: {sources}")

        return QueryResponse(
            answer=answer,
            source=sources,
            session_id=session_id,
        )

    except Exception as e:
        logger.error(f"[{session_id}] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
def root():
    """Root endpoint — redirects to docs."""
    return {
        "message": "AI Agent RAG API is running. Visit /docs for Swagger UI.",
        "endpoints": {
            "POST /ask": "Query the AI agent",
            "GET /health": "Health check",
            "GET /docs": "Swagger UI documentation",
        },
    }
