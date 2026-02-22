"""
AutoFixOps v2 — Real AI-Powered Environment Debugging Assistant

Request lifecycle:
  1. RequestIDMiddleware assigns UUID4 → stored in ContextVar → echoed in X-Request-ID header
  2. Route handler calls ClaudeDebugAgent with request_id for correlated logging
  3. Response time measured server-side and returned in payload + recorded to MetricsStore
  4. MetricsStore records request_id so a full session can be reconstructed
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from backend.claude_agent import ClaudeDebugAgent
from backend.executor import SafeCommandExecutor
from backend.metrics import store
from backend.tracing import RequestIDMiddleware, configure_logging, get_request_id


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    for d in ["frontend/static", "frontend/templates"]:
        if not Path(d).exists():
            raise RuntimeError(f"Missing required directory: {d}/")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — AI analysis unavailable.")
    else:
        logger.info("AutoFixOps v2 started — Claude AI reasoning enabled.")
    yield


app = FastAPI(
    title="AutoFixOps v2",
    description="AI-Powered Environment Debugging — Claude LLM + ReAct Agent Loop",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────
# Must be added before any routes are registered.
# Every request gets a UUID4 in X-Request-ID header and ContextVar.
app.add_middleware(RequestIDMiddleware)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")


# ── Request / Response Models ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    log: str
    context: str = ""


class ExecuteRequest(BaseModel):
    commands: list[str]
    safe_mode: bool = True


class FeedbackLoopRequest(BaseModel):
    original_log: str
    applied_fix: str
    new_log: str


class AnalysisResponse(BaseModel):
    request_id: str           # UUID for this debugging session — correlates all steps
    error_type: str
    error_category: str
    severity: str
    confidence: str
    reasoning_summary: str    # Curated summary — internal reasoning is hidden
    root_cause: str
    explanation: str
    fix_steps: list[str]
    safe_commands: list[str]
    prevention_tips: list[str]
    model_used: str
    response_time_ms: int     # Measured server-side


class ExecutionResult(BaseModel):
    command: str
    stdout: str
    stderr: str
    exit_code: int
    safe: bool


class FeedbackResponse(BaseModel):
    request_id: str
    fix_worked: bool
    analysis: str
    next_steps: list[str]
    still_broken: bool


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_log(payload: AnalyzeRequest):
    """
    LLM-powered analysis via Claude API.
    - Request ID flows from middleware → agent call → response → metrics record.
    - Response time measured server-side; recorded to rolling p50/p95/p99 window.
    """
    req_id   = get_request_id()
    log_text = payload.log.strip()

    if not log_text:
        raise HTTPException(status_code=400, detail="No error log provided.")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not configured. Claude AI is unavailable."
        )

    logger.info("Analysis request received (log_len=%d)", len(log_text))

    agent = ClaudeDebugAgent(api_key=api_key)
    t0    = time.perf_counter()
    result = await agent.analyze(
        error_log=log_text,
        context=payload.context,
        request_id=req_id,       # ← propagated for correlated logging inside the agent
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "Analysis complete — category=%s severity=%s elapsed=%dms",
        result.get("error_category"), result.get("severity"), elapsed_ms,
    )

    is_fallback   = result.get("error_type") == "Analysis Unavailable"
    is_ratelimit  = "rate limit"  in result.get("reasoning_summary", "").lower()
    is_timeout    = "unavailable" in result.get("reasoning_summary", "").lower()
    is_parse_err  = "parsing"     in result.get("reasoning_summary", "").lower()

    await metrics.record_analysis(
        request_id=req_id,
        category=result.get("error_category", "unknown"),
        severity=result.get("severity", "medium"),
        response_time_s=elapsed_ms / 1000,
        api_error=is_fallback,
        rate_limited=is_ratelimit,
        timed_out=is_timeout,
        parse_error=is_parse_err,
    )

    return AnalysisResponse(request_id=req_id, response_time_ms=elapsed_ms, **result)


@app.post("/execute", response_model=list[ExecutionResult])
async def execute_commands(payload: ExecuteRequest):
    """
    ReAct 'Act' step: run whitelisted diagnostic commands, return stdout/stderr.
    """
    req_id   = get_request_id()
    executor = SafeCommandExecutor(safe_mode=payload.safe_mode)

    results = []
    for cmd in payload.commands:
        results.append(ExecutionResult(**(await executor.run(cmd))))

    logger.info(
        "Execution batch complete — %d commands, %d succeeded",
        len(payload.commands),
        sum(1 for r in results if r.exit_code == 0),
    )

    await metrics.record_execution(
        request_id=req_id,
        commands_run=len(payload.commands),
    )
    return results


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback_loop(payload: FeedbackLoopRequest):
    """
    ReAct 'Reflect' step: Claude evaluates whether the applied fix worked.
    """
    req_id  = get_request_id()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set.")

    agent  = ClaudeDebugAgent(api_key=api_key)
    result = await agent.evaluate_fix(
        original_log=payload.original_log,
        applied_fix=payload.applied_fix,
        new_log=payload.new_log,
        request_id=req_id,
    )

    logger.info("Feedback evaluation — fix_worked=%s", result.get("fix_worked"))
    await metrics.record_feedback(
        request_id=req_id,
        fix_worked=result.get("fix_worked", False),
    )

    return FeedbackResponse(request_id=req_id, **result)


@app.get("/metrics")
async def get_metrics():
    """
    Live service metrics: totals, p50/p95/p99 latency, category/severity
    breakdowns, fix success rate, API health, recent request IDs.
    """
    return await metrics.snapshot()


@app.get("/health")
async def health():
    snap = await store.snapshot()
    return {
        "status":         "ok",
        "service":        "AutoFixOps v2",
        "ai_enabled":     bool(os.getenv("ANTHROPIC_API_KEY")),
        "model":          "claude-sonnet-4-20250514",
        "uptime":         snap["uptime_human"],
        "total_analyses": snap["totals"]["analyses"],
    }


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "request_id": get_request_id()},
    )

@app.exception_handler(500)
async def server_error(request: Request, exc):
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": get_request_id()},
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
