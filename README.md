# ⚡ AutoFixOps v2 — AI-Powered Environment Debugging Assistant

A full-stack AI-assisted debugging platform using the **Anthropic Claude API** for dynamic
LLM-powered terminal error analysis, safe command execution, and a self-healing feedback loop.

---
## Problem Statement

Developers frequently copy raw stack traces into search engines or forums.
AutoFixOps streamlines this workflow by programmatically analyzing error logs,
classifying root causes, and suggesting actionable remediation steps using
LLM-powered reasoning.

---
## Architecture

```
User (browser)
    │
    ▼
FastAPI  (backend/main.py)           ← Routing, Pydantic validation, metrics wiring
    │
    ├── ClaudeDebugAgent             ← Anthropic Messages API client
    │   (backend/claude_agent.py)       Structured reasoning summary (not raw CoT)
    │                                   Retry: exponential backoff on 429 / 5xx
    │                                   Structured error responses on all failure paths
    │                                   Input truncation for token cost control
    │
    ├── SafeCommandExecutor          ← ReAct "Act → Observe" step
    │   (backend/executor.py)           create_subprocess_exec (no shell=True)
    │                                   shlex.split prevents injection
    │                                   Whitelist + forbidden-pattern enforcement
    │
    └── MetricsStore                 ← In-process observability
        (backend/metrics.py)            Response time percentiles (p50/p95/p99)
                                        Error category + severity breakdowns
                                        Fix success rate, API error rate, uptime
```

### ReAct Agent Loop

Reason → Claude analyzes log → structured reasoning summary returned

---

## Engineering Decisions

### 1. Structured Reasoning Summary

Claude is prompted to reason **internally** before producing output.
The response surfaces a `reasoning_summary` field — a curated 1–2 sentence
explanation of how the conclusion was reached — rather than raw chain-of-thought.

```
"reasoning_summary": "The error unambiguously identifies a missing pip package.
 Confidence is high because the module name is explicit and the error type
 leaves no ambiguity."
```

This is the correct production LLM pattern: structured transparency without
leaking internal prompt mechanics or fragile reasoning traces.

### 2. Retry with Exponential Backoff

```python
# _post_with_retry() in claude_agent.py

for attempt in range(MAX_RETRIES):          # MAX_RETRIES = 3
    response = await client.post(...)

    if response.status_code == 429:
        wait = retry_after_header + jitter  # Respects Retry-After
        await asyncio.sleep(wait)
        continue

    if response.status_code >= 500:
        wait = BASE_BACKOFF ** attempt + jitter
        await asyncio.sleep(wait)
        continue

    return response.json()

# Exhausted → raise RateLimitError or APIError
# Caught at call site → structured fallback dict returned to user
```

All failure paths return structured dicts. The user sees a clean message
("Rate limit reached. Please wait..."), never a raw stack trace.

### 3. Live Metrics Dashboard

`/metrics` endpoint returns a real-time snapshot. The UI polls it every 15 seconds.

```
Totals         → analyses run, commands executed, feedback evals
Breakdowns     → by error category (python/docker/npm/...), by severity
Response times → p50 / p95 / p99 latency in ms (rolling 200-sample window)
API health     → error rate %, rate limit hits, timeout hits, parse errors
Fix success    → % of feedback evaluations where Claude confirmed fix worked
Uptime         → human-readable session duration
```

Implemented in `backend/metrics.py` as an async-safe in-process store.
For production scale: swap the in-memory store for Redis or Prometheus counters.

### 4. Shell Injection Prevention

```python
# ❌ Before — shell interpretation active
proc = await asyncio.create_subprocess_shell(cmd, ...)

# ✅ After — shell disabled entirely
argv = shlex.split(cmd)
proc = await asyncio.create_subprocess_exec(*argv, ...)
```

---
## Production Considerations

- Replace in-memory metrics store with Redis or Prometheus
- Add request rate limiting (e.g., SlowAPI)
- Add circuit breaker around external LLM calls
- Add request body size limits to prevent abuse
- Introduce structured logging (JSON logs)
- Add caching for repeated identical error logs

---
## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`       | Frontend UI |
| `POST` | `/analyze` | LLM-powered error analysis |
| `POST` | `/execute` | Safe diagnostic command execution |
| `POST` | `/feedback` | Self-healing feedback loop |
| `GET`  | `/metrics` | Live system metrics snapshot |
| `GET`  | `/health`  | Service health + AI key status |

### GET /metrics — Response Shape

```json
{
  "uptime_human": "4m 12s",
  "totals": { "analyses": 14, "commands_run": 31, "feedback_evals": 5 },
  "by_category": { "python": 8, "docker": 4, "npm": 2 },
  "by_severity":  { "high": 9, "medium": 3, "critical": 2 },
  "response_time_ms": { "p50": 1840, "p95": 3120, "p99": 4500, "samples": 14 },
  "fix_success_rate_pct": 80.0,
  "api_health": { "error_rate_pct": 0.0, "rate_limit_hits": 0, "timeout_hits": 0 }
}
```

---

## Quick Start

```bash
git clone https://github.com/yourname/autofixops.git
cd autofixops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-your-key-here
python -m uvicorn backend.main:app --reload --port 8000
```

### Docker

```bash
docker build -t autofixops .
docker run -e ANTHROPIC_API_KEY=sk-ant-your-key-here -p 8000:8000 autofixops
```


---

## Project Structure
autofixops/
├── backend/
│   ├── main.py            # FastAPI app, routes, metrics wiring
│   ├── claude_agent.py    # Anthropic API client, retry, fallbacks
│   ├── executor.py        # Safe subprocess execution engine
│   └── metrics.py         # In-process observability store
├── frontend/
│   ├── templates/
│   │   └── index.html     # Single-page UI + live metrics dashboard
│   └── static/
│       └── style.css      # Industrial terminal aesthetic
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI |
| AI/LLM | Anthropic Claude API (claude-3-sonnet-20240229) |
| HTTP Client | httpx (async) |
| Frontend | HTML5, CSS3, Vanilla JS |
| Templating | Jinja2 |
| Container | Docker |

