"""
MetricsStore — In-process observability for AutoFixOps v2.

Tracks per request_id so a full debugging session can be reconstructed:
  analyze → execute → feedback — correlated by the same UUID.

Production note: this is intentionally in-memory.
For production scale, replace with Redis (shared across workers)
or Prometheus counters (scrapeable by Grafana).
The interface is identical either way — only the backend changes.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


RESPONSE_TIME_WINDOW = 200    # Rolling window for latency percentiles
RECENT_SESSIONS_CAP  = 20     # How many recent request IDs to surface in /metrics


@dataclass
class SessionRecord:
    """One entry per request_id — links all steps of a debugging session."""
    request_id:   str
    timestamp:    float
    category:     str   = "unknown"
    severity:     str   = "medium"
    response_ms:  int   = 0
    executed:     bool  = False
    commands_run: int   = 0
    feedback_run: bool  = False
    fix_worked:   bool  = False
    had_error:    bool  = False


class MetricsStore:
    """
    Async-safe in-process metrics collector.
    All mutation goes through async methods guarded by a single asyncio.Lock.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

        # Counters
        self.total_analyses:       int = 0
        self.total_executions:     int = 0
        self.total_commands_run:   int = 0
        self.total_feedback_evals: int = 0
        self.fixes_confirmed:      int = 0

        # Breakdowns
        self.by_category: dict[str, int] = defaultdict(int)
        self.by_severity:  dict[str, int] = defaultdict(int)

        # API health
        self.api_errors:      int = 0
        self.rate_limit_hits: int = 0
        self.timeout_hits:    int = 0
        self.parse_errors:    int = 0

        # Latency (rolling window)
        self._response_times: deque[float] = deque(maxlen=RESPONSE_TIME_WINDOW)

        # Session index (request_id → SessionRecord)
        self._sessions: dict[str, SessionRecord]  = {}
        self._recent:   deque[str]                 = deque(maxlen=RECENT_SESSIONS_CAP)

        self.started_at: float = time.time()

    # ── Write ──────────────────────────────────────────────────────────────

    async def record_analysis(
        self,
        request_id: str,
        category: str,
        severity: str,
        response_time_s: float,
        api_error:   bool = False,
        rate_limited: bool = False,
        timed_out:   bool = False,
        parse_error: bool = False,
    ) -> None:
        async with self._lock:
            self.total_analyses += 1
            self.by_category[category] += 1
            self.by_severity[severity]  += 1
            self._response_times.append(response_time_s)

            if api_error:    self.api_errors      += 1
            if rate_limited: self.rate_limit_hits  += 1
            if timed_out:    self.timeout_hits      += 1
            if parse_error:  self.parse_errors      += 1

            # Create / update session record
            rec = self._sessions.setdefault(
                request_id,
                SessionRecord(request_id=request_id, timestamp=time.time())
            )
            rec.category    = category
            rec.severity    = severity
            rec.response_ms = int(response_time_s * 1000)
            rec.had_error   = api_error

            if request_id not in self._recent:
                self._recent.append(request_id)

    async def record_execution(self, request_id: str, commands_run: int) -> None:
        async with self._lock:
            self.total_executions   += 1
            self.total_commands_run += commands_run

            rec = self._sessions.get(request_id)
            if rec:
                rec.executed     = True
                rec.commands_run = commands_run

    async def record_feedback(self, request_id: str, fix_worked: bool) -> None:
        async with self._lock:
            self.total_feedback_evals += 1
            if fix_worked:
                self.fixes_confirmed += 1

            rec = self._sessions.get(request_id)
            if rec:
                rec.feedback_run = True
                rec.fix_worked   = fix_worked

    # ── Read ───────────────────────────────────────────────────────────────

    def _percentile(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        s   = sorted(data)
        idx = max(0, int(len(s) * p / 100) - 1)
        return round(s[idx], 3)

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            times   = list(self._response_times)
            uptime  = int(time.time() - self.started_at)
            total   = self.total_analyses

            fix_rate = (
                round(self.fixes_confirmed / self.total_feedback_evals * 100, 1)
                if self.total_feedback_evals > 0 else None
            )
            err_rate = round(self.api_errors / total * 100, 1) if total > 0 else 0.0

            # Recent sessions summary (newest first)
            recent = [
                {
                    "request_id":   rid[:8],      # First 8 chars for readability
                    "category":     self._sessions[rid].category,
                    "severity":     self._sessions[rid].severity,
                    "response_ms":  self._sessions[rid].response_ms,
                    "executed":     self._sessions[rid].executed,
                    "feedback_run": self._sessions[rid].feedback_run,
                    "fix_worked":   self._sessions[rid].fix_worked,
                    "had_error":    self._sessions[rid].had_error,
                }
                for rid in reversed(list(self._recent))
                if rid in self._sessions
            ]

            return {
                "uptime_seconds": uptime,
                "uptime_human":   _fmt_uptime(uptime),
                "totals": {
                    "analyses":     total,
                    "executions":   self.total_executions,
                    "commands_run": self.total_commands_run,
                    "feedback_evals": self.total_feedback_evals,
                },
                "by_category":    dict(self.by_category),
                "by_severity":    dict(self.by_severity),
                "response_time_ms": {
                    "p50":     round(self._percentile(times, 50)  * 1000),
                    "p95":     round(self._percentile(times, 95)  * 1000),
                    "p99":     round(self._percentile(times, 99)  * 1000),
                    "samples": len(times),
                },
                "fix_success_rate_pct": fix_rate,
                "api_health": {
                    "error_rate_pct":  err_rate,
                    "rate_limit_hits": self.rate_limit_hits,
                    "timeout_hits":    self.timeout_hits,
                    "parse_errors":    self.parse_errors,
                },
                "recent_sessions": recent,
            }


def _fmt_uptime(s: int) -> str:
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


store = MetricsStore()
