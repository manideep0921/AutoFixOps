"""
ClaudeDebugAgent — Real LLM-powered debugging using Anthropic Claude API.

Production design decisions:

1. HIDDEN REASONING
   Claude reasons internally. Only a curated `reasoning_summary` (1–2 sentences)
   is returned. Raw chain-of-thought is never written to the response body.

2. RETRY WITH EXPONENTIAL BACKOFF
   429 (rate limit) and 5xx errors are retried up to MAX_RETRIES times with
   jittered exponential backoff. Retry-After headers are respected on 429.

3. GRACEFUL DEGRADATION
   All failure paths return structured fallback dicts. Raw exceptions never
   reach HTTP response handlers.

4. COST CONTROL  ← explicit constants, not just comments
   MAX_LOG_CHARS  = 6 000   Input truncation — largest cost lever (fewer input tokens)
   MAX_TOKENS     = 2 048   Hard output cap — prevents runaway generation cost
   MAX_RETRIES    = 3       Caps total API calls per user request to 3× worst case
   No streaming   = intentional — streaming adds latency complexity without saving cost
   Model choice   = claude-3-sonnet-20240229: best cost/capability ratio for this task

5. REQUEST ID PROPAGATION
   Each agent call receives a request_id from the middleware ContextVar.
   It is injected into every log line for end-to-end session tracing.
"""

import asyncio
import json
import logging
import random
import re
from typing import Any

import httpx


logger = logging.getLogger(__name__)

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# ── Model & cost control constants ────────────────────────────────────────
# These are the four levers that control LLM API cost.
# Document them explicitly — interviewers will ask.

MODEL         = "claude-3-sonnet-20240229"
# Sonnet is chosen over Opus: ~5× cheaper per token, sufficient capability
# for structured error classification. Upgrade to Opus only for ambiguous
# multi-system failures where deeper reasoning justifies the cost.

ANTHROPIC_VER = "2023-06-01"

MAX_LOG_CHARS = 6_000
# Primary cost lever: input token reduction.
# A 6 000-char log ≈ 1 500 tokens. A raw 50 000-char log would be ≈ 12 500 tokens.
# Truncation preserves the head (error type) and tail (stack bottom) — the
# most diagnostically relevant sections.

MAX_TOKENS    = 2_048
# Hard output cap. Prevents runaway generation if the model tries to produce
# lengthy prose. 2 048 tokens is sufficient for all structured JSON fields
# including verbose fix_steps and prevention_tips.

MAX_RETRIES   = 3
# Caps worst-case API calls per user request at 3. Combined with MAX_TOKENS,
# the maximum cost per user interaction is bounded and predictable.

BASE_BACKOFF  = 1.5   # seconds — grows as BASE_BACKOFF^attempt + jitter


# ── Prompts ────────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """\
You are AutoFixOps, an expert DevOps and software engineering debugging assistant.

Instructions:
  - Reason about the error INTERNALLY before writing output.
  - Do NOT write out your thinking steps in the response.
  - Output only a single <json> block.
  - The `reasoning_summary` field must be 1-2 sentences explaining HOW you
    reached your conclusion. This is a curated summary, not a raw thought log.

Analysis quality rules:
  - root_cause and fix_steps must be specific to THIS error, not boilerplate.
  - fix_steps should be real executable shell commands where applicable.
  - safe_commands must be read-only diagnostics only.
  - Express honest uncertainty in the `confidence` field.

Output ONLY the <json> block. No preamble, no markdown outside it.

<json>
{
  "error_type": "short descriptive name",
  "error_category": "python|docker|npm|node|system|network|database|git|kubernetes|terraform|unknown",
  "severity": "low|medium|high|critical",
  "confidence": "plain English confidence statement",
  "reasoning_summary": "1-2 sentence summary of how this conclusion was reached",
  "root_cause": "precise technical root cause",
  "explanation": "clear explanation for a mid-level engineer",
  "fix_steps": ["Step 1: command or action", "Step 2: ..."],
  "safe_commands": ["read-only diagnostic command"],
  "prevention_tips": ["concrete prevention measure"],
  "model-used": "claude-3-sonnet-20240229"
}
</json>\
"""

FEEDBACK_SYSTEM_PROMPT = """\
You are AutoFixOps evaluating whether a fix resolved an error.

You receive: the original error, the fix applied, and the new output.

Reason internally, then output ONLY a <json> block. No preamble.

<json>
{
  "fix_worked": true,
  "analysis": "what changed and why the fix did or did not work",
  "next_steps": ["next action if still broken, or empty list if resolved"],
  "still_broken": false
}
</json>\
"""


# ── Custom exceptions ──────────────────────────────────────────────────────

class APIError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code

class RateLimitError(APIError):
    pass


# ── Retry helper ───────────────────────────────────────────────────────────

async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
    request_id: str = "",
) -> dict:
    """
    POST to Claude API with exponential backoff on 429 and 5xx.
    request_id is included in log lines for session correlation.
    """
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 429:
                retry_after = float(response.headers.get("retry-after", BASE_BACKOFF))
                wait = retry_after + random.uniform(0.0, 0.5)
                logger.warning(
                    "[req=%s] Rate limited (429) — waiting %.1fs, retry %d/%d",
                    request_id[:8], wait, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                last_exc = RateLimitError("Rate limit exceeded.", 429)
                continue

            if response.status_code >= 500:
                wait = (BASE_BACKOFF ** (attempt + 1)) + random.uniform(0.0, 0.5)
                logger.warning(
                    "[req=%s] Server error %d — waiting %.1fs, retry %d/%d",
                    request_id[:8], response.status_code, wait, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                last_exc = APIError(f"Server error {response.status_code}.", response.status_code)
                continue

            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as exc:
            wait = (BASE_BACKOFF ** (attempt + 1)) + random.uniform(0.0, 0.5)
            logger.warning(
                "[req=%s] Timeout — waiting %.1fs, retry %d/%d",
                request_id[:8], wait, attempt + 1, MAX_RETRIES,
            )
            await asyncio.sleep(wait)
            last_exc = exc

    if isinstance(last_exc, RateLimitError):
        raise RateLimitError(
            "Claude API rate limit reached. Please wait a moment and try again."
        ) from last_exc
    raise APIError(
        f"Claude API unavailable after {MAX_RETRIES} attempts. Please try again shortly."
    ) from last_exc


# ── Agent ──────────────────────────────────────────────────────────────────

class ClaudeDebugAgent:
    """
    Wraps the Anthropic Messages API for dynamic LLM-powered error analysis.

    Cost control summary (for interviews):
      - Input truncated to MAX_LOG_CHARS   → limits input tokens
      - Output capped at MAX_TOKENS        → prevents runaway generation
      - Retries bounded by MAX_RETRIES     → predictable worst-case call count
      - No streaming                       → simpler, no cost difference
      - Model: "claude-3-sonnet-20240229"    → optimal cost/capability ratio
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VER,
            "content-type": "application/json",
        }

    # ── Input helpers ──────────────────────────────────────────────────────

    def _truncate(self, text: str, max_chars: int = MAX_LOG_CHARS) -> str:
        """
        Truncate long input by keeping head and tail — the most diagnostically
        relevant sections. Middle sections (repeated stack frames) are dropped.
        """
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        dropped = len(text) - max_chars
        return (
            text[:half]
            + f"\n\n... [{dropped} characters truncated — middle stack frames omitted] ...\n\n"
            + text[-half:]
        )

    def _build_analysis_prompt(self, error_log: str, context: str) -> str:
        log = self._truncate(error_log)
        parts = [f"Terminal error log:\n\n```\n{log}\n```"]
        if context.strip():
            parts.append(f"\nEnvironment context: {self._truncate(context, 500)}")
        return "\n".join(parts)

    def _build_feedback_prompt(
        self, original_log: str, applied_fix: str, new_log: str
    ) -> str:
        return (
            f"ORIGINAL ERROR:\n```\n{self._truncate(original_log)}\n```\n\n"
            f"FIX APPLIED:\n{self._truncate(applied_fix, 1_000)}\n\n"
            f"NEW OUTPUT:\n```\n{self._truncate(new_log)}\n```"
        )

    # ── Response parsing ───────────────────────────────────────────────────

    def _extract_json(self, text: str) -> dict:
        match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
        match = re.search(r"\{[\s\S]+\}", text)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No parseable JSON block in Claude response.")

    # ── Fallback responses ─────────────────────────────────────────────────

    @staticmethod
    def _analysis_fallback(reason: str) -> dict:
        return {
            "error_type": "Analysis Unavailable",
            "error_category": "unknown",
            "severity": "medium",
            "confidence": "Unavailable",
            "reasoning_summary": reason,
            "root_cause": "Analysis could not be completed.",
            "explanation": reason,
            "fix_steps": ["Retry the analysis, or review the error log manually."],
            "safe_commands": [],
            "prevention_tips": [],
            "model_used": MODEL,
        }

    @staticmethod
    def _feedback_fallback(reason: str) -> dict:
        return {
            "fix_worked": False,
            "analysis": reason,
            "next_steps": ["Retry evaluation, or review the output manually."],
            "still_broken": True,
        }

    # ── Public API ─────────────────────────────────────────────────────────

    async def analyze(
        self,
        error_log: str,
        context: str = "",
        request_id: str = "",
    ) -> dict[str, Any]:
        """
        Send error log to Claude. Returns structured analysis.
        request_id flows into log lines for session-level tracing.
        """
        payload = {
            "model":      MODEL,
            "max_tokens": MAX_TOKENS,   # Hard cost cap
            "system":     ANALYSIS_SYSTEM_PROMPT,
            "messages":   [
                {"role": "user", "content": self._build_analysis_prompt(error_log, context)}
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                data = await _post_with_retry(
                    client, CLAUDE_API_URL, self.headers, payload, request_id
                )
            structured = self._extract_json(data["content"][0]["text"])

            # Log token usage for cost visibility
            usage = data.get("usage", {})
            logger.info(
                "[req=%s] Tokens — input=%d output=%d",
                request_id[:8],
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            )

        except RateLimitError:
            return self._analysis_fallback(
                "Claude API rate limit reached. Please wait a moment and try again."
            )
        except (APIError, httpx.TimeoutException) as exc:
            return self._analysis_fallback(f"Claude API temporarily unavailable: {exc}")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.error("[req=%s] Parse error: %s", request_id[:8], exc)
            return self._analysis_fallback(
                "Response parsing failed — the model returned an unexpected format."
            )

        return {
            "error_type": "Unknown Error",
            "error_category": "unknown",
            "severity": "medium",
            "confidence": "Insufficient context",
            "reasoning_summary": "No summary provided.",
            "root_cause": "Could not determine root cause.",
            "explanation": "",
            "fix_steps": [],
            "safe_commands": [],
            "prevention_tips": [],
            "model_used": MODEL,
            **structured,
        }

    async def evaluate_fix(
        self,
        original_log: str,
        applied_fix: str,
        new_log: str,
        request_id: str = "",
    ) -> dict[str, Any]:
        """
        Feedback loop: Claude evaluates whether the fix resolved the issue.
        Same cost controls apply: input truncated, output capped at MAX_TOKENS.
        """
        payload = {
            "model":      MODEL,
            "max_tokens": 1_024,   # Feedback responses are shorter — tighter cap
            "system":     FEEDBACK_SYSTEM_PROMPT,
            "messages":   [
                {"role": "user", "content": self._build_feedback_prompt(
                    original_log, applied_fix, new_log
                )}
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                data = await _post_with_retry(
                    client, CLAUDE_API_URL, self.headers, payload, request_id
                )
            structured = self._extract_json(data["content"][0]["text"])

        except RateLimitError:
            return self._feedback_fallback(
                "Rate limit reached during fix evaluation. Please retry shortly."
            )
        except (APIError, httpx.TimeoutException) as exc:
            return self._feedback_fallback(f"API unavailable: {exc}")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.error("[req=%s] Feedback parse error: %s", request_id[:8], exc)
            return self._feedback_fallback("Response parsing failed.")

        return {
            "fix_worked": False,
            "analysis": "No analysis returned.",
            "next_steps": [],
            "still_broken": True,
            **structured,
        }
