"""
SafeCommandExecutor — Execution feedback loop for AutoFixOps v2.

Security design:
  - Uses create_subprocess_exec + shlex.split instead of create_subprocess_shell.
    Shell interpretation is fully disabled, eliminating shell-injection risk even
    when a whitelisted command contains unexpected input.
  - Strict whitelist: only read-only diagnostic commands are auto-executable.
  - Separate forbidden-pattern list blocks destructive commands at all times.
  - 15-second timeout per command prevents runaway processes.
"""

import asyncio
import re
import shlex
from typing import Any


# ── Whitelist: read-only diagnostic commands ───────────────────────────────
SAFE_COMMAND_PATTERNS = [
    r"^python3?\s+--version$",
    r"^pip3?\s+show\s+[\w\-]+$",
    r"^pip3?\s+list$",
    r"^pip3?\s+check$",
    r"^which\s+\w+$",
    r"^docker\s+ps(\s+-a)?$",
    r"^docker\s+images$",
    r"^docker\s+inspect\s+[\w\-]+$",
    r"^docker\s+logs\s+[\w\-]+$",
    r"^docker\s+version$",
    r"^docker\s+info$",
    r"^npm\s+list(\s+--depth=\d+)?$",
    r"^npm\s+doctor$",
    r"^npm\s+audit$",
    r"^node\s+--version$",
    r"^npm\s+--version$",
    r"^ls(\s+-[lahrtR]+)?(\s+[\w\.\/\-]+)?$",
    r"^cat\s+[\w\.\/\-]+$",
    r"^pwd$",
    r"^df\s+-h$",
    r"^free\s+-h$",
    r"^ps\s+aux(\s+--sort=[-\w]+)?(\s+\|\s+head\s+-\d+)?$",
    r"^env$",
    r"^printenv(\s+\w+)?$",
    r"^echo\s+\$\w+$",
    r"^uname\s+-[a-z]+$",
    r"^systemctl\s+status\s+\w+$",
    r"^journalctl\s+-u\s+\w+(\s+-n\s+\d+)?$",
    r"^git\s+status$",
    r"^git\s+log(\s+--\S+)*$",
    r"^git\s+branch(\s+-[a-z]+)?$",
    r"^curl\s+-[vIsf]+\s+https?://\S+$",
    r"^ping\s+-c\s+\d+\s+\S+$",
    r"^nslookup\s+\S+$",
    r"^dig\s+\S+$",
    r"^netstat\s+-[a-z]+$",
    r"^ss\s+-[a-z]+$",
]

# ── Hard block: never execute, even in unsafe mode ─────────────────────────
FORBIDDEN_PATTERNS = [
    r"rm\s+-rf",
    r"sudo\s+rm",
    r":\(\)\{.*\}",           # Fork bomb
    r">\s*/dev/sd",           # Disk wipe
    r"mkfs\.",                # Filesystem format
    r"dd\s+if=",              # Disk copy/wipe
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"curl\s+.*\|\s*(ba)?sh", # Remote code execution
    r"wget\s+.*\|\s*(ba)?sh",
    r"base64\s+-d.*\|",
    r"eval\s+",
    r"exec\s+",
]

COMPILED_SAFE      = [re.compile(p) for p in SAFE_COMMAND_PATTERNS]
COMPILED_FORBIDDEN = [re.compile(p) for p in FORBIDDEN_PATTERNS]


class SafeCommandExecutor:
    """
    Executes shell commands for the ReAct agent loop: Act → Observe.

    Security properties:
    - create_subprocess_exec (not shell=True) prevents shell injection.
    - Whitelist enforced before any execution attempt.
    - Forbidden patterns checked as final hard block.
    - 15-second per-command timeout.
    """

    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode

    def _is_forbidden(self, command: str) -> bool:
        return any(p.search(command) for p in COMPILED_FORBIDDEN)

    def _is_safe(self, command: str) -> bool:
        return any(p.match(command.strip()) for p in COMPILED_SAFE)

    async def run(self, command: str) -> dict[str, Any]:
        """
        Execute one command. Returns structured stdout/stderr/exit_code.

        Uses create_subprocess_exec with shlex.split — shell interpretation
        is disabled, eliminating shell injection risk entirely.
        """
        cmd = command.strip()

        # 1. Hard block forbidden commands regardless of mode
        if self._is_forbidden(cmd):
            return {
                "command": cmd,
                "stdout": "",
                "stderr": "BLOCKED: Command matches a forbidden pattern and will never execute.",
                "exit_code": -1,
                "safe": False,
            }

        is_safe = self._is_safe(cmd)

        # 2. In safe_mode, only whitelisted commands proceed
        if self.safe_mode and not is_safe:
            return {
                "command": cmd,
                "stdout": "",
                "stderr": (
                    f"BLOCKED (safe_mode=True): '{cmd}' is not in the diagnostic whitelist. "
                    "Set safe_mode=false to allow non-diagnostic commands."
                ),
                "exit_code": -2,
                "safe": False,
            }

        # 3. Parse into argv — no shell, no injection
        try:
            argv = shlex.split(cmd)
        except ValueError as exc:
            return {
                "command": cmd,
                "stdout": "",
                "stderr": f"PARSE ERROR: Could not tokenize command — {exc}",
                "exit_code": -5,
                "safe": is_safe,
            }

        if not argv:
            return {
                "command": cmd,
                "stdout": "",
                "stderr": "PARSE ERROR: Empty command after tokenisation.",
                "exit_code": -5,
                "safe": is_safe,
            }

        # 4. Execute without shell
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)

            return {
                "command": cmd,
                "stdout": stdout.decode("utf-8", errors="replace").strip(),
                "stderr": stderr.decode("utf-8", errors="replace").strip(),
                "exit_code": proc.returncode,
                "safe": is_safe,
            }

        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return {
                "command": cmd,
                "stdout": "",
                "stderr": "TIMEOUT: Command exceeded 15-second limit and was terminated.",
                "exit_code": -3,
                "safe": is_safe,
            }
        except FileNotFoundError:
            return {
                "command": cmd,
                "stdout": "",
                "stderr": f"NOT FOUND: '{argv[0]}' is not installed or not in PATH.",
                "exit_code": -6,
                "safe": is_safe,
            }
        except Exception as exc:
            return {
                "command": cmd,
                "stdout": "",
                "stderr": f"EXECUTION ERROR: {exc}",
                "exit_code": -4,
                "safe": is_safe,
            }

    def classify_commands(self, commands: list[str]) -> dict[str, list[str]]:
        """
        Partition commands into safe / unsafe / forbidden buckets.
        Used by the frontend to communicate execution intent to the user.
        """
        safe      = [c for c in commands if self._is_safe(c) and not self._is_forbidden(c)]
        unsafe    = [c for c in commands if not self._is_safe(c) and not self._is_forbidden(c)]
        forbidden = [c for c in commands if self._is_forbidden(c)]
        return {"safe": safe, "unsafe": unsafe, "forbidden": forbidden}
