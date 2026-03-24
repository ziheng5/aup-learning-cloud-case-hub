from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


class SubprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int


def _truncate_bytes(b: bytes, limit: int, *, keep: str = "tail") -> tuple[bytes, bool]:
    if limit <= 0:
        return b"", True
    if len(b) <= limit:
        return b, False
    if keep == "head":
        return b[:limit], True
    # default tail: tracebacks are usually at the end
    return b[-limit:], True


class SubprocessRunner:
    def __init__(self, *, max_stdout_bytes: int, max_stderr_bytes: int) -> None:
        self._max_stdout_bytes = max_stdout_bytes
        self._max_stderr_bytes = max_stderr_bytes

    def run(self, cmd: list[str], *, cwd: str, timeout_s: int, env: dict[str, str] | None = None) -> RunResult:
        started = time.time()
        try:
            p = subprocess.run(
                cmd,
                cwd=cwd,
                timeout=timeout_s,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            # User-facing: avoid English in UI.
            raise SubprocessError(f"未找到命令：{cmd[0]}") from e
        except subprocess.TimeoutExpired as e:
            duration_ms = int((time.time() - started) * 1000)
            # User-facing: avoid English in UI.
            raise TimeoutError(f"命令超时（{timeout_s} 秒）：{cmd}") from e

        duration_ms = int((time.time() - started) * 1000)
        stdout_b, stdout_trunc = _truncate_bytes(p.stdout or b"", self._max_stdout_bytes, keep="tail")
        stderr_b, stderr_trunc = _truncate_bytes(p.stderr or b"", self._max_stderr_bytes, keep="tail")
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if stdout_trunc:
            # User-facing: avoid English in UI.
            stdout = "[输出已截断]\n" + stdout
        if stderr_trunc:
            # User-facing: avoid English in UI.
            stderr = "[输出已截断]\n" + stderr
        return RunResult(exit_code=int(p.returncode), stdout=stdout, stderr=stderr, duration_ms=duration_ms)
