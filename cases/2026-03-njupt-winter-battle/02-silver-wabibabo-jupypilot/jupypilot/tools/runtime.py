from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from ..config import Config
from ..types import SessionState, ToolResult
from .subprocess_runner import SubprocessError, SubprocessRunner


def _err(tool: str, code: str, message: str, *, details: dict[str, Any] | None = None) -> ToolResult:
    return {"ok": False, "tool": tool, "data": None, "error": {"code": code, "message": message, "details": details or {}}}


def _ok(tool: str, data: dict[str, Any]) -> ToolResult:
    return {"ok": True, "tool": tool, "data": data, "error": None}


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_repo_relpath(repo_root: Path, relpath: str) -> Path:
    if not relpath or not isinstance(relpath, str):
        raise ValueError("路径必须为非空字符串")
    p = Path(relpath)
    # If the model passes an absolute path, try to convert it to a relative
    # path under repo_root so the call still succeeds.
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(repo_root.resolve())
        except ValueError:
            raise ValueError("绝对路径不在仓库目录内")
    resolved = (repo_root / p).resolve()
    if not _is_within(repo_root, resolved):
        raise ValueError("路径越过了仓库根目录")
    return resolved


def _iter_repo_files(repo_root: Path, *, ignore_dirs: tuple[str, ...], ignore_globs: tuple[str, ...], max_file_bytes: int) -> list[Path]:
    files: list[Path] = []
    ignore_dirs_set = frozenset(ignore_dirs)  # O(1) membership vs O(n) for tuple
    for root, dirs, filenames in os.walk(repo_root):
        root_path = Path(root)
        # prune ignored dirs
        dirs[:] = [d for d in dirs if d not in ignore_dirs_set]
        for name in filenames:
            rel = (root_path / name).relative_to(repo_root)
            rel_str = str(rel).replace("\\", "/")
            if any(fnmatch.fnmatch(rel_str, g) for g in ignore_globs):
                continue
            try:
                st = (repo_root / rel).stat()
            except OSError:
                continue
            if st.st_size > max_file_bytes:
                continue
            files.append(repo_root / rel)
    return files


class ToolRuntime:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._runner = SubprocessRunner(
            max_stdout_bytes=config.limits.max_stdout_bytes,
            max_stderr_bytes=config.limits.max_stderr_bytes,
        )

    @property
    def config(self) -> Config:
        return self._config

    def execute(self, tool: str, args: dict[str, Any], session: SessionState) -> ToolResult:
        if tool == "search_code":
            return self._search_code(args, session)
        if tool == "open_file":
            return self._open_file(args, session)
        if tool == "run_task":
            return self._run_task(args, session)
        if tool == "git_apply_check":
            return self._git_apply_check(args, session)
        if tool == "write_files":
            return self._write_files(args, session)
        return _err(tool, "E_TOOL", f"未知工具：{tool}")

    # ---------------------------
    # search_code
    # ---------------------------
    def _search_code(self, args: dict[str, Any], session: SessionState) -> ToolResult:
        tool = "search_code"
        # Accept "pattern" as an alias for "query" (some models use it).
        query = args.get("query") or args.get("pattern")
        if not isinstance(query, str) or not query.strip():
            return _err(tool, "E_VALIDATION", "参数 query 必须为非空字符串")
        glob = args.get("glob", "**/*")
        if not isinstance(glob, str) or not glob.strip():
            return _err(tool, "E_VALIDATION", "参数 glob 必须为字符串")
        max_results = args.get("max_results", 50)
        try:
            max_results_i = int(max_results)
        except Exception:
            return _err(tool, "E_VALIDATION", "参数 max_results 必须为整数")
        max_results_i = max(1, min(self._config.limits.search_max_results, max_results_i))

        repo_root = Path(session.repo_path).resolve()
        matches: list[dict[str, Any]] = []

        # Prefer ripgrep if available.
        try:
            cmd = [
                "rg",
                "-n",
                "--no-heading",
                "--color",
                "never",
                "--glob",
                glob,
            ]
            # Keep rg aligned with RAG scan policy to avoid surfacing cache files
            # (e.g. .ruff_cache) as "source" paths.
            for d in self._config.rag.ignore_dirs:
                cmd.extend(["--glob", f"!{d}/**"])
            for g in self._config.rag.ignore_globs:
                cmd.extend(["--glob", f"!{g}"])
            cmd.extend([query, "."])
            rr = self._runner.run(cmd, cwd=str(repo_root), timeout_s=self._config.limits.rg_timeout_s)
            if rr.exit_code not in (0, 1):
                return _err(tool, "E_SUBPROCESS", "rg failed", details={"exit_code": rr.exit_code, "stderr": rr.stderr})
            for line in rr.stdout.splitlines():
                if not line:
                    continue
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                path_s, line_s, text = parts
                try:
                    ln = int(line_s)
                except ValueError:
                    continue
                matches.append({"path": path_s, "line": ln, "text": text})
                if len(matches) >= max_results_i:
                    break
        except TimeoutError:
            return _err(tool, "E_TIMEOUT", "rg 搜索超时", details={"timeout_s": self._config.limits.rg_timeout_s})
        except SubprocessError:
            # fallback to Python scanning
            matches = self._python_search(repo_root, query=query, glob=glob, max_results=max_results_i)

        return _ok(tool, {"matches": matches})

    def _python_search(self, repo_root: Path, *, query: str, glob: str, max_results: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        files = _iter_repo_files(
            repo_root,
            ignore_dirs=self._config.rag.ignore_dirs,
            ignore_globs=self._config.rag.ignore_globs,
            max_file_bytes=self._config.rag.max_file_bytes,
        )
        for path in files:
            rel = path.relative_to(repo_root).as_posix()
            if not fnmatch.fnmatch(rel, glob.replace("**/", "*")) and not fnmatch.fnmatch(rel, glob):
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for idx, line in enumerate(f, start=1):
                        if query in line:
                            results.append({"path": rel, "line": idx, "text": line.rstrip("\n\r")})
                            if len(results) >= max_results:
                                return results
            except OSError:
                continue
        return results

    # ---------------------------
    # open_file
    # ---------------------------
    def _open_file(self, args: dict[str, Any], session: SessionState) -> ToolResult:
        tool = "open_file"
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            return _err(tool, "E_VALIDATION", "参数 path 必须为非空字符串")

        max_lines = self._config.limits.open_file_max_lines

        # start_line / end_line are now optional.  When omitted, read from
        # line 1 up to max_lines.
        raw_start = args.get("start_line")
        raw_end = args.get("end_line")

        # Detect duplicate full-file reads of the same path.
        # If the model already read this file (same path, no line range), return
        # a short hint instead of the full content again — this prevents the
        # context from being stuffed with repeated file dumps.
        opened_files: set[str] = session.flags.setdefault("_opened_files", set())
        norm_path = path.strip().replace("\\", "/")
        is_full_read = raw_start is None and raw_end is None
        if is_full_read and norm_path in opened_files:
            return _err(
                tool,
                "E_DUPLICATE",
                f"文件 `{norm_path}` 已在本轮会话中读取过，无需重复打开。请直接基于已有内容进行分析。",
            )
        if raw_start is not None and raw_end is not None:
            try:
                start_line = int(raw_start)
                end_line = int(raw_end)
            except Exception:
                return _err(tool, "E_VALIDATION", "参数 start_line/end_line 必须为整数")
            if start_line < 1 or end_line < 1 or end_line < start_line:
                return _err(tool, "E_VALIDATION", "行号范围不合法")
            if (end_line - start_line + 1) > max_lines:
                return _err(
                    tool,
                    "E_VALIDATION",
                    "请求的行数过多",
                    details={"max_lines": max_lines},
                )
        else:
            start_line = 1
            end_line = max_lines

        repo_root = Path(session.repo_path).resolve()
        try:
            full = _resolve_repo_relpath(repo_root, path)
        except ValueError as e:
            return _err(tool, "E_PATH", str(e))

        try:
            with full.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError as e:
            # 通用提示：路径问题时引导模型主动用 search_code 查找正确路径，而不是死磕当前相对路径。
            # 这样文件名，都可以通过“先搜索再打开”的模式自我纠正。
            if getattr(e, "errno", None) == 2:
                msg = (
                    f"读取文件失败：{e}。"
                    "可能文件不存在，或路径不在当前项目根目录内。"
                    "建议先使用 search_code 按文件名或关键字搜索出完整相对路径，再用该路径重试 open_file。"
                )
            else:
                msg = f"读取文件失败：{e}"
            return _err(tool, "E_IO", msg)

        # Clamp end_line to actual file length.
        end_line = min(end_line, len(lines))
        start_i = start_line - 1
        content_lines: list[str] = []
        for i in range(start_i, end_line):
            ln = i + 1
            content_lines.append(f"{ln:6d}| {lines[i]}")
        rel = full.relative_to(repo_root).as_posix()
        truncated = len(lines) > end_line

        # For large files without explicit line range, return a compact
        # outline (first 30 + last 20 lines with a gap) so the model gets
        # the structure without blowing up the context.
        # 300: 单文件分析常见规模（如 config.py ~270 行）可一次看全，减少行号/漏类错误。
        _COMPACT_THRESHOLD = 300  # lines
        if is_full_read and len(content_lines) > _COMPACT_THRESHOLD:
            head = content_lines[:30]
            tail = content_lines[-20:]
            skipped = len(content_lines) - 50
            content_lines = head + [f"       ... 省略中间 {skipped} 行 ... (可用 start_line/end_line 查看详细内容)"] + tail
            truncated = True

        # Track this file as opened (for duplicate detection).
        if is_full_read:
            opened_files.add(norm_path)

        data: dict[str, Any] = {
            "path": rel,
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": len(lines),
            "content": "\n".join(content_lines),
        }
        if truncated:
            data["truncated"] = True
            data["hint"] = f"文件共 {len(lines)} 行，仅显示前 {end_line} 行。可用 start_line/end_line 查看其余部分。"
        return _ok(tool, data)

    # ---------------------------
    # run_task
    # ---------------------------
    def _run_task(self, args: dict[str, Any], session: SessionState) -> ToolResult:
        tool = "run_task"
        task = args.get("task")
        if task not in ("ruff_check", "pytest_q"):
            return _err(tool, "E_POLICY", "不允许执行该任务", details={"allowed": ["ruff_check", "pytest_q"]})
        cmds: list[list[str]]
        if task == "ruff_check":
            cmds = [["ruff", "check", "."]]
        else:
            # Prefer pytest if available; fallback to stdlib unittest to reduce external deps.
            cmds = [
                ["pytest", "-q"],
                ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
            ]

        repo_root = Path(session.repo_path).resolve()
        env = os.environ.copy()
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        rr = None
        last_subprocess_err: SubprocessError | None = None
        for cmd in cmds:
            try:
                rr = self._runner.run(cmd, cwd=str(repo_root), timeout_s=self._config.limits.subprocess_timeout_s, env=env)
                break
            except TimeoutError:
                return _err(
                    tool,
                    "E_TIMEOUT",
                    f"{task} 超时",
                    details={"timeout_s": self._config.limits.subprocess_timeout_s},
                )
            except SubprocessError as e:
                last_subprocess_err = e
                # If pytest is missing, try unittest fallback.
                if task == "pytest_q" and len(cmds) > 1 and cmd == cmds[0] and "未找到命令" in str(e):
                    continue
                return _err(tool, "E_SUBPROCESS", str(e))
        if rr is None:
            return _err(tool, "E_SUBPROCESS", str(last_subprocess_err) if last_subprocess_err else "子进程执行失败")

        data = {
            "task": task,
            "exit_code": rr.exit_code,
            "stdout": rr.stdout,
            "stderr": rr.stderr,
            "duration_ms": rr.duration_ms,
        }
        # pytest exit code 5 means "no tests collected" – treat as error.
        if rr.exit_code == 5 and task == "pytest_q":
            return {
                "ok": False,
                "tool": tool,
                "data": data,
                "error": {
                    "code": "E_EXIT",
                    "message": "未收集到任何测试",
                    "details": {"task": task, "exit_code": rr.exit_code},
                },
            }
        if rr.exit_code == 0:
            # For pytest_q, verify tests actually ran (guard against empty runs).
            if task == "pytest_q" and "passed" not in rr.stdout and "error" not in rr.stdout.lower():
                no_tests = "no tests ran" in rr.stdout.lower() or rr.stdout.strip() == ""
                if no_tests:
                    return {
                        "ok": False,
                        "tool": tool,
                        "data": data,
                        "error": {
                            "code": "E_EXIT",
                            "message": "未收集到任何测试",
                            "details": {"task": task, "exit_code": rr.exit_code},
                        },
                    }
            return _ok(tool, data)
        # Non-zero exit means the task ran but reported failure.
        return {
            "ok": False,
            "tool": tool,
            "data": data,
            "error": {
                "code": "E_EXIT",
                "message": "任务执行失败（退出码非 0）",
                "details": {"task": task, "exit_code": rr.exit_code},
            },
        }

    # ---------------------------
    # git_apply_check
    # ---------------------------
    def _git_apply_check(self, args: dict[str, Any], session: SessionState) -> ToolResult:
        tool = "git_apply_check"
        if not self._config.tools.allow_git_apply_check:
            return _err(tool, "E_POLICY", "配置已禁用 git_apply_check")
        diff = args.get("diff")
        if not isinstance(diff, str) or not diff.strip():
            return _err(tool, "E_VALIDATION", "参数 diff 必须为非空字符串")
        repo_root = Path(session.repo_path).resolve()
        started = time.time()
        try:
            p = subprocess.run(
                ["git", "apply", "--check", "-"],
                cwd=str(repo_root),
                input=diff.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._config.limits.subprocess_timeout_s,
            )
        except FileNotFoundError:
            return _err(tool, "E_SUBPROCESS", "未找到 git")
        except subprocess.TimeoutExpired:
            return _err(tool, "E_TIMEOUT", "git apply --check 超时")

        duration_ms = int((time.time() - started) * 1000)
        stdout = (p.stdout or b"")[: self._config.limits.max_stdout_bytes].decode("utf-8", errors="replace")
        stderr = (p.stderr or b"")[: self._config.limits.max_stderr_bytes].decode("utf-8", errors="replace")
        ok_to_apply = p.returncode == 0
        return _ok(tool, {"ok_to_apply": ok_to_apply, "stdout": stdout, "stderr": stderr, "duration_ms": duration_ms})

    # ---------------------------
    # write_files
    # ---------------------------
    def _write_files(self, args: dict[str, Any], session: SessionState) -> ToolResult:
        tool = "write_files"
        if not self._config.tools.allow_write_files:
            return _err(tool, "E_POLICY", "配置已禁用 write_files")
        if not bool(session.flags.get("write_enabled", False)):
            return _err(tool, "E_POLICY", "需要在 UI 打开“允许写入文件”开关")

        plan = args.get("plan")
        if not isinstance(plan, dict):
            return _err(tool, "E_VALIDATION", "参数 plan 必须为对象")
        files = plan.get("files")
        if not isinstance(files, list) or not files:
            return _err(tool, "E_VALIDATION", "plan.files 必须为非空数组")
        dry_run = bool(args.get("dry_run", True))
        overwrite = bool(args.get("overwrite", False))

        repo_root = Path(session.repo_path).resolve()
        written: list[str] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for item in files:
            if not isinstance(item, dict):
                errors.append({"error": "文件条目必须为对象"})
                continue
            relpath = item.get("path")
            content = item.get("content")
            if not isinstance(relpath, str) or not isinstance(content, str):
                errors.append({"path": relpath, "error": "path/content 必须为字符串"})
                continue
            try:
                full = _resolve_repo_relpath(repo_root, relpath)
            except ValueError as e:
                errors.append({"path": relpath, "error": f"路径不合法：{e}"})
                continue

            if full.exists() and not overwrite:
                skipped.append({"path": relpath, "reason": "已存在"})
                continue

            if dry_run:
                written.append(relpath)
                continue

            try:
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content, encoding="utf-8")
                written.append(relpath)
            except OSError as e:
                errors.append({"path": relpath, "error": str(e)})

        return _ok(tool, {"dry_run": dry_run, "written": written, "skipped": skipped, "errors": errors})
