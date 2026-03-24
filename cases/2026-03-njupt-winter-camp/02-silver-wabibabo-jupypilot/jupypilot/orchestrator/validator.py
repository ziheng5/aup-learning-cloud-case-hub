from __future__ import annotations

import json
import re
from typing import Any

from ..types import Envelope, FinalResponse, ToolCall


class EnvelopeParseError(ValueError):
    pass


class ValidationError(ValueError):
    pass


_DIFF_BLOCK_RE = re.compile(r"```diff\s*\r?\n(.*?)\r?\n```", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```json\s*\r?\n(.*?)\r?\n```", re.DOTALL)
_GENERIC_FENCE_RE = re.compile(r"```\s*\r?\n(.*?)\r?\n```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(s: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3)."""
    result = _THINK_RE.sub("", s)
    # Handle unclosed <think> tag (model output truncated during thinking).
    idx = result.find("<think>")
    if idx != -1:
        result = result[:idx]
    return result.strip()


def _strip_code_fence(s: str) -> str:
    s2 = s.strip()
    m = _JSON_FENCE_RE.fullmatch(s2)
    if m:
        return m.group(1).strip()
    m = _GENERIC_FENCE_RE.fullmatch(s2)
    if m:
        return m.group(1).strip()
    return s2


def _extract_all_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract all valid JSON objects from text that may contain multiple concatenated objects.

    Models sometimes output multiple JSON objects on separate lines, e.g.:
        {"kind":"tool","tool":"open_file","args":{...}}
        {"kind":"final","format":"markdown","content":"..."}

    Returns a list of parsed dicts in order of appearance.
    """
    # Fast path: single JSON object.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Slow path: extract all JSON objects via raw_decode.
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    idx = 0
    length = len(text)
    while idx < length:
        # Skip whitespace between objects.
        while idx < length and text[idx] in " \t\r\n":
            idx += 1
        if idx >= length:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                objects.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            break

    if not objects:
        raise EnvelopeParseError("JSON 解析失败：无法从模型输出中提取有效 JSON 对象")

    return objects


def _obj_to_envelope(obj: dict[str, Any]) -> Envelope:
    """Convert a parsed JSON dict to a typed Envelope (ToolCall or FinalResponse)."""
    kind = obj.get("kind")
    if kind == "tool":
        _validate_allowed_keys(obj, {"kind", "tool", "args"})
        tool = obj.get("tool")
        if tool not in {"search_code", "open_file", "run_task", "git_apply_check", "write_files"}:
            raise EnvelopeParseError(f"未知工具：{tool!r}")
        args = obj.get("args")
        if not isinstance(args, dict):
            raise EnvelopeParseError("tool.args 必须为对象")
        return ToolCall(kind="tool", tool=tool, args=args)  # type: ignore[arg-type]

    if kind == "final":
        _validate_allowed_keys(obj, {"kind", "format", "content"})
        fmt = obj.get("format")
        if fmt not in {"markdown", "json"}:
            raise EnvelopeParseError(f"final.format 不合法：{fmt!r}")
        content = obj.get("content")
        if not isinstance(content, str):
            raise EnvelopeParseError("final.content 必须为字符串")
        return FinalResponse(kind="final", format=fmt, content=content)  # type: ignore[arg-type]

    raise EnvelopeParseError("kind 必须为 'tool' 或 'final'")


def parse_envelopes(raw_text: str) -> list[Envelope]:
    """Parse LLM output into a list of Envelope objects.

    Handles the common case where the model outputs multiple JSON objects
    concatenated together (e.g. multiple tool calls, or tool calls + final).
    All JSON objects are parsed and returned; the tool-loop may execute
    multiple tools in one round to improve reasoning speed and reduce rounds.

    (Previously: Early Stop limited to one tool per turn; removed to allow
     multiple tools per round for faster reasoning.)
    """
    text = _strip_think_tags(raw_text)
    text = _strip_code_fence(text)
    if not text:
        raise EnvelopeParseError("模型输出为空（可能仅包含思考过程，未生成有效 JSON）")
    objects = _extract_all_json_objects(text)
    envelopes: list[Envelope] = []
    for obj in objects:
        env = _obj_to_envelope(obj)
        envelopes.append(env)
    return envelopes


def parse_envelope(raw_text: str) -> Envelope:
    """
    Parse the LLM output envelope (returns the best single envelope).

    When the model outputs multiple JSON objects, this returns the first
    ``final`` envelope if one exists, otherwise the first envelope.
    For full multi-envelope handling, use :func:`parse_envelopes`.

    Tolerance:
    - Leading/trailing whitespace
    - Optional ```json ... ``` fences (extracted before parsing)
    - Multiple concatenated JSON objects
    """
    envelopes = parse_envelopes(raw_text)
    # Prefer the first "final" envelope.
    for env in envelopes:
        if env["kind"] == "final":
            return env
    return envelopes[0]


def _validate_allowed_keys(obj: dict[str, Any], allowed: set[str]) -> None:
    extra = set(obj.keys()) - allowed
    if extra:
        raise EnvelopeParseError(f"输出包含未允许的字段：{sorted(extra)}")


def extract_single_diff_block(markdown: str) -> str:
    blocks = _DIFF_BLOCK_RE.findall(markdown)
    if len(blocks) != 1:
        raise ValidationError(f"期望恰好 1 个 ```diff 代码块，实际 {len(blocks)} 个")
    return blocks[0].strip("\r\n")


def _try_fix_diff_header(diff_text: str) -> str:
    """Attempt to auto-fix common diff format issues from qwen3-coder.

    The model frequently outputs diffs missing the 'diff --git' header line
    but with valid --- / +++ / @@ content.  This function detects and repairs
    such cases so the diff passes validation without a costly retry round.
    """
    has_git_header = "diff --git a/" in diff_text and " b/" in diff_text
    has_minus = "--- a/" in diff_text or "--- " in diff_text
    has_plus = "+++ b/" in diff_text or "+++ " in diff_text
    has_hunk = "@@" in diff_text

    if has_git_header:
        return diff_text  # Already valid.

    if not (has_hunk and (has_minus or has_plus)):
        return diff_text  # Not a recognizable diff at all.

    # Extract the file path from --- or +++ lines.
    path = None
    for line in diff_text.splitlines():
        if line.startswith("--- a/"):
            path = line[6:].strip()
            break
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            break
        if line.startswith("--- ") and not line.startswith("--- /dev/null"):
            path = line[4:].strip()
            break
        if line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            path = line[4:].strip()
            break

    if not path:
        return diff_text

    # Normalize --- / +++ lines to include a/ b/ prefix if missing.
    lines = diff_text.splitlines()
    fixed_lines: list[str] = []
    for line in lines:
        if line.startswith("--- ") and not line.startswith("--- a/") and not line.startswith("--- /dev/null"):
            fixed_lines.append(f"--- a/{line[4:].strip()}")
        elif line.startswith("+++ ") and not line.startswith("+++ b/") and not line.startswith("+++ /dev/null"):
            fixed_lines.append(f"+++ b/{line[4:].strip()}")
        else:
            fixed_lines.append(line)

    # Prepend the git header.
    header = f"diff --git a/{path} b/{path}"
    return header + "\n" + "\n".join(fixed_lines)


def _try_fix_json_invalid_escapes(json_text: str) -> str:
    """
    Best-effort repair for invalid JSON escape sequences inside strings.

    Common failure cases from LLM output:
    - Windows paths with backslashes can create invalid escapes like ``\\U`` in JSON strings
    - Regex / code snippets can create invalid escapes like ``\\d`` / ``\\s`` / ``\\w``

    JSON only allows these escapes: \\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX.
    Any other "\\X" inside a JSON string is invalid and breaks json.loads().

    This function scans the JSON text and, when inside a JSON string, turns
    invalid escapes like "\\U" / "\\d" into a literal backslash ("\\\\U", "\\\\d").
    """

    out: list[str] = []
    in_str = False
    i = 0
    n = len(json_text)
    while i < n:
        ch = json_text[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue

        # Inside a JSON string.
        if ch == '"':
            out.append(ch)
            in_str = False
            i += 1
            continue

        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        # Backslash inside string: validate escape.
        if i + 1 >= n:
            # Trailing backslash inside a string: treat it as a literal backslash.
            out.append("\\\\")
            i += 1
            continue

        nxt = json_text[i + 1]
        if nxt in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
            out.append(ch)
            out.append(nxt)
            i += 2
            continue

        if nxt == "u":
            digits = json_text[i + 2 : i + 6]
            if len(digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in digits):
                out.append(ch)
                out.append(nxt)
                out.append(digits)
                i += 6
                continue
            # Invalid unicode escape: make the backslash literal.
            out.append("\\\\")
            out.append(nxt)
            i += 2
            continue

        # Invalid escape: make the backslash literal.
        out.append("\\\\")
        out.append(nxt)
        i += 2

    return "".join(out)


def validate_diff_contract(diff_text: str) -> None:
    if "diff --git a/" not in diff_text or " b/" not in diff_text:
        raise ValidationError("diff 必须包含 'diff --git a/... b/...'")
    if "--- a/" not in diff_text and "--- /dev/null" not in diff_text:
        raise ValidationError("diff 必须包含 '--- a/...'")
    if "+++ b/" not in diff_text and "+++ /dev/null" not in diff_text:
        raise ValidationError("diff 必须包含 '+++ b/...'")
    if "@@" not in diff_text:
        raise ValidationError("diff 至少要包含一个 '@@' 块头")


def parse_json_content(final: FinalResponse) -> Any:
    if final["format"] != "json":
        raise ValidationError("final.format 必须为 'json'")
    try:
        return json.loads(final["content"])
    except json.JSONDecodeError as e:
        raise ValidationError(f"final.content 必须是合法的 JSON 字符串：{e}") from e












def validate_memory_summary_schema(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValidationError("历史摘要必须为对象")
    for k in ("constraints", "decisions", "progress", "todo", "pitfalls"):
        v = obj.get(k)
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValidationError(f"memory_summary.{k} 必须为字符串数组")
    return obj


def validate_scaffold_plan_schema(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValidationError("脚手架计划必须为对象")
    files = obj.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationError("scaffold plan.files 必须为非空数组")
    for i, f in enumerate(files):
        if not isinstance(f, dict):
            raise ValidationError(f"files[{i}] 必须为对象")
        path = f.get("path")
        content = f.get("content")
        if not isinstance(path, str) or not path.strip():
            raise ValidationError(f"files[{i}].path 必须为非空字符串")
        if not isinstance(content, str):
            raise ValidationError(f"files[{i}].content 必须为字符串")
    return obj


def validate_refactor_suggestions_schema(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValidationError("重构建议输出必须为对象")
    findings = obj.get("findings")
    if not isinstance(findings, list):
        raise ValidationError("refactor.findings 必须为数组")
    for i, item in enumerate(findings):
        if not isinstance(item, dict):
            raise ValidationError(f"findings[{i}] 必须为对象")
        for k in ("path", "line", "issue", "suggestion", "confidence"):
            if k not in item:
                raise ValidationError(f"findings[{i}] 缺少字段：{k}")
        if not isinstance(item["path"], str):
            raise ValidationError(f"findings[{i}].path 必须为字符串")
        if not isinstance(item["line"], int) or item["line"] < 1:
            raise ValidationError(f"findings[{i}].line 必须为正整数")
        if not isinstance(item["issue"], str):
            raise ValidationError(f"findings[{i}].issue 必须为字符串")
        if not isinstance(item["suggestion"], str):
            raise ValidationError(f"findings[{i}].suggestion 必须为字符串")
        if not isinstance(item["confidence"], (int, float)):
            raise ValidationError(f"findings[{i}].confidence 必须为数字")
    return obj
