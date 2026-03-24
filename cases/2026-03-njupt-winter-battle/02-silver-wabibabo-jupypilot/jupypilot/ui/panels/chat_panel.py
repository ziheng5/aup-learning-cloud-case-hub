from __future__ import annotations

import html
import json
import logging
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import HTML, Markdown, display

from ...orchestrator.orchestrator import Orchestrator
from ...types import SessionState
from ..event_format import format_event_line_zh, task_kind_to_zh

# ── Debug logger: writes to .jupypilot/debug_ui.log ──
_debug_log_path = Path(".jupypilot/debug_ui.log")
_debug_log_path.parent.mkdir(parents=True, exist_ok=True)
_dbg = logging.getLogger("jupypilot.ui.debug")
_dbg.setLevel(logging.DEBUG)
_dbg.propagate = False
if not _dbg.handlers:
    _fh = logging.FileHandler(str(_debug_log_path), encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _dbg.addHandler(_fh)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
_TITLE_HTML = """
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 16px 20px; border-radius: 10px;
    margin-bottom: 12px; font-family: sans-serif;
">
    <h2 style="margin:0 0 4px 0; font-size:22px;">🛠️ JupyPilot · 代码辅助编程专家</h2>
    <span style="opacity:0.85; font-size:13px;">代码解释 · 错误调试 · 测试生成 · 脚手架 · 风格检查</span>
</div>
"""

_SECTION_STYLE = "border:1px solid #e0e0e0; border-radius:8px; padding:10px 14px; margin-bottom:8px; background:#fafafa;"
_CARD_STYLE = "border:1px solid #ddd; border-radius:8px; padding:12px; background:white;"


def _layout(**kwargs: str) -> widgets.Layout:
    """Build Layout by setting attributes after construction to avoid traitlets 4.2+ DeprecationWarning (unrecognized args to super().__init__)."""
    L = widgets.Layout()
    for k, v in kwargs.items():
        setattr(L, k, v)
    return L


class ChatPanel:
    """All-in-one chat panel with code assistant features."""

    def __init__(self, orchestrator: Orchestrator, session: SessionState) -> None:
        self._orchestrator = orchestrator
        self._session = session
        self._ctx_window_tokens = 32768
        self._last_usage: dict[str, int] | None = None

        # ── Title ──
        self._title = widgets.HTML(value=_TITLE_HTML)

        # ── Token bar ──
        self.token_bar = widgets.IntProgress(
            value=0, min=0, max=self._ctx_window_tokens,
            description="", bar_style="info",
            layout=_layout(width="75%", height="18px"),
        )
        self.token_text = widgets.HTML(value=self._format_usage_html(None))
        token_section = widgets.HBox(
            [widgets.HTML("<b style='font-size:12px;color:#555;'>上下文：</b>"), self.token_bar, self.token_text],
            layout=_layout(align_items="center", gap="6px"),
        )

        # ── Project path ──
        self.repo_path = widgets.Text(
            value=str(session.repo_path), placeholder="项目根目录路径",
            layout=_layout(width="75%"),
        )
        self.repo_btn = widgets.Button(description="📂 切换项目", button_style="warning",
                                       layout=_layout(width="120px"))
        self.repo_btn.on_click(self._on_switch_repo)
        repo_section = widgets.HBox(
            [widgets.HTML("<b style='font-size:12px;color:#555;'>项目路径：</b>"), self.repo_path, self.repo_btn],
            layout=_layout(align_items="center", gap="6px"),
        )

        # ── Pinned constraints ──
        self.pinned = widgets.Text(
            value="", placeholder="例如：所有代码必须兼容 Python 3.12",
            layout=_layout(width="75%"),
        )
        self.pin_btn = widgets.Button(description="📌 添加", button_style="info",
                                      layout=_layout(width="120px"))
        self.pin_btn.on_click(self._on_pin)
        pin_section = widgets.HBox(
            [widgets.HTML("<b style='font-size:12px;color:#555;'>固定约束：</b>"), self.pinned, self.pin_btn],
            layout=_layout(align_items="center", gap="6px"),
        )

        config_box = widgets.VBox(
            [token_section, repo_section, pin_section],
            layout=_layout(object_position="left", padding="8px 12px",
                          border="1px solid #e0e0e0", border_radius="8px",
                          margin="0 0 10px 0"),
        )

        # ── Mode selector (task kind) ──
        self.mode = widgets.ToggleButtons(
            options=[
                ("💬 代码解释", "code_qa"),
                ("🐛 错误调试", "code_patch"),
                ("🧪 生成测试", "testgen"),
                ("📦 生成脚手架", "scaffold"),
                ("✨ 风格检查", "refactor"),
            ],
            value="code_qa",
            tooltips=[
                "解释代码逻辑，逐段说明",
                "分析错误并给出调试建议",
                "为指定代码生成测试用例",
                "生成 Python/Java/C++ 项目脚手架",
                "代码风格检查与重构建议",
            ],
            layout=_layout(width="100%"),
            style={"button_width": "auto", "font_weight": "bold"},
        )

        # ── Input area + Send button (combined row) ──
        self.input = widgets.Textarea(
            value="", placeholder="在这里输入你的问题、代码片段、文件路径或报错信息...",
            layout=_layout(flex="1 1 auto", height="100px"),
        )
        self.send_btn = widgets.Button(
            description="🚀 发送", button_style="primary",
            layout=_layout(width="80px", height="100px"),
        )
        self.send_btn.on_click(self._on_send)

        input_row = widgets.HBox(
            [self.input, self.send_btn],
            layout=_layout(width="100%", margin="4px 0"),
        )

        # ── Loading ──
        self.loading_bar = widgets.IntProgress(
            value=0, min=0, max=100, bar_style="info",
            layout=_layout(width="100%", height="6px", display="none"),
        )
        self.status_text = widgets.HTML(value="", layout=_layout(display="none"))

        # ── Events log ──
        self.events = widgets.Output(
            layout=_layout(border="1px solid #e8e8e8", border_radius="6px",
                          padding="8px", height="180px", overflow="auto",
                          background="#f9f9f9"),
        )

        # ── Result output ──
        self.output = widgets.Output(
            layout=_layout(border="1px solid #d0d0d0", border_radius="8px",
                          padding="12px", min_height="100px", background="white"),
        )

        # ── Assemble layout ──
        self.widget = widgets.VBox([
            self._title,
            config_box,
            self.mode,
            input_row,
            self.loading_bar,
            self.status_text,
            widgets.HTML("<div style='font-size:12px;color:#888;margin:6px 0 2px;'>📋 事件日志</div>"),
            self.events,
            widgets.HTML("<div style='font-size:12px;color:#888;margin:6px 0 2px;'>📝 结果输出</div>"),
            self.output,
        ], layout=_layout(max_width="900px", padding="0 8px"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _format_usage_html(self, usage: dict[str, int] | None) -> str:
        if not usage:
            return "<span style='font-size:11px;color:#999;'>暂无</span>"
        total = usage.get("total_tokens", 0)
        prompt = usage.get("prompt_tokens", 0)
        comp = usage.get("completion_tokens", 0)
        return (
            f"<span style='font-size:11px;color:#555;'>"
            f"总 {total} · 输入 {prompt} · 输出 {comp}"
            f"</span>"
        )

    def _update_token_dashboard(self, usage: dict[str, int]) -> None:
        total = usage.get("total_tokens", 0)
        self.token_bar.value = min(total, self._ctx_window_tokens)
        self.token_text.value = self._format_usage_html(usage)
        self._last_usage = usage

    def _set_loading(self, active: bool) -> None:
        if active:
            self.loading_bar.layout.display = ""
            self.loading_bar.value = 0
            self.loading_bar.bar_style = "info"
            self.send_btn.disabled = True
        else:
            self.loading_bar.layout.display = "none"
            self.send_btn.disabled = False

    def _set_status(self, text: str, *, progress: int | None = None) -> None:
        if text:
            self.status_text.layout.display = ""
            self.status_text.value = (
                f"<span style='font-size:12px;color:#666;'>{html.escape(text)}</span>"
            )
        else:
            self.status_text.layout.display = "none"
            self.status_text.value = ""
        if progress is not None:
            self.loading_bar.value = min(max(progress, 0), 100)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------
    def _on_switch_repo(self, _btn: Any) -> None:
        new_path = self.repo_path.value.strip()
        if not new_path:
            return
        p = Path(new_path)
        if not p.is_dir():
            self._set_status(f"路径不存在：{new_path}")
            return
        self._session.repo_path = str(p.resolve())
        self._set_status(f"已切换到：{self._session.repo_path}")

    def _on_pin(self, _btn: Any) -> None:
        text = self.pinned.value.strip()
        if not text:
            return
        self._orchestrator.pin_requirement(self._session, text)
        self.pinned.value = ""
        self._set_status(f"已添加固定约束：{text}")

    # ------------------------------------------------------------------
    # Send handler
    # ------------------------------------------------------------------
    def _on_send(self, _btn: Any) -> None:
        user_text = self.input.value.strip()
        if not user_text:
            return

        task_kind = self.mode.value  # code_qa | code_patch | testgen | scaffold | refactor
        _dbg.debug("[_on_send] task=%s text=%r", task_kind, user_text[:100])
        self.events.clear_output()
        self.output.clear_output()
        self._set_loading(True)
        self._set_status("正在处理...")

        def _worker() -> None:
            _dbg.debug("[_worker] START task=%s", task_kind)
            try:
                final = self._orchestrator.handle_user_request(
                    self._session,
                    task_kind=task_kind,
                    user_text=user_text,
                    on_event=lambda e: self._handle_ui_event(e, task_kind=task_kind),
                )
                _dbg.debug("[_worker] GOT FINAL kind=%s format=%s content_len=%d",
                           final.get("kind"), final.get("format"), len(final.get("content", "")))
                self._render_result(task_kind, final)
                _dbg.debug("[_worker] RENDER DONE")
            except Exception as exc:
                _dbg.debug("[_worker] EXCEPTION %s: %s\n%s", type(exc).__name__, exc, traceback.format_exc())
                self.output.append_display_data(HTML(
                    f"<div style='color:#c0392b;padding:8px;border:1px solid #e74c3c;"
                    f"border-radius:6px;background:#fdf0ef;'>"
                    f"<b>发生错误：</b>{html.escape(type(exc).__name__)}<br>"
                    f"{html.escape(str(exc))}</div>"
                ))
            finally:
                self._set_loading(False)
                self._set_status("")
                _dbg.debug("[_worker] END")

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # UI event handler (called from orchestrator on_event callback)
    # ------------------------------------------------------------------
    def _handle_ui_event(self, e: dict[str, Any], *, task_kind: str = "") -> None:
        event = e.get("event", "")
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        _dbg.debug("[_handle_ui_event] event=%s data_keys=%s", event, list(data.keys()) if data else "none")

        # Update token dashboard on usage events
        if event == "llm_usage":
            self._update_token_dashboard(data)

        # Update status on retry events
        if event == "llm_retry":
            hint = str(data.get("hint_zh", "")).strip() or "正在重试..."
            attempt = data.get("attempt", 0)
            max_r = data.get("max_retries", "?")
            self._set_status(f"{hint}（{attempt + 1}/{max_r}）")

        # Update loading progress on loop iterations
        if event == "tool_loop_iter":
            it = data.get("iter", 0)
            self._set_status(f"推理中... 第 {it + 1} 轮", progress=min((it + 1) * 20, 90))

        # Log all events to the events panel
        line = format_event_line_zh(e)
        _dbg.debug("[_handle_ui_event] formatted_line=%r", line[:200] if line else "EMPTY")
        if line:
            try:
                self.events.append_stdout(line + "\n")
                _dbg.debug("[_handle_ui_event] append_stdout OK")
            except Exception as ex:
                _dbg.debug("[_handle_ui_event] append_stdout FAILED: %s", ex)

    # ------------------------------------------------------------------
    # Result rendering
    # ------------------------------------------------------------------
    def _render_result(self, task_kind: str, final: dict[str, Any]) -> None:
        content = final.get("content", "")
        fmt = final.get("format", "markdown")
        _dbg.debug("[_render_result] task=%s fmt=%s content_len=%d", task_kind, fmt, len(content))

        try:
            if task_kind == "scaffold" and fmt == "json":
                self._render_scaffold(content)
            elif task_kind == "refactor" and fmt == "json":
                self._render_refactor(content)
            else:
                # code_qa, code_patch, testgen, refactor_diff → markdown
                self.output.append_display_data(Markdown(content))
                # 若内容过短，提示可能是生成长度被截断（事件日志中的“N 字符”即实际收到的长度）
                if len(content) < 150:
                    self.output.append_display_data(HTML(
                        "<p style='font-size:12px;color:#888;margin-top:8px;'>"
                        "⚠️ 回答较短，可能是模型生成长度受限。可尝试重新提问，或缩小问题范围后再试。</p>"
                    ))
            _dbg.debug("[_render_result] display OK")
        except Exception as ex:
            _dbg.debug("[_render_result] display FAILED: %s\n%s", ex, traceback.format_exc())

    def _render_scaffold(self, content: str) -> None:
        """Render scaffold JSON (files array) as styled cards."""
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            self.output.append_display_data(Markdown(content))
            return

        files = obj.get("files", [])
        if not files:
            self.output.append_display_data(HTML("<p style='color:#999;'>脚手架未生成任何文件。</p>"))
            return

        self.output.append_display_data(HTML(
            f"<div style='font-size:14px;font-weight:bold;color:#2c3e50;margin-bottom:8px;'>"
            f"📦 脚手架生成了 {len(files)} 个文件</div>"
        ))
        for f in files:
            path = f.get("path", "未知路径")
            code = html.escape(f.get("content", ""))
            self.output.append_display_data(HTML(
                f"<details style='{_CARD_STYLE} margin-bottom:6px;'>"
                f"<summary style='cursor:pointer;font-weight:bold;color:#2980b9;'>"
                f"📄 {html.escape(path)}</summary>"
                f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px;"
                f"overflow-x:auto;font-size:12px;margin-top:6px;'>{code}</pre>"
                f"</details>"
            ))

    def _render_refactor(self, content: str) -> None:
        """Render refactor JSON (findings array) as styled cards."""
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            self.output.append_display_data(Markdown(content))
            return

        findings = obj.get("findings", [])
        if not findings:
            summary = str(obj.get("summary", "") or "").strip()
            mentor = str(obj.get("mentor_deep_dive", "") or "").strip()

            # 若模型提供了 summary/mentor_deep_dive，区分「无法分析」与「确实没有问题」
            if summary or mentor:
                details_html = ""
                if summary:
                    details_html += f"<p style='margin:4px 0 0 0;font-size:12px;color:#555;'>{html.escape(summary)}</p>"
                if mentor:
                    details_html += (
                        "<details style='margin-top:4px;font-size:12px;color:#555;'>"
                        "<summary style='cursor:pointer;color:#2980b9;'>查看模型说明</summary>"
                        f"<div style='margin-top:4px;'>{html.escape(mentor)}</div>"
                        "</details>"
                    )
                self.output.append_display_data(HTML(
                    "<div style='color:#e67e22;padding:8px;'>"
                    "⚠️ 未能对目标代码给出具体风格建议，模型认为可能是路径或文件不存在所致。"
                    "</div>" + details_html
                ))
            else:
                self.output.append_display_data(HTML(
                    "<div style='color:#27ae60;padding:8px;'>✅ 未发现风格问题，代码质量良好。</div>"
                ))
            return

        self.output.append_display_data(HTML(
            f"<div style='font-size:14px;font-weight:bold;color:#2c3e50;margin-bottom:8px;'>"
            f"✨ 发现 {len(findings)} 条建议</div>"
        ))
        for item in findings:
            path = html.escape(str(item.get("path", "")))
            line = item.get("line", "?")
            issue = html.escape(str(item.get("issue", "")))
            suggestion = html.escape(str(item.get("suggestion", "")))
            confidence = item.get("confidence", 0)
            conf_pct = f"{confidence * 100:.0f}%" if isinstance(confidence, (int, float)) else str(confidence)
            color = "#e74c3c" if confidence >= 0.8 else "#f39c12" if confidence >= 0.5 else "#95a5a6"
            self.output.append_display_data(HTML(
                f"<div style='{_CARD_STYLE} margin-bottom:6px;border-left:3px solid {color};'>"
                f"<div style='font-size:12px;color:#888;'>{path}:{line} · 置信度 {conf_pct}</div>"
                f"<div style='color:#c0392b;margin:4px 0;'>⚠️ {issue}</div>"
                f"<div style='color:#27ae60;'>💡 {suggestion}</div>"
                f"</div>"
            ))
