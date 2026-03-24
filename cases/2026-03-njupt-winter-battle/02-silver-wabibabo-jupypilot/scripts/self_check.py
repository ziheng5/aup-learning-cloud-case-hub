from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jupypilot.config import Config  # noqa: E402
from jupypilot.orchestrator.orchestrator import Orchestrator  # noqa: E402
from jupypilot.orchestrator.validator import (  # noqa: E402
    extract_single_diff_block,
    parse_envelope,
    validate_diff_contract,
)
from jupypilot.orchestrator.session import new_session  # noqa: E402
from jupypilot.tools.runtime import ToolRuntime  # noqa: E402


def main() -> None:
    cfg = Config(repo_path=".", artifacts_dir=".jupypilot")

    # Backend boot
    orch = Orchestrator(cfg)
    sess = orch.start_session()
    orch.pin_requirement(sess, "测试：Pinned requirements 不应在压缩时丢失")

    # ToolRuntime basic
    tools = ToolRuntime(cfg)
    s2 = new_session(repo_path=str(cfg.repo_root()))
    of = tools.execute("open_file", {"path": "架构设计_修订稿.md", "start_line": 1, "end_line": 3}, s2)
    assert of["ok"], of

    sc = tools.execute("search_code", {"query": "Orchestrator", "glob": "**/*.py", "max_results": 3}, s2)
    assert sc["ok"], sc
    assert isinstance(sc["data"]["matches"], list)

    # Envelope parser and diff validator
    env = parse_envelope(
        '{"kind":"final","format":"markdown","content":"```diff\\ndiff --git a/a b/a\\n--- a/a\\n+++ b/a\\n@@ -1 +1 @@\\n-a\\n+b\\n```"}'
    )
    diff = extract_single_diff_block(env["content"])
    validate_diff_contract(diff)

    print("SELF_CHECK_OK")


if __name__ == "__main__":
    main()
