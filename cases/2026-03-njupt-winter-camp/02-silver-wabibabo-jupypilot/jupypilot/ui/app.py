from __future__ import annotations

from dataclasses import dataclass

from IPython.display import display

from ..orchestrator.orchestrator import Orchestrator
from ..types import SessionState
from .panels.chat_panel import ChatPanel


@dataclass
class JupyPilotApp:
    orchestrator: Orchestrator
    repo_path: str | None = None

    def __post_init__(self) -> None:
        self.session: SessionState = self.orchestrator.start_session(repo_path=self.repo_path)
        self.chat_panel = ChatPanel(self.orchestrator, self.session)

    def render(self) -> None:
        display(self.chat_panel.widget)
