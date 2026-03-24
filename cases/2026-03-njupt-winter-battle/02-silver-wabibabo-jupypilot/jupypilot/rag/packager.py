from __future__ import annotations

from dataclasses import dataclass

from .retriever import RetrievedContext


@dataclass(frozen=True)
class PackagedContext:
    retrieved_context: str
    items: list[RetrievedContext]


class ContextPackager:
    def package(self, contexts: list[RetrievedContext]) -> PackagedContext:
        lines: list[str] = []
        for i, c in enumerate(contexts, start=1):
            lines.append(f"[CONTEXT {i}] path={c.path} lines={c.start_line}-{c.end_line}")
            lines.append(c.snippet.rstrip("\n\r"))
            lines.append("")
        return PackagedContext(retrieved_context="\n".join(lines).strip(), items=contexts)

