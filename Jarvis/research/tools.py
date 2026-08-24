"""The research agent, exposed as a tool a project can call.

Kept separate from :mod:`tools.web` because the two answer different questions.
``web_search`` returns links; this returns cited findings. A project that needs
to decide something is served by the second and drowned by the first -- eight
URLs in a 24k context window is most of the budget spent on things the model
must then fetch and read itself.
"""

from __future__ import annotations

from typing import Any

from tools.registry import RiskLevel, ToolContext, ToolSpec


def make_research_tools(*, brain: Any = None, graph: Any = None) -> list[ToolSpec]:
    """Build the research tool.  ``brain`` is what extracts findings."""

    def research(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        from research.agent import ResearchAgent

        question = str(arguments.get("question", "")).strip()
        if not question:
            return {"ok": False, "error": "question is required"}

        agent = ResearchAgent(brain=brain, graph=graph)
        report = agent.research(
            question,
            max_sources=int(arguments.get("max_sources", 3) or 3),
            max_seconds=float(arguments.get("max_seconds", 150) or 150),
        )
        return {
            "ok": report.grounded,
            "question": question,
            "summary": report.summary,
            "report": report.as_text()[:6000],
            "sources": [item.url for item in report.sources if item.ok],
            "grounded": report.grounded,
            "offline": report.offline,
        }

    return [
        ToolSpec(
            name="research",
            purpose=(
                "Answer a technical question from public documentation. Returns findings with "
                "the source and a verbatim quote for each, so claims can be checked."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "max_sources": {"type": "integer"},
                },
                "required": ["question"],
            },
            adapter=research,
            risk=RiskLevel.MODERATE,
            tags=("research", "web", "investigate"),
            example='{"name": "research", "arguments": {"question": "how do I read a WAV file with the Python standard library?"}}',
        )
    ]
