"""
QualityAgent: evaluates menu description quality and suggests improvements
to boost conversion, clarity and SEO on delivery platforms.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from api.settings import settings
from ingestion.pipeline import get_retriever

SYSTEM = """You are a conversion optimization and UX writing specialist for
food delivery platforms (iFood, Rappi, Uber Eats).

When given menu content, evaluate:

1. **Clarity** (0-10): Are dish names and descriptions clear and specific?
2. **Appetite Appeal** (0-10): Do descriptions trigger desire? Sensory words?
3. **Completeness** (0-10): Are portions, ingredients and allergens mentioned?
4. **SEO/Searchability** (0-10): Do names match what customers typically search?
5. **Overall Score** (0-10): Weighted average.

Then provide:
- Top 3 specific improvement suggestions with before/after rewrite examples.
- Items most at risk of low conversion and why.

Be direct, actionable and data-driven.
"""


class QualityAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
        )

    def run(self, query: str, menu_name: str | None = None) -> str:
        retriever = get_retriever(menu_name)
        docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)

        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(
                content=(
                    f"Menu content:\n{context}\n\n"
                    f"User request: {query}\n\n"
                    "Provide your quality evaluation:"
                )
            ),
        ]
        response = self.llm.invoke(messages)
        return response.content
