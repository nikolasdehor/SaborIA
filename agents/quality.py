"""
QualityAgent: evaluates menu description quality and suggests improvements
to boost conversion, clarity and SEO on delivery platforms.
"""

from __future__ import annotations

import logging

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from agents.retry import async_retry_with_backoff, retry_with_backoff
from api.settings import settings
from ingestion.pipeline import get_retriever

logger = logging.getLogger(__name__)

SYSTEM = """You are a conversion optimization and UX writing specialist for
food delivery platforms (iFood, Rappi, Uber Eats).

When given menu content, evaluate:

1. Clarity (0-10): Are dish names and descriptions clear and specific?
2. Appetite Appeal (0-10): Do descriptions trigger desire? Sensory words?
3. Completeness (0-10): Are portions, ingredients and allergens mentioned?
4. SEO/Searchability (0-10): Do names match what customers typically search?
5. Overall Score (0-10): Weighted average.

Then provide:
- Top 3 specific improvement suggestions with before/after rewrite examples.
- Items most at risk of low conversion and why.

Be direct, actionable and data-driven.
Use plain text only. Do NOT use markdown formatting (no **, no ##, no *).
Always respond in the same language the user used.
"""


def _build_prompt(system: str):
    from langchain.prompts import PromptTemplate

    template = f"{system}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:"
    return PromptTemplate(input_variables=["context", "question"], template=template)


class QualityAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
        )

    def _build_chain(self, menu_name: str | None = None) -> RetrievalQA:
        retriever = get_retriever(menu_name)
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "verbose": False,
                "prompt": _build_prompt(SYSTEM),
            },
        )

    @retry_with_backoff(max_retries=3)
    def run(self, query: str, menu_name: str | None = None) -> str:
        chain = self._build_chain(menu_name)
        result = chain.invoke({"query": query})
        return result["result"]

    @async_retry_with_backoff(max_retries=3)
    async def arun(self, query: str, menu_name: str | None = None) -> str:
        chain = self._build_chain(menu_name)
        result = await chain.ainvoke({"query": query})
        return result["result"]
