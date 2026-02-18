"""
QualityAgent: evaluates menu description quality and suggests improvements
to boost conversion, clarity and SEO on delivery platforms.
"""

from __future__ import annotations

from langchain.chains import RetrievalQA
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

    def run(self, query: str, menu_name: str | None = None) -> str:
        retriever = get_retriever(menu_name)
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "verbose": False,
                "prompt": _build_prompt(SYSTEM),
            },
        )
        result = chain.invoke({"query": query})
        return result["result"]
