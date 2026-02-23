"""NutritionAgent: handles dietary restrictions, allergies and caloric analysis."""

from __future__ import annotations

import logging

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from agents.retry import async_retry_with_backoff, retry_with_backoff
from api.settings import settings
from ingestion.pipeline import get_retriever

logger = logging.getLogger(__name__)

SYSTEM = """You are a nutrition and dietary specialist. Using ONLY the menu
context provided, answer questions about:
- Dietary restrictions (vegan, vegetarian, gluten-free, lactose-free, etc.)
- Allergens and ingredients
- Estimated calorie ranges
- Healthiest options

CRITICAL RULES:
- Always check the allergen/dietary tags in parentheses for each dish.
- A dish tagged "Contém laticínios" is NOT vegan. A dish tagged "Contém glúten"
  is NOT gluten-free. Respect every tag strictly.
- If information is not in the context, say so explicitly.
- Use plain text only. Do NOT use markdown formatting (no **, no ##, no *).
- Always respond in the same language the user used.
"""


class NutritionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0,
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


def _build_prompt(system: str):
    from langchain.prompts import PromptTemplate

    template = f"{system}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:"
    return PromptTemplate(input_variables=["context", "question"], template=template)
