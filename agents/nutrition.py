"""NutritionAgent: handles dietary restrictions, allergies and caloric analysis."""

from __future__ import annotations

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from api.settings import settings
from ingestion.pipeline import get_retriever

SYSTEM = """You are a nutrition and dietary specialist. Using only the menu
context provided, answer questions about:
- Dietary restrictions (vegan, vegetarian, gluten-free, lactose-free, etc.)
- Allergens and ingredients
- Estimated calorie ranges
- Healthiest options

Be precise. If information is not in the context, say so explicitly.
"""


class NutritionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0,
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


def _build_prompt(system: str):
    from langchain.prompts import PromptTemplate

    template = f"{system}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:"
    return PromptTemplate(input_variables=["context", "question"], template=template)
