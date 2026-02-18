"""RecommendationAgent: builds personalized combos and dish suggestions."""

from __future__ import annotations

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from api.settings import settings
from ingestion.pipeline import get_retriever

SYSTEM = """You are an expert food sommelier and menu consultant. Using ONLY the
menu context provided, help users by:
- Building personalized meal combos (starter + main + dessert + drink)
- Suggesting dishes by budget, preference, occasion or group size
- Highlighting chef's specials or most popular items
- Pairing suggestions

CRITICAL RULES:
- Before including ANY dish in a combo, CHECK its allergen/dietary tags in the
  context. If the user asks for vegan options, NEVER include dishes tagged with
  "Contém laticínios", "Contém ovos", or any animal-derived ingredient.
- Only recommend dishes that are explicitly present in the context.
- Format combos clearly with item names and prices when available.
- Use plain text only. Do NOT use markdown formatting (no **, no ##, no *).
- Always respond in the same language the user used.
"""


class RecommendationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.2,
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
