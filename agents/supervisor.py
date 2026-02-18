"""
Supervisor agent: routes user queries to the right specialist agent and
consolidates the final response.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.nutrition import NutritionAgent
from agents.quality import QualityAgent
from agents.recommendation import RecommendationAgent
from api.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Voce e o SaborAI, um assistente inteligente de alimentacao
especializado em analisar cardapios de restaurantes. Voce tem acesso a tres
agentes especialistas:

1. NutritionAgent  – restricoes alimentares, alergias, info calorica.
2. RecommendationAgent – combos personalizados, sugestoes por orcamento,
   preferencia ou ocasiao.
3. QualityAgent – avalia qualidade das descricoes e sugere melhorias de
   conversao e clareza.

Leia a consulta do usuario, agregue as respostas dos agentes especialistas e
entregue uma resposta final clara e estruturada.

REGRAS OBRIGATORIAS:
- Use texto puro. NAO use formatacao markdown (nada de **, ##, *).
- SEMPRE responda em portugues brasileiro, independentemente do idioma da
  consulta ou das respostas dos agentes.
- Se um agente retornou erro ou nao teve dados, informe ao usuario de forma
  amigavel que ele precisa primeiro ingerir um cardapio antes de consultar.
"""

ROUTING_PROMPT = """Given the user query below, output ONLY a JSON array with
the agent names to invoke. Choose from: ["nutrition", "recommendation", "quality"].

ROUTING RULES:
- "nutrition" = dietary restrictions, allergens, ingredients, calories.
- "recommendation" = combos, suggestions by budget/occasion, pairings.
- "quality" = evaluate menu descriptions, suggest copywriting improvements.
- Select MORE THAN ONE only when the query EXPLICITLY covers multiple domains.

Examples:
- "Quais pratos são veganos?" -> ["nutrition"]
- "Monte um combo por R$60" -> ["recommendation"]
- "Monte um combo vegano por R$60" -> ["nutrition", "recommendation"]
- "Avalie a qualidade do cardápio" -> ["quality"]
- "Opções sem glúten e avalie as descrições" -> ["nutrition", "quality"]

Query: {query}

JSON array:"""


class SupervisorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.2,
        )
        self.agents = {
            "nutrition": NutritionAgent(),
            "recommendation": RecommendationAgent(),
            "quality": QualityAgent(),
        }

    def _route(self, query: str) -> list[str]:
        import json

        prompt = ROUTING_PROMPT.format(query=query)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # Strip markdown fences if present
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            agents = json.loads(raw)
            return [a for a in agents if a in self.agents]
        except Exception:
            logger.warning("Routing failed, defaulting to recommendation. Raw: %s", raw)
            return ["recommendation"]

    def run(self, query: str, menu_name: str | None = None) -> dict:
        selected = self._route(query)
        logger.info("Routing query to agents: %s", selected)

        agent_outputs: dict[str, str] = {}
        for name in selected:
            try:
                agent_outputs[name] = self.agents[name].run(query, menu_name)
            except Exception as exc:
                logger.error("Agent '%s' failed: %s", name, exc)
                agent_outputs[name] = f"[Erro no agente {name}: {exc}]"

        # Consolidate
        consolidation_input = "\n\n".join(
            f"[{name.upper()} AGENT]\n{output}" for name, output in agent_outputs.items()
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Consulta do usuario: {query}\n\n"
                    f"Respostas dos agentes especialistas:\n{consolidation_input}\n\n"
                    "Agora escreva a resposta final consolidada para o usuario em portugues."
                )
            ),
        ]
        final = self.llm.invoke(messages)

        return {
            "query": query,
            "agents_used": selected,
            "agent_outputs": agent_outputs,
            "response": final.content,
        }
