"""
Supervisor agent: routes user queries to the right specialist agent and
consolidates the final response.

Supports both sync (`run`) and async (`arun`) execution. The async path
invokes specialist agents in **parallel** via `asyncio.gather`, cutting
latency roughly by `1/N` where N is the number of routed agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.nutrition import NutritionAgent
from agents.quality import QualityAgent
from agents.recommendation import RecommendationAgent
from agents.retry import async_retry_with_backoff, retry_with_backoff
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

    @retry_with_backoff(max_retries=2)
    def _route(self, query: str) -> list[str]:
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

    @async_retry_with_backoff(max_retries=2)
    async def _aroute(self, query: str) -> list[str]:
        prompt = ROUTING_PROMPT.format(query=query)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            agents = json.loads(raw)
            return [a for a in agents if a in self.agents]
        except Exception:
            logger.warning("Routing failed, defaulting to recommendation. Raw: %s", raw)
            return ["recommendation"]

    def _consolidate(self, query: str, agent_outputs: dict[str, str]) -> str:
        """Ask the LLM to merge specialist answers into one final response."""
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
        return self.llm.invoke(messages).content

    async def _aconsolidate(self, query: str, agent_outputs: dict[str, str]) -> str:
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
        resp = await self.llm.ainvoke(messages)
        return resp.content

    # ── Sync path (backward-compatible) ──────────────────────────────────

    def run(self, query: str, menu_name: str | None = None) -> dict:
        t0 = time.perf_counter()
        selected = self._route(query)
        logger.info("Routing query to agents: %s", selected)

        agent_outputs: dict[str, str] = {}
        for name in selected:
            try:
                agent_outputs[name] = self.agents[name].run(query, menu_name)
            except Exception as exc:
                logger.error("Agent '%s' failed: %s", name, exc)
                agent_outputs[name] = f"[Erro no agente {name}: {exc}]"

        final = self._consolidate(query, agent_outputs)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("Query completed in %.0f ms (agents: %s)", latency_ms, selected)

        return {
            "query": query,
            "agents_used": selected,
            "agent_outputs": agent_outputs,
            "response": final,
            "latency_ms": round(latency_ms, 1),
        }

    # ── Async path (parallel agent execution) ────────────────────────────

    async def arun(self, query: str, menu_name: str | None = None) -> dict:
        """Run the full pipeline asynchronously.

        Specialist agents are invoked **in parallel** via asyncio.gather,
        reducing wall-clock latency when multiple agents are selected.
        """
        t0 = time.perf_counter()
        selected = await self._aroute(query)
        logger.info("Routing query to agents (async): %s", selected)

        async def _invoke_agent(name: str) -> tuple[str, str]:
            try:
                result = await self.agents[name].arun(query, menu_name)
                return name, result
            except Exception as exc:
                logger.error("Agent '%s' failed: %s", name, exc)
                return name, f"[Erro no agente {name}: {exc}]"

        results = await asyncio.gather(*[_invoke_agent(n) for n in selected])
        agent_outputs = dict(results)

        final = await self._aconsolidate(query, agent_outputs)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("Query completed in %.0f ms (async, agents: %s)", latency_ms, selected)

        return {
            "query": query,
            "agents_used": selected,
            "agent_outputs": agent_outputs,
            "response": final,
            "latency_ms": round(latency_ms, 1),
        }
