"""
Synthetic menu data generator for SaborAI.

Generates realistic restaurant menus using LLM, with control over:
- Cuisine type (Italian, Japanese, Brazilian, Mexican, etc.)
- Price tier (budget, mid-range, premium)
- Dietary diversity (vegan, gluten-free, allergen-labeled items)

Usage:
    python -m scripts.generate_synthetic_menus --count 5 --output data/sample_menus/
    python -m scripts.generate_synthetic_menus --cuisine japanese --count 2 --ingest
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

CUISINES = [
    "Italiana",
    "Japonesa",
    "Brasileira",
    "Mexicana",
    "Francesa",
    "Indiana",
    "Tailandesa",
    "Mediterranea",
    "Vegetariana/Vegana",
    "Americana",
]

PRICE_TIERS = {
    "budget": {"min": 12, "max": 45},
    "mid-range": {"min": 25, "max": 90},
    "premium": {"min": 45, "max": 180},
}

MENU_GENERATION_PROMPT = """Voce e um consultor criativo de restaurantes.
Gere um cardapio COMPLETO e REALISTA de restaurante em portugues brasileiro.

Detalhes do restaurante:
- Nome: {restaurant_name}
- Cozinha: {cuisine}
- Faixa de preco: {price_tier} (R${price_min} a R${price_max})

Requisitos:
- Inclua secoes: ENTRADAS (4-5 itens), PRATOS PRINCIPAIS (5-7 itens),
  SOBREMESAS (3-4 itens), BEBIDAS (5-6 itens), INFORMACOES
- Cada prato deve ter: nome, preco em R$, descricao apetitosa de 1-2 frases
- Apos CADA descricao, na mesma linha, adicione tags de alergenos/dieta entre parenteses
  no formato: (Tag1 | Tag2 | Tag3)
  Tags validas: Vegano, Vegetariano, Sem Gluten, Sem Lacticinios, Sem Lactose,
  Contem gluten, Contem laticinios, Contem ovo, Contem frutos do mar,
  Contem peixe, Contem oleaginosas, Picante
- Garanta ao menos 2 opcoes veganas e 3 opcoes sem gluten no cardapio
- Use separadores de secao com linhas de ===
- Precos devem ser realistas para a faixa {price_tier}
- Na secao INFORMACOES, inclua horario de funcionamento e politica de alergenos

Retorne SOMENTE o texto do cardapio, sem comentarios adicionais. \
Comece com o nome do restaurante."""


class MenuItem(BaseModel):
    """Metadata for a generated menu, used for tracking and validation."""

    restaurant_name: str
    cuisine: str
    price_tier: str
    num_items: int
    has_vegan: bool
    has_gluten_free: bool
    file_path: str


def generate_restaurant_name(client: OpenAI, cuisine: str, model: str) -> str:
    """Generate a creative restaurant name for the given cuisine."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Gere UM nome criativo para um restaurante de cozinha {cuisine} "
                    f"no Brasil. Responda APENAS com o nome, nada mais."
                ),
            }
        ],
        temperature=0.9,
        max_tokens=30,
    )
    return resp.choices[0].message.content.strip().strip('"')


def generate_menu(
    client: OpenAI,
    cuisine: str,
    price_tier: str,
    model: str,
    restaurant_name: str | None = None,
) -> tuple[str, str]:
    """Generate a complete menu. Returns (restaurant_name, menu_text)."""
    if not restaurant_name:
        restaurant_name = generate_restaurant_name(client, cuisine, model)

    tier = PRICE_TIERS[price_tier]
    prompt = MENU_GENERATION_PROMPT.format(
        restaurant_name=restaurant_name,
        cuisine=cuisine,
        price_tier=price_tier,
        price_min=tier["min"],
        price_max=tier["max"],
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000,
    )
    menu_text = resp.choices[0].message.content.strip()
    return restaurant_name, menu_text


def save_menu(menu_text: str, restaurant_name: str, output_dir: Path) -> Path:
    """Save generated menu to a .txt file."""
    safe_name = restaurant_name.lower().replace(" ", "_").replace("/", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
    file_path = output_dir / f"{safe_name}.txt"
    file_path.write_text(menu_text, encoding="utf-8")
    return file_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic restaurant menus")
    parser.add_argument("--count", type=int, default=3, help="Number of menus to generate")
    parser.add_argument(
        "--cuisine",
        type=str,
        default=None,
        help=f"Specific cuisine ({', '.join(CUISINES)})",
    )
    parser.add_argument(
        "--price-tier",
        type=str,
        default="mid-range",
        choices=list(PRICE_TIERS.keys()),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_menus/",
        help="Output directory for generated menus",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Auto-ingest generated menus into ChromaDB",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model to use (default: from settings)",
    )
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s | %(message)s")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load settings for API key and default model
    from api.settings import settings

    api_key = settings.openai_api_key
    model = args.model or settings.openai_model

    client = OpenAI(api_key=api_key)

    # Pick cuisines
    if args.cuisine:
        cuisines = [args.cuisine] * args.count
    else:
        cuisines = random.choices(CUISINES, k=args.count)

    # Distribute price tiers for diversity
    tiers = list(PRICE_TIERS.keys())
    price_tiers = [tiers[i % len(tiers)] for i in range(args.count)]

    manifest: list[MenuItem] = []

    for i, (cuisine, tier) in enumerate(zip(cuisines, price_tiers)):
        logger.info("Generating menu %d/%d: %s (%s)...", i + 1, args.count, cuisine, tier)

        name, text = generate_menu(client, cuisine, tier, model)
        path = save_menu(text, name, output_dir)
        logger.info("  Saved: %s", path)

        entry = MenuItem(
            restaurant_name=name,
            cuisine=cuisine,
            price_tier=tier,
            num_items=text.count("R$"),
            has_vegan="Vegano" in text or "vegano" in text,
            has_gluten_free="Sem Gluten" in text or "Sem gluten" in text,
            file_path=str(path),
        )
        manifest.append(entry)

        if args.ingest:
            from ingestion.pipeline import ingest_file

            result = ingest_file(str(path), name)
            logger.info("  Ingested: %d chunks", result["total_chunks"])

    # Save manifest
    manifest_path = output_dir / "synthetic_manifest.json"
    manifest_path.write_text(
        json.dumps([m.model_dump() for m in manifest], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Manifest saved to %s", manifest_path)
    logger.info("Done! Generated %d menus.", len(manifest))


if __name__ == "__main__":
    main()
