"""
LLM Experiment Runner — Compare models, prompts, temperatures and chunk sizes.

Usage:
    python -m experiments.compare_models
    python -m experiments.compare_models --models gpt-4o-mini gpt-4o --temperatures 0 0.2 0.5
"""

from experiments import main

if __name__ == "__main__":
    main()
