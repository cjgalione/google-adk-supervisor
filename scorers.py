"""Reusable Braintrust scorer definitions for `braintrust push`."""

from __future__ import annotations

import os
from typing import Any

import braintrust
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROJECT_NAME = os.environ.get("BRAINTRUST_PROJECT", "google-adk-supervisor")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini")


class StepEfficiencyParams(BaseModel):
    output: list[dict[str, Any]]
    max_steps: int = 8


async def step_efficiency_scorer(output):
    """Score based on total number of output messages."""
    max_steps = 8
    if isinstance(output, dict):
        num_steps = len(output.get("messages", []))
    elif isinstance(output, list):
        # Online scoring root spans often store output as a list of message-like items.
        num_steps = len(output)
    elif isinstance(output, str):
        num_steps = 1 if output.strip() else 0
    else:
        num_steps = 0

    if num_steps <= max_steps:
        return 1.0
    return max(0.0, 1.0 - (num_steps - max_steps) / max_steps)


RESPONSE_QUALITY_PROMPT = """
You are an expert evaluator of AI assistant responses.

User Question: {{input}}
AI Response: {{output}}

Evaluate the response based on:
1. ACCURACY
2. COMPLETENESS
3. CLARITY
4. RELEVANCE

Scoring guidance:
- For pure arithmetic questions, a concise correct numeric answer is acceptable.
- For compound questions that ask for both a factual lookup and a calculation,
  the response must include both the factual answer and the computed result.
- Do not mark a response incorrect merely for brevity if it fully answers the question.

Respond with:
EXCELLENT
GOOD
FAIR
POOR
""".strip()


project = braintrust.projects.create(name=PROJECT_NAME)

project.scorers.create(
    name="Step Efficiency",
    slug="step-efficiency",
    description="Penalizes excessive step counts in the final message trace.",
    parameters=StepEfficiencyParams,
    handler=step_efficiency_scorer,
)

project.scorers.create(
    name="Response Quality (LLM Judge)",
    slug="response-quality-llm-judge",
    description=(
        "LLM-as-a-judge scorer for overall response quality, with guidance for "
        "concise math answers and compound research+math questions."
    ),
    # Use chat-style messages to avoid OpenAI errors about missing `messages`.
    messages=[
        {
            "role": "user",
            "content": RESPONSE_QUALITY_PROMPT,
        }
    ],
    model=JUDGE_MODEL,
    use_cot=True,
    choice_scores={
        "EXCELLENT": 1.0,
        "GOOD": 0.75,
        "FAIR": 0.5,
        "POOR": 0.0,
    },
)
