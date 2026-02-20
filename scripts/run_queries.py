#!/usr/bin/env python3
"""Generate test questions and run them through the supervisor concurrently."""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

from braintrust_adk import setup_adk
from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_BRAINTRUST_PROJECT = "google-adk-supervisor"

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import AgentConfig
from src.helpers import run_adk_agent

load_dotenv()

MODEL_POOL = ["gemini-2.0-flash-lite"]


def _openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")
    return OpenAI(api_key=api_key)


def generate_questions(num_questions: int, seed: Optional[int] = None) -> list[str]:
    """Generate realistic, varied questions with natural language variation."""
    rng = random.Random(seed)
    client = _openai_client()

    prompt = f"""Generate exactly {num_questions} realistic user questions that test an AI multi-agent system.

Create a diverse mix of:
- Pure math questions
- Pure research questions
- Hybrid questions (research + math)
- Edge cases (ambiguous, conversational, frustrated)

Output requirements:
- Return ONLY a valid JSON array of strings
- No markdown, no explanation
- Keep each question under 200 characters
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
    )
    text = response.output_text.strip()

    try:
        questions = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Question generator returned non-JSON output: {text[:300]}") from exc

    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise RuntimeError("Question generator did not return a JSON array of strings")

    rng.shuffle(questions)
    return questions[:num_questions]


async def run_question(question: str) -> tuple[str, bool]:
    """Run one question through the supervisor with a random model assignment."""
    from src.agent_graph import get_supervisor

    selected_model = random.choice(MODEL_POOL)
    config = AgentConfig(
        supervisor_model=selected_model,
        research_model=selected_model,
        math_model=selected_model,
    )
    supervisor = get_supervisor(config=config, force_rebuild=True)

    try:
        result = await run_adk_agent(
            agent=supervisor,
            query=question,
            app_name="google-adk-supervisor-batch",
        )
        print(f"✅ {question[:80]} -> {str(result.get('final_output', ''))[:80]}")
        return question, True
    except Exception as exc:
        print(f"❌ {question[:80]} -> {exc}")
        return question, False


async def main_async(args: argparse.Namespace) -> None:
    num_questions = args.num_questions if args.num_questions is not None else random.randint(1, 100)
    questions = generate_questions(num_questions=num_questions, seed=args.seed)

    print(f"Generated {len(questions)} questions")
    print(f"Running with concurrency={args.concurrency}")
    print(f"Model pool: {', '.join(MODEL_POOL)}")
    print("=" * 80)

    successes = 0
    failures = 0

    for i in range(0, len(questions), args.concurrency):
        batch = questions[i : i + args.concurrency]
        results = await asyncio.gather(*(run_question(q) for q in batch))
        for _, ok in results:
            if ok:
                successes += 1
            else:
                failures += 1
        print()

    print("=" * 80)
    print(f"Completed. successes={successes} failures={failures}")
    print("=" * 80)

    if args.fail_on_error and failures > 0:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random questions and run through supervisor locally"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "3")),
        help="Number of concurrent questions to process (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Exact number of questions to generate (default: random 1-100)",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any request fails",
    )
    args = parser.parse_args()

    if os.environ.get("BRAINTRUST_API_KEY"):
        setup_adk(
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
            project_name=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
        )

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
