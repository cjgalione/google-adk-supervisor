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

from dotenv import load_dotenv
from google import genai

DEFAULT_BRAINTRUST_PROJECT = "google-adk-supervisor"

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import AgentConfig
from src.helpers import run_adk_agent
from src.tracing import configure_adk_tracing

load_dotenv()

MODEL_POOL = ["gemini-2.0-flash-lite"]
QUESTION_GENERATOR_MODEL = "gemini-2.0-flash-lite"

QUESTION_BANK = [
    "What is 37 * 24?",
    "Who won the first modern Olympic Games and in what year?",
    "If a supernova releases 10^44 joules, how many 60W lightbulb-hours is that?",
    "What's the capital of Japan and what is 18% of 250?",
    "Hey, can you help me quickly estimate 15% tip on $86.40?",
    "When was the Eiffel Tower completed?",
    "Compute (1250 / 5) - 73.",
    "I'm frustrated. Just tell me if 144 divided by 12 is actually 11 or 12.",
    "What is the population of Canada and what is 2% of that number?",
    "Convert 10^6 joules to horsepower-seconds.",
    "What is the square root of 2025?",
    "Can you summarize what a quasar is in one sentence?",
    "If GDP is $2.1T and growth is 3.2%, what is the increase?",
    "Who discovered penicillin and in what year?",
    "What is (48 + 72) / 6?",
]


def _extract_json_array(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.startswith("json"):
                text = text[4:].strip()

    parsed = json.loads(text)
    if not isinstance(parsed, list) or not all(isinstance(q, str) for q in parsed):
        raise RuntimeError("Question generator did not return a JSON array of strings")
    return parsed


def generate_questions(num_questions: int, seed: Optional[int] = None) -> list[str]:
    """Generate realistic, varied questions with Gemini."""
    rng = random.Random(seed)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment")

    client = genai.Client(api_key=api_key)
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
    response = client.models.generate_content(
        model=QUESTION_GENERATOR_MODEL,
        contents=prompt,
    )
    text = (response.text or "").strip()
    try:
        questions = _extract_json_array(text)
        rng.shuffle(questions)
        return questions[:num_questions]
    except Exception:
        questions = QUESTION_BANK.copy()
        rng.shuffle(questions)
        if num_questions <= len(questions):
            return questions[:num_questions]
        out: list[str] = []
        while len(out) < num_questions:
            remaining = num_questions - len(out)
            out.extend(questions[:remaining])
            rng.shuffle(questions)
        return out


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
        configure_adk_tracing(
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
            project_name=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
        )

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
