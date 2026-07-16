#!/usr/bin/env python3
"""Generate test questions and run them through the supervisor concurrently."""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
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
from src.agent_graph import run_supervisor_with_critic
from src.model_resolver import make_wrapped_openai_client
from src.tracing import configure_adk_tracing

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values or default


MODEL_POOL = _env_csv("MODEL_POOL", ["gemini-2.0-flash-lite"])
QUESTION_GENERATOR_MODEL = (
    os.environ.get("QUESTION_GENERATOR_MODEL", MODEL_POOL[0]).strip() or MODEL_POOL[0]
)

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


def _fallback_questions(num_questions: int, rng: random.Random) -> list[str]:
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


def _is_resource_exhausted_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota exceeded" in text or "error code 429" in text


def _is_hard_quota_exhausted(exc: Exception) -> bool:
    text = str(exc).lower()
    return "generaterequestsperday" in text or "limit: 0" in text


def _is_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "api key expired" in text
        or "invalid api key" in text
        or "incorrect api key" in text
        or "authentication" in text
        or "unauthorized" in text
        or "401" in text
    )


def _retry_delay_seconds(exc: Exception) -> float | None:
    text = str(exc)

    m = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.search(r"'retryDelay': '([0-9]+)s'", text)
    if m:
        return float(m.group(1))

    return None


def _generate_model_text(prompt: str) -> str:
    if _env_bool("BRAINTRUST_USE_GATEWAY", False):
        client = make_wrapped_openai_client()
        response = client.chat.completions.create(
            model=QUESTION_GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY in environment. To run without a Google API key, "
            "set BRAINTRUST_USE_GATEWAY=true and provide BRAINTRUST_API_KEY or "
            "BRAINTRUST_GATEWAY_API_KEY."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=QUESTION_GENERATOR_MODEL,
        contents=prompt,
    )
    return response.text or ""


def generate_questions(num_questions: int, seed: Optional[int] = None) -> list[str]:
    """Generate realistic, varied questions with Gemini."""
    rng = random.Random(seed)

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
    try:
        text = _generate_model_text(prompt).strip()
        questions = _extract_json_array(text)
        rng.shuffle(questions)
        return questions[:num_questions]
    except Exception:
        return _fallback_questions(num_questions=num_questions, rng=rng)


def _quota_preflight_ok() -> tuple[bool, str]:
    return True, ""


def _preflight_failure_category(exc: Exception) -> str:
    if _is_auth_error(exc):
        return "authentication"
    if _is_hard_quota_exhausted(exc):
        return "quota"
    if _is_resource_exhausted_error(exc):
        return "transient"
    return "provider"


def _run_preflight() -> dict[str, str]:
    missing = [name for name in ("BRAINTRUST_API_KEY", "EXA_API_KEY") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    from src.agents.research_agent import _search_exa

    for attempt in range(1, 4):
        try:
            _generate_model_text("Reply with exactly: OK")
            _search_exa(query="Braintrust", max_results=1)
            return {"model": "ok", "exa": "ok"}
        except Exception as exc:
            category = _preflight_failure_category(exc)
            if category == "transient" and attempt < 3:
                time.sleep(2**attempt)
                continue
            # Do not include provider exceptions: some SDKs echo request headers in errors.
            raise RuntimeError(f"Provider preflight failed ({category}).") from exc

    raise RuntimeError("Provider preflight failed (transient).")


def _write_summary(
    path: str | None,
    *,
    preflight: dict[str, str],
    total: int,
    successes: int,
    failures: int,
) -> None:
    if not path:
        return
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "preflight": preflight,
                "total": total,
                "successes": successes,
                "failures": failures,
            },
            indent=2,
        )
        + "\n"
    )


async def run_question(
    question: str,
    *,
    max_retries: int,
    base_retry_seconds: float,
) -> tuple[str, bool, bool]:
    """Run one question through the supervisor with a random model assignment."""
    from src.agent_graph import get_supervisor

    selected_model = random.choice(MODEL_POOL)
    config = AgentConfig.from_env(
        supervisor_model=selected_model,
        research_model=selected_model,
        math_model=selected_model,
    )
    supervisor = get_supervisor(config=config, force_rebuild=True)

    attempt = 0
    while True:
        attempt += 1
        try:
            result = await run_supervisor_with_critic(
                supervisor=supervisor,
                query=question,
                app_name="google-adk-supervisor-batch",
            )
            print(f"✅ {question[:80]} -> {str(result.get('final_output', ''))[:80]}")
            return question, True, False
        except Exception as exc:
            if not _is_resource_exhausted_error(exc):
                print(f"❌ {question[:80]} -> {exc}")
                return question, False, False

            if _is_hard_quota_exhausted(exc):
                print(f"⏹️ {question[:80]} -> hard quota exhausted ({exc})")
                return question, False, True

            if attempt > max_retries:
                print(f"❌ {question[:80]} -> exhausted retries ({exc})")
                return question, False, False

            suggested = _retry_delay_seconds(exc)
            backoff = base_retry_seconds * (2 ** (attempt - 1))
            sleep_s = max(suggested or 0.0, backoff)
            print(f"⏳ {question[:80]} -> retrying in {sleep_s:.1f}s after quota error")
            await asyncio.sleep(sleep_s)


async def main_async(args: argparse.Namespace) -> None:
    preflight = {} if args.skip_preflight else _run_preflight()
    if args.preflight_only:
        _write_summary(
            args.summary_path,
            preflight=preflight,
            total=0,
            successes=0,
            failures=0,
        )
        print("Provider preflight passed.")
        return

    num_questions = args.num_questions if args.num_questions is not None else random.randint(1, 100)
    rng = random.Random(args.seed)
    questions = (
        _fallback_questions(num_questions=num_questions, rng=rng)
        if args.question_source == "bank"
        else generate_questions(num_questions=num_questions, seed=args.seed)
    )

    print(f"Generated {len(questions)} questions")
    print(f"Running with concurrency={args.concurrency}")
    print(f"Model pool: {', '.join(MODEL_POOL)}")
    print(f"Question source: {args.question_source}")
    print("=" * 80)

    successes = 0
    failures = 0
    hard_quota_stop = False

    for i in range(0, len(questions), args.concurrency):
        if hard_quota_stop:
            break
        batch = questions[i : i + args.concurrency]
        results = await asyncio.gather(
            *(
                run_question(
                    q,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                )
                for q in batch
            )
        )
        for _, ok, hard_stop in results:
            if ok:
                successes += 1
            else:
                failures += 1
            if hard_stop:
                hard_quota_stop = True
        if hard_quota_stop:
            print("Hard quota exhausted; stopping remaining questions to avoid repeated 429s.")
            break
        if args.inter_question_delay_seconds > 0:
            await asyncio.sleep(args.inter_question_delay_seconds)
        print()

    print("=" * 80)
    print(f"Completed. successes={successes} failures={failures}")
    print("=" * 80)
    _write_summary(
        args.summary_path,
        preflight=preflight,
        total=len(questions),
        successes=successes,
        failures=failures,
    )

    if args.fail_on_error and failures > 0:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random questions and run through supervisor locally"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "1")),
        help="Number of concurrent questions to process (default: 1)",
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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("MAX_RETRIES", "3")),
        help="Max retries for transient quota errors (default: 3)",
    )
    parser.add_argument(
        "--base-retry-seconds",
        type=float,
        default=float(os.environ.get("BASE_RETRY_SECONDS", "15")),
        help="Base retry delay used for exponential backoff (default: 15)",
    )
    parser.add_argument(
        "--inter-question-delay-seconds",
        type=float,
        default=float(os.environ.get("INTER_QUESTION_DELAY_SECONDS", "2")),
        help="Delay between processed batches to reduce burst rate (default: 2)",
    )
    parser.add_argument(
        "--quota-preflight",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("QUOTA_PREFLIGHT", "1") != "0",
        help="Run a lightweight Gemini call before batch and skip run if daily quota is exhausted",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify the configured model and Exa adapter without running questions",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip provider preflight after a separate successful preflight step",
    )
    parser.add_argument(
        "--question-source",
        choices=("generated", "bank"),
        default=os.environ.get("QUESTION_SOURCE", "generated"),
        help="Question source: generated or deterministic bank",
    )
    parser.add_argument(
        "--summary-path",
        default=os.environ.get("QUERY_SUMMARY_PATH", ""),
        help="Optional path for a JSON query result summary artifact",
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
