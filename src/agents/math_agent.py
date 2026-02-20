"""Math agent with arithmetic capabilities."""

from typing import Any, Callable

from google.adk import Agent

from src.config import DEFAULT_MATH_AGENT_PROMPT


def add(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b and return the quotient."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


def get_math_agent(
    system_prompt: str | None = None,
    model: str = "gemini-2.0-flash-lite",
    extra_tools: list[Callable[..., Any]] | None = None,
) -> Agent:
    """Create the math agent with optional custom prompt and model."""
    prompt = system_prompt if system_prompt is not None else DEFAULT_MATH_AGENT_PROMPT

    tools: list[Callable[..., Any]] = [add, subtract, multiply, divide]
    if extra_tools:
        tools.extend(extra_tools)

    return Agent(
        name="MathAgent",
        model=model,
        instruction=prompt,
        tools=tools,
    )
