"""Math agent with arithmetic capabilities."""

from typing import Any, Callable

from google.adk import Agent
from pint import UnitRegistry

from src.config import DEFAULT_MATH_AGENT_PROMPT

_UREG = UnitRegistry()
_UNIT_ALIASES = {
    "j": "joule",
    "joules": "joule",
    "hp": "horsepower",
    "horsepower-seconds": "horsepower * second",
    "horsepower seconds": "horsepower * second",
    "horsepower-seconds(s)": "horsepower * second",
    "horsepower-hours": "horsepower * hour",
    "horsepower hours": "horsepower * hour",
    "hp-s": "horsepower * second",
    "hp*s": "horsepower * second",
    "hp-hr": "horsepower * hour",
    "hp*h": "horsepower * hour",
}


def _normalize_unit(unit: str) -> str:
    lowered = unit.strip().lower()
    return _UNIT_ALIASES.get(lowered, lowered)


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


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert numeric values between compatible units."""
    source = _normalize_unit(from_unit)
    target = _normalize_unit(to_unit)
    quantity = float(value) * _UREG.parse_units(source)
    converted = quantity.to(_UREG.parse_units(target))
    return float(converted.magnitude)


def get_math_agent(
    system_prompt: str | None = None,
    model: str = "gemini-2.0-flash-lite",
    extra_tools: list[Callable[..., Any]] | None = None,
) -> Agent:
    """Create the math agent with optional custom prompt and model."""
    prompt = system_prompt if system_prompt is not None else DEFAULT_MATH_AGENT_PROMPT

    tools: list[Callable[..., Any]] = [add, subtract, multiply, divide, convert_units]
    if extra_tools:
        tools.extend(extra_tools)

    return Agent(
        name="MathAgent",
        model=model,
        instruction=prompt,
        tools=tools,
    )
