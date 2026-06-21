"""Research agent with web search capabilities."""

import os
from typing import Any, Callable

from braintrust import SpanTypeAttribute, start_span
from google.adk import Agent
from tavily import TavilyClient

from src.config import DEFAULT_RESEARCH_AGENT_PROMPT


def _get_tavily_client() -> TavilyClient:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=api_key)


def _get_exa_client() -> Any:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set")

    from exa_py import Exa

    return Exa(api_key=api_key)


def _build_tavily_output(response: dict[str, Any]) -> str:
    lines: list[str] = []
    answer = response.get("answer")
    if answer:
        lines.append(f"Answer: {answer}")

    results = response.get("results", []) or []
    if not results:
        if lines:
            return "\n\n".join(lines)
        return "No search results found."

    for i, item in enumerate(results, start=1):
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip()
        block = (
            f"{i}. {title or 'Untitled'}\n"
            f"URL: {url or 'N/A'}\n"
            f"Summary: {content or 'N/A'}"
        )
        lines.append(block)
    return "\n\n".join(lines)


def _result_value(result: Any, name: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _build_exa_output(response: Any) -> str:
    lines: list[str] = []
    results = _result_value(response, "results", []) or []

    if not results:
        return "No search results found."

    for i, result in enumerate(results, start=1):
        title = str(_result_value(result, "title", "") or "").strip()
        url = str(_result_value(result, "url", "") or "").strip()
        highlights = _result_value(result, "highlights", []) or []
        summary = str(_result_value(result, "summary", "") or "").strip()
        text = str(_result_value(result, "text", "") or "").strip()

        if isinstance(highlights, str):
            content = highlights.strip()
        else:
            content = " ".join(str(item).strip() for item in highlights if str(item).strip())

        if not content:
            content = summary or text[:800]

        block = (
            f"{i}. {title or 'Untitled'}\n"
            f"URL: {url or 'N/A'}\n"
            f"Summary: {content or 'N/A'}"
        )
        lines.append(block)
    return "\n\n".join(lines)


def _search_tavily(query: str, max_results: int) -> str:
    response = _get_tavily_client().search(
        query=query,
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
    )
    return _build_tavily_output(response)


def _search_exa(query: str, max_results: int) -> str:
    with start_span(
        name="exa_search",
        type=SpanTypeAttribute.TOOL,
        input={"query": query, "max_results": max_results},
        metadata={"provider": "exa", "search_type": "auto", "contents": "highlights"},
    ) as tool_span:
        response = _get_exa_client().search(
            query,
            type="auto",
            num_results=max_results,
            contents={"highlights": True},
        )
        output = _build_exa_output(response)
        tool_span.log(output=output)
        return output


def tavily_search(query: str, max_results: int = 3) -> str:
    """Search the web and return summarized results with links."""
    limited_max_results = max(1, min(max_results, 5))

    with start_span(
        name="tavily_search",
        type=SpanTypeAttribute.TOOL,
        input={"query": query, "max_results": limited_max_results},
        metadata={"provider": "tavily"},
    ) as tool_span:
        try:
            output = _search_tavily(query=query, max_results=limited_max_results)
            tool_span.log(output=output)
            return output
        except Exception as tavily_error:
            if not os.environ.get("EXA_API_KEY"):
                raise

            output = _search_exa(query=query, max_results=limited_max_results)
            tool_span.log(
                output=output,
                metadata={
                    "fallback_provider": "exa",
                    "fallback_reason": str(tavily_error),
                },
            )
            return output


def get_research_agent(
    system_prompt: str | None = None,
    model: Any = "gemini-2.0-flash-lite",
    extra_tools: list[Callable[..., Any]] | None = None,
) -> Agent:
    """Create the research agent with optional custom prompt and model."""
    prompt = system_prompt if system_prompt is not None else DEFAULT_RESEARCH_AGENT_PROMPT

    tools: list[Callable[..., Any]] = [tavily_search]
    if extra_tools:
        tools.extend(extra_tools)

    return Agent(
        name="ResearchAgent",
        model=model,
        instruction=prompt,
        tools=tools,
    )
