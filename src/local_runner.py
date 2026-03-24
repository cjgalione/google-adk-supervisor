"""Local CLI runner for the Google ADK supervisor."""

import asyncio
import getpass
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.agent_graph import get_supervisor, run_supervisor_with_critic
from src.cache import SemanticCache, make_cache_from_env
from src.tracing import configure_adk_tracing

DEFAULT_BRAINTRUST_PROJECT = "google-adk-supervisor"


def _set_if_undefined(var: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


def _is_gateway_enabled() -> bool:
    return os.environ.get("BRAINTRUST_USE_GATEWAY", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def _run_chat() -> None:
    load_dotenv()
    if _is_gateway_enabled():
        if not (
            os.environ.get("BRAINTRUST_GATEWAY_API_KEY")
            or os.environ.get("BRAINTRUST_API_KEY")
        ):
            _set_if_undefined("BRAINTRUST_GATEWAY_API_KEY")
    else:
        _set_if_undefined("GOOGLE_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")

    if os.environ.get("BRAINTRUST_API_KEY"):
        configure_adk_tracing(
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
            project_name=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
        )

    console = Console()
    supervisor = get_supervisor()
    cache: SemanticCache | None = make_cache_from_env()

    welcome_text = Text("Google ADK Supervisor Chat", style="bold cyan")
    cache_note = " · cache ON" if cache is not None else ""
    welcome_panel = Panel(
        welcome_text,
        subtitle=f"Type 'quit' or 'q' to exit{cache_note}",
        border_style="cyan",
    )
    console.print(welcome_panel)
    console.print()

    while True:
        user_input = Prompt.ask("[bold green]You[/bold green]", console=console)

        if user_input.lower() in {"q", "quit", "exit"}:
            if cache is not None:
                stats = cache.stats()
                console.print(
                    f"\n[dim]Cache stats — hits: {stats['hits']}, "
                    f"misses: {stats['misses']}, "
                    f"hit rate: {stats['hit_rate']:.1%}, "
                    f"entries: {stats['entries']}[/dim]"
                )
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not user_input.strip():
            continue

        with console.status("[bold blue]Processing...", spinner="dots"):
            run_result = await run_supervisor_with_critic(
                supervisor=supervisor,
                query=user_input,
                app_name="google-adk-supervisor-local",
                cache=cache,
            )

        final_output = run_result.get("final_output", "")
        cache_hit = run_result.get("cache_hit", False)
        title = "[dim]Assistant (cached)[/dim]" if cache_hit else "Assistant"
        console.print(
            Panel(
                str(final_output) if final_output else "(No response generated)",
                title=title,
                border_style="blue" if not cache_hit else "dim",
            )
        )
        console.print()


def main() -> None:
    asyncio.run(_run_chat())


if __name__ == "__main__":
    main()
