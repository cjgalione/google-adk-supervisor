"""Compatibility module that exports the ADK-backed supervisor.

This keeps the original import path stable for scripts and evals.
"""

# Import the new deep agent implementation
from src.agents.deep_agent import get_supervisor

# Export for backward compatibility
__all__ = ["get_supervisor"]
