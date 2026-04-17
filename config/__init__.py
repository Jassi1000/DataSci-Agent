"""Configuration module initialization.

Exports settings and prompts:
    from config import get_settings, AGENT_PROMPTS
"""

from .settings import Settings, get_settings
from .prompts import AGENT_PROMPTS

__all__ = ["Settings", "get_settings", "AGENT_PROMPTS"]
