"""Configuration module initialization.

Exports settings and prompts:
    from auto_ds_agent.config import get_settings, AGENT_PROMPTS
"""

from auto_ds_agent.config.settings import Settings, get_settings
from auto_ds_agent.config.prompts import AGENT_PROMPTS

__all__ = ["Settings", "get_settings", "AGENT_PROMPTS"]
