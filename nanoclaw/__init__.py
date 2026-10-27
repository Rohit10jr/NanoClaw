"""NanoClaw: a minimal personal AI agent."""

from loguru import logger

__version__ = "0.1.0"

# Library use and tests stay quiet: nothing under ``nanoclaw`` logs until a CLI
# command turns logs on with ``nanoclaw.cli.log_control.set_nanoclaw_logs``.
# (nanobot leaves its logger enabled and disables it per command instead.)
logger.disable("nanoclaw")
