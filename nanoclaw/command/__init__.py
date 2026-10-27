"""Slash commands, independent of any channel (nanobot/command/)."""

from nanoclaw.command.builtin import BUILTIN_COMMAND_SPECS, BuiltinCommandSpec, build_help_text, register_builtin_commands
from nanoclaw.command.router import CommandContext, CommandRouter, command_text, normalize_command_text

__all__ = [
    "BUILTIN_COMMAND_SPECS",
    "BuiltinCommandSpec",
    "CommandContext",
    "CommandRouter",
    "build_help_text",
    "command_text",
    "normalize_command_text",
    "register_builtin_commands",
]
