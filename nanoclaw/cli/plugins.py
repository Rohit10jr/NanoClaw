"""``nanoclaw plugins list | enable NAME | disable NAME``: Agent Plugins.

nanobot enables Agent Plugins only from its web UI Apps page
(webui/mcp_presets_api.py ``mcp_presets_settings_action`` → ``plugin-<name>``
rows). NanoClaw has no Apps page, so the same ``set_agent_plugin_enabled``
call is made from the command line. ``enable`` shows what the package
brings (its fingerprint, declared permissions, skills and MCP servers, with
the commands they run) and asks before enabling unless ``--yes`` is given.
The fingerprint shown is the one enabled: if the package changes between
the review and the write, the plugin is disabled again and the command fails.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from nanoclaw.agent import plugins as agent_plugins
from nanoclaw.agent.plugins import AgentPlugin, discover_agent_plugins, set_agent_plugin_enabled
from nanoclaw.agent.skills import parse_skill_metadata, valid_skill_metadata

__all__ = ["add_plugins_parser", "plugins_command"]


def add_plugins_parser(
    commands: argparse._SubParsersAction, common: argparse.ArgumentParser, workspace_help: str
) -> None:
    plugins = commands.add_parser(
        "plugins", help="list, enable or disable the Agent Plugins in <workspace>/plugins (no API key needed)"
    )
    actions = plugins.add_subparsers(dest="plugins_action", required=True)
    listing = actions.add_parser("list", parents=[common], help="list installed Agent Plugins and whether they are enabled")
    listing.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    enable = actions.add_parser(
        "enable", parents=[common], help="review an Agent Plugin and enable its skills and MCP servers"
    )
    enable.add_argument("name", help="the plugin's name (plugin.json \"name\")")
    enable.add_argument("-y", "--yes", action="store_true", help="enable without asking for confirmation")
    enable.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)
    disable = actions.add_parser("disable", parents=[common], help="disable an Agent Plugin")
    disable.add_argument("name", help="the plugin's name")
    disable.add_argument("-w", "--workspace", type=Path, default=None, help=workspace_help)


def plugins_command(args: argparse.Namespace, workspace: Path, *, ask: Callable[[str], str] | None = None) -> int:
    if args.plugins_action == "list":
        return _list(workspace)
    if args.plugins_action == "enable":
        return _enable(workspace, args.name, yes=args.yes, ask=ask or input)
    return _disable(workspace, args.name)


def _find(workspace: Path, name: str) -> AgentPlugin | None:
    plugin = next((item for item in discover_agent_plugins(workspace) if item.name == name), None)
    if plugin is None:
        print(
            f"Unknown Agent Plugin '{name}' (none with a valid plugin.json in {workspace / 'plugins'}; "
            "see nanoclaw plugins list).",
            file=sys.stderr,
        )
    return plugin


def _skill_names(plugin: AgentPlugin) -> tuple[list[str], list[str]]:
    """``(valid, ignored)`` skill folder names, read from the package as it is now."""
    valid: list[str] = []
    ignored: list[str] = []
    for name, path in agent_plugins._plugin_skill_files(plugin.name, plugin.root):
        try:
            content = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except (OSError, UnicodeError):
            content = ""
        metadata = parse_skill_metadata(content)
        (valid if metadata is not None and valid_skill_metadata(metadata, name) else ignored).append(name)
    return valid, ignored


def _list(workspace: Path) -> int:
    plugins = discover_agent_plugins(workspace)
    if not plugins:
        print(f"No Agent Plugins in {workspace / 'plugins'}.")
        return 0
    lines = [f"Agent Plugins in {workspace / 'plugins'} ({len(plugins)}):", ""]
    for plugin in plugins:
        state = "enabled" if plugin.enabled else "disabled"
        title = f"{plugin.display_name} " if plugin.display_name != plugin.name else ""
        lines.append(f"- **{plugin.name}** {title}[{state}] — {plugin.description or 'Agent Plugin'}")
        skills, _ignored = _skill_names(plugin)
        if skills:
            lines.append(f"  skills: {', '.join(skills)}")
        if plugin.mcp_servers:
            lines.append(f"  MCP servers: {', '.join(plugin.mcp_servers)}")
    print("\n".join(lines))
    return 0


def _review(workspace: Path, plugin: AgentPlugin, fingerprint: str) -> str:
    lines = [
        f"Agent Plugin: {plugin.name}" + (f" ({plugin.display_name})" if plugin.display_name != plugin.name else ""),
        f"  folder:      {plugin.root}",
        f"  fingerprint: sha256:{fingerprint}",
    ]
    if plugin.description:
        lines.append(f"  description: {plugin.description}")
    if plugin.repository:
        lines.append(f"  repository:  {plugin.repository}")
    lines.append(f"  permissions: {', '.join(plugin.permissions) if plugin.permissions else '(none declared)'}")
    skills, ignored = _skill_names(plugin)
    lines.append(f"  skills:      {', '.join(skills) if skills else '(none)'}")
    if ignored:
        lines.append(f"  ignored skills (invalid SKILL.md): {', '.join(ignored)}")
    servers = agent_plugins._plugin_mcp_servers(workspace, plugin)
    if servers:
        lines.append("  MCP servers (run as local programs with your permissions):")
        for name, server in sorted(servers.items()):
            command = " ".join([server.command, *server.args])
            lines.append(f"    {name}: {command}  (cwd {server.cwd})")
    else:
        lines.append("  MCP servers: (none)")
    lines.append(
        "Enabling lets the agent read these skills and start these servers. "
        "Any change to the package disables it again."
    )
    return "\n".join(lines)


def _enable(workspace: Path, name: str, *, yes: bool, ask: Callable[[str], str]) -> int:
    plugin = _find(workspace, name)
    if plugin is None:
        return 1
    fingerprint = agent_plugins._package_fingerprint(plugin.root)
    if fingerprint is None:
        print(f"Agent Plugin '{name}' cannot be read (unreadable or special files).", file=sys.stderr)
        return 1
    print(_review(workspace, plugin, fingerprint))
    if plugin.enabled:
        print(f"{name} is already enabled.")
        return 0
    if not yes:
        try:
            answer = ask(f"Enable {name}? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Not enabled.")
            return 1
    try:
        set_agent_plugin_enabled(workspace, name, True)
    except (ValueError, RuntimeError) as exc:
        print(f"Could not enable {name}: {exc}", file=sys.stderr)
        return 1
    # Bind the approval to the package that was reviewed above.
    if agent_plugins._package_fingerprint(plugin.root) != fingerprint:
        set_agent_plugin_enabled(workspace, name, False)
        print(f"Could not enable {name}: the package changed while it was being reviewed.", file=sys.stderr)
        return 1
    print(f"{name} enabled. Its skills are offered from the next turn; its MCP servers start "
          "with the next chat, ask or gateway.")
    return 0


def _disable(workspace: Path, name: str) -> int:
    if _find(workspace, name) is None:
        return 1
    try:
        set_agent_plugin_enabled(workspace, name, False)
    except (ValueError, RuntimeError) as exc:
        print(f"Could not disable {name}: {exc}", file=sys.stderr)
        return 1
    print(f"{name} disabled.")
    return 0
