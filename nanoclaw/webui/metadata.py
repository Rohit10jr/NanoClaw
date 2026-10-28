"""Shared web UI metadata keys. (nanobot webui/metadata.py)"""

from nanoclaw.bus.events import INBOUND_META_TURN_ID

WEBUI_TURN_METADATA_KEY = INBOUND_META_TURN_ID
WEBUI_SYSTEM_COMMAND_TURN_PREFIX = "webui-system:"
# Set by authenticated web UI ingress, never copied from an envelope's metadata.
WEBUI_RESTART_CONTROL_METADATA_KEY = "_webui_restart_control"
WEBSOCKET_TURN_OWNER_METADATA_KEY = "_websocket_turn_owner"
WEBUI_MESSAGE_SOURCE_METADATA_KEY = "_webui_message_source"
# A shell command typed by the user (nanobot bus/events.py:20): its output is
# shown as literal text.
INBOUND_META_USER_SHELL = "_user_shell"
