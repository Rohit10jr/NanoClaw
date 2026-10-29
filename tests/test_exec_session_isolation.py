"""Adapted from nanobot tests/agent/test_exec_session_isolation.py (c4dac64).

nanobot checks that two AgentLoops each own an ExecSessionManager shared by
their exec, exec_session and list_exec_sessions tools. That loop-level test
is in test_exec_session_wiring.py (task 16.3); here the same property is
checked one level down, on ``build_default_tools``.
"""

from nanoclaw.agent.tools import build_default_tools
from nanoclaw.agent.tools.exec_session import DEFAULT_EXEC_SESSION_MANAGER, ExecSessionManager


def test_tool_sets_do_not_share_exec_session_managers(tmp_path):
    manager_a = ExecSessionManager()
    manager_b = ExecSessionManager()
    tools_a = build_default_tools(tmp_path / "a", exec_session_manager=manager_a)
    tools_b = build_default_tools(tmp_path / "b", exec_session_manager=manager_b)

    assert tools_a.get("exec")._session_manager is manager_a
    assert tools_a.get("exec_session")._manager is manager_a
    assert tools_a.get("list_exec_sessions")._manager is manager_a
    assert tools_b.get("exec")._session_manager is manager_b
    assert manager_a is not manager_b


def test_default_tool_set_uses_the_default_manager(tmp_path):
    tools = build_default_tools(tmp_path)
    assert tools.get("exec")._session_manager is DEFAULT_EXEC_SESSION_MANAGER
    assert tools.get("exec_session")._manager is DEFAULT_EXEC_SESSION_MANAGER
    assert tools.get("list_exec_sessions")._manager is DEFAULT_EXEC_SESSION_MANAGER
