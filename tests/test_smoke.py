from nanoclaw import __version__
from nanoclaw.__main__ import main


def test_version_flag(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"nanoclaw {__version__}"
