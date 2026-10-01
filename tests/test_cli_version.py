import pytest

from nanoclaw import __version__
from nanoclaw.__main__ import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"nanoclaw {__version__}"
