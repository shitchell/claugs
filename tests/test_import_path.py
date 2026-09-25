"""Guard: the suite must test the working tree, not an installed claugs.

`[tool.pytest.ini_options] pythonpath` used to list only "tests". With an
installed claugs on the path, `python -m pytest` then imported
site-packages instead of ``src/`` -- the whole suite silently validated
the released wheel, and a genuine source fix looked like a test failure.
"""

from pathlib import Path

import pytest

import claude_logs

REPO_SRC = Path(__file__).resolve().parent.parent / "src"


def test_claude_logs_is_imported_from_the_working_tree():
    if not (REPO_SRC / "claude_logs").is_dir():
        pytest.skip("not running from a source checkout")

    imported = Path(claude_logs.__file__).resolve()
    assert imported.is_relative_to(REPO_SRC.resolve()), (
        f"claude_logs was imported from {imported}, not {REPO_SRC}. "
        "Check `pythonpath` in [tool.pytest.ini_options] -- 'src' must "
        "come before any installed copy."
    )
