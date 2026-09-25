# Copyright 2025 SuperX SLAM / AirLab, Carnegie Mellon University
# Copyright 2026 SuperMap fork contributors
#
# Licensed under the MIT License; see the LICENSE file at the repository root.
"""ament_pep257 over the package.

The code base writes a summary sentence that may wrap onto a second line,
and multi-line docstrings start on the first line, so the corresponding
formatting checks are added to the ament convention's ignore list.
"""
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]

STYLE_EXCEPTIONS = ['D204', 'D205', 'D209', 'D213', 'D301', 'D400', 'D401', 'D403', 'D413', 'D415', 'D417']


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257(monkeypatch):
    from ament_pep257.main import main

    monkeypatch.chdir(PACKAGE_DIR)
    argv = ['--add-ignore', *STYLE_EXCEPTIONS, '--', 'semantic_mapping', 'launch', 'test', 'examples']
    rc = main(argv=argv)
    assert rc == 0, 'Found code style errors / warnings'
