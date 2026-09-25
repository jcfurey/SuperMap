# Copyright 2025 SuperX SLAM / AirLab, Carnegie Mellon University
# Copyright 2026 SuperMap fork contributors
#
# Licensed under the MIT License; see the LICENSE file at the repository root.
"""ament_flake8 over the package, with the style exceptions in setup.cfg [flake8]."""
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8(monkeypatch):
    from ament_flake8.main import main_with_errors

    monkeypatch.chdir(PACKAGE_DIR)
    rc, errors = main_with_errors(argv=['--config', str(PACKAGE_DIR / 'setup.cfg')])
    assert rc == 0, 'Found %d code style errors / warnings:\n' % len(errors) + '\n'.join(errors)
