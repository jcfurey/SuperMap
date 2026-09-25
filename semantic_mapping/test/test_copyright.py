# Copyright 2025 SuperX SLAM / AirLab, Carnegie Mellon University
# Copyright 2026 SuperMap fork contributors
#
# Licensed under the MIT License; see the LICENSE file at the repository root.
"""ament_copyright over the package.

The sources inherited from upstream carry no per-file copyright header; the
repository-level LICENSE covers them. Skipped (as ament_python templates do
for generated sources) until headers are added file by file.
"""
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.skip(reason='Upstream sources have no per-file copyright headers; see LICENSE.')
@pytest.mark.copyright
@pytest.mark.linter
def test_copyright(monkeypatch):
    from ament_copyright.main import main

    monkeypatch.chdir(PACKAGE_DIR)
    rc = main(argv=['.', 'test'])
    assert rc == 0, 'Found errors'
