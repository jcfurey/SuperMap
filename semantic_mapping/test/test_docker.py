"""Check the full-image dependency failure path without downloading packages."""
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('fail_call', [1, 2, 3])
def test_detector_install_failures_abort_the_build(fail_call):
    completed = _run_install_block(fail_call=fail_call)
    assert completed.returncode == 42
    assert 'python check' not in completed.stdout


@pytest.mark.parametrize('detectors', [0, 1])
def test_successful_detector_and_lite_install_paths(detectors):
    completed = _run_install_block(detectors=detectors)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count('pip call') == (3 if detectors else 0)
    assert 'import cv2, numpy' in completed.stdout
    assert ('import torch, torchvision, ultralytics' in completed.stdout) == bool(detectors)


def _run_install_block(fail_call=0, detectors=1):
    dockerfile = Path(__file__).resolve().parents[2] / 'docker' / 'Dockerfile'
    lines = dockerfile.read_text().replace('\\\n', ' ').splitlines()
    block, = [line.removeprefix('RUN ') for line in lines if line.startswith('RUN ') and 'INSTALL_DETECTORS' in line]
    stub = f"""
INSTALL_DETECTORS={detectors}
review_pip_calls=0
pip3() {{
    review_pip_calls=$((review_pip_calls + 1))
    echo "pip call $review_pip_calls"
    if [ "$review_pip_calls" = "{fail_call}" ]; then return 42; fi
}}
python3() {{ echo "python check $*"; }}
"""
    return subprocess.run(['bash', '-o', 'pipefail', '-c', stub + block], capture_output=True, text=True)
