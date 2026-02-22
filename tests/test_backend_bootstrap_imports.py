from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_backend_package_bootstrap_allows_root_import() -> None:
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        f"sys.path.insert(0, '{str(root)}'); "
        "import backend.v2_router; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
