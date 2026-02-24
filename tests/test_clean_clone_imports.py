from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_clean_clone_imports_without_artifact_python_sources(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "import training.train_liquid\n"
        "import backend.v2_repository\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"

