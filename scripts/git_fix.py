#!/usr/bin/env python3
"""Run pylint from the repository root (used by `git fix`).

Configure once per clone:
    git config alias.fix '!python "$(git rev-parse --show-toplevel)/scripts/git_fix.py"'
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "pylint",
        "--recursive=y",
        str(root),
    ]
    return subprocess.call(cmd, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
