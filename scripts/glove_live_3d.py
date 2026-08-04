#!/usr/bin/env python3
"""Compatibility wrapper for ``main.py stream --show-plot3d``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from main import main


def compatibility_main() -> None:
    sys.argv = [sys.argv[0], "stream", "--show-plot3d", *sys.argv[1:]]
    main()


if __name__ == "__main__":
    compatibility_main()
