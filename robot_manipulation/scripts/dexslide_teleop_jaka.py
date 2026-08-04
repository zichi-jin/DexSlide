#!/usr/bin/env python3
"""Backward-compatible alias for the JAKA incremental teleop entry point."""

try:
    from .jaka_dexslide_incremental_teleop import main
except ImportError:
    from jaka_dexslide_incremental_teleop import main


if __name__ == "__main__":
    raise SystemExit(main())
