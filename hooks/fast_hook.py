#!/usr/bin/env python3
"""Shim — the canonical fast hook client lives in src/cairn/hooks/fast_hook.py.

This file exists only so settings that reference the repo-root path keep
working. Running the canonical file in place (runpy, not import) makes its
_fallback() resolve hook scripts relative to src/cairn/hooks/, which is the
complete script set (including pre_add_guard.py, absent from this directory).
"""
import os
import runpy
import sys

_CANON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "cairn", "hooks", "fast_hook.py",
)
sys.argv[0] = _CANON
runpy.run_path(_CANON, run_name="__main__")
