#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thin wrapper for sensitivity plots: delegate to augment.plots.sens_lines as the single source of truth.
- Keeps outputs under augment/results/real/sens/figs_pub/{elec,amp}
- Adds sensible defaults for targets and annotations based on metric
- Prints a deprecation notice to encourage using augment/plots/sens_lines.py directly
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path


def _import_core():
    """Import the unified plotting entry from augment.plots.sens_lines as `core`."""
    # Ensure project root is on sys.path so that `import augment.plots.sens_lines` works
    here = Path(__file__).resolve()
    project_root = here.parents[4]  # c:/WSN-Intel-Lab-Project
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from augment.plots import sens_lines as core  # type: ignore
        return core
    except Exception:
        # Fallback: add augment/plots to sys.path and import as a module
        aug_plots_dir = here.parents[3] / "plots"  # augment/plots
        if str(aug_plots_dir) not in sys.path:
            sys.path.insert(0, str(aug_plots_dir))
        import sens_lines as core  # type: ignore
        return core


def _has_flag(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def _get_value(flag: str) -> str | None:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    for arg in sys.argv[1:]:
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def main() -> None:
    warnings.warn(
        "This script is a thin wrapper. Please use augment/plots/sens_lines.py directly for future use.",
        RuntimeWarning,
        stacklevel=2,
    )

    core = _import_core()

    # Build additional default args if not provided by user
    extra: list[str] = []

    # Default output directories mapped to sens/figs_pub for backward compatibility
    if not _has_flag("--outdir-elec"):
        extra += ["--outdir-elec", "augment/results/real/sens/figs_pub/elec"]
    if not _has_flag("--outdir-amp"):
        extra += ["--outdir-amp", "augment/results/real/sens/figs_pub/amp"]

    # Default export formats
    if not _has_flag("--formats"):
        extra += ["--formats", "svg,png"]

    # Prefer to show annotations unless explicitly disabled at the core level
    if not _has_flag("--annotate-crossing"):
        extra += ["--annotate-crossing"]

    # Enable QC by default with strict fail, unless explicitly overridden
    if not _has_flag("--qc"):
        extra += ["--qc"]
    if not _has_flag("--qc-strict"):
        extra += ["--qc-strict"]

    # Target line defaults depend on metric; only set if user does not supply one
    if not _has_flag("--target-y"):
        metric = _get_value("--metric") or "pdr"
        defaults = {"pdr": "0.95", "ctrl_percent": "20", "delay_s": "0.5"}
        if metric in defaults:
            extra += ["--target-y", defaults[metric]]

    # Execute the unified core CLI with augmented argv
    orig_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + extra + sys.argv[1:]
        core.main()
    finally:
        sys.argv = orig_argv


if __name__ == "__main__":
    main()