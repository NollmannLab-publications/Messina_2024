#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Removes the previous outputs (cleanup_figures.py), then runs every Figure_*.py script
(main figures first, then supplementary figures in numerical order).
"""

from pathlib import Path
import re
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent


def figure_order(path):
    """Figure_1 ... Figure_6, then Figure_S1 ... Figure_S16."""
    match = re.match(r"Figure_(S?)(\d+)$", path.stem)
    return (match.group(1) == "S", int(match.group(2))) if match else (True, 999)


def run(script_path):
    """Run one script in its own process, printing its output live. Returns True if it succeeded."""
    result = subprocess.run([sys.executable, str(script_path)], cwd=SCRIPT_DIR)
    return result.returncode == 0


start = time.time()

# 1. Cleanup
print("=" * 60)
print("Cleaning previous outputs")
print("=" * 60)
if not run(SCRIPT_DIR / "cleanup_figures.py"):
    sys.exit("✗ Cleanup failed, figures not generated.")

# 2. All figures
figure_scripts = sorted(SCRIPT_DIR.glob("Figure_*.py"), key=figure_order)
print(f"\nFound {len(figure_scripts)} figure scripts.\n")

failed = []
for script_path in figure_scripts:
    print("=" * 60)
    print(f"▶ {script_path.name}")
    print("=" * 60)
    if run(script_path):
        print(f"✓ {script_path.name} completed\n")
    else:
        print(f"✗ {script_path.name} failed\n")
        failed.append(script_path.name)

# 3. Summary
minutes = (time.time() - start) / 60
print("=" * 60)
if failed:
    print(f"⚠ {len(failed)} script(s) failed: {', '.join(failed)} ({minutes:.1f} min)")
    sys.exit(1)
print(f"🎉 All figures generated in {minutes:.1f} min")
