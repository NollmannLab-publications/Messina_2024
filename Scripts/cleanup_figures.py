#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Removes all generated panels (everything inside Figures/). Source data are not touched.
"""

from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "Figures"

print("🧹 Removing generated figures...")

FIGURES_DIR.mkdir(exist_ok=True)
for folder_path in sorted(FIGURES_DIR.glob("Figure_*")):
    if folder_path.is_dir():
        try:
            shutil.rmtree(folder_path)
            print(f"   Removed: Figures/{folder_path.name}/")
        except Exception as e:
            print(f"   Error removing {folder_path.name}: {e}")

print("✅ Cleanup completed\n")
