#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S8 Messina et al. (mouse Hi-M)
------------------------------------------------------------------------------------------------------------------------

                   ############################
                   #           .-"-.          #
                   #          /  ~0_0         #
                   #          \_  (__\        #
                   #          //   \\         #
                   #         ((  ®  ))        #
                   #==========""===""=========#
                   #            |||           #
                   #            '|'           #
                   #             |            #
                   ############################

S8b, S8c and S8d reuse the light files of Figures S5b, S6b and S7c: run Figure_S5.py, Figure_S6.py and Figure_S7.py first.

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ====================== SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_DATA = REPO_ROOT / "Source_data"     # full source data, only read to create the light files
SOURCE_LIGHT = REPO_ROOT / "Source_light"   # light (precomputed) source data

# Add custom functions
sys.path.append(str(SCRIPT_DIR))
from Function_Messina_et_al import *

# Colormaps
Colormap_topic = 'RdBu_r'
Colormap_distance = 'RdGy'

print("Starting Figure S8 plotting with light source data...\n")

# Weight thresholds and line colors
Weights = {0.05: 'blue', 0.10: 'black', 0.15: 'gray', 0.20: 'orange'}


def plot_representativity(decomposition_file, Domains, Loops, title, output_dir, sort_by=None):
    """Representativity of each CFM at 5, 10, 15 and 20% weight; markers colored by CFM class.
    sort_by: weight (e.g. 0.10) used to order the CFMs by decreasing representativity, or None to keep the file order."""
    decomposition_path = SOURCE_LIGHT / decomposition_file
    if not decomposition_path.exists():
        raise FileNotFoundError(f"{decomposition_file} not found: run the script that creates it first (see header).")
    Decomposition = np.load(decomposition_path)

    # Order of the CFMs on the x axis (original CFM numbers)
    CFM_order = np.arange(Decomposition.shape[1])
    if sort_by is not None:
        CFM_order = np.argsort(topic_representativity_frequency(Decomposition, sort_by))[::-1]
        Decomposition = Decomposition[:, CFM_order]
        print(f"   CFM order (decreasing representativity @{int(sort_by * 100)}%): {CFM_order.tolist()}")

    CFM_index = np.arange(len(CFM_order))
    markerfacecolors = ['red' if cfm in Domains else 'green' if cfm in Loops else 'gray' for cfm in CFM_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    for weight, color in Weights.items():
        frequency = topic_representativity_frequency(Decomposition, weight)
        ax.plot(CFM_index, frequency, linewidth=2, color=color, label=f'@{int(weight * 100)}%')
        ax.scatter(CFM_index, frequency, s=30, c=markerfacecolors, edgecolors='black', linewidths=1, zorder=3)

    # Legend: CFM classes + weight thresholds
    ax.scatter([], [], s=30, c='green', edgecolors='black', label='Loops')
    ax.scatter([], [], s=30, c='red', edgecolors='black', label='Domains')
    ax.scatter([], [], s=30, c='gray', edgecolors='black', label='N/A')
    ax.legend(frameon=False)

    ax.set_xlabel('Chromatin Folding Motifs (CFMs)')
    ax.set_ylabel('Representativity')
    ax.set_title(title)
    plt.savefig(output_dir / 'Topic_representativity.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Topic_representativity.svg', bbox_inches='tight')
    plt.close()
# ===================================================

#%% Figure S8a - Representativity, Pdx1 library
print("Plotting S8a: Representativity, Pdx1 library")
output_dir = REPO_ROOT / "Figures/Figure_8_supp" / "Figure_S8a"
output_dir.mkdir(parents=True, exist_ok=True)

plot_representativity('Matrix_Decomposition_PDX1LR_Pancreas.npy',
                      Domains=[0, 1, 2, 3, 4, 5, 6, 7, 8, 17],
                      Loops=[9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28],
                      title='Pdx1 Hi-M library', output_dir=output_dir)

print("   ✓ S8a done\n")

#%% Figure S8b - Representativity, Isl1 library
print("Plotting S8b: Representativity, Isl1 library")
output_dir = REPO_ROOT / "Figures/Figure_8_supp" / "Figure_S8b"
output_dir.mkdir(parents=True, exist_ok=True)

# Ordered Isl1 decomposition, created by Figure_S5.py
plot_representativity('figureS5b_decomposition_ordered_Isl1.npy',
                      Domains=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                      Loops=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27],
                      title='Isl1 Hi-M library', output_dir=output_dir)

print("   ✓ S8b done\n")

#%% Figure S8c - Representativity, Ins2 library
print("Plotting S8c: Representativity, Ins2 library")
output_dir = REPO_ROOT / "Figures/Figure_8_supp" / "Figure_S8c"
output_dir.mkdir(parents=True, exist_ok=True)

# Ordered Ins2 decomposition, created by Figure_S6.py
plot_representativity('figureS6b_decomposition_ordered_Ins2.npy',
                      Domains=[0, 1, 2, 3, 4, 5, 7, 8, 19],
                      Loops=[6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28],
                      title='Ins2 Hi-M library', output_dir=output_dir)

print("   ✓ S8c done\n")

#%% Figure S8d - Representativity, human PDX1 library
print("Plotting S8d: Representativity, human PDX1 library")
output_dir = REPO_ROOT / "Figures/Figure_8_supp" / "Figure_S8d"
output_dir.mkdir(parents=True, exist_ok=True)

# Human decomposition, made by Figure_S7.py (S7c). CFM classes as in Figure S7d (original CFM numbers)
plot_representativity('figureS7c_decomposition_ordered_Human_PDX1.npy',
                      Domains=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      Loops=[10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                      title='PDX1 Human Hi-M library', output_dir=output_dir,
                      sort_by=0.10)   # human CFMs are not pre-sorted: order them by decreasing representativity @10%

print("   ✓ S8d done\n")

print("🎉 All Figure S8 panels completed!")