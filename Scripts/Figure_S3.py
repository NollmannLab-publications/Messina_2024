#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S3 Messina et al. (mouse Hi-M)
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

print("Starting Figure S3 plotting with light source data...\n")


def plot_all_CFMs(decomposition_file, cutoff, output_dir, n_barcodes=25):
    """Plot every CFM of a 3DTopic decomposition."""
    Decomposition = np.load(SOURCE_LIGHT / decomposition_file)

    for i in range(Decomposition.shape[0]):
        matrix_tempo = fill_matrix_from_vector(Decomposition[i, :], n_barcodes)

        fig, ax = plt.subplots(1)
        im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
        ax.set_title(f'CFM {i} ({cutoff})')
        plt.colorbar(im1, ax=ax)

        plt.savefig(output_dir / f'Matrix_topic_{cutoff}_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
# ===================================================

#%% Figure S3a - 3DTopic decomposition, cutoff 50 nm
print("Plotting S3a: 3DTopic decomposition, cutoff 50 nm")
output_dir = REPO_ROOT / "Figures/Figure_3_supp" / "Figure_S3a"
output_dir.mkdir(parents=True, exist_ok=True)

plot_all_CFMs('components_50nm_0fd69c.npy', '50nm', output_dir)

print("   ✓ S3a done\n")

#%% Figure S3b - 3DTopic decomposition, cutoff 150 nm
print("Plotting S3b: 3DTopic decomposition, cutoff 150 nm")
output_dir = REPO_ROOT / "Figures/Figure_3_supp" / "Figure_S3b"
output_dir.mkdir(parents=True, exist_ok=True)

plot_all_CFMs('components_150nm_16d148.npy', '150nm', output_dir)

print("   ✓ S3b done\n")

#%% Figure S3c - 3DTopic decomposition, cutoff 200 nm
print("Plotting S3c: 3DTopic decomposition, cutoff 200 nm")
output_dir = REPO_ROOT / "Figures/Figure_3_supp" / "Figure_S3c"
output_dir.mkdir(parents=True, exist_ok=True)

plot_all_CFMs('components_200nm_28abc9.npy', '200nm', output_dir)

print("   ✓ S3c done\n")

print("🎉 All Figure S3 panels completed!")