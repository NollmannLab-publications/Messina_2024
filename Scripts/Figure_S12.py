#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S12 Messina et al. (mouse Hi-M)
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

print("Starting Figure S12 plotting with light source data...\n")

Tissues = ['Pancreas', 'Brain', 'Kidney', 'Liver', 'Lung', 'Lymphnode', 'Thymus']


def load_tissue_maps(tissue, library):
    """Single-allele PWD maps (n_alleles, n_barcodes, n_barcodes) of one tissue and one library."""
    if tissue == 'Pancreas' and library == 'PDX1LR':
        return np.concatenate((np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy'),
                               np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')), axis=0)
    return np.load(SOURCE_DATA / f'Distance_maps_{tissue}_ND_all_missing_bc_corrected_{library}.npy')


def tissue_light_maps(library):
    """Median PWD and contact (100 nm) maps of every tissue for one library; computed once, then read from a light file."""
    light_file = SOURCE_LIGHT / f"figureS12_tissues_{library}_maps.npz"

    if light_file.exists():
        return dict(np.load(light_file))

    maps = {}
    for tissue in Tissues:
        Matrix = load_tissue_maps(tissue, library)
        print(f"   {library} {tissue}: {Matrix.shape[0]} alleles")

        matrix_median = np.nanmedian(Matrix, axis=0)
        np.fill_diagonal(matrix_median, 0)
        matrix_contact, _ = PWD_to_Contact(np.transpose(Matrix, (1, 2, 0)), 1000, 100)
        np.fill_diagonal(matrix_contact, 1)

        maps[f'median_{tissue}'] = matrix_median
        maps[f'contact_{tissue}'] = matrix_contact

    np.savez(light_file, **maps)
    print(f"   saved light file: {light_file.name}")
    return maps
# ===================================================

#%% Figure S12a - Pdx1 PWD maps across tissues
print("Plotting S12a: Pdx1 PWD maps across tissues")
output_dir = REPO_ROOT / "Figures/Figure_12_supp" / "Figure_S12a"
output_dir.mkdir(parents=True, exist_ok=True)

maps_PDX1LR = tissue_light_maps('PDX1LR')

for tissue in Tissues:
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(maps_PDX1LR[f'median_{tissue}'], cmap=Colormap_distance, vmin=0.1, vmax=0.42)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'{tissue} PWD')
    plt.savefig(output_dir / f'PWD_PDX1LR_{tissue}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S12a done\n")

#%% Figure S12b - Ins2 PWD maps across tissues
print("Plotting S12b: Ins2 PWD maps across tissues")
output_dir = REPO_ROOT / "Figures/Figure_12_supp" / "Figure_S12b"
output_dir.mkdir(parents=True, exist_ok=True)

maps_INS2HR = tissue_light_maps('INS2HR')

for tissue in Tissues:
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(maps_INS2HR[f'median_{tissue}'], cmap=Colormap_distance, vmin=0.1, vmax=0.45)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'{tissue} PWD')
    plt.savefig(output_dir / f'PWD_INS2HR_{tissue}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S12b done\n")

#%% Figure S12c - Ins2 contact maps across tissues
print("Plotting S12c: Ins2 contact maps across tissues")
output_dir = REPO_ROOT / "Figures/Figure_12_supp" / "Figure_S12c"
output_dir.mkdir(parents=True, exist_ok=True)

for tissue in Tissues:
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(maps_INS2HR[f'contact_{tissue}'], cmap=Colormap_topic, vmin=0, vmax=0.2)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'{tissue} Contact matrix')
    plt.savefig(output_dir / f'Matrix_contact_INS2HR_{tissue}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S12c done\n")

print("🎉 All Figure S12 panels completed!")