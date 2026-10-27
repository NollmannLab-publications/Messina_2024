#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S10 Messina et al. (mouse Hi-M)
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
import h5py
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

print("Starting Figure S10 plotting with light source data...\n")

n_barcodes_bintu = 44

# Bintu et al. 2018, HCT116 chr21 (single-allele PWDs in nm)
Bintu_files = {'WT': 'bintu2018Superresolution_Hs-HCT116-40-83_wt.h5',
               'RAD21': 'bintu2018Superresolution_Hs-HCT116-40-83_RAD21del.h5'}

def bintu_maps(condition):
    """Median PWD map and contact map (150 nm) of a Bintu dataset; computed once, then read from a light file."""
    light_file = SOURCE_LIGHT / f"figureS10_Bintu_{condition}_maps.npz"

    if light_file.exists():
        maps = np.load(light_file)
        return maps['median'], maps['contact']

    h5_file = SOURCE_DATA / Bintu_files[condition]
    if not h5_file.exists():
        raise FileNotFoundError(f"{h5_file.name} not found in Source_data: copy it (or fix the name in Bintu_files).")

    with h5py.File(h5_file, mode="r") as handler:
        data = handler["pwds"][:]

    # Half-matrix vectors -> symmetric matrices (n_alleles, 44, 44)
    expected_len = n_barcodes_bintu * (n_barcodes_bintu - 1) // 2
    if data.shape[1] != expected_len:
        raise ValueError(f"Second dimension ({data.shape[1]}) does not match half-matrix length ({expected_len})")

    tri_u = np.triu_indices(n_barcodes_bintu, k=1)
    reshaped = np.zeros((data.shape[0], n_barcodes_bintu, n_barcodes_bintu))
    reshaped[:, tri_u[0], tri_u[1]] = data
    reshaped = reshaped + np.transpose(reshaped, (0, 2, 1))
    print(f"   {condition}: {data.shape[0]} alleles")

    HiM = np.transpose(reshaped, (1, 2, 0))
    Matrix_contact, _ = PWD_to_Contact(HiM, 1, 150)
    Matrix_median = np.nanmedian(HiM, axis=2)

    np.savez(light_file, median=Matrix_median, contact=Matrix_contact)
    print(f"   saved light file: {light_file.name}")
    return Matrix_median, Matrix_contact


def plot_bintu(condition, output_dir):
    Matrix_median, Matrix_contact = bintu_maps(condition)

    # PWD map (upper plot)
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Matrix_median, cmap=Colormap_distance, vmin=100, vmax=900)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'Bintu {condition} Distance')
    plt.savefig(output_dir / f'Matrix_Distance_Bintu_{condition}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Contact map at 150 nm (lower plot)
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Matrix_contact, cmap=Colormap_topic, vmin=0, vmax=0.2)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'Bintu {condition} Contact 150nm')
    plt.savefig(output_dir / f'Matrix_Contact_Bintu_{condition}.png', dpi=300, bbox_inches='tight')
    plt.close()
# ===================================================

#%% Figure S10a - HCT116 wild-type PWD and contact maps
print("Plotting S10a: HCT116 wild-type PWD and contact maps")
output_dir = REPO_ROOT / "Figures/Figure_10_supp" / "Figure_S10a"
output_dir.mkdir(parents=True, exist_ok=True)

plot_bintu('WT', output_dir)

print("   ✓ S10a done\n")

#%% Figure S10b - HCT116 RAD21-depleted PWD and contact maps
print("Plotting S10b: HCT116 RAD21-depleted PWD and contact maps")
output_dir = REPO_ROOT / "Figures/Figure_10_supp" / "Figure_S10b"
output_dir.mkdir(parents=True, exist_ok=True)

plot_bintu('RAD21', output_dir)

print("   ✓ S10b done\n")

#%% Figure S10c - Correlation with Hi-C vs threshold
print("Plotting S10c: Correlation with Hi-C vs threshold")
output_dir = REPO_ROOT / "Figures/Figure_10_supp" / "Figure_S10c"
output_dir.mkdir(parents=True, exist_ok=True)

thresholds = range(50, 1050, 50)

# Load light data (precomputed correlation curves)
Bintu_correlation = np.load(SOURCE_LIGHT / 'Bintu_correlation.npy')
Pancreas_correlation = np.load(SOURCE_LIGHT / 'Pancreas_correlation.npy')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, Bintu_correlation, marker='o', markersize=12, linewidth=2, color='black', markerfacecolor='red', label='Bintu')
plt.plot(thresholds, Pancreas_correlation, marker='o', markersize=12, linewidth=2, color='black', markerfacecolor='blue', label='Hi-M Pancreas')
plt.xticks(range(0, max(thresholds) + 1, 100))
plt.xlabel('Threshold (nm)')
plt.ylabel('Correlation with Hi-C')
plt.legend()
plt.savefig(output_dir / 'Bintu_HiM_Correlation.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Bintu_HiM_Correlation.svg', bbox_inches='tight')
plt.close()

print("   ✓ S10c done\n")

#%% Figure S10d - Hi-C contact map
print("Plotting S10d: Hi-C contact map")
output_dir = REPO_ROOT / "Figures/Figure_10_supp" / "Figure_S10d"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (same Hi-C map and cropping as in the correlation of S10c)
Hi_C = np.load(SOURCE_LIGHT / 'Bintu_GM12878_10kb_interpolated.npy')[41:, 41:]
Hi_C = np.triu(Hi_C) + np.triu(Hi_C, k=1).T   # symmetrize from the upper triangle

# Plot
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Hi_C, cmap=Colormap_topic, vmin=0, vmax=50)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Hi-C Contact')
plt.savefig(output_dir / 'Matrix_Contact_HiC_Bintu.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S10d done\n")

#%% Figure S10e - npmi coherence wild-type / RAD21-depleted
# JB panels
print("   ⚠ S10e: panels from JB, not added yet\n")

#%% Figure S10f - Cluster maps wild-type vs RAD21-depleted
# JB panels
print("   ⚠ S10f: panels from JB, not added yet\n")

print("🎉 All Figure S10 panels completed!")