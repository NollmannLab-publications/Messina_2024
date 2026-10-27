#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S16 Messina et al. (mouse Hi-M)
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
from scipy import stats
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

print("Starting Figure S16 plotting with light source data...\n")


def load_endocrine(diet, library):
    """Single-allele PWD maps (n_alleles, n_barcodes, n_barcodes) of endocrine cells, ND or HFD."""
    return np.load(SOURCE_DATA / f'Distance_maps_Pancreas_{diet}_endocrine_missing_bc_corrected_{library}.npy')


def contact_map(diet, library):
    """Contact map at 100 nm; computed once, then read from a light file."""
    light_file = SOURCE_LIGHT / f"figureS16_contact_{library}_endocrine_{diet}.npy"
    if light_file.exists():
        return np.load(light_file)

    Matrix = load_endocrine(diet, library)
    print(f"   {library} endocrine {diet}: {Matrix.shape[0]} alleles")
    contact, _ = PWD_to_Contact(np.transpose(Matrix, (1, 2, 0)), 1000, 100)
    np.save(light_file, contact)
    print(f"   saved light file: {light_file.name}")
    return contact


def differential_PWD(library):
    """log2(median PWD ND / median PWD HFD), endocrine cells; computed once, then read from a light file."""
    light_file = SOURCE_LIGHT / f"figureS16_differential_PWD_{library}_ND_HFD.npy"
    if light_file.exists():
        return np.load(light_file)

    median_ND = np.nanmedian(load_endocrine('ND', library), axis=0)
    median_HFD = np.nanmedian(load_endocrine('HFD', library), axis=0)
    np.fill_diagonal(median_ND, np.nan)
    np.fill_diagonal(median_HFD, np.nan)

    Differential_matrix = np.log2(median_ND / median_HFD)
    np.save(light_file, Differential_matrix)
    print(f"   saved light file: {light_file.name}")
    return Differential_matrix


def volcano_PWD(library):
    """Differential PWD and Wilcoxon p-values (endocrine ND vs HFD); computed once, then read from a light file."""
    light_file = SOURCE_LIGHT / f"figureS16_volcano_PWD_{library}_ND_HFD.npz"
    if light_file.exists():
        volcano = np.load(light_file)
        return volcano['differential'], volcano['pvalues']

    Matrix_ND = np.transpose(load_endocrine('ND', library), (1, 2, 0))
    Matrix_HFD = np.transpose(load_endocrine('HFD', library), (1, 2, 0))
    differential, pvalues = VolcanoplotHiM(Matrix_ND, Matrix_HFD)

    np.savez(light_file, differential=differential, pvalues=pvalues)
    print(f"   saved light file: {light_file.name}")
    return differential, pvalues


def plot_volcano_PWD(differential, pvalues, annotations, xlim, ylim, filename, output_dir):
    """Volcano plot of all barcode pairs: gray if p > 0.05, otherwise colored by the sign of the change."""
    cmap = plt.get_cmap('RdBu')
    y_values = -np.log10(pvalues)
    threshold = -np.log10(0.05)

    colors = ['gray' if y < threshold else cmap(0.0) if x < 0 else cmap(1.0)
              for x, y in zip(differential.flatten(), y_values.flatten())]

    fig, ax = plt.subplots()
    ax.scatter(differential.flatten(), y_values.flatten(), c=colors, alpha=0.33)
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    for idx, idy, label in annotations:
        xytext_score = (25, -50) if idx % 2 == 1 else (-25, 50)
        ax.annotate(label, (differential[idx, idy], y_values[idx, idy]),
                    xytext=xytext_score, textcoords='offset points',
                    ha='center', color='black', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='black'))

    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.svg', bbox_inches='tight')
    plt.close()


def plot_contact(contact, vmax, filename, output_dir):
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(contact, cmap=Colormap_topic, vmin=0, vmax=vmax)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title('Pancreas Contact 100nm')
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
# ===================================================

#%% Figure S16a - PWD volcano plot Pdx1 ND / HFD
print("Plotting S16a: PWD volcano plot Pdx1 ND / HFD")
output_dir = REPO_ROOT / "Figures/Figure_16_supp" / "Figure_S16a"
output_dir.mkdir(parents=True, exist_ok=True)

differential, pvalues = volcano_PWD('PDX1LR')
annotations = [
    (1, 16, 'Pdx1_CRE1'),
    (2, 16, 'Pdx1_CRE2'),
    (5, 16, 'Pdx1_CRE3'),
    (9, 16, 'Pdx1_CRE4'),
    (10, 16, 'Pdx1_CRE5'),
    (11, 16, 'Pdx1_CRE5'),
    (15, 16, 'Pdx1_CRE6'),
    (16, 16, 'Pdx1_CRE7'),
    (17, 16, 'Pdx1_CRE7'),
    (18, 16, 'Pdx1_CRE8')]
plot_volcano_PWD(differential, pvalues, annotations, (-0.04, 0.04), (-5, 17),
                 'Volcano_plot_Endocrine_ND_HFD_PWD', output_dir)

print("   ✓ S16a done\n")

#%% Figure S16b - Isl1 contact maps ND / HFD
print("Plotting S16b: Isl1 contact maps ND / HFD")
output_dir = REPO_ROOT / "Figures/Figure_16_supp" / "Figure_S16b"
output_dir.mkdir(parents=True, exist_ok=True)

for diet in ['ND', 'HFD']:
    plot_contact(contact_map(diet, 'ISL1'), 0.12, f'Contact_Pancreas_endocrine{diet}', output_dir)

print("   ✓ S16b done\n")

#%% Figure S16c - Isl1 differential PWD ND / HFD
print("Plotting S16c: Isl1 differential PWD ND / HFD")
output_dir = REPO_ROOT / "Figures/Figure_16_supp" / "Figure_S16c"
output_dir.mkdir(parents=True, exist_ok=True)

fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(differential_PWD('ISL1'), cmap='RdBu', vmin=-0.35, vmax=0.35)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Ratio PWD matrix')
plt.savefig(output_dir / 'Matrix_EndocrineND_EndocrineHFD_differential_PWD_log2_ISL1.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S16c done\n")

#%% Figure S16d - Isl1 PWD volcano plot ND / HFD
print("Plotting S16d: Isl1 PWD volcano plot ND / HFD")
output_dir = REPO_ROOT / "Figures/Figure_16_supp" / "Figure_S16d"
output_dir.mkdir(parents=True, exist_ok=True)

differential, pvalues = volcano_PWD('ISL1')
annotations = [
    (0, 6, 'Isl1_CRE1'),
    (4, 6, 'Isl1_CRE2'),
    (6, 6, 'Isl1_CRE3'),
    (10, 6, 'Isl1_CRE4')]
plot_volcano_PWD(differential, pvalues, annotations, (-0.35, 0.35), (0, 15),
                 'Volcano_plot_EndocrineND_EndocrineHFD_PWD_ISL1', output_dir)

print("   ✓ S16d done\n")

#%% Figure S16e - Isl1 CFM volcano plot ND / HFD
print("Plotting S16e: Isl1 CFM volcano plot ND / HFD")
output_dir = REPO_ROOT / "Figures/Figure_16_supp" / "Figure_S16e"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the odds ratio distributions if not available yet)
volcano_file = SOURCE_LIGHT / "figureS16e_volcano_CFM_ISL1_ND_HFD.npz"

if volcano_file.exists():
    volcano = np.load(volcano_file)
    x_values, y_values = volcano['x_values'], volcano['y_values']
else:
    Distribution_CFMs = np.load(SOURCE_DATA / 'Distribution_OR_CFMs_Reference_00_Correlation_0.8_Pancreas_ND_endocrine_ISL1.npy', allow_pickle=True)
    Distribution_CFMs_subset = Distribution_CFMs[3, :]   # Tissues_nb = 3 (Pancreas_HFD_endocrine)

    x_values = np.zeros(Distribution_CFMs_subset.shape[0])
    y_values = np.zeros(Distribution_CFMs_subset.shape[0])
    for i in range(Distribution_CFMs_subset.shape[0]):
        _, p_val = stats.ttest_1samp(Distribution_CFMs_subset[i], 0)
        y_values[i] = -np.log10(p_val)
        x_values[i] = np.nanmedian(Distribution_CFMs_subset[i])

    np.savez(volcano_file, x_values=x_values, y_values=y_values)
    print("   saved light file: figureS16e_volcano_CFM_ISL1_ND_HFD.npz")

# Parameters
cmap = plt.get_cmap('BrBG_r')
Domains_Isl1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
Loops_Isl1 = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27]
artifact = [24, 28, 29]
Isl1_related = [6, 7, 12, 13, 14, 15, 17, 21]
annotations = {i: f'CFM{i}-Isl1' if i in Isl1_related else f'CFM{i}' for i in range(30)}

# Colors: negative / positive odds ratio
colors = [cmap(0.0) if x < 0 else cmap(1.0) for x in x_values]
x_values = x_values.copy()
y_values = y_values.copy()
x_values[artifact] = np.nan
y_values[artifact] = np.nan

# Plot volcano (squares = domains, circles = loops)
fig, ax = plt.subplots()
for i in range(len(x_values)):
    if i in Loops_Isl1:
        ax.scatter(x_values[i], y_values[i], color=colors[i], alpha=0.8, s=100, marker='o')
    elif i in Domains_Isl1:
        ax.scatter(x_values[i], y_values[i], color=colors[i], alpha=0.8, s=100, marker='s')

plt.xlim(-0.25, 0.25)
plt.ylim(0, 50)
for idx, label in annotations.items():
    ax.annotate(label, (x_values[idx], y_values[idx]), xytext=(0, 0), textcoords='offset points',
                ha='center', color='black', fontsize=10)

ax.invert_xaxis()
plt.savefig(output_dir / 'Volcanoplot_CFM_Pancreas_EndocrineND_EndocrineHFD.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Volcanoplot_CFM_Pancreas_EndocrineND_EndocrineHFD.svg', bbox_inches='tight')
plt.close()

print("   ✓ S16e done\n")

print("🎉 All Figure S16 panels completed!")