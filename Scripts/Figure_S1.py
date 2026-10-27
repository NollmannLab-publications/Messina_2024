#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S1 Messina et al. (mouse Hi-M)
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

Schematic / microscopy panels (no script): S1a

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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

print("Starting Figure S1 plotting with light source data...\n")


def load_pancreas_PDX1LR():
    """All pancreas alleles (endocrine + exocrine), Pdx1 library, shape (n_alleles, 25, 25)."""
    Matrix_endocrine = np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_endocrine_missing_bc_corrected_PDX1LR.npy')
    Matrix_exocrine = np.load(SOURCE_DATA / 'Distance_maps_Pancreas_ND_exocrine_missing_bc_corrected_PDX1LR.npy')
    return np.concatenate((Matrix_endocrine, Matrix_exocrine), axis=0)
# ===================================================

#%% Figure S1b - Distance histograms + standard error of the median
print("Plotting S1b: Distance histograms")
output_dir = REPO_ROOT / "Figures/Figure_1_supp" / "Figure_S1b"
output_dir.mkdir(parents=True, exist_ok=True)

bins = np.arange(0, 2, 0.05)

# Load light data (same KDE as plot_distance_histograms, computed once if not available yet)
kde_file = SOURCE_LIGHT / "figureS1b_distance_KDE.npy"

if kde_file.exists():
    KDE_values = np.load(kde_file)
else:
    Matrix = np.transpose(load_pancreas_PDX1LR(), (1, 2, 0))
    n_bc = Matrix.shape[0]
    KDE_values = np.full((n_bc, n_bc, len(bins)), np.nan)
    for i in range(n_bc):
        for j in range(n_bc):
            data = Matrix[i, j, :]
            data = data[np.isfinite(data)]
            if len(data) > 1:
                try:
                    KDE_values[i, j, :] = gaussian_kde(data, bw_method='silverman')(bins)
                except np.linalg.LinAlgError:
                    pass
    np.save(kde_file, KDE_values)
    print("   saved light file: figureS1b_distance_KDE.npy")


def plot_KDE_grid(KDE, filename):
    """Same layout as plot_distance_histograms, from precomputed KDE values."""
    n_bin = KDE.shape[0]
    fig, axs = plt.subplots(figsize=(n_bin, n_bin), ncols=n_bin, nrows=n_bin, squeeze=False)
    for i in range(n_bin):
        for j in range(n_bin):
            kde_vals = KDE[i, j, :]
            if np.all(np.isfinite(kde_vals)):
                axs[i, j].plot(bins, kde_vals, 'b-', linewidth=1)
                axs[i, j].fill_between(bins, kde_vals, color='lightblue', alpha=0.5)
                axs[i, j].axvline(x=bins[np.argmax(kde_vals)], color='red', linestyle='--', linewidth=1)
            axs[i, j].set_xlim(bins[0], bins[-1])   # same x-axis for all pairs
            axs[i, j].tick_params(labelleft=False, labelbottom=False)
    plt.savefig(filename, dpi=300)
    plt.close()


# Plot all pairs + zoom
plot_KDE_grid(KDE_values, output_dir / 'Histogram_Matrix_Pancreas.png')
plot_KDE_grid(KDE_values[2:4, 19:21, :], output_dir / 'Histogram_Matrix_zoom_Pancreas.png')

# Lower plot (standard error of the median map): not in the previous repository
print("   ⚠ S1b standard error of the median map: not added yet")
print("   ✓ S1b done\n")

#%% Figure S1c - Detection efficiency
print("Plotting S1c: Detection efficiency")
output_dir = REPO_ROOT / "Figures/Figure_1_supp" / "Figure_S1c"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
N_file = SOURCE_LIGHT / "figureS1c_N_matrix_Pancreas.npy"

if N_file.exists():
    normalized_matrix = np.load(N_file)
else:
    Matrix = np.transpose(load_pancreas_PDX1LR(), (1, 2, 0))
    summed_matrix = np.sum(~np.isnan(Matrix), axis=2)
    normalized_matrix = (summed_matrix / np.max(summed_matrix)) * 100
    np.fill_diagonal(normalized_matrix, 100)
    np.save(N_file, normalized_matrix)
    print("   saved light file: figureS1c_N_matrix_Pancreas.npy")

# Plot detection efficiency per barcode
N_matrix_sum = np.sum(normalized_matrix, axis=1)
normalized_vector = (N_matrix_sum / np.max(N_matrix_sum)) * 100
indices = np.arange(len(normalized_vector)) + 1
labels = [f'RT{i}' for i in indices]

fig, ax = plt.subplots()
ax.bar(indices, normalized_vector)
ax.set_xticks(indices)
ax.set_xticklabels(labels, rotation=90)
ax.set_ylim(0, 100)
ax.set_xlabel('barcode #')
ax.set_ylabel('Detection efficiency')
plt.tight_layout()
plt.savefig(output_dir / 'Efficiency_Matrix_Pancreas.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot detection efficiency map
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(normalized_matrix, cmap='Blues', vmin=0, vmax=100)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Pancreas N matrix')
plt.savefig(output_dir / 'N_Matrix_Pancreas.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S1c done\n")

#%% Figure S1d - Bootstrapping
print("Plotting S1d: Bootstrapping")
output_dir = REPO_ROOT / "Figures/Figure_1_supp" / "Figure_S1d"
output_dir.mkdir(parents=True, exist_ok=True)

Number_Cells = np.array([1, 5, 10, 25, 50, 100])

# Load light data (bootstrapped once from the single-allele distance maps if not available yet)
boot_file = SOURCE_LIGHT / "figureS1d_bootstrapping_HiM.npy"

if boot_file.exists():
    Matrix_bootstrapping_HiM = list(np.load(boot_file, allow_pickle=True))
else:
    HiM = np.transpose(load_pancreas_PDX1LR(), (1, 2, 0))
    HiM_contact, _ = PWD_to_Contact(HiM, 1000, 100)
    data_raw_filtered = Filtering_efficiency(HiM, 0)

    np.random.seed(0)   # reproducible bootstrapping
    Matrix_bootstrapping_HiM = bootstrapping_HiM_correlation(data_raw_filtered, Number_Cells, 50, 100, HiM_contact)

    to_save = np.empty(len(Matrix_bootstrapping_HiM), dtype=object)
    to_save[:] = Matrix_bootstrapping_HiM
    np.save(boot_file, to_save, allow_pickle=True)
    print("   saved light file: figureS1d_bootstrapping_HiM.npy")

# Plot
fig, ax = plt.subplots()
vp = ax.violinplot(Matrix_bootstrapping_HiM, showmeans=False, showmedians=True)
colors = ['blue', 'green', 'orange', 'red', 'purple', 'yellow', 'cyan']
for patch, color in zip(vp['bodies'], colors):
    patch.set_facecolor(color)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp[partname].set_color('black')

ax.set_xticks(np.arange(1, len(Number_Cells) + 1))
ax.set_xticklabels(Number_Cells)
ax.set_ylim(0.35, 1.05)
ax.set_xlabel('Number of alleles')
ax.set_ylabel('Pearson correlation')
plt.savefig(output_dir / 'Bootstrapping_HiM.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S1d done\n")

#%% Figure S1e - PWD maps per replicate
print("Plotting S1e: PWD maps per replicate")
output_dir = REPO_ROOT / "Figures/Figure_1_supp" / "Figure_S1e"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the normalized 3D distance maps if not available yet)
replicate_file = SOURCE_LIGHT / "figureS1e_median_PWD_replicates.npz"

if replicate_file.exists():
    Replicates = dict(np.load(replicate_file))
else:
    Matrix_Pancreas_ID = np.load(SOURCE_DATA / 'All_experiment_ID_PDX1LR_Pancreas.npy')
    Matrix_Pancreas_all = np.load(SOURCE_DATA / 'All_distance_maps_normalized_PDX1LR_Pancreas.npy')
    Matrix_Pancreas_all = np.sqrt(Matrix_Pancreas_all[:, :, :, 0]**2 + Matrix_Pancreas_all[:, :, :, 1]**2 + Matrix_Pancreas_all[:, :, :, 2]**2)

    Replicates = {}
    for rep in range(1, 6):
        selection = Matrix_Pancreas_ID == f'exp_{rep}'
        print(f"   replicate {rep}: {np.sum(selection)} alleles")
        Replicates[f'rep_{rep}'] = np.nanmedian(Matrix_Pancreas_all[selection, :, :], axis=0)

    np.savez(replicate_file, **Replicates)
    print("   saved light file: figureS1e_median_PWD_replicates.npz")

# Plot
for rep in range(1, 6):
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Replicates[f'rep_{rep}'], cmap=Colormap_distance, vmin=0.1, vmax=0.5)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'Pancreas Median PWD matrix Replicate {rep}')
    plt.savefig(output_dir / f'Matrix_PWD_HiM_Pancreas_rep_{rep}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S1e done\n")

#%% Figure S1f - Cross correlation between replicates
print("Plotting S1f: Cross correlation between replicates")
output_dir = REPO_ROOT / "Figures/Figure_1_supp" / "Figure_S1f"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (same replicate medians as S1e)
Replicates = dict(np.load(SOURCE_LIGHT / "figureS1e_median_PWD_replicates.npz"))
matrices = [Replicates[f'rep_{rep}'] for rep in range(1, 6)]
correlation_matrix = np.corrcoef([matrix.flatten() for matrix in matrices])

# Plot
replicate_labels = [f"Replicate {i + 1}" for i in range(len(matrices))]
fig, ax = plt.subplots()
im1 = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=0, vmax=1)
plt.colorbar(im1, ax=ax)
ax.set_title("Cross-Correlation Hi-M replicates")
ax.set_xticks(range(len(matrices)))
ax.set_xticklabels(replicate_labels, rotation=45)
ax.set_yticks(range(len(matrices)))
ax.set_yticklabels(replicate_labels)
plt.savefig(output_dir / 'Correlation_Pancreas_replicates.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S1f done\n")

#%% Figure S1g - Pancreas PWD map (Pdx1)
print("Plotting S1g: Pancreas PWD map (Pdx1)")
output_dir = REPO_ROOT / "Figures/Figure_1_supp" / "Figure_S1g"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
median_file = SOURCE_LIGHT / "figureS1g_median_PWD_Pancreas_PDX1LR.npy"

if median_file.exists():
    Matrix_median = np.load(median_file)
else:
    Matrix_median = np.nanmedian(load_pancreas_PDX1LR(), axis=0)
    np.fill_diagonal(Matrix_median, 0)
    np.save(median_file, Matrix_median)
    print("   saved light file: figureS1g_median_PWD_Pancreas_PDX1LR.npy")

# Plot
fig, axs = plt.subplots(figsize=(10, 8))
im1 = axs.imshow(Matrix_median, cmap=Colormap_distance, vmin=0.1, vmax=0.42)
fig.colorbar(im1, ax=axs, shrink=1)
plt.title('Pancreas Median PWD matrix')
plt.savefig(output_dir / 'Matrix_PWD_Pancreas.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S1g done\n")

#%% Figure S1h - Hi-C contact maps (Leidescher et al. 2022)
print("Plotting S1h: Hi-C contact maps (Leidescher et al. 2022)")
output_dir = REPO_ROOT / "Figures/Figure_1_supp" / "Figure_S1h"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data: (sample, vmax)
HiC_samples = {'Tubes': 0.003, 'Thyroid': 0.0018, 'Blasts': 0.0025, 'Bladder': 0.0025}

# Plot
for sample, vmax in HiC_samples.items():
    Hi_C = np.load(SOURCE_LIGHT / f'PDX1LR_LR_HiC_{sample}_4kb_interpolated_020524.npy')

    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Hi_C, cmap=Colormap_topic, vmin=0, vmax=vmax)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'HiC {sample} Contact')
    plt.savefig(output_dir / f'Matrix_Contact_HiC_{sample}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S1h done\n")

#%% Figure S1i - Hi-C / Hi-M correlation vs threshold
print("Plotting S1i: Hi-C / Hi-M correlation vs threshold")
output_dir = REPO_ROOT / "Figures/Figure_1_supp" / "Figure_S1i"
output_dir.mkdir(parents=True, exist_ok=True)

thresholds = np.arange(50, 1050, 50)

# Load light data (computed once from the single-allele distance maps if not available yet)
correlation_file = SOURCE_LIGHT / "figureS1i_HiC_HiM_correlation.npy"

if correlation_file.exists():
    correlation = np.load(correlation_file)
else:
    Hi_C = np.load(SOURCE_LIGHT / 'PDX1LR_LR_HiC_Tubes_4kb_interpolated_020524.npy')
    HiM = np.transpose(load_pancreas_PDX1LR(), (1, 2, 0))

    correlation = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        Pancreas_Matrix_contact, _ = PWD_to_Contact(HiM, 1000, threshold)
        np.fill_diagonal(Pancreas_Matrix_contact, np.nan)

        valid_indices = np.isfinite(Hi_C.flatten()) & np.isfinite(Pancreas_Matrix_contact.flatten())
        correlation[i] = np.corrcoef(Hi_C.flatten()[valid_indices], Pancreas_Matrix_contact.flatten()[valid_indices])[0, 1]

    np.save(correlation_file, correlation)
    print("   saved light file: figureS1i_HiC_HiM_correlation.npy")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, correlation, marker='o', markersize=12, linewidth=2, color='black', markerfacecolor='blue')
plt.xticks(range(0, max(thresholds) + 1, 100))
plt.xlabel('Threshold (nm)')
plt.ylabel('Correlation Hi-C / Hi-M')
plt.savefig(output_dir / 'HiC_HiM_Correlation.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S1i done\n")

print("🎉 All Figure S1 panels completed!")