#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S15 Messina et al. (mouse Hi-M)
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

Schematic / microscopy panels (no script): S15f

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

print("Starting Figure S15 plotting with light source data...\n")

# Weight of each mouse over time (16 ND mice, 19 HFD mice)
Weight = pd.read_excel(SOURCE_LIGHT / 'weight.xlsx')
ND_weight = Weight.loc[:, ['ND'] + [f'Unnamed: {i}' for i in range(2, 17)]]
HFD_weight = Weight.loc[:, ['HFD'] + [f'Unnamed: {i}' for i in range(18, 36)]]

# IPGTT of each mouse (column 1 = time; 16 ND and 16 HFD mice)
ipgtt = pd.read_excel(SOURCE_LIGHT / 'IPGTT.xlsx')
ipgtt.columns = [f'Column{i}' for i in range(1, ipgtt.shape[1] + 1)]
ND_ipgtt = ipgtt[[f'Column{i}' for i in list(range(2, 10)) + list(range(18, 26))]]
HFD_ipgtt = ipgtt[[f'Column{i}' for i in list(range(10, 18)) + list(range(26, 34))]]
ipgtt_rows = slice(1, 7)   # time points used for the curve and the AUC


def plot_two_groups(ND_values, HFD_values, ylim, ylabel, filename, output_dir):
    """Strip plot of ND vs HFD mice with mean ± s.d. (red), and Mann-Whitney test printed."""
    data = np.concatenate((np.array(ND_values, dtype=float), np.array(HFD_values, dtype=float)))
    labels = ['ND'] * len(ND_values) + ['HFD'] * len(HFD_values)

    fig, ax = plt.subplots()
    sns.stripplot(x='Label', y='Data', data={'Data': data, 'Label': labels}, hue='Label',
                  palette={'ND': 'black', 'HFD': 'blue'}, jitter=True, legend=False, ax=ax)
    ax.errorbar(0, np.mean(ND_values), yerr=np.std(ND_values, ddof=1), color='red', fmt='o')
    ax.errorbar(1, np.mean(HFD_values), yerr=np.std(HFD_values, ddof=1), color='red', fmt='o')
    ax.axhline(np.nanmean(ND_values), color='red', linestyle='-', xmin=0.28, xmax=0.39)
    ax.axhline(np.nanmean(HFD_values), color='red', linestyle='-', xmin=0.61, xmax=0.72)
    ax.set_ylim(*ylim)
    ax.set_xlim(-1, 2)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.svg', bbox_inches='tight')
    plt.close()

    _, p_value = mannwhitneyu(ND_values, HFD_values, alternative='two-sided')
    print(f"   Mann-Whitney ND vs HFD: p = {p_value:.2e}")
# ===================================================

#%% Figure S15a - Weight over time ND / HFD
print("Plotting S15a: Weight over time ND / HFD")
output_dir = REPO_ROOT / "Figures/Figure_15_supp" / "Figure_S15a"
output_dir.mkdir(parents=True, exist_ok=True)

x = Weight['Time (days)']

fig, ax = plt.subplots()
ax.errorbar(x, ND_weight.mean(axis=1), yerr=ND_weight.std(axis=1), fmt='-o', color='black', markerfacecolor='black', label='ND')
ax.errorbar(x, HFD_weight.mean(axis=1), yerr=HFD_weight.std(axis=1), fmt='-o', color='blue', markerfacecolor='blue', label='HFD')
ax.set_ylim(20, 47)
ax.set_xlabel('Weeks of treatment')
ax.set_ylabel('Weight (g)')
ax.legend(loc='upper left')
plt.savefig(output_dir / 'Weight_ND_HFD.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Weight_ND_HFD.svg', bbox_inches='tight')
plt.close()

print("   ✓ S15a done\n")

#%% Figure S15b - Weight at week 14
print("Plotting S15b: Weight at week 14")
output_dir = REPO_ROOT / "Figures/Figure_15_supp" / "Figure_S15b"
output_dir.mkdir(parents=True, exist_ok=True)

plot_two_groups(ND_weight.iloc[10], HFD_weight.iloc[10], (20, 60), 'Weight (g) week 14',
                'Weight_ND_HFD_at_14_weeks', output_dir)

print("   ✓ S15b done\n")

#%% Figure S15c - IPGTT
print("Plotting S15c: IPGTT")
output_dir = REPO_ROOT / "Figures/Figure_15_supp" / "Figure_S15c"
output_dir.mkdir(parents=True, exist_ok=True)

x = ipgtt['Column1'].iloc[ipgtt_rows]   # time (min)

fig, ax = plt.subplots()
ax.errorbar(x, ND_ipgtt.iloc[ipgtt_rows].mean(axis=1), yerr=ND_ipgtt.iloc[ipgtt_rows].std(axis=1),
            fmt='-o', color='black', markerfacecolor='black', label='ND')
ax.errorbar(x, HFD_ipgtt.iloc[ipgtt_rows].mean(axis=1), yerr=HFD_ipgtt.iloc[ipgtt_rows].std(axis=1),
            fmt='-o', color='blue', markerfacecolor='blue', label='HFD')
ax.set_ylim(0, 650)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Blood glucose (mg/dl)')
ax.legend(loc='upper right')
plt.savefig(output_dir / 'IPGTT.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'IPGTT.svg', bbox_inches='tight')
plt.close()

print("   ✓ S15c done\n")

#%% Figure S15d - IPGTT area under the curve
print("Plotting S15d: IPGTT area under the curve")
output_dir = REPO_ROOT / "Figures/Figure_15_supp" / "Figure_S15d"
output_dir.mkdir(parents=True, exist_ok=True)

# Area under the glucose curve of each mouse (trapezoidal rule, same time points as S15c)
trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz   # name depends on the numpy version
time_points = ipgtt['Column1'].iloc[ipgtt_rows].astype(float).values
AUC_ND = [trapezoid(ND_ipgtt[mouse].iloc[ipgtt_rows].astype(float).values, time_points) for mouse in ND_ipgtt.columns]
AUC_HFD = [trapezoid(HFD_ipgtt[mouse].iloc[ipgtt_rows].astype(float).values, time_points) for mouse in HFD_ipgtt.columns]

plot_two_groups(AUC_ND, AUC_HFD, (0, 80000), 'AUC IPGTT (AU)', 'AUC_IPGTT_ND_HFD', output_dir)

print("   ✓ S15d done\n")

#%% Figure S15e - RNA-seq volcano plot ND / HFD
print("Plotting S15e: RNA-seq volcano plot HFD vs SD")
output_dir = REPO_ROOT / "Figures/Figure_15_supp" / "Figure_S15e"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (DESeq2 results, GSE131941)
df = pd.read_csv(SOURCE_LIGHT / "GSE131941_DESeq2_results_HFD_vs_SD_young.txt", sep="\t")
df = df.dropna(subset=['pvalue', 'log2FoldChange', 'padj'])
df['neg_log10_pvalue'] = -np.log10(df['pvalue'])

# Thresholds and color coding
y_threshold = -np.log10(0.05)
logfc_threshold = 0
highlight_genes = ['PDX1', 'NKX2-2', 'NKX6-1', 'MAFA']

df['color'] = 'grey'
df.loc[(df['neg_log10_pvalue'] > y_threshold) & (df['log2FoldChange'] > logfc_threshold), 'color'] = 'red'
df.loc[(df['neg_log10_pvalue'] > y_threshold) & (df['log2FoldChange'] < -logfc_threshold), 'color'] = 'blue'

# Broken y axis
cut_low, cut_high, ylim_top = 26, 45, 50
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(hspace=0.05)

for color in ['blue', 'grey', 'red']:
    subset = df[df['color'] == color]
    subset_high = subset[subset['neg_log10_pvalue'] > cut_high]
    subset_low = subset[subset['neg_log10_pvalue'] <= cut_low]
    ax1.scatter(subset_high['log2FoldChange'], subset_high['neg_log10_pvalue'], c=color, alpha=0.1)
    ax2.scatter(subset_low['log2FoldChange'], subset_low['neg_log10_pvalue'], c=color, alpha=0.1, label=color.capitalize())

# Highlight selected genes
for gene in highlight_genes:
    gene_row = df[df['Gene_name'].str.upper() == gene.upper()]
    if not gene_row.empty:
        x = gene_row['log2FoldChange'].values[0]
        y = gene_row['neg_log10_pvalue'].values[0]
        ax = ax1 if y > cut_high else ax2
        ax.scatter(x, y, c='green', edgecolors='black', s=40, zorder=7)
        ax.text(x, y + 1, gene, fontsize=9, color='black', ha='center')

ax2.axhline(y_threshold, linestyle='--', color='black')
ax1.set_ylim(cut_high, ylim_top)
ax2.set_ylim(0, cut_low)

# Broken axis effect
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax2.set_xlabel('log2(Fold Change)')
ax2.set_ylabel('-log10(p-value)')
ax1.set_title('Volcano Plot with Broken Y-Axis: HFD vs SD (young)')
ax2.set_xlim(-2.2, 2.2)
ax2.legend(loc='upper right', fontsize=9)
plt.savefig(output_dir / 'VolcanoPlot_HFD_vs_SD_young_broken_axis.svg', bbox_inches='tight')
plt.savefig(output_dir / 'VolcanoPlot_HFD_vs_SD_young_broken_axis.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S15e done\n")

#%% Figure S15g - CTCF fluorescence per islet
# Marie / JB panels
print("   ⚠ S15g: not added yet\n")

#%% Figure S15h - PWD maps beta cells ND / HFD
print("Plotting S15h: PWD maps endocrine cells ND / HFD")
output_dir = REPO_ROOT / "Figures/Figure_15_supp" / "Figure_S15h"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (computed once from the single-allele distance maps if not available yet)
median_file = SOURCE_LIGHT / "figureS15h_median_PWD_endocrine_ND_HFD.npz"

if median_file.exists():
    Medians = dict(np.load(median_file))
else:
    Medians = {diet: np.nanmedian(np.load(SOURCE_DATA / f'Distance_maps_Pancreas_{diet}_endocrine_missing_bc_corrected_PDX1LR.npy'), axis=0)
               for diet in ['ND', 'HFD']}
    np.savez(median_file, **Medians)
    print("   saved light file: figureS15h_median_PWD_endocrine_ND_HFD.npz")

# Plot
for diet in ['ND', 'HFD']:
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Medians[diet], cmap=Colormap_distance, vmin=0.1, vmax=0.4)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title(f'Pancreas Median Endocrine PWD matrix ({diet})')
    plt.savefig(output_dir / f'Matrix_PWD_Pancreas_Endocrine_{diet}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S15h done\n")

print("🎉 All Figure S15 panels completed!")