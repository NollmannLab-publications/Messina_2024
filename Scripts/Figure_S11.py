#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S11 Messina et al. (mouse Hi-M)
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
from matplotlib.colors import LinearSegmentedColormap
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

print("Starting Figure S11 plotting with light source data...\n")
# ===================================================

#%% Figure S11a - Tabula Muris tSNE + gene expression
print("Plotting S11a: Tabula Muris tSNE + gene expression")
output_dir = REPO_ROOT / "Figures/Figure_11_supp" / "Figure_S11a"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data
tabula_muris = pd.read_csv(SOURCE_LIGHT / 'Tabula_muris_filtered_gene_expression_data.csv')

# Plot tissues (same tab10 colors as a scatter colored by tissue_code)
codes = tabula_muris['tissue_code']
cmap_tissues = plt.get_cmap('tab10')
norm = plt.Normalize(vmin=codes.min(), vmax=codes.max())

plt.figure(figsize=(10, 10))
for code in sorted(codes.unique()):
    tissue_cells = tabula_muris[codes == code]
    plt.scatter(tissue_cells['tSNE_1'], tissue_cells['tSNE_2'], color=cmap_tissues(norm(code)),
                alpha=1, label=tissue_cells['tissue'].iloc[0])
plt.legend(title="Tissues")
plt.xlim(-60, 60)
plt.ylim(-60, 60)
plt.xlabel('tSNE 1')
plt.ylabel('tSNE 2')
plt.savefig(output_dir / 'Tabula_Muris_Tissues_ID.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot gene expression (cells with expression > 50 colored, others gray)
# 'Prhoxnb' is the former name of Urad
genes = {'Polr1d': 'Polr1d', 'Gsx1': 'Gsx1', 'Pdx1': 'Pdx1', 'Prhoxnb': 'Urad'}
custom_cmap = LinearSegmentedColormap.from_list('custom_blues', [(0.5, 0.5, 1), (0, 0, 0.5)], N=50)

for column, gene_name in genes.items():
    plt.figure(figsize=(10, 10))
    plt.scatter(tabula_muris['tSNE_1'], tabula_muris['tSNE_2'], color=[0.8, 0.8, 0.8], alpha=1)

    masked_expression = np.ma.masked_where(tabula_muris[column] <= 50, tabula_muris[column])
    scatter = plt.scatter(tabula_muris['tSNE_1'], tabula_muris['tSNE_2'], c=masked_expression,
                          cmap=custom_cmap, alpha=1, vmin=1, vmax=100, edgecolor='none')

    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.title(gene_name)
    plt.colorbar(scatter, label=f'Log({gene_name} Expression)')
    plt.savefig(output_dir / f'Tabula_Muris_Expression_{gene_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ S11a done\n")

print("🎉 All Figure S11 panels completed!")