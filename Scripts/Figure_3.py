#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure 3 Messina et al. (mouse Hi-M)
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

# CFM classes of the reference Pdx1 decomposition
Domains = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17]
Loops = [9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28]

print("Starting Figure 3 plotting with light source data...\n")
# ===================================================

#%% Figure 3a - Block-copolymer simulated CFMs
print("Plotting 3a: Block-copolymer simulated CFMs")
output_dir = REPO_ROOT / "Figures/Figure_3" / "Figure_3a"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data
Decomposition = np.load(SOURCE_LIGHT / 'components_topics_heteropolymer_PDX1.npy')
representativity_CFM_toy = np.sum(Decomposition, axis=1) * 100 / np.sum(Decomposition)

Subselection = [0, 1, 3, 4, 5, 6, 7, 22]

# Plot CFMs
for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'CFM{cfm}_BC - representativity: {representativity_CFM_toy[cfm]:.2f}%')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_BC_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 3a done\n")

#%% Figure 3b - Loop-extrusion simulated CFMs
print("Plotting 3b: Loop-extrusion simulated CFMs")
output_dir = REPO_ROOT / "Figures/Figure_3" / "Figure_3b"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data
Decomposition = np.load(SOURCE_LIGHT / 'components_topics_extrusion_PDX1.npy')
representativity_CFM_toy = np.sum(Decomposition, axis=1) * 100 / np.sum(Decomposition)

Subselection = [0, 1, 2, 4, 6, 7, 12, 17]

# Plot CFMs
for cfm in Subselection:
    matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], 25)

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
    ax.set_title(f'CFM{cfm}_LE - representativity: {representativity_CFM_toy[cfm]:.2f}%')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'Matrix_topic_LE_{cfm}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 3b done\n")

#%% Figure 3c - Correlation experimental vs simulated CFMs
print("Plotting 3c: Correlation experimental vs simulated CFMs")
output_dir = REPO_ROOT / "Figures/Figure_3" / "Figure_3c"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data
correlation_reorganized = np.load(SOURCE_LIGHT / 'correlation_CFM_consensus_exp_simu.npy')
merged_list = np.load(SOURCE_LIGHT / 'merged_list_correlation.npy')
yticks = np.load(SOURCE_LIGHT / 'yticks_correlation.npy')

# Row labels (experimental CFMs) and column labels (simulated CFMs)
mapping = {**{val: f'CFM_domain_{val}' for val in Domains}, **{val: f'CFM_loop_{val}' for val in Loops}}
yticks = [mapping[val] for val in yticks]
renamed_merged_list = [f'CFM_{value}_block' if value <= 19 else f'CFM_{value - 20}_LE' for value in merged_list]

quadrants = [
    {'rows': (0, 10), 'cols': (0, 20), 'title': 'Quadrant 1'},
    {'rows': (0, 10), 'cols': (20, 46), 'title': 'Quadrant 2'},
    {'rows': (10, 27), 'cols': (0, 19), 'title': 'Quadrant 3'},
    {'rows': (10, 27), 'cols': (20, 46), 'title': 'Quadrant 4'}
]

# Plot
for a, quad in enumerate(quadrants, start=1):
    row_start, row_end = quad['rows']
    col_start, col_end = quad['cols']

    fig, ax = plt.subplots(figsize=(10, 8))
    im1 = ax.imshow(correlation_reorganized[row_start:row_end, col_start:col_end], cmap=Colormap_topic, vmin=-0.3, vmax=0.7)

    ax.set_xticks(range(col_end - col_start))
    ax.set_xticklabels(renamed_merged_list[col_start:col_end], rotation=45, ha='right')
    ax.set_yticks(range(row_end - row_start))
    ax.set_yticklabels(yticks[row_start:row_end])
    ax.set_title(quad['title'])

    cbar = plt.colorbar(im1, ax=ax, shrink=0.5)
    cbar.set_label('Correlation Coefficient')

    plt.tight_layout()
    plt.savefig(output_dir / f'Cross_Correlation_CFM_exp_simu_quadrant_{a}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 3c done\n")

#%% Figure 3d - Experiment vs simulation CFM examples
print("Plotting 3d: Experiment vs simulation CFM examples")
output_dir = REPO_ROOT / "Figures/Figure_3" / "Figure_3d"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data
CFM_domain_experiment = np.load(SOURCE_LIGHT / 'CFM_nonconsensus_domain_experiment.npy')
CFM_loop_experiment = np.load(SOURCE_LIGHT / 'CFM_nonconsensus_loop_experiment.npy')
CFM_simu_ATAC = np.load(SOURCE_LIGHT / 'CFM_consensus_domain_simu.npy')
CFM_simu_CTCF = np.load(SOURCE_LIGHT / 'CFM_consensus_loop_simu.npy')

# (matrices, index, title number, file name)
Examples = [
    (CFM_domain_experiment, 3, 3, 'Matrix_domain_exp_3'),                                 # CFM3
    (CFM_simu_ATAC, 19, 19, 'Matrix_domain_simu_19'),                                     # CFM19_BC
    (CFM_loop_experiment, 12, 12 + len(CFM_domain_experiment), 'Matrix_loop_exp_12'),     # CFM23
    (CFM_simu_CTCF, 6, 6 + len(CFM_simu_ATAC), 'Matrix_loop_simu_6'),                     # CFM6_LE
]

# Plot
for CFMs, idx, title_number, titlename in Examples:
    fig, ax = plt.subplots(1)
    im1 = ax.imshow(CFMs[idx, :, :], cmap=Colormap_topic)
    ax.set_title(f'Matrix_{title_number}')
    plt.colorbar(im1, ax=ax)

    plt.savefig(output_dir / f'{titlename}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("   ✓ 3d done\n")

#%% Figure 3e - Block-copolymer vs loop-extrusion scores
print("Plotting 3e: Block-copolymer vs loop-extrusion scores")
output_dir = REPO_ROOT / "Figures/Figure_3" / "Figure_3e"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (uses correlation_reorganized and yticks from 3c)
correlation_reorganized = np.load(SOURCE_LIGHT / 'correlation_CFM_consensus_exp_simu.npy')

# Relative score of block and LE: best correlation of each experimental CFM with each model
simulation_score = np.zeros((correlation_reorganized.shape[0], 2))
simulation_score[:, 0] = np.max(correlation_reorganized[:, 0:19], axis=1)   # block-copolymer
simulation_score[:, 1] = np.max(correlation_reorganized[:, 20:46], axis=1)  # loop extrusion

# Plot scatter
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(simulation_score[0:10, 1], simulation_score[0:10, 0], color='red', label='BC', s=100, linewidth=1, edgecolor='k')
ax.scatter(simulation_score[10:27, 1], simulation_score[10:27, 0], color='blue', label='LE', s=100, linewidth=1, edgecolor='k')
for i in range(27):
    ax.text(simulation_score[i, 1], simulation_score[i, 0], yticks[i], fontsize=9, ha='right')

ax.plot([0, 1], [0, 1], color='k', linestyle='--')   # x = y line
ax.set_ylabel('Block_copolymer_score')
ax.set_xlabel('LF_score')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
plt.savefig(output_dir / 'Scatter_simulation_score.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Scatter_simulation_score.svg', bbox_inches='tight')
plt.close()

print("   ✓ 3e done\n")

#%% Figure 3f-3k - HCT116 wild-type / RAD21-depleted
# Gautham panels
print("   ⚠ 3f-3k: panels from Gautham, See folder Script_Figure3_FK yet\n")

print("🎉 All Figure 3 panels completed!")
