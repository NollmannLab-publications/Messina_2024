#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S9 Messina et al. (mouse Hi-M)
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

default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

print("Starting Figure S9 plotting with light source data...\n")


def plot_toy_CFMs(decomposition_file, Subselection, output_dir, n_barcodes=25):
    """CFMs of a toy simulation, with their representativity (% of the total weight)."""
    Decomposition = np.load(SOURCE_LIGHT / decomposition_file)
    representativity_CFM_toy = np.sum(Decomposition, axis=1) * 100 / np.sum(Decomposition)

    for cfm in Subselection:
        matrix_tempo = fill_matrix_from_vector(Decomposition[cfm, :], n_barcodes)

        fig, ax = plt.subplots(1)
        im1 = ax.imshow(matrix_tempo, cmap=Colormap_topic)
        ax.set_title(f'CFM {cfm} - representativity = {representativity_CFM_toy[cfm]:.2f}%')
        plt.colorbar(im1, ax=ax)

        plt.savefig(output_dir / f'Matrix_topic_{cfm}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_all_CFM_matrices(CFMs, prefix, output_dir):
    """Every CFM of a stack of matrices (n_CFMs, n_barcodes, n_barcodes)."""
    for i in range(CFMs.shape[0]):
        fig, ax = plt.subplots(1)
        im1 = ax.imshow(CFMs[i, :, :], cmap=Colormap_topic)
        ax.set_title(f'CFM {i}')
        plt.colorbar(im1, ax=ax)

        plt.savefig(output_dir / f'Matrix_topic_{prefix}_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
# ===================================================

#%% Figure S9a - Block-copolymer simple simulation
print("Plotting S9a: Block-copolymer simple simulation")
output_dir = REPO_ROOT / "Figures/Figure_9_supp" / "Figure_S9a"
output_dir.mkdir(parents=True, exist_ok=True)

plot_toy_CFMs('components_topics_simple_heteropolymer_toy.npy', [0, 1, 2], output_dir)

print("   ✓ S9a done\n")

#%% Figure S9b - Polymer potentials
print("Plotting S9b: Polymer potentials")
output_dir = REPO_ROOT / "Figures/Figure_9_supp" / "Figure_S9b"
output_dir.mkdir(parents=True, exist_ok=True)


# Homopolymer repulsive potential
def polynomial_repulsive(r, r_rep, E_rep):
    term = (r * np.sqrt(6 / 7)) / r_rep
    return E_rep * (1 + (term ** 12) * (term ** 2 - 1) * (823543 / 46656))


# Heteropolymer repulsive + attractive potentials stitched together
def ssw(r, r_att, r_rep, E_rep_common, E_att_common, E_rep, E_att):
    r_min12 = np.sqrt(6 / 7)
    e_min12 = 46656 / 823543
    delta = (r_att - r_rep) / 2

    rshft = (r - r_rep - delta) / delta * r_min12
    E_att_total = E_att_common + E_att * 1
    E_att_final = - rshft ** 12 * (rshft ** 2 - 1) * (E_att_total / e_min12) - E_att_total

    rsc = r / r_rep * r_min12
    E_rep_total = E_rep_common + E_rep * 1
    E_rep_final = rsc ** 12 * (rsc ** 2 - 1) * (E_rep_total / e_min12) + E_rep_total

    return np.heaviside(r_rep - r, 1) * E_rep_final + np.heaviside(r - r_rep, 1) * E_att_final


# Repulsive potential (radius 1.05, max energy 3)
r_rep, E_rep = 1.05, 3.0
r = np.linspace(0, r_rep, 100)
plt.figure(figsize=(12, 7))
plt.plot(r, polynomial_repulsive(r, r_rep, E_rep), linewidth=3, color=default_colours[0] + "99")
plt.ylim((0, E_rep + 0.2))
plt.grid(lw=0.2)
plt.xlabel("Distance from center of the monomer", labelpad=10, fontsize=13)
plt.ylabel("Non-bonded repulsive potential", labelpad=10, fontsize=13)
plt.savefig(output_dir / 'polynomial_repulsive_potential-scale.svg', bbox_inches='tight')
plt.savefig(output_dir / 'polynomial_repulsive_potential-scale.png', dpi=300, bbox_inches='tight')
plt.close()

# Repulsive + attractive potential (radii 1.05 / 1.8)
r = np.linspace(0, 1.8, 100)
V = ssw(r, r_att=1.8, r_rep=1.05, E_rep_common=3.0, E_att_common=0, E_rep=0, E_att=0.2)
plt.figure(figsize=(12, 7))
plt.axhline(y=0, color="#000")
plt.plot(r, V, linewidth=3, color=default_colours[0] + "99")
plt.grid(lw=0.2)
plt.xlabel("Distance from center of the monomer", labelpad=10, fontsize=13)
plt.ylabel("Non-bonded potential", labelpad=10, fontsize=13)
plt.savefig(output_dir / 'heteropolymer_ssw_potential-scale.svg', bbox_inches='tight')
plt.savefig(output_dir / 'heteropolymer_ssw_potential-scale.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S9b done\n")

#%% Figure S9c - Loop-extrusion simple simulation
print("Plotting S9c: Loop-extrusion simple simulation")
output_dir = REPO_ROOT / "Figures/Figure_9_supp" / "Figure_S9c"
output_dir.mkdir(parents=True, exist_ok=True)

plot_toy_CFMs('components_topics_simple_extrusion_loop_toy.npy', [0, 1, 2], output_dir)

print("   ✓ S9c done\n")

#%% Figure S9d - LEF capture probability vs peak intensity
print("Plotting S9d: LEF capture probability vs peak intensity")
output_dir = REPO_ROOT / "Figures/Figure_9_supp" / "Figure_S9d"
output_dir.mkdir(parents=True, exist_ok=True)

# Load light data (CTCF ChIP-seq peaks, Pdx1 locus, pancreas)
raw_data = pd.read_csv(SOURCE_LIGHT / 'ctcf_chip-seq_pdx1_pancreas', sep="\t",
                       names=["Chromosome", "Start", "End", "ID", "Intensity",
                              "unkown1", "unkown2", "unkown3", "unkown4", "unkown5"])
intensities = raw_data["Intensity"].to_numpy()
midpt = np.mean(intensities) + 50
monotonic_slope = 0.01

# Plot
ints_range = np.linspace(0, np.max(intensities) + 10, 100)
plt.figure(figsize=(12, 7))
plt.plot(ints_range, sigmoid(ints_range, midpt, monotonic_slope), linewidth=3, color=default_colours[0] + "75")
plt.scatter(intensities, sigmoid(intensities, midpt, monotonic_slope), s=100, linewidth=2,
            facecolors="none", edgecolors=default_colours[0])
plt.xlabel("Peak intensity", labelpad=10, fontsize=12)
plt.ylabel("LEF capture probability", labelpad=10, fontsize=13)
plt.grid(False)
plt.savefig(output_dir / 'sigmoid_function_CTCF_capture_prob_log-scale.svg', bbox_inches='tight')
plt.savefig(output_dir / 'sigmoid_function_CTCF_capture_prob_log-scale.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ S9d done\n")

#%% Figure S9e - Consensus simulated CFMs, block-copolymer
print("Plotting S9e: Consensus simulated CFMs, block-copolymer")
output_dir = REPO_ROOT / "Figures/Figure_9_supp" / "Figure_S9e"
output_dir.mkdir(parents=True, exist_ok=True)

plot_all_CFM_matrices(np.load(SOURCE_LIGHT / 'CFM_consensus_domain_simu.npy'), 'domain_simu', output_dir)

print("   ✓ S9e done\n")

#%% Figure S9f - Consensus simulated CFMs, loop extrusion
print("Plotting S9f: Consensus simulated CFMs, loop extrusion")
output_dir = REPO_ROOT / "Figures/Figure_9_supp" / "Figure_S9f"
output_dir.mkdir(parents=True, exist_ok=True)

plot_all_CFM_matrices(np.load(SOURCE_LIGHT / 'CFM_consensus_loop_simu.npy'), 'loop_simu', output_dir)

print("   ✓ S9f done\n")

#%% Figure S9g - Loop-extrusion simulation (domains)
print("Plotting S9g: Loop-extrusion simulation (domains)")
output_dir = REPO_ROOT / "Figures/Figure_9_supp" / "Figure_S9g"
output_dir.mkdir(parents=True, exist_ok=True)

plot_toy_CFMs('components_topics_simple_extrusion_domain_toy.npy', [0, 1, 2], output_dir)

print("   ✓ S9g done\n")

print("🎉 All Figure S9 panels completed!")