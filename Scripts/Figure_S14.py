#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
@author: Messina Olivier

Created on Oct, 2026

------------------------------------------------------------------------------------------------------------------------
This code is compiling all the pannels for Figure S14 Messina et al. (mouse Hi-M)
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

print("Starting Figure S14 plotting with light source data...\n")

# Single-allele distance maps (endocrine / exocrine) of each library
Distance_files = {
    'PDX1LR': {cell: f'Distance_maps_Pancreas_ND_{cell}_missing_bc_corrected_PDX1LR.npy' for cell in ['endocrine', 'exocrine']},
    'PDX1HR': {cell: f'Distance_maps_Pancreas_ND_{cell}_missing_bc_corrected_PDX1HR.npy' for cell in ['endocrine', 'exocrine']},
    'ISL1': {cell: f'Distance_maps_Pancreas_ND_{cell}_missing_bc_corrected_ISL1.npy' for cell in ['endocrine', 'exocrine']},
}


def load_maps(library, cell_types):
    """Single-allele PWD maps (n_alleles, n_barcodes, n_barcodes) of the given cell types, concatenated."""
    return np.concatenate([np.load(SOURCE_DATA / Distance_files[library][cell]) for cell in cell_types], axis=0)


def contact_map(library, cell_types):
    """Contact map at 100 nm; computed once, then read from a light file."""
    light_file = SOURCE_LIGHT / f"figureS14_contact_{library}_{'_'.join(cell_types)}.npy"
    if light_file.exists():
        return np.load(light_file)

    Matrix = load_maps(library, cell_types)
    print(f"   {library} {'+'.join(cell_types)}: {Matrix.shape[0]} alleles")
    contact, _ = PWD_to_Contact(np.transpose(Matrix, (1, 2, 0)), 1000, 100)
    np.save(light_file, contact)
    print(f"   saved light file: {light_file.name}")
    return contact


def differential_PWD(library):
    """log2(median PWD endocrine / median PWD exocrine); computed once, then read from a light file."""
    light_file = SOURCE_LIGHT / f"figureS14_differential_PWD_{library}.npy"
    if light_file.exists():
        return np.load(light_file)

    median_endocrine = np.nanmedian(load_maps(library, ['endocrine']), axis=0)
    median_exocrine = np.nanmedian(load_maps(library, ['exocrine']), axis=0)
    np.fill_diagonal(median_endocrine, np.nan)
    np.fill_diagonal(median_exocrine, np.nan)

    Differential_matrix = np.log2(median_endocrine / median_exocrine)
    np.save(light_file, Differential_matrix)
    print(f"   saved light file: {light_file.name}")
    return Differential_matrix


def volcano_PWD(library):
    """Differential PWD and Wilcoxon p-values (endocrine vs exocrine); computed once, then read from a light file."""
    light_file = SOURCE_LIGHT / f"figureS14_volcano_PWD_{library}.npz"
    if light_file.exists():
        volcano = np.load(light_file)
        return volcano['differential'], volcano['pvalues']

    Matrix_endocrine = np.transpose(load_maps(library, ['endocrine']), (1, 2, 0))
    Matrix_exocrine = np.transpose(load_maps(library, ['exocrine']), (1, 2, 0))
    differential, pvalues = VolcanoplotHiM(Matrix_endocrine, Matrix_exocrine)

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


def volcano_CFM(distribution_file, light_name, tissue_nb=1):
    """Median log10(odds ratio) and t-test p-value of each CFM; computed once, then read from a light file."""
    light_file = SOURCE_LIGHT / f"figureS14_volcano_CFM_{light_name}.npz"
    if light_file.exists():
        volcano = np.load(light_file)
        return volcano['x_values'], volcano['y_values']

    Distribution_CFMs_subset = np.load(SOURCE_DATA / distribution_file, allow_pickle=True)[tissue_nb, :]
    x_values = np.zeros(Distribution_CFMs_subset.shape[0])
    y_values = np.zeros(Distribution_CFMs_subset.shape[0])
    for i in range(Distribution_CFMs_subset.shape[0]):
        _, p_val = stats.ttest_1samp(Distribution_CFMs_subset[i], 0)
        y_values[i] = -np.log10(p_val)
        x_values[i] = np.nanmedian(Distribution_CFMs_subset[i])

    np.savez(light_file, x_values=x_values, y_values=y_values)
    print(f"   saved light file: {light_file.name}")
    return x_values, y_values


def plot_volcano_CFM(x_values, y_values, Domains, Loops, artifact, annotations, xlim, ylim, output_dir):
    """Volcano plot of CFMs: squares = domains, circles = loops, colored by the sign of the odds ratio."""
    cmap = plt.get_cmap('BrBG_r')
    colors = [cmap(0.0) if x < 0 else cmap(1.0) for x in x_values]

    x_values = x_values.copy()
    y_values = y_values.copy()
    x_values[artifact] = np.nan
    y_values[artifact] = np.nan

    fig, ax = plt.subplots()
    for i in range(len(x_values)):
        if i in Loops:
            ax.scatter(x_values[i], y_values[i], color=colors[i], alpha=0.8, s=100, marker='o')
        elif i in Domains:
            ax.scatter(x_values[i], y_values[i], color=colors[i], alpha=0.8, s=100, marker='s')

    plt.xlim(*xlim)
    plt.ylim(*ylim)

    for idx, label in annotations.items():
        ax.annotate(label, (x_values[idx], y_values[idx]), xytext=(0, 0), textcoords='offset points',
                    ha='center', color='black', fontsize=10)

    ax.invert_xaxis()
    plt.savefig(output_dir / 'Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Volcanoplot_CFM_Pancreas_Exocrine_Endocrine.svg', bbox_inches='tight')
    plt.close()


def plot_expression(tabula_muris, gene, gene_label, output_dir):
    """Violin plot of a gene expression (normalized by Actb) in exocrine (acinar) and beta cells."""
    normalization = tabula_muris[gene] / tabula_muris['Actb']
    Exocrine_cells = remove_outliers(normalization[tabula_muris['cell_ontology_class'] == 'pancreatic acinar cell'])
    Beta_cells = remove_outliers(normalization[tabula_muris['cell_ontology_class'] == 'type B pancreatic cell'])

    fig, ax = plt.subplots(figsize=(12, 6))
    vp = ax.violinplot([Exocrine_cells, Beta_cells], showmeans=False, showmedians=True)
    for pc, color in zip(vp['bodies'], ['blue', 'red']):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp[partname].set_color('black')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Exocrine Cells', 'Beta Cells'])
    ax.set_title(f'Violin Plot of {gene_label} Expression in Different Pancreatic Cell Types')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel(f'{gene_label} Expression (/Actb)')
    ax.set_ylim(-0.1, 1)
    plt.savefig(output_dir / f'Expression_{gene_label}_exocrine_beta_cells.svg', bbox_inches='tight')
    plt.savefig(output_dir / f'Expression_{gene_label}_exocrine_beta_cells.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_contact(contact, vmax, filename, output_dir):
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(contact, cmap=Colormap_topic, vmin=0, vmax=vmax)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title('Pancreas Contact 100nm')
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_differential(Differential_matrix, vmax, filename, output_dir):
    fig, axs = plt.subplots(figsize=(10, 8))
    im1 = axs.imshow(Differential_matrix, cmap='RdBu', vmin=-vmax, vmax=vmax)
    fig.colorbar(im1, ax=axs, shrink=1)
    plt.title('Ratio PWD matrix')
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
# ===================================================

#%% Figure S14a - Pdx1 / Plut expression
print("Plotting S14a: Pdx1 / Plut expression")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14a"
output_dir.mkdir(parents=True, exist_ok=True)
tabula_muris = pd.read_csv(SOURCE_LIGHT / 'Tabula_muris_filtered_gene_expression_data_S14.csv')
plot_expression(tabula_muris, 'Pdx1', 'Pdx1', output_dir)
plot_expression(tabula_muris, '2210019I11Rik', 'Pluto', output_dir)   # 2210019I11Rik = Plut (Pluto)

print("   ✓ S14a done\n")

#%% Figure S14b - Pdx1 low / high-resolution contact maps
print("Plotting S14b: Pdx1 low / high-resolution contact maps")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14b"
output_dir.mkdir(parents=True, exist_ok=True)

plot_contact(contact_map('PDX1LR', ['endocrine', 'exocrine']), 0.15, 'Contact_Pancreas_LR', output_dir)
plot_contact(contact_map('PDX1HR', ['endocrine', 'exocrine']), 0.15, 'Contact_Pancreas_HR', output_dir)

print("   ✓ S14b done\n")

#%% Figure S14c - Exocrine / beta cells, Pdx1 high resolution
print("Plotting S14c: Exocrine / endocrine cells, Pdx1 high resolution")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14c"
output_dir.mkdir(parents=True, exist_ok=True)

plot_contact(contact_map('PDX1HR', ['endocrine']), 0.15, 'Contact_Pancreas_HR_endocrine', output_dir)
plot_contact(contact_map('PDX1HR', ['exocrine']), 0.15, 'Contact_Pancreas_HR_exocrine', output_dir)

print("   ✓ S14c done\n")

#%% Figure S14d - Isl1 expression
print("Plotting S14d: Isl1 expression")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14d"
output_dir.mkdir(parents=True, exist_ok=True)

plot_expression(tabula_muris, 'Isl1', 'Isl1', output_dir)

print("   ✓ S14d done\n")

#%% Figure S14e - Isl1 exocrine / beta cell contact maps
print("Plotting S14e: Isl1 exocrine / endocrine contact maps")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14e"
output_dir.mkdir(parents=True, exist_ok=True)

plot_contact(contact_map('ISL1', ['exocrine']), 0.12, 'Contact_Pancreas_exocrine', output_dir)
plot_contact(contact_map('ISL1', ['endocrine']), 0.12, 'Contact_Pancreas_endocrine', output_dir)

print("   ✓ S14e done\n")

#%% Figure S14f - Isl1 differential PWD beta / exocrine
print("Plotting S14f: Isl1 differential PWD endocrine / exocrine")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14f"
output_dir.mkdir(parents=True, exist_ok=True)

plot_differential(differential_PWD('ISL1'), 0.35, 'Matrix_Endocrine_Exocrine_differential_PWD_log2_ISL1', output_dir)

print("   ✓ S14f done\n")

#%% Figure S14g - PWD volcano plot Pdx1
print("Plotting S14g: PWD volcano plot Pdx1")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14g"
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
plot_volcano_PWD(differential, pvalues, annotations, (-0.17, 0.17), (-10, 40),
                 'Volcano_plot_Endocrine_Exocrine_PWD', output_dir)

print("   ✓ S14g done\n")

#%% Figure S14h - CFM volcano plot Pdx1 high resolution
print("Plotting S14h: CFM volcano plot Pdx1 high resolution")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14h"
output_dir.mkdir(parents=True, exist_ok=True)

x_values, y_values = volcano_CFM('Distribution_OR_CFMs_Reference_00_Correlation_0.8_Pancreas_ND_exocrine_PDX1HR.npy', 'PDX1HR')
Pdx1_related = [2, 4, 12, 13, 14, 15, 19, 22, 28]
annotations = {i: f'CFM{i}-Pdx1' if i in Pdx1_related else f'CFM{i}' for i in range(30)}
plot_volcano_CFM(x_values, y_values,
                 Domains=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 Loops=[13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29],
                 artifact=[16, 26], annotations=annotations,
                 xlim=(-0.25, 0.25), ylim=(0, 100), output_dir=output_dir)

print("   ✓ S14h done\n")

#%% Figure S14i - PWD volcano plot Pdx1 high resolution
print("Plotting S14i: PWD volcano plot Pdx1 high resolution")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14i"
output_dir.mkdir(parents=True, exist_ok=True)

differential, pvalues = volcano_PWD('PDX1HR')
annotations = [
    (8, 10, 'Pdx1_CRE1'),
    (9, 10, 'Pdx1_CRE2'),
    (10, 10, 'Pdx1_CRE3'),
    (13, 10, 'Pdx1_CRE4')]
plot_volcano_PWD(differential, pvalues, annotations, (-0.25, 0.25), (0, 17),
                 'Volcano_plot_Endocrine_Exocrine_PWD_PDX1HR', output_dir)

print("   ✓ S14i done\n")

#%% Figure S14j - CFM volcano plot Isl1
print("Plotting S14j: CFM volcano plot Isl1")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14j"
output_dir.mkdir(parents=True, exist_ok=True)

x_values, y_values = volcano_CFM('Distribution_OR_CFMs_Reference_00_Correlation_0.8_Pancreas_ND_exocrine_ISL1.npy', 'ISL1')
Isl1_related = [6, 7, 12, 13, 14, 15, 17, 21]
annotations = {i: f'CFM{i}-Isl1' if i in Isl1_related else f'CFM{i}' for i in range(30)}
plot_volcano_CFM(x_values, y_values,
                 Domains=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                 Loops=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27],
                 artifact=[24, 28, 29], annotations=annotations,
                 xlim=(-0.55, 0.55), ylim=(0, 70), output_dir=output_dir)

print("   ✓ S14j done\n")

#%% Figure S14k - PWD volcano plot Isl1
print("Plotting S14k: PWD volcano plot Isl1")
output_dir = REPO_ROOT / "Figures/Figure_14_supp" / "Figure_S14k"
output_dir.mkdir(parents=True, exist_ok=True)

differential, pvalues = volcano_PWD('ISL1')
annotations = [
    (0, 6, 'Isl1_CRE1'),
    (4, 6, 'Isl1_CRE2'),
    (6, 6, 'Isl1_CRE3'),
    (10, 6, 'Isl1_CRE4')]
plot_volcano_PWD(differential, pvalues, annotations, (-0.35, 0.35), (0, 20),
                 'Volcano_plot_Endocrine_Exocrine_PWD_ISL1', output_dir)

print("   ✓ S14k done\n")

print("🎉 All Figure S14 panels completed!")
