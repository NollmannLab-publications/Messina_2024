from os import path
from glob import glob

from matplotlib import lines
import numpy as np
import pandas as pd
import umap
import h5py
import joblib as jl

import matplotlib.pyplot as plt
import seaborn as sns


# Results of topic model training were organized by a unique tag and the parameters used
tag_wt = "bintu2018Superresolution_Hs-HCT116-40-83_wt"
tag_mut = "bintu2018Superresolution_Hs-HCT116-40-83_RAD21del"
dir_root_training = "./training/"       # change to directory where are trainings are stored
dir_results = "./results/"       # change to directory where results are saved
num_components_wt = 20
num_components_mut = 15
maxiters = 4000
contact_threshold = 150

# Optimized UMAP parameters
MIN_DIST = 0.8
NUM_NEIGHBORS = 12

# Pooling components (a.k.a CFMs) and their source dataset labels; format for training results is HDF5 only
sources = []
components_all_wt = []
results = sorted(glob(path.join(dir_root_training,
                                tag_wt,
                                f"{num_components_wt}-components*threshold={contact_threshold:.1f}*maxiters={maxiters}*",
                                "results.h5")))
for fp in results:
    with h5py.File(fp) as handler:
        components_all_wt.append(handler["components"][:])
components_all_wt = np.concatenate(components_all_wt, axis=0)
sources.extend([tag_wt] * len(components_all_wt))
pseudocounts_wt = np.max(components_all_wt, axis=1)

components_all_mut = []
results = sorted(glob(path.join(dir_root_training,
                                tag_mut,
                                f"{num_components_mut}-components*{contact_threshold:.1f}*maxiters={maxiters}*",
                                "results.h5")))                                
for fp in results:
    with h5py.File(fp) as handler:
        components_all_mut.append(handler["components"][:])
components_all_mut = np.concatenate(components_all_mut, axis=0)
sources.extend([tag_mut] * len(components_all_mut))
pseudocounts_mut = np.max(components_all_mut, axis=1)

# Pseudocounts are not normalized, and doing so does not change the results
components_all = np.concatenate((components_all_wt, components_all_mut), axis=0)
pseudocounts_all = np.concatenate((pseudocounts_wt, -pseudocounts_mut), axis=0)

# Getting an optimal UMAP model already trained with the pooled data
model_id = f"{tag_wt}-{tag_mut}-combined-20-15_UMAP-components_model_md={MIN_DIST}_nn={NUM_NEIGHBORS}"
reducer = jl.load(path.join(dir_results,
                            "optimization",
                            f"{model_id}.jl"))
# Embedding CFMs in the UMAP space
embedding = reducer.transform(components_all)
embedding_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])

# Annotation of each point with source and max pseudocount
labels = ["Wild-type" if "wt" in src else "RAD21-depleted" for src in sources]
embedding_df["label"] = labels
embedding_df["pseudocount"] = pseudocounts_all

# NOTE saving format in svg for manuscript/panel preparation, with pseudounts
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
fd = {"fontfamily": "sans-serif", "fontsize": 30}
palette = sns.color_palette("RdBu", as_cmap=True)
fig = plt.figure(figsize=(12.5, 10))
ax = sns.scatterplot(
    data=embedding_df,
    x="UMAP_1", y="UMAP_2",
    hue="pseudocount",
    style="label",
    palette=palette,
    markers=["o", "X"],
    s=60,
    alpha=0.7,
    edgecolor="none",
    legend=False,
)
sns.despine()
plt.xlabel("UMAP_1", fontdict=fd, labelpad=20)
plt.ylabel("UMAP_2", fontdict=fd, labelpad=20)
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
norm = plt.Normalize(embedding_df["pseudocount"].min(), embedding_df["pseudocount"].max())
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.6, fraction=0.1, location="right", pad=0.1, aspect=25)
cb.set_ticks(ticks=[-3000, -2000, -1000, 0, 1000, 2000, 3000], labels=[3000, 2000, 1000, 0, 1000, 2000, 3000])
cb.ax.yaxis.set_label_position("right")
cb.ax.yaxis.set_ticks_position("left")
cb.set_label(label="maximum pseudocount", labelpad=22, fontsize=22,)
handles = []
col = sns.color_palette("RdBu", n_colors=10)[-2]
handles.append(lines.Line2D([], [],
                            color="#0d0d0d", linewidth=0,
                            marker="o", markersize=20,
                            markerfacecolor=col,
                            # markeredgecolor="#fff",
                            label="Wild-type"))
col = sns.color_palette("RdBu", n_colors=10)[1]
handles.append(lines.Line2D([], [],
                            color="#0d0d0d", linewidth=0,
                            marker="X", markersize=20,
                            markerfacecolor=col,
                            # markeredgecolor="#fff",
                            label="RAD21-depleted"))
ax.legend(handles=handles, loc="upper left", fontsize="20", frameon=False, borderpad=0.5)
plt.tight_layout()
# Replace saving path as required
path_to_save = path.join(dir_results,
                         "UMAP-components_with_sources_pseudocounts.svg")   # NOTE include file extension
plt.savefig(path_to_save, format="svg")
plt.close()