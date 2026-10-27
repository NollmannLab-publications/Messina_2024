from os import path
from glob import glob

import numpy as np
import pandas as pd
from itertools import product
import umap
import h5py
import joblib as jl
from scipy.spatial.distance import squareform, pdist

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

components_all = np.concatenate((components_all_wt, components_all_mut), axis=0)


# UMAP parameters list for optimization
METRIC = "cosine"   # this is default; if using Jensen-Shannon divergence, set "precomputed" 
N_UMAP_COMPONENTS = 2
MIN_DIST_ALL = [0.01, 0.1, 1]               # second round: [0.2, 0.4, 0.6, 0.8]
NUM_NEIGHBORS_ALL = [2, 22, 42, 62, 82]     # second round: [4, 8, 12, 16, 20]
PARAMSETS = list(product(MIN_DIST_ALL, NUM_NEIGHBORS_ALL))

fig, axes = plt.subplots(nrows=len(MIN_DIST_ALL), ncols=len(NUM_NEIGHBORS_ALL), sharex=True, sharey=True)
fig.set_figwidth(len(NUM_NEIGHBORS_ALL) * 4)
fig.set_figheight(len(MIN_DIST_ALL) * 4)

# By default, pass pooled components; if using Jensen-Shannon divergence, a pairwise distance matrix is passed
data = squareform(pdist(components_all, metric="jensenshannon")) if METRIC == "precomputed" else components_all

for i, (MIN_DIST, NUM_NEIGHBORS) in enumerate(PARAMSETS):

    reducer = umap.UMAP(
        n_neighbors=NUM_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=METRIC,
        n_components=2,
        random_state=42,
    )
    embedding = reducer.fit_transform(data)
    jl.dump(reducer, path.join(dir_results,
                               "optimization",
                               f"{tag_wt}-{tag_mut}-combined-{num_components_wt}-{num_components_mut}_UMAP-components" \
                               f"_model_md={MIN_DIST}_nn={NUM_NEIGHBORS}.jl"))
    embedding_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])

    sns.scatterplot(data=embedding_df,
                    x="UMAP_1", y="UMAP_2",
                    s=10, alpha=0.4, color="#216eff", edgecolor="none",
                    legend=False,
                    ax=axes.flat[i])
    axes.flat[i].set_title(f"min_dist={MIN_DIST}, num_neighbors={NUM_NEIGHBORS}")

fig.tight_layout()
fig.savefig(path.join(dir_results,
                      "optimization",
                      f"{tag_wt}-{tag_mut}-combined-{num_components_wt}-{num_components_mut}_UMAP-components_grid.svg"))
plt.close(fig)