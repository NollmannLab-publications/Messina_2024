from os import path
from glob import glob

from matplotlib.path import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import umap
import hdbscan
import h5py
import joblib as jl
from scipy.spatial import ConvexHull
import json

import matplotlib.pyplot as plt
import seaborn as sns
from visualization import plot_pyramid_topic


def plot_2d_triangle_bintu(matrix:np.ndarray, path_to_save:str, cmap="Reds", label=" "):
    """Plot a 2D matrix as a triangular matrix with a colorbar
    
    Parameters
    --------
    path_to_save : str
        path to save the image, image type is taken based on the extension
    cmap : 
        colormap for the plot (default: Reds)
    label : str
        label for the colorbar (default: )
    """

    from matplotlib.transforms import Affine2D

    # Take only upper triangle by masking the lower one
    data = squareform(matrix) if matrix.ndim == 1 else matrix[:]
    lower = np.tril(data, k=-1)
    mask = np.tril(np.ones_like(data, dtype=bool), k=-1)
    data_masked = np.where(mask, lower, np.nan)
    num_bc = data_masked.shape[0]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), constrained_layout=True)

    # Plot with pcolormesh (preferred over imshow for rotation)
    X, Y = np.meshgrid(np.arange(num_bc+1), np.arange(num_bc+1))
    pc = ax.pcolormesh(X, Y, data_masked, cmap=cmap, shading="auto")

    # Create a 45° rotation transform around center of the data and apply
    trans = Affine2D().rotate_deg_around(num_bc//2, num_bc//2, -45)
    pc.set_transform(trans + ax.transData)

    # Adjust axis limits to fit rotated content
    xf = 0.22 * num_bc
    yf = 0.46 * num_bc
    ax.set_xlim(-xf, num_bc+xf)
    ax.set_ylim(yf, num_bc*1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    
    # TODO align it to the triangle edge
    cb = fig.colorbar(pc, shrink=0.75, aspect=20, orientation="horizontal")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.yaxis.set_ticks_position("left")
    cb.set_label(label=label, labelpad=12, fontsize=16,)
    min_val = np.nanmin(data_masked)
    max_val = np.nanmax(data_masked)
    mid_val = (max_val + min_val) / 2
    cb.set_ticks(ticks=[min_val, mid_val, max_val],
                labels=[f"{min_val:.2f}", f"{mid_val:.2f}", f"{max_val:.2f}"],
                fontsize=12)
    
    fig.savefig(path_to_save, dpi=300)
    plt.close(fig)


def plot_2d_square_bintu(matrix:np.ndarray, path_to_save:str,
                         mark_pos: list | np.ndarray = None,
                         title: str = "", cmap: str = "Reds", cb_label: str = ""):
    """Plot a 2D matrix as a square symmetric matrix with a colorbar
    
    Parameters
    --------
    path_to_save : str
        path to save the image, image type is taken based on the extension
    cmap : str
        colormap for the plot (default: Reds)
    label : str
        label for the colorbar (default: )
    """

    # Take only upper triangle by masking the lower one
    data = squareform(matrix) if matrix.ndim == 1 else matrix[:]
    num_bc = data.shape[0]
    start_bc = 39
    step_bc = 5
    if mark_pos is None:
        mark_pos = [40, 43, 46, 47, 48, 49, 50, 52, 54, 56, 58, 59, 60, 62, 65,\
                    66, 67, 69, 70, 71, 74, 76, 77, 78, 79, 81, 82]
    mark_pos = [m - start_bc for m in mark_pos]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.25, 7), constrained_layout=True)

    # # Plot with pcolormesh (preferred over imshow for rotation), but edge linewidths don't work correctly
    # x_vals, y_vals = np.meshgrid(np.arange(num_bc+1), np.arange(num_bc+1))
    # pc = ax.pcolormesh(x_vals, y_vals, data, cmap=cmap, edgecolors="#b1b1b1", linewidths=0.04, shading="auto")
    # ax.invert_yaxis()

    # Plotting with imshow
    pc = ax.imshow(data, cmap=cmap, interpolation="none", aspect="equal")
    
    ax.scatter(mark_pos, mark_pos, marker="o", s=32, color=sns.color_palette("RdBu", n_colors=20)[-1])
    ax.set_aspect("equal")
    
    ax.set_title(title, fontsize=24, pad=20, fontweight="bold")
    ax.text(x=1, y=num_bc-3, s="WT", fontsize=20)
    ax.text(x=num_bc-15, y=4, s="dRAD21", fontsize=20)
    ax.text

    ax.set_xticks(ticks=np.arange(0, num_bc, step_bc), labels=range(start_bc, start_bc+num_bc, step_bc))
    ax.set_yticks(ticks=np.arange(0, num_bc, step_bc), labels=range(start_bc, start_bc+num_bc, step_bc))
    ax.set_xticks(ticks=np.arange(0.5, num_bc, 1), minor=True)
    ax.set_yticks(ticks=np.arange(0.5, num_bc, 1), minor=True)
    ax.tick_params(axis="both", which="major", right=True, left=True, top=True, bottom=True)
    ax.tick_params(axis="both", which="minor", right=False, top=False, bottom=False, left=False)
    ax.tick_params(axis="both", which="major", labelsize=16, width=1)
    ax.tick_params(axis="both", which="minor", grid_color="0.9", grid_linewidth=0.3)
    ax.grid(which="minor")
    
    ax.spines[:].set(color="0.6", linewidth=0.8)
    
    cb = fig.colorbar(pc, shrink=0.7, aspect=22, orientation="horizontal")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.yaxis.set_ticks_position("left")
    cb.set_label(label=cb_label, labelpad=12, fontsize=20,)
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    ticks = [data_min, data_min+data_max/2, data_max]
    cb.set_ticks(ticks=ticks,
                labels=[f"{t:.2f}" for t in ticks],
                fontsize=18)
    cb.ax.tick_params(length=3, width=1.25)
    
    fig.savefig(path_to_save, dpi=300)
    plt.close(fig)


# Results of topic model training were organized by a unique tag and the parameters used
tag_wt = "bintu2018Superresolution_Hs-HCT116-40-83_wt"
tag_mut = "bintu2018Superresolution_Hs-HCT116-40-83_RAD21del"
tag = "bintu2018Superresolution_Hs-HCT116-40-83_combined"
dir_root_training = "./training/"
dir_results = "./results/"
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

# Getting an optimal UMAP model already trained with the pooled data
model_id = f"{tag_wt}-{tag_mut}-combined-20-15_UMAP-components_model_md={MIN_DIST}_nn={NUM_NEIGHBORS}"
reducer = jl.load(path.join(dir_results,
                            "optimization",
                            f"{model_id}.jl"))
# Embedding wild-type CFMs in the UMAP space
embedding_wt = reducer.transform(components_all_wt)
embedding_wt_df = pd.DataFrame(embedding_wt, columns=["UMAP_1", "UMAP_2"])

# Manually optimized HDBSCAN parameters for the pooled components from WT
MIN_CLUSTER_SIZE = len(components_all_wt) // 50
MIN_SAMPLES = 2
CLUSTER_SELECTION_EPSILON = 0.45
CLUSTER_SELECTION_METHOD = "leaf"
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
    cluster_selection_method=CLUSTER_SELECTION_METHOD,
    prediction_data=True,
)
clusterer = clusterer.fit(embedding_wt)
embedding_wt_df["cluster"] = clusterer.labels_.astype(int)

clusters_unique = sorted(list(np.unique(clusterer.labels_).astype(int)))
clusters_unique.pop(0)
clusters_unique = np.asarray(clusters_unique)
num_clusters = len(clusters_unique)     # NOTE noise is NOT included as a cluster

embedding_clusters_wt_df = embedding_wt_df[embedding_wt_df["cluster"] != -1].copy()
embedding_noise_df = embedding_wt_df[embedding_wt_df["cluster"] == -1].copy()

# NOTE saving format in svg for manuscript/panel preparation
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
fd = {"fontfamily": "sans-serif", "fontsize": 30}

# Choosing a suitable color palette
palette = sns.color_palette("hls", n_colors=num_clusters-1)
palette.append("#BCDCE0")
cluster_to_color = dict(zip(clusters_unique, palette[::-1]))

# Plotting only WT pooled components annotated by cluster label
plt.figure(figsize=(11, 10))
sns.scatterplot(
data=embedding_clusters_wt_df,
x="UMAP_1", y="UMAP_2",
hue="cluster",
palette=cluster_to_color,
s=60,
alpha=0.85,
legend=False,
)
sns.scatterplot(
    data=embedding_noise_df,
    x="UMAP_1", y="UMAP_2",
    color="gainsboro",
    s=60,
    alpha=0.85,
)
sns.despine()
plt.xlabel("UMAP_1", fontdict=fd, labelpad=20)
plt.ylabel("UMAP_2", fontdict=fd, labelpad=20)
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
# marking cluster labels near the centroid of each cluster
centroids = embedding_clusters_wt_df.groupby("cluster")[["UMAP_1", "UMAP_2"]].mean()
for cluster_id, row in centroids.iterrows():
    plt.text(row["UMAP_1"]+0.1, row["UMAP_2"]+0.1, str(cluster_id), fontsize=18, weight='bold',
                ha='center', va='center', color='black',
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.4'))
# Finding the areas occupied by clusters using convex hulls to later see which points in dRAD21 fall within them
hulls_all = []
points_wt_all = []
for cluster_id in clusters_unique:
    df_by_custer_id = embedding_clusters_wt_df[embedding_clusters_wt_df["cluster"] == cluster_id]
    points_wt = df_by_custer_id[["UMAP_1", "UMAP_2"]].to_numpy()
    points_wt_all.append(points_wt)
    hull = ConvexHull(points_wt)
    hulls_all.append(hull)
    # for simplex in hull.simplices:    # to draw the hulls in the plot
    #     plt.plot(points_wt[simplex, 0], points_wt[simplex, 1], color=cluster_to_color[cluster_id])

plt.tight_layout()
plt.savefig(path.join(dir_results, "UMAP_components-wt_clusters.svg"), format="svg")
plt.close()

# Embedding the dRAD21 pooled components
embedding_mut = reducer.transform(components_all_mut)
embedding_mut_df = pd.DataFrame(embedding_mut, columns=["UMAP_1", "UMAP_2"])
points_mut_from_df = embedding_mut_df[["UMAP_1", "UMAP_2"]].to_numpy()  # get only UMAP point coordinates

plt.figure(figsize=(11, 10))
# For each cluster and corresponding convex hull from WT UMAP, check which dRAD21 UMAP points fall within the hull
# and label them accordingly, rest will be noise
points_mut_all = []     # points that fall within are noted to separate them from the rest
embedding_mut_df["cluster"] = -1       # initialize DataFrame with everything labeled as noise
for cluster_id, points_wt, hull in zip(clusters_unique, points_wt_all, hulls_all):
    polygon_path = Path(points_wt[hull.vertices])
    is_inside = polygon_path.contains_points(points_mut_from_df)
    points_mut = points_mut_from_df[is_inside]  # points that fall within the hull are extracted
    points_mut_all.append(points_mut)
    plt.scatter(points_mut[:, 0], points_mut[:, 1],
                s=60, color=cluster_to_color[cluster_id], edgecolors="white")   # plot the points in same color as WT
    for simplex in hull.simplices:  # to draw the hulls in the plot
        plt.plot(points_wt[simplex, 0], points_wt[simplex, 1], color=cluster_to_color[cluster_id], ls=":")
    embedding_mut_df.loc[is_inside, "cluster"] = cluster_id
for cluster_id, row in centroids.iterrows():    # to label them with the correct cluster from WT
    plt.text(row["UMAP_1"]+0.1, row["UMAP_2"]+0.1, str(cluster_id), fontsize=18, weight='bold',
                ha='center', va='center', color='black',
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.4'))

# Remaining dRAD21 points are plotted as noise
points_mut_all = np.vstack(points_mut_all).tolist()
points_mut_from_df = points_mut_from_df.tolist()
points_mut_outside = np.asarray([pt for pt in points_mut_from_df if pt not in points_mut_all])
plt.scatter(points_mut_outside[:, 0], points_mut_outside[:, 1],
            s=60, color="gainsboro", edgecolors="white")

# Clustering on dRAD21 pooled components
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
    cluster_selection_method=CLUSTER_SELECTION_METHOD,
    prediction_data=True,
)
clusterer = clusterer.fit(embedding_mut)
# manually assign label to the most important cluster (in terms of pseudocounts and number of points)
is_to_be_assigned = np.logical_and(clusterer.labels_ == 0, embedding_mut_df["cluster"] == -1)
ckey = max(embedding_clusters_wt_df["cluster"]) + 1     # cluster label continues WT label list
embedding_mut_df.loc[is_to_be_assigned, "cluster"] = ckey
# Get points to plot from the updated DataFrame
df_section = embedding_mut_df[embedding_mut_df["cluster"] == ckey][["UMAP_1", "UMAP_2"]].to_numpy()
centroid = df_section.mean(axis=0)
plt.scatter(df_section[:, 0], df_section[:, 1],     # plotting the points
            s=60, color="#131342", edgecolors="white")
plt.text(centroid[0]+0.1, centroid[1]+0.1, ckey, fontsize=18, weight='bold',
         ha='center', va='center', color='black',   # indicating the label
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.4'))

sns.despine()
plt.xlabel("UMAP_1", fontdict=fd, labelpad=20)
plt.ylabel("UMAP_2", fontdict=fd, labelpad=20)
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.savefig(path.join(dir_results, "UMAP_components-RAD21del_clusters.svg"), format="svg")
plt.close()

# Saving DataFrame results to avoid unnecessary recomputation
embedding_wt_df.to_csv(path.join(dir_results, "embedding_components_WT.csv"), index=False)
embedding_mut_df.to_csv(path.join(dir_results, "embedding_components_dRAD21.csv"), index=False)

# Plotting the most important clusters in either case manually
font_info = {"title": 12,       # for triangular representation
             "tick_label": 8,
             "legend": 8}
req_clusters = {ckey: "mut", 14: "wt", 10: "wt", 11: "wt", 5: "wt"}      # cluster-source mapping
src_map = {"wt": embedding_clusters_wt_df, "mut": embedding_mut_df}
for cluster, src in req_clusters.items():
    df = src_map[src]
    idxs = df.index[df["cluster"] == cluster]
    comps = components_all_wt[idxs,:] if src == "wt" else components_all_mut[idxs,:]
    prop_recurrence = len(comps) / len(components_all_wt if src == "wt" else components_all_mut)
    components_avg = np.mean(comps, axis=0)     # consensus representation of CFMs within a cluster by averaging
    # # NOTE use plot_pyramid_topic for odds ratio computation based on loci; plotting needs to be improved
    # plot_pyramid_topic(topic_n=cluster,
    #                    topic=components_avg,
    #                    topic_repr=prop_recurrence,
    #                    lib=pd.DataFrame({"barcode_id": np.arange(39, 83)}),
    #                    locus_dict=json.load(open("./helper/json/bintu2.json")),
    #                    figname=path.join(dir_results, "maps", f"temp_component-avg_{src}_{cluster}.png"),
    #                    font_info=font_info,
    #                    line_sep=0.02)
    plot_2d_triangle_bintu(matrix=components_avg,   # NOTE this is a simpler representation used for the manuscript
                           path_to_save=path.join(dir_results,
                                                  "maps",
                                                  f"{tag_mut}_UMAP-components_cluster-{cluster}_avg.png"),
                           label="mean pseudocount")

# Plot square representations for comparison between WT and dRAD21 based on clusters
comparison_pairs = zip(clusters_unique, clusters_unique)
for i, (c0, c1) in enumerate(comparison_pairs):
    idxs_wt = embedding_clusters_wt_df.index[embedding_clusters_wt_df["cluster"] == c0]
    idxs_mut = embedding_mut_df.index[embedding_mut_df["cluster"] == c1]
    components_wt_avg = np.mean(components_all_wt[idxs_wt,:], axis=0)       # consensus representation of CFMs
    components_mut_avg = np.mean(components_all_mut[idxs_mut,:], axis=0)    # same for dRAD21
    assert components_wt_avg.shape[0] == components_mut_avg.shape[0]
    components_joined_avg = np.tril(squareform(components_wt_avg))
    components_joined_avg[np.triu_indices_from(components_joined_avg, k=1)] = components_mut_avg
    np.fill_diagonal(components_joined_avg, np.nan)
    plot_2d_square_bintu(matrix=components_joined_avg,
                         path_to_save=path.join(dir_results,
                                                "maps",
                                                f"wt_RAD21del_mean-component-comparison_cluster-{i}.svg"),
                         title=f"Cluster {i}",
                         cb_label="mean pseudocount")