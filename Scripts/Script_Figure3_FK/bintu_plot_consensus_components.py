from os import path
from glob import glob

import numpy as np
import pandas as pd
import json
import h5py

import matplotlib.pyplot as plt
from visualization import plot_pyramid_topic


def _plot_routine(df:pd.DataFrame, cluster:int, data:np.ndarray, src:str = ""):
    idxs = df.index[df["cluster"] == cluster]
    data_filt = data[idxs,:]
    print(f"Found {len(data_filt)} components for cluster {cluster} in {src}")
    prop = len(data_filt) / len(data)
    data_filt_avg = np.mean(data_filt, axis=0)
    font_info = {"title": 12,       # for triangular representation
             "tick_label": 8,
             "legend": 8}
    plot_pyramid_topic(topic_n=cluster,
                       topic=data_filt_avg,
                       topic_repr=prop,
                       lib=pd.DataFrame({"barcode_id": np.arange(39, 83)}),
                       locus_dict=json.load(open("./helper/json/bintu.json")),
                       figname=path.join(dir_results, "maps", f"component-avg_{src}_{cluster}.svg"),
                       font_info=font_info,
                       line_sep=0.02)


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

# Pooling components (a.k.a CFMs) and their source dataset labels; format for training results is HDF5 only
components_all_wt = []
results = sorted(glob(path.join(dir_root_training,
                                tag_wt,
                                f"{num_components_wt}-components*threshold={contact_threshold:.1f}*maxiters={maxiters}*",
                                "results.h5")))
for fp in results:
    with h5py.File(fp) as handler:
        components_all_wt.append(handler["components"][:])
components_all_wt = np.concatenate(components_all_wt, axis=0)

components_all_mut = []
results = sorted(glob(path.join(dir_root_training,
                                tag_mut,
                                f"{num_components_mut}-components*{contact_threshold:.1f}*maxiters={maxiters}*",
                                "results.h5")))                                
for fp in results:
    with h5py.File(fp) as handler:
        components_all_mut.append(handler["components"][:])
components_all_mut = np.concatenate(components_all_mut, axis=0)

# Loading UMAP embedding results from existing DataFrame
embedding_wt_df = pd.read_csv(path.join(dir_results, "embedding_components_WT.csv"))
embedding_mut_df = pd.read_csv(path.join(dir_results, "embedding_components_dRAD21.csv"))

# Plotting all the clusters
clusters_all = set(embedding_wt_df["cluster"].unique())
clusters_all.update(embedding_mut_df["cluster"].unique())
clusters_all.remove(-1)
for cluster in clusters_all:
    if cluster in embedding_wt_df["cluster"].unique():
        _plot_routine(df=embedding_wt_df,
                      cluster=cluster,
                      data=components_all_wt,
                      src="WT")
    if cluster in embedding_mut_df["cluster"].unique():
        _plot_routine(df=embedding_mut_df,
                      cluster=cluster,
                      data=components_all_mut,
                      src="dRAD21")