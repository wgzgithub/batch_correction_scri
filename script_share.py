#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:15:57 2022

@author: zfd297
"""

import scanpy as sc
import numpy as np
import re


#Load datasets using scanpy
#change the path as your folder
control_data = sc.read_10x_mtx("/Users/zfd297/workspace/scrna_timetag/Raw_data/Control/filtered_feature_bc_matrix")
D5_data = sc.read_10x_mtx("/Users/zfd297/workspace/scrna_timetag/Raw_data/Sequential1_D5/filtered_feature_bc_matrix")
D14_data = sc.read_10x_mtx("/Users/zfd297/workspace/scrna_timetag/Raw_data/Sequential2_D14/filtered_feature_bc_matrix")
D30_data = sc.read_10x_mtx("/Users/zfd297/workspace/scrna_timetag/Raw_data/Sequential3_D30/filtered_feature_bc_matrix")

#remove duplicate genes
control_data.var_names_make_unique()
D5_data.var_names_make_unique()
D14_data.var_names_make_unique()
D30_data.var_names_make_unique()

#merge four datasets as large data
adata = control_data.concatenate(D5_data, join='inner',batch_key="batch")
adata = adata.concatenate(D14_data, join='inner',batch_key="batch")
adata = adata.concatenate(D30_data, join='inner',batch_key="batch")

#add batch labels
batch_label = np.array(["batch" for i in range(adata.shape[0])])
D0_batch = ["D0" for i in range(control_data.shape[0])]
D5_batch = ["D5" for i in range(D5_data.shape[0])]
D14_batch = ["D14" for i in range(D14_data.shape[0])]
D30_batch = ["D30" for i in range(D30_data.shape[0])]
all_batch = D0_batch+D5_batch+D14_batch+D30_batch
adata.obs["batch"] = all_batch

#remove doublelet
sc.external.pp.scrublet(adata, expected_doublet_rate=0.1)
adata = adata[~adata.obs['predicted_doublet'],]

#check counts and genes for cells
mito_genes = adata.var_names.str.startswith('mt-')
adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
adata.obs['n_counts'] = adata.X.sum(axis=1).A1
adata.obs['n_genes'] = np.count_nonzero(adata.X.A,axis=1)

#violinplot for basic information
sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
             jitter=0.4, multi_panel=True)


adata = adata[adata.obs['n_genes'] < 7000, :]
adata = adata[adata.obs['percent_mito'] < 0.4, :]

#normalize data
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
#remove batch effect
sc.pp.combat(adata)

#save normalized data to adata_raw
adata_raw = adata.copy()

#pick up highli variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata)
adata = adata[:,adata.var['highly_variable']]

#find cell population using louvain and embed data to umap
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.pca(adata)

sc.tl.louvain(adata,resolution=0.25)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['louvain'])
sc.pl.umap(adata, color=['batch'])


#check gene expression
dopaminergic_mker = ["Lmx1b","Foxa2","Th"]
hvgs = np.array(adata.var_names)[np.array(adata.var['highly_variable'])]
hvgs = list(hvgs) + dopaminergic_mker
hvgs = np.array(hvgs)
hvgs = np.unique(hvgs)
adata_raw_hvgs = adata_raw[:, hvgs]

adata_raw_hvgs.obsm["X_umap"] = adata.obsm["X_umap"]
adata_raw_hvgs.obs["louvain"] = adata.obs["louvain"]

sc.pl.umap(adata_raw_hvgs, color=dopaminergic_mker)




