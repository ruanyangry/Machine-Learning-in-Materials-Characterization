# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Reference: https://pycroscopy.github.io/pycroscopy/auto_examples/plot_spectral_unmixing.html

Purpose: Spectral Unmixing
Data type: AFM

Machine Learning methods:

1. KMeans Clustering
2. Non-negative Matrix Factorization
3. Principal Component Analysis
'''

# First make sure the necessary module installed?

from __future__ import division,unicode_literals,print_function

# plot

import matplotlib.pyplot as plt

# numpy

import numpy as np

# read .h5 file

import h5py

# multivariaye analysis

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# system

import wget
import os
import subprocess
import sys

def install(package):
	subprocess.call([sys.executable,"-m","pip","install",package])
	
try:
	import pyUSID as usid
except ImportError:
	print('pyUSID not found.  Will install with pip.')
	import pip
	install('pyUSID')
	import pyUSID as usid

try:
	import pycroscopy as px
except ImportError:
	print("pycroscopy not found.  Will install with pip.")
	import pip
	install('pycroscopy')
	import pycroscopy as px
	
from pycroscopy.viz import cluster_utils

# Data type: advanced atomic force microscopes (AFM)
# Band Excitation Piezoresponse Force Microscopy (BE-PFM)
# Data format: a spectra was collected for each position in a \
# two dimensional grid of spatial locations
# 3 dimensional dataset === flattened to 2 dimensional matrix

# Summary: all statistical analysis, machine learning, spectral unmixing \
# algorithms, etc. only accept data that is formatted in the same manner \
# of [position x spectra] in a two dimensional matrix

# Download the datasets

data_file_path="temp_um.h5"
url='https://raw.githubusercontent.com/pycroscopy/pycroscopy/master/data/BELine_0004.h5'
data_file_path=wget.download(url,data_file_path,bar=None)

# Using h5py module read download data

h5_file=h5py.File(data_file_path,mode="r+")

# Get the tree structures of input data

print("#-----------------------------------#")
print("Contents of data file:")
usid.hdf_utils.print_tree(h5_file)
print("#-----------------------------------#")
print("\n")

h5_meas_grp=h5_file["Measurement_000"]

# Extracting some basic parameters

num_rows=usid.hdf_utils.get_attr(h5_meas_grp,'grid_num_rows')
num_cols=usid.hdf_utils.get_attr(h5_meas_grp,'grid_num_cols')

# Getting a reference to the main dataset

h5_main=usid.USIDataset(h5_meas_grp['Channel_000/Raw_Data'])
usid.hdf_utils.write_simple_attrs(h5_main,{'quantity':'Deflection','units':'V'})

# Extracting the X axis - vector of frequencies

h5_spec_vals=usid.hdf_utils.get_auxiliary_datasets(h5_main,'Spectroscopic_Values')[-1]
freq_vec=np.squeeze(h5_spec_vals.value)*1E-3

print("#-----------------------------------#")
print("Data currently of shape:",h5_main.shape)
print("#-----------------------------------#")
print("\n")

x_label="Frequency (kHz)"
y_label="Amplitude (a.u.)"

# SVD: eigenvector decomposition
# SVD results in three matrics

# V:Eigenvectors sorted by variance in descending order
# U:corresponding abundance maps
# S:Variance or importance of each of these components

# Using API in pycroscopy 

decomposer=px.processing.svd_utils.SVD(h5_main,num_components=100)
h5_svd_group=decomposer.compute()

# Get the matrics

h5_u=h5_svd_group["U"]
h5_v=h5_svd_group["V"]
h5_s=h5_svd_group["S"]

# Since the two spatial dimensions (x, y) have been collapsed to one, \
# we need to reshape the abundance maps

# Like 3D density map === 2D density matrix

abun_maps=np.reshape(h5_u[:,:25],(num_rows,num_cols,-1))

# plot

#usid.plot_utils.plot_map_stack(abun_maps,num_comps=9,title="SVD Abundance Maps",\
#reverse_dims=True,color_bar_mode="single",cmap="inferno",title_yoffset=0.95)
#
#plt.savefig("spectral-unmixing-1.jpg",dpi=300)
#
## Visualize the variance/statistical importance of each component
#
#usid.plot_utils.plot_scree(h5_s,title="Note the exponential drop of \
#variance with number of components")
#
#plt.savefig("spectral-unmixing-2.jpg",dpi=300)
#
#_=usid.plot_utils.plot_complex_spectra(h5_v[:9,:],x_label=x_label,y_label=y_label,\
#title="SVE Eigenvectors",evenly_spaced=False)
#
#plt.savefig("spectral-unmixing-3.jpg",dpi=300)
#plt.show()

# KMeans Clustering

num_clusters = 4

estimator = px.processing.Cluster(h5_main, KMeans(n_clusters=num_clusters))
h5_kmeans_grp = estimator.compute(h5_main)
h5_kmeans_labels = h5_kmeans_grp['Labels']
h5_kmeans_mean_resp = h5_kmeans_grp['Mean_Response']

cluster_utils.plot_cluster_h5_group(h5_kmeans_grp)

plt.savefig("spectral-unmixing-4.jpg",dpi=300)
plt.show()

# NMF: useful towards unmixing of spectral data
# Limit: It only works on data with positive real values

# V=WH determined W,H Vector

num_comps=4

# get the non-negative portion of the dataset

data_mat=np.abs(h5_main)

# NMF
model=NMF(n_components=num_comps,init="random",random_state=0)
model.fit(data_mat)

fig,axis=plt.subplots(figsize=(5.5,5))
usid.plot_utils.plot_line_family(axis,freq_vec,model.components_,\
label_prefix='NMF Component')
axis.set_xlabel(x_label,fontsize=12)
axis.set_ylabel(y_label,fontsize=12)
axis.set_title("NMF Components",fontsize=14)
axis.legend(bbox_to_anchor=[1.0,1.0],fontsize=12)

plt.savefig("spectral-unmixing-5.jpg",dpi=300)
plt.show()

h5_file.close()
os.remove(data_file_path)
