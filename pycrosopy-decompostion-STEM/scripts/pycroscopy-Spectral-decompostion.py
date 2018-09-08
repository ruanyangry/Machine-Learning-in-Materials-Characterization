# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Purpose: Spectral Unmixing

Methods: 1. Singular Value Decomposition
         2. Non-Negative Matrix Factorization
         3. Independent Component Analysis
         4. n-FINDR
         
STEM image
'''

# import modules

from __future__ import division, print_function, absolute_import, unicode_literals

# numpy
import numpy as np

# h5py
import h5py

# matplotlib
import matplotlib.pyplot as plt

# downloading files

import wget

# system
import os
import sys
import subprocess

# multivariate analysis

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF,FastICA

# defined function

def install(package):
	subprocess.call([sys.executable,"-m","pip","install",package])

try:
	import pyUSID as usid
except ImportError:
	print("pyUSID not found. Will install with pip.")
	import pip
	install('pyUSID')
	import pyUSID as usid
	
try:
	import pycroscopy as px
except ImportError:
	print("pycroscopy not found. Will install with pip")
	import  pip
	install('pycroscopy')
	import pycroscopy as px
	
# import pycroscopy plot function

from pycroscopy.viz import cluster_utils

# Dataset: STEM image

# First download dataset

#data_file_path=wget.download('https://ftp.ornl.gov/ftp_out/uP116H9fphkL/SuperImage2.h5')
data_file_path=r"SuperImage2.h5"
h5_file=h5py.File(data_file_path,mode='r+')

print("#-----------------------------------------#")
print(usid.hdf_utils.print_tree(h5_file))
print("#-----------------------------------------#")
print("\n")

# Get the image

h5_meas_grp=h5_file['Measurement_000']
h5_image=h5_meas_grp['Channel_000/Raw_Data']

# h5_image=h5_file['Measurement_000/Channel_000/Raw_Data']

# convert to USID dataset

h5_image=usid.USIDataset(h5_image)

# Getting a reference to the main spectral dataset
# what's the function of the reference?

h5_main=usid.USIDataset(h5_meas_grp['Channel_000/Raw_Data-FFT_Window_000/FFT_Data'])

[num_rows,num_cols]=h5_main.pos_dim_sizes
num_spect=h5_main.spec_dim_sizes[0]

print("#-----------------------------------------#")
print(h5_main)
print("#-----------------------------------------#")
print("\n")

# Plot the image and some spectra

fig,axes=plt.subplots()
axes.imshow(np.flipud(h5_image.get_n_dim_form()[:,:,0]))
axes.set_title('STEM Image of catalyst')
axes.set_xticks([])
axes.set_yticks([])
fig.savefig("spectral-decomposition-1.jpg",dpi=300)
plt.show()

FFT_mat=h5_main.get_n_dim_form().reshape(h5_main.pos_dim_sizes[0]*\
h5_main.pos_dim_sizes[1], h5_main.spec_dim_sizes[0], h5_main.spec_dim_sizes[1])

random_indices=np.random.randint(low=0,high=FFT_mat.shape[0],size=9)

# Now plot some FFT windows: sliding window method

usid.plot_utils.use_nice_plot_params()
usid.plot_utils.plot_map_stack(FFT_mat[random_indices,:,:],subtitle="FFT",title="")

plt.savefig("spectral-decomposition-2.jpg",dpi=300)
plt.show()

# SVD
# Output for SVD methods
# V - Eigenvectors sorted by variance in descending order
# U - corresponding abundance maps
# S - Variance or importance of each of these components

decomposer=px.processing.svd_utils.SVD(h5_main,num_components=100)
test=decomposer.test()
h5_svd_group=decomposer.compute()

h5_u=h5_svd_group['U']
h5_v=h5_svd_group['V']
h5_s=h5_svd_group['S']

# Since the two spatial dimensions (x, y) have been collapsed to one, 
# we need to reshape the abundance maps

abun_maps=np.reshape(h5_u[:,:25],(num_rows,num_cols,-1))

usid.plot_utils.plot_map_stack(abun_maps,num_comps=9,title="SVD Abundance Maps",\
reverse_dims=True,color_bar_mode='single',cmap='viridis',title_yoffset=0.95)

plt.savefig("spectral-decomposition-3.jpg",dpi=300)
plt.show()

# Visualize the variance / statistical importance of each component

usid.plot_utils.plot_scree(h5_s,title='Scree')

plt.savefig("spectral-decomposition-4.jpg",dpi=300)
plt.show()

# Visualize the eigenvectors

_=usid.plot_utils.plot_map_stack(h5_v[:9,:].reshape(9,h5_main.spec_dim_sizes[0],\
h5_main.spec_dim_sizes[1]),title='SVD Eigenvectors',evenly_spaced=False)

plt.savefig("spectral-decomposition-5.jpg",dpi=300)
plt.show()

# NMF

num_comps=4

# choose the NMFmodel from sklearn

nmf_model=NMF(n_components=num_comps,init="random",random_state=0)

# get the data,ensuring it is positive only.

data_mat=np.abs(h5_main[:])

# pycroscopy can handle this decompostion given an sklearn estimator object

decomposer=px.processing.decomposition.Decomposition(h5_main,estimator=nmf_model)
h5_nmf_group=decomposer.compute()

components=h5_nmf_group['Components']
projection=h5_nmf_group['Projection']

# nmf_coeffs_fig
usid.plot_utils.plot_map_stack(projection[:].reshape(h5_main.pos_dim_sizes[0],\
h5_main.pos_dim_sizes[1],-1), num_comps=9, title='NMF Abundance Maps',reverse_dims=True, \
color_bar_mode='single', cmap='viridis', title_yoffset=0.95)

plt.savefig("spectral-decomposition-6.jpg",dpi=300)
plt.show()

# nmf_components_fig

usid.plot_utils.plot_map_stack(components[:].reshape(num_comps, \
h5_main.spec_dim_sizes[0], h5_main.spec_dim_sizes[1]), title = '')

plt.savefig("spectral-decomposition-7.jpg",dpi=300)
plt.show()

# ICA: can be a useful tool when the signal arises from multiple independent sources

num_comps=4

ica_model=FastICA(n_components=num_comps,max_iter=200)

# get the data

data_mat=h5_main[:]

decomposer=px.processing.decomposition.Decomposition(h5_main, estimator = ica_model)
h5_ica_group=decomposer.compute()

decomposer=px.processing.decomposition.Decomposition(h5_main, estimator =ica_model)
h5_ica_group=decomposer.compute()

components=h5_ica_group['Components']
projection=h5_ica_group['Projection']

# ica_coeffs_fig

usid.plot_utils.plot_map_stack(projection[:].reshape(num_rows,num_cols,-1),\
num_comps=9, title='ICA Abundance Maps', reverse_dims=True,color_bar_mode='single',\
cmap='viridis', title_yoffset=0.95)

plt.savefig("spectral-decomposition-8.jpg",dpi=300)
plt.show()

# visualzie the components
# ica_components_fig

usid.plot_utils.plot_map_stack(components[:].reshape(num_comps,\
h5_main.spec_dim_sizes[0], h5_main.spec_dim_sizes[1]), title = '')

plt.savefig("spectral-decomposition-9.jpg",dpi=300)
plt.show()

# N-FINDR
# useful in situations where the basis spectra are already present in the dataset
# New modules: pysptools

from pysptools.eea import nfindr
import pysptools.abundance_maps as amp

num_endmembers=3

# finde the endmembers

comps=nfindr.NFINDR(h5_main[:].copy(),num_endmembers)[0]

# calculate abundance maps

nnls=amp.NNLS()
abundances=nnls.map(h5_main[:].copy().reshape(h5_main.pos_dim_sizes[0],\
h5_main.pos_dim_sizes[1], -1),comps)

# plot the components

usid.plot_utils.plot_map_stack(comps.reshape(num_endmembers,\
h5_main.spec_dim_sizes[0], h5_main.spec_dim_sizes[1]),title = '',\
color_bar_mode='each')

plt.savefig("spectral-decomposition-10.jpg",dpi=300)
plt.show()

# plot the abundances

usid.plot_utils.plot_map_stack(abundances.transpose(2,0,1), title = '', color_bar_mode='each')

plt.savefig("spectral-decomposition-11.jpg",dpi=300)
plt.show()

h5_file.close()

