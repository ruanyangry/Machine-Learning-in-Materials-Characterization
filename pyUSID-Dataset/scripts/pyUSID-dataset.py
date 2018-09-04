# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Reference: https://pycroscopy.github.io/pyUSID/auto_examples/beginner/\
plot_usi_dataset.html#sphx-glr-auto-examples-beginner-plot-usi-dataset-py

Purpose: Studying USID datasets

USID datasets format:

1. All spatial dimensions are collapsed to a single dimension
2. All spectroscopic dimensions are also collapsed to a single dimension

Data is stored as a two-dimensional (N x P) matrix with N spatial \
locations each with P spectroscopic data points

Matrix = N x P
N: spatial locations
P: spectroscopic data 

Main Datasets

1. the recorded physical quantity
2. units of the data
3. names of the position and spectroscopic dimensions
4. dimensionality of the data in its original N dimensional form

USIDatasets == USID Main Datasets
This type list in pyUSID

USIDatasets have all the capabilities of standard HDF5 / h5py Dataset objects

Examples:

Band Excitation Polarization Switching (BEPS) dataset acquired from \
advanced atomic force microscopes
'''

# Load necessary library

from __future__ import print_function, division, unicode_literals

# system

import os
from warnings import warn
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
try:
	import wget
except ImportError:
    warn('wget not found.  Will install with pip.')
    import pip
    install('wget')
    import wget
    
# read .h5 file

import h5py

# numpy

import numpy as np

# plot
 
import matplotlib.pyplot as plt

try:
    import pyUSID as usid
except ImportError:
    warn('pyUSID not found.  Will install with pip.')
    import pip
    install('pyUSID')
    import pyUSID as usid
    
# Load the dataset

url = 'https://raw.githubusercontent.com/pycroscopy/pyUSID/master/data/BEPS_small.h5'
h5_path = 'temp.h5'
_ = wget.download(url, h5_path, bar=None)

print("#--------------------------#")
print('Working on:\n' + h5_path)
print("#--------------------------#")
print("\n")

# Note that opening the file does not cause the contents to be \
# automatically loaded to memory.

h5_path='temp.h5'
h5_f=h5py.File(h5_path,mode='r')

# Through print_tree get the contents of .h5 file

print("#--------------------------#")
print("Contents of the H5 file:")
usid.hdf_utils.print_tree(h5_f)
print("#--------------------------#")
print("\n")

# We can check the datasets was the main set

h5_raw=h5_f["/Measurement_000/Channel_000/Raw_Data"]

print("#--------------------------#")
print(h5_raw)
print("h5_raw is a main dataset? {}".format(usid.hdf_utils.check_if_main(h5_raw)))
print("#--------------------------#")
print("\n")

# Convert Main Dataset to USIDataset
# All one needs for creating a USIDataset object is a Main dataset

pd_raw=usid.USIDataset(h5_raw)

print("#--------------------------#")
print(pd_raw)
print("#--------------------------#")
print("\n")

# Remember: USIDataset == Supercharged(h5py.Dataset)

print("#--------------------------#")
print(pd_raw==h5_raw)
print("#--------------------------#")
print("\n")

# Access to information stored in USIDataset

# pd_raw: USIDataset format
h5_spec_inds_1=pd_raw.h5_spec_inds
print("#--------------------------#")
print(h5_spec_inds_1)
print("#--------------------------#")
print("\n")

h5_spec_inds_2=usid.hdf_utils.get_auxiliary_datasets(h5_raw,"Spectroscopic_Indices")[0]

print("#--------------------------#")
print(h5_spec_inds_2)
print("#--------------------------#")
print("\n")

h5_spec_inds_3=usid.hdf_utils.get_auxiliary_datasets(h5_raw,"Spectroscopic_Indices")

print("#--------------------------#")
print(h5_spec_inds_3)
print("#--------------------------#")
print("\n")

print("#--------------------------#")
print(h5_spec_inds_1 == h5_spec_inds_2)
print("#--------------------------#")
print("\n")

print("#--------------------------#")
print("Desctiption of physical quantity in the main dataset:")
print(pd_raw.data_descriptor)
print("Position Dimension names and sizes:")
for name,length in zip(pd_raw.pos_dim_labels,pd_raw.pos_dim_sizes):
	print("{} : {}".format(name,length))
	
print("Position Dimensions:")
print(pd_raw.pos_dim_descriptors)

print("Spectroscopic Dimensions:")
print(pd_raw.spec_dim_descriptors)
print("#--------------------------#")
print("\n")

# values for each dimension
# position: get_pos_values()
# spectroscopic: get_spec_values()

# Get the DC_Offset

dim_name="DC_Offset"
dc_vec=pd_raw.get_spec_values(dim_name)
fig,axis=plt.subplots(figsize=(3.5,3.5))
axis.plot(dc_vec)
axis.set_xlabel("Points in dimension")
axis.set_title(dim_name)
fig.tight_layout()

plt.savefig("dataset-1.jpg",dpi=300)
plt.show()

# Reshaping to N dimensions

ndim_form=pd_raw.get_n_dim_form()
print("#--------------------------#")
print("Shape of the N dimensional form of the dataset:")
print(ndim_form.shape)
print("And these are the dimensions")
print(pd_raw.n_dim_labels)
print("#--------------------------#")
print("\n")

# Slicing 
# Lets try to get the spatial map for the following conditions without \
# loading the entire dataset in its N dimensional form and then slicing it

# 14th index of DC Offset
# 1st index of cycle
# 0th index of Field (remember Python is 0 based)
# 43rd index of frequency

# Input slice like dict

#spat_map_1,success=pd_raw.slice({"frequency":43,"DC_Offset":14,"Field":0,\
#"Cycle":1})

spat_map_1, success = pd_raw.slice({'Frequency': 43, 'DC_Offset': 14, 'Field': 0, 'Cycle': 1})

# Verification

spat_map_2=np.squeeze(ndim_form[:,:,43,14,0,1])
print("2D slicing == ND slicing: {}".format(np.allclose(spat_map_1,spat_map_2)))

# Interactive Visualization (Just in Jupyternotebook)

pd_raw.visualize(slice_dict={"Field":0,"Cycle":1})

plt.savefig("dataset-2.jpg",dpi=300)
plt.show()

# close input Datasets

h5_f.close()
os.remove(h5_path)



