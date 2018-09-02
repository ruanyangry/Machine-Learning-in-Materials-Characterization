# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Reference: https://pycroscopy.github.io/pycroscopy/auto_examples/plot_image_registration.

Purpose: Image alignment and registration

Compare similar positions in a series pictures

STM (scanning tunneling microscope) imaging 
3D scanning tunneling spectroscopic (STS) 

Often scientists find themselves wanting to compare data of various \
origins on the same sample that has a location of interest where multiple \
experiments have been carried out. Often, these data sets may not have the \
same resolution, not have been captured over the exact same region, be \
distorted in different ways due to drift, and even have different \
dimensionality. In this example, we will make use of algorithms which \
attempt to find the best alignment transformation between data sets.
'''

# First make sure the necessary module installed?

from __future__ import division,unicode_literals,print_function

# plot

import matplotlib.pyplot as plt

# numpy and scipy

import numpy as np
from scipy import interpolate,stats

# .h5 file

import h5py

# skimage

from skimage import transform

# system

from warnings import warn
import os
import subprocess
import sys

# defined install function

def install(package):
	subprocess.call([sys.executable,"-m","pip","install",package])
	
# try ... except

try:
	import wget
except ImportError:
	print("wget not found. will install with pip.")
	import pip
	install("wget")
	import wget
	
try:
	import pyUSID as usid
except ImportError:
	print("pyUSID not found. will install with pip")
	import pip
	install("pyUSID")
	import pyUSID as usid
	
# Defining a few handy functions that will be reused multiple times

def twin_image_plot(images,titles,cmap=plt.cm.viridis):
	'''
	Purpose: handy function that plots two images side by side with colorbars
	
	Parameters:
	
	images: list or array-like
	        list of two images defined as 2D numpy arrays
	titles: list ot array-like
	        list of the titles for each image
	cmap(optional):matplotlib.pyplot colormap object or string
	               colormap to use for displaying the images. 
	
	Returns:
	fig: Figure
	     Figure containing the plots 
	axes: 1D array_like of axes objects
	     Axes of the individual plots within 'fig'
	'''
	
	fig,axes=plt.subplots(ncols=2,figsize=(10,5))
	for axis,img,title in zip(axes.flat,images,titles):
		usid.plot_utils.plot_map(axis,img,cmap=cmap)
		axis.set_title(title)
	fig.tight_layout()
	return fig,axes
	
def normalize_image(image):
	'''
	Normalizes the provided image from 0 - 1
	
	Parameters:
	
	image: np.array object
	       Image to be normalized
	       
	Returns
	
	image: np.array object
	       Image normalized from 0 - 1
	'''
	return (image-np.amin(image))/(np.amax(image)-np.amin(image))
	
# Load h5 file

url='https://raw.githubusercontent.com/pycroscopy/pycroscopy/master/data/sts_data_image_registration.h5'
h5_path="temp.h5"
_=wget.download(url,h5_path,bar=None)
print("#---------------------------------------#")
print("Working on: \n"+h5_path)
print("#---------------------------------------#")
print("\n")

# Check data

with h5py.File(h5_path,mode="r") as h5_f:
	sts_spectral_data=h5_f['sts_spectra'][()]   # STS spectral data set
	high_res_topo=h5_f['stm_topography'][()]    # STM image
	sts_z_contr=h5_f['sts_z_contrast'][()]      # STS Z contrast image
	# print h5 file trees
	#usid.hdf_utils.print_tree(h5_f)
	
h5_f=h5py.File(h5_path,mode="r")
usid.hdf_utils.print_tree(h5_f)

# Normalize images

high_res_topo=normalize_image(high_res_topo)
sts_z_contr=normalize_image(sts_z_contr)

# Get the shape of datasets

print("#---------------------------------------#")
print("STS Spectra shape:",sts_spectral_data.shape)
print("STM Topography shape: ",high_res_topo.shape)
print("STS Z contrast shape: ",sts_z_contr.shape)
print("#---------------------------------------#")
print("\n")

# plot

fig,axes=twin_image_plot([high_res_topo,sts_z_contr],["STM topography",\
"STS Z contrast"])

plt.savefig("image-align-1.jpg",dpi=300)
plt.show()

# Interpolate image and Z channel data
# our goal is to maximize overlap between the two datasets

z_shape=sts_z_contr.shape
topo_shape=high_res_topo.shape

z_upscaler=interpolate.RectBivariateSpline(np.arange(z_shape[0]),np.arange(z_shape[1]),\
sts_z_contr)

z_upscaled=z_upscaler(np.arange(0,z_shape[0],z_shape[0]/topo_shape[0]),\
np.arange(0,z_shape[1],z_shape[1]/topo_shape[1]))

topo_downscaler=interpolate.RectBivariateSpline(np.arange(0,z_shape[0],z_shape[0]/topo_shape[0]),\
np.arange(0,z_shape[1],z_shape[1]/topo_shape[1]),high_res_topo)

topo_downscaled=topo_downscaler(np.arange(z_shape[0]),np.arange(z_shape[1]))

fig,axes=twin_image_plot([topo_downscaled,z_upscaled],["Downscaled Topography to Z contrast size",\
"Z contrast upscaled to Topography size"])

plt.savefig("image-align-2.jpg",dpi=300)
plt.show()

# preparing for image registration
# calculated transformation matrix between two images
# transform matrix == essentially a coordinate matching problem

# First normalize the up and downscaled images

z_upscaled=normalize_image(z_upscaled)
topo_downscaled=normalize_image(topo_downscaled)

# define the topography as the image that is fixed and the upscaled Z \
# contrast image as the one that moves during the image registration

fixed=high_res_topo
moving=z_upscaled

# Define the points that are common

src = [(536, 482),
       (100, 785),
       (745, 294),
       (604, 918)]
dst = [(561, 527),
       (193, 800),
       (749, 332),
       (678, 946)]
       
# First plot the two images

fig,axes=twin_image_plot([high_res_topo,z_upscaled],['Downscaled Topography to Z contrast size',\
'Z contrast upscaled to topography size'],cmap="gray")

# defined function used to plot marker in  image

def plot_markers(axis,coordinates,colors):
	for clr,point in zip(colors,coordinates):
		axis.scatter(point[0],point[1],color=clr,s=40)
		
# Add the markers in two images

pointer_colors=['b','y', 'g', 'r']
plot_markers(axes[0],src,pointer_colors)
plot_markers(axes[1],dst,pointer_colors)

plt.savefig("image-align-3.jpg",dpi=300)
plt.show()

# There have a problems: high_res_topo and topo_downscaled have different shapes.

fig,axes=twin_image_plot([high_res_topo,topo_downscaled],['Downscaled Topography to Z contrast size',\
'Z contrast downscaled to topography size'],cmap="gray")

# Add the markers in two images

pointer_colors=['b','y', 'g', 'r']
plot_markers(axes[0],src,pointer_colors)
plot_markers(axes[1],dst,pointer_colors)

plt.savefig("image-align-4.jpg",dpi=300)
plt.show()

# look at the overlaid raw data to gauge the difficulty of the \
# transformation prior to starting

fig,axis=plt.subplots(figsize=(5,5))
axis.imshow(fixed,cmap="Reds",alpha=0.8)
axis.imshow(moving,cmap="Blues",alpha=0.8)
axis.set_title("Images overlayed")

plt.savefig("image-align-5.jpg",dpi=300)
plt.show()

# Image registrations
# Below lists the transformation methods
# Just try

# Translation: translation types of distortion
# Rigid: translation and rotation types of distortion
# Similarity: translation, rotation and scale types of distortion
# Affine: translation, rotation, scale and shear types of distortion

# Using Pearson Correlation to determined which trnsformation methods best

trans_names=['similarity','affine','piecewise-affine','projective']

fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,10))


for tform_type,axis in zip(trans_names,axes.flat):
	tform=transform.estimate_transform(tform_type,np.array(src),np.array(dst))
	raw_corrected_Z=transform.warp(moving,inverse_map=tform.inverse,\
	output_shape=np.shape(moving))
	corr=stats.pearsonr(np.reshape(fixed,[1024*1024,1]),np.reshape(raw_corrected_Z,\
	[1024*1024,1]))[0][0]
	axis.set_title(tform_type+' - Pearson corr: '+str(np.round(corr,3)))
	axis.imshow(raw_corrected_Z)
	
fig.suptitle('Different transforms applied to the images',y=1.03)
fig.tight_layout()

plt.savefig("image-align-6.jpg",dpi=300)
plt.show()

os.remove(h5_path)
