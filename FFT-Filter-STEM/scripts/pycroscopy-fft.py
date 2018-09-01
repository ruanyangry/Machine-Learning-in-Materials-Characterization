# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Reference: https://pycroscopy.github.io/pycroscopy/auto_examples/\
plot_fft_2d_filtering.html#sphx-glr-auto-examples-plot-fft-2d-filtering-py

Purpose: FFT & Filtering of Atomically Resolved Images

FFT advantage:  The power of the Fast Fourier Transform (FFT) is due in \
part to (as the name suggests) its speed and also to the fact that \
complex operations such as convolution and differentiation/integration \
are made much simpler when performed in the Fourier domain

Examples:

1. Load a image
2. Fourier transform it
3. Apply a smoothing filter
4. Transform it back
'''

# First make sure the necessary module installed?

from __future__ import division,unicode_literals,print_function

# plot

import matplotlib.pyplot as plt

# numpy

import numpy as np
import numpy.fft as npf

# system

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
	
# Download input data file.

data_file_path="temp_STEM_STO.txt"

url="https://raw.githubusercontent.com/pycroscopy/pycroscopy/master/data/STEM_STO_2_20.txt"

_=wget.download(url,data_file_path,bar=None)
#_=wget.download(url,data_file_path,bar=True)

# Get the data stored in download file.

image_raw=np.loadtxt(data_file_path,dtype="str",delimiter="\t")

# delete the temporarily downloaded file:

os.remove(data_file_path)

# convert the file from a string array to a numpy array of floating point numbers

image_raw=np.array(image_raw)
image_raw=image_raw[0:,0:-1].astype(np.float)
#image_raw=image_raw[:,:].astype(np.float) --- wrong

print("#----------------------------------------#")
print(" raw data shape : = {}".format(image_raw.shape))
print("#----------------------------------------#")
print("\n")

# subtract out the mean of the image

image_raw=image_raw-np.mean(image_raw)

# keeping track of units between transformations

x_pixels,y_pixels=np.shape(image_raw)
x_edge_length=5.0 # nm
y_edge_length=5.0 # nm
x_sampling=x_pixels/x_edge_length
y_sampling=y_pixels/y_edge_length
x_axis_vec=np.linspace(-x_edge_length/2.0,x_edge_length/2.0,x_pixels)
y_axis_vec=np.linspace(-y_edge_length/2.0,y_edge_length/2.0,y_pixels)
x_mat,y_mat=np.meshgrid(x_axis_vec,y_axis_vec)

# The axes in the Fourier domain are defined below

u_max=x_sampling/2
v_max=y_sampling/2
u_axis_vec=np.linspace(-u_max/2,u_max/2,x_pixels)
v_axis_vec=np.linspace(-v_max/2,v_max/2,y_pixels)

# below: matrices of u-positions and v-positions
u_mat,v_mat=np.meshgrid(u_axis_vec,v_axis_vec)

# STEM image of STO

fig,axis=plt.subplots(figsize=(5,5))
_=usid.plot_utils.plot_map(axis,image_raw,cmap=plt.cm.inferno,clim=[0,6],\
x_vec=x_axis_vec,y_vec=y_axis_vec,num_ticks=5)
axis.set_title("original image of STO captured via STEM")
plt.savefig("fft-1.jpg",dpi=300)
plt.show()

# FFT transform by numpy.fft2

fft_image_raw=npf.fft2(image_raw)

# Plotting the magnitude 2D-FFT on a vertical log scales shows something\
# unexpected: there appears to be peaks at the corners and no information\
# at the center. This is because the output for the 'fft2' function flips\
# the frequency axes so that low frequencies are at the ends, and the \
# highest frequency is in the middle.

fig,axis=plt.subplots(figsize=(5,5))
_=usid.plot_utils.plot_map(axis,np.abs(fft_image_raw),cmap=plt.cm.OrRd,\
clim=[0,3E+3])
axis.set_title("FFT2 of Image")
plt.savefig("fft-2.jpg",dpi=300)
plt.show()

# To correct this, use the 'fftshift' command. fftshift brings the lowest\
# frequency components of the FFT back to the center of the plot

fft_image_raw=npf.fftshift(fft_image_raw)
fft_abs_image_raw=np.abs(fft_image_raw)

def crop_center(image,cent_size=128):
	return image[image.shape[0]//2-cent_size//2:image.shape[0]//2+cent_size//2,\
	image.shape[1]//2-cent_size//2:image.shape[1]//2+cent_size//2]
	
# After the fftshift, the FFT looks right

fig,axes=plt.subplots(ncols=2,figsize=(10,5))
for axis,img,title in zip(axes,[fft_abs_image_raw,crop_center(fft_abs_image_raw)],\
["FFT after fftshift-ing","Zoomed view around origin"]):
	_=usid.plot_utils.plot_map(axis,img,cmap=plt.cm.OrRd,clim=[0,1E+4])
	axis.set_title(title)

fig.tight_layout()

plt.savefig("fft-3.jpg",dpi=300)
plt.show()

# redefine the Fourier domain in polar coordinates to make building the \
# radially symmetric function easier.
# convert cartesian coordinates to polar radius

r=np.sqrt(u_mat**2+v_mat**2)

# An expression for the filter is given below. 
# Note, the width of the filter is defined in terms of the real space dimensions for ease of use.

# inverse width of gaussian, units same as real space axes

filter_width=0.15
gauss_filter=np.e**(-(r*filter_width)**2)

fig,axes=plt.subplots(ncols=2,figsize=(10,5))
_=usid.plot_utils.plot_map(axes[0],gauss_filter,cmap=plt.cm.OrRd)
axes[0].set_title("Gaussian Filter")
axes[1].plot(gauss_filter[gauss_filter.shape[0]//2])
axes[1].set_title("Cross section of filter")
fig.tight_layout()
plt.savefig("fft-4.jpg",dpi=300)
plt.show()

# Application of the filter to the data in the Fourier domain is done \
# simply by dot-multiplying the two matrices.

F_m1_filtered=gauss_filter*fft_image_raw

# To view the filtered data in the space domain, simply use the inverse fast Fourier transform ('ifft2'). 

image_filtered=npf.ifft2(npf.ifftshift(F_m1_filtered))
image_filtered=np.real(image_filtered)

fig,axes=plt.subplots(ncols=2,figsize=(10,5))
for axis,img,title in zip(axes,[image_raw,image_filtered],['original','filtered']):
	_=usid.plot_utils.plot_map(axis,img,cmap=plt.cm.inferno,x_vec=x_axis_vec,y_vec=y_axis_vec,\
	num_ticks=5)
	axis.set_title(title)

fig.tight_layout()

plt.savefig("fft-5.jpg",dpi=300)
plt.show()

# artificially add a background to the original image

background_distottion=0.2*(x_mat+y_mat+np.sin(2*np.pi*x_mat/x_edge_length))
image_w_background=image_raw+background_distottion

fig,axes=plt.subplots(figsize=(10,5),ncols=2)
for axis,img,title in zip(axes,[background_distottion,image_w_background],\
['background', 'image with background']):
	_=usid.plot_utils.plot_map(axis,img,cmap=plt.cm.inferno,x_vec=x_axis_vec,\
	y_vec=y_axis_vec,num_ticks=5)
	axis.set_title(title)

fig.tight_layout()

plt.savefig("fft-6.jpg",dpi=300)
plt.show()

# inverse width of gaussian, units same as real space axes

filter_width=2
inverse_gaus_filter=1-np.e**(-(r*filter_width)**2)

fig,axis=plt.subplots()
_=usid.plot_utils.plot_map(axis,inverse_gaus_filter,cmap=plt.cm.OrRd)
axis.set_title("background filter")
plt.savefig("fft-7.jpg",dpi=300)
plt.show()

# Let perform the same process of taking the FFT of the image\
# multiplying with the filter and taking the inverse Fourier transform\
# of the image to get the filtered image.

# take the fft of the image

fft_image_w_background=npf.fftshift(npf.fft2(image_w_background))
fft_abs_image_background=np.abs(fft_image_w_background)

# Apply the filter
fft_image_corrected=fft_image_w_background*inverse_gaus_filter

# perform the inverse fourier transform on the filtered data
image_corrected = np.real(npf.ifft2(npf.ifftshift(fft_image_corrected)))

# find what was removed from the image by filtering
filtered_background = image_w_background - image_corrected

fig,axes=plt.subplots(ncols=2,figsize=(10,5))
for axis,img,title in zip(axes,[image_corrected,filtered_background],\
["image with background subtracted","background component that was removed"]):
	_=usid.plot_utils.plot_map(axis,img,cmap=plt.cm.inferno,x_vec=x_axis_vec,\
	y_vec=y_axis_vec,num_ticks=5)
	axis.set_title(title)
	
fig.tight_layout() 
plt.savefig("fft-8.jpg",dpi=300)
plt.show()
	
	
