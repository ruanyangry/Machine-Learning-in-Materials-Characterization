# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Purpose: Image processing

scikit-image: https://scikit-image.org/
'''

# import modules

# numpy
import numpy as np

# plot

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# skimage

import skimage
from skimage import io
from skimage import filters
from skimage.morphology import disk
from skimage.feature import blob_log
from skimage import data
from skimage import measure
from skimage.filters import threshold_isodata,threshold_minimum,threshold_otsu,\
threshold_triangle,threshold_li
from skimage import exposure

# reading images

stem_image=io.imread(r"C:\Users\RY\Desktop\CNMS_UM_2018_SPIMA\data\LLTO_800.tif")

# images are just matrices of numbers

print("#-------------------------------------#")
print("Image read as : {}".format(type(stem_image)))
print("Image of shape: {} and precision: {}".format(stem_image.shape,stem_image.dtype))
print("#-------------------------------------#")
print("\n")

# take a look at the contents of a single pixel

print("#-------------------------------------#")
print(stem_image[5,4])
print("#-------------------------------------#")
print("\n")

# visualizing images

fig,axis=plt.subplots()
axis.imshow(stem_image,cmap='gray')
fig.savefig("scikit-image-1.jpg",dpi=300)
plt.show()

# changing the image datatype to interger or float is very useful
# for next analysis
# skimage.img_as_float() change value to [0,1]

print("#-------------------------------------#")
print('Before normalization: data type: {}, Min: {}, max: {}'.format(stem_image.dtype,\
stem_image.min(), stem_image.max()))

# Normalizing here
stem_image=skimage.img_as_float(stem_image)

print('After normalization: data type: {}, Min: {}, max: {}'.format(stem_image.dtype,\
stem_image.min(), stem_image.max()))

print("#-------------------------------------#")
print("\n")

# image manipulation
# remenber image just matrices

# copy image,not chaning the raw image

modified_image=stem_image.copy()

# Let's set a rectangle within the image to some new value

modified_image[30:150,350:450]=0

# visualize the modified image

fig,axis=plt.subplots()
axis.imshow(modified_image,cmap="gray")
axis.set_title('Modified image')
fig.savefig("scikit-image-2.jpg",dpi=300)
plt.show()

# change cmap types
fig,axis=plt.subplots()
axis.imshow(modified_image,cmap="jet")
axis.set_title('Modified JET image')
fig.savefig("scikit-image-3.jpg",dpi=300)
plt.show()

# cropping images
# Images are cropped in the same way that numpy arrays are sliced

cropped_images=stem_image[:128,:128]

fig,axis=plt.subplots(ncols=2,figsize=(12,6))
axis[0].imshow(cropped_images,cmap="gray")
axis[0].set_title("Cropped Images gray")
axis[1].imshow(cropped_images,cmap="jet")
axis[1].set_title("Cropped Images jet")
fig.savefig("scikit-image-4.jpg",dpi=300)
plt.show()

# Pseudo-color or false-color

fig,axes=plt.subplots(ncols=2,figsize=(12,6))

im_handle=axes[0].imshow(cropped_images)
axes[0].set_title("cmap = \"viridis\"")
cbar=plt.colorbar(im_handle,ax=axes[0],orientation='vertical',\
fraction=0.046, pad=0.04, use_gridspec=True)

im_handle = axes[1].imshow(cropped_images, cmap='jet')
axes[1].set_title('cmap="jet"')
cbar = plt.colorbar(im_handle, ax=axes[1], orientation='vertical',\
fraction=0.046, pad=0.04, use_gridspec=True)
fig.tight_layout()

fig.savefig("scikit-image-5.jpg",dpi=300)
plt.show()

# channels
# Black and white images have a single channel of information while color\
# images have three (red, green, blue) channels of information.

cat=data.chelsea()

print("#-------------------------------------#")
print("Image shape: ",cat.shape)
print("#-------------------------------------#")
print("\n")

fig,axis=plt.subplots()
axis.imshow(cat)

fig.savefig("scikit-image-6.jpg",dpi=300)
plt.show()

# visualizing the three channels separately
# R G B

fig,axes=plt.subplots(ncols=3,figsize=(15,3))

for channel,axis,colmap,title in zip(range(cat.shape[2]),axes.flat,[plt.cm.Reds,\
plt.cm.Greens,plt.cm.Blues],["Red","Green","Blue"]):
	im_handle=axis.imshow(np.squeeze(cat[:,:,channel]),cmap=colmap)
	axis.set_title(title)
	
	cbar=plt.colorbar(im_handle,ax=axis,fraction=0.046,pad=0.04)
	axis.set_title("FFT2 of image")

fig.savefig("scikit-image-7.jpg",dpi=300)
plt.show()

# converting color image to grayscale
# function: rgb2gray()

gscale_cat=skimage.color.rgb2gray(cat)
print("#-------------------------------------#")
print('Original shape: {}, grayscale shape: {}'.format(cat.shape, gscale_cat.shape))
print("#-------------------------------------#")
print("\n")

fig, axis = plt.subplots()
axis.imshow(gscale_cat, cmap='gray')
fig.savefig("scikit-image-8.jpg",dpi=300)
plt.show()

# shapes of images when loaded into python
# 2D grayscale     (row,column)
# 2D multichannel  (row,column,channel)
# 3D grayscale     (plane,row,column)
# 3D multichannel  (plane,row,column,channel)

# frequency space
# A lot of information cam be gained by visualizing the same image in \
# the frequency domain
# FFT transform

fft_image_raw=np.fft.fft2(stem_image)

fig,axis=plt.subplots(figsize=(5,5))
im_hande=axis.imshow(np.abs(fft_image_raw),cmap=plt.cm.OrRd)
cbar=plt.colorbar(im_handle,ax=axis,fraction=0.046,pad=0.04)
axis.set_title('FFT2 of image')

fig.savefig("scikit-image-9.jpg",dpi=300)
plt.show()

# looking at the log10 plot

fig,axis=plt.subplots(figsize=(5, 5))
im_handle=axis.imshow(np.log10(np.abs(fft_image_raw)),cmap=plt.cm.OrRd,vmin=2,vmax=3.25)
cbar=plt.colorbar(im_handle,ax=axis,fraction=0.046,pad=0.04)
axis.set_title('FFT2 of image')
fig.savefig("scikit-image-10.jpg",dpi=300)
plt.show()

# fftshift brings the lowest frequency components of the FFT back to the center of the plot

fft_image_raw_shifted = np.fft.fftshift(fft_image_raw)
fig, axis = plt.subplots(figsize=(5, 5))
im_handle = axis.imshow(np.log10(np.abs(fft_image_raw_shifted)),\
cmap=plt.cm.OrRd, vmin=2, vmax=3.25)

# add vertical and Horizontal line
axis.axvline(x=fft_image_raw_shifted.shape[0] // 2, color='k', linestyle=':')
axis.axhline(y=fft_image_raw_shifted.shape[1]// 2, color='k', linestyle=':')

cbar=plt.colorbar(im_handle, ax=axis, fraction=0.046, pad=0.04)
axis.set_title('FFT after fftshift-ing')

fig.savefig("scikit-image-11.jpg",dpi=300)
plt.show()

# image filters
# Gaussian filter

# get part image of stem image

cropped_image=stem_image[:128,:128]

gaus_sigma=2.0
gaussian_filtered=filters.gaussian(cropped_image,sigma=gaus_sigma)

fig,axes=plt.subplots(ncols=3,figsize=(12,4),sharey=True)
axes[0].imshow(cropped_image,cmap='gray')
axes[0].set_title('Original image')
axes[1].imshow(gaussian_filtered,cmap='gray')
axes[1].set_title('Gaussian filtered, Sigma = {}'.format(gaus_sigma))
axes[2].imshow(cropped_image-gaussian_filtered,cmap='gray')
axes[2].set_title('Removed noise')
fig.tight_layout()

fig.savefig("scikit-image-12.jpg",dpi=300)
plt.show()

# Visualize the effect of filtering in the frequency domain

fft_raw=np.fft.fftshift(np.fft.fft2(cropped_image))
fft_noise_rem=np.fft.fftshift(np.fft.fft2(cropped_image-gaussian_filtered))
fft_gaus_filt=np.fft.fftshift(np.fft.fft2(gaussian_filtered))

fig,axes=plt.subplots(ncols=3,figsize=(12,4))
for axis,fft_img,title in zip(axes.flat,[fft_raw,fft_gaus_filt,fft_noise_rem],\
['Raw Image','Filtered','Noise Removed']):
	axis.imshow(np.log10(np.abs(fft_img)),cmap=plt.cm.OrRd,vmin=[0.5, 2.5])
	axis.set_title(title)
	
fig.tight_layout()

fig.savefig("scikit-image-13.jpg",dpi=300)
plt.show()

# Testing gaussian filters parameter sigma

cropped_image=stem_image[:128,:128]

sigma_values=np.linspace(0,5,num=9)

fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(10, 10),sharex=True,sharey=True)

for axis,gaus_sigma in zip(axes.flat,sigma_values):
	gaussian_filtered = filters.gaussian(cropped_image,sigma=gaus_sigma)
	axis.imshow(gaussian_filtered, cmap='gray')
	axis.set_title('Sigma = {}'.format(gaus_sigma))
	
fig.suptitle('Gaussian Filter', fontsize=14, y=1.03)
fig.tight_layout()

fig.savefig("scikit-image-14.jpg",dpi=300)
plt.show()

# More filters
# Local mean
# Median
# Gaussian
# Unsupervised Wiener
# Bi-lateral Mean
# Wiener deconvolution

# All filters parameters must adjusted

cropped_image=stem_image[:128,:128]

# Local mean filter
loc_mean=filters.rank.mean(cropped_image,disk(4))

# Median filter
median_filtered=filters.median(cropped_image,disk(5))

# Gaussian Filter
gaussian_filtered=filters.gaussian(cropped_image,sigma=1.875)

# Bilateral mean
noisy_image=skimage.img_as_ubyte(cropped_image)
bilat_filtered=filters.rank.mean_bilateral(noisy_image.astype(np.uint16),\
disk(10),s0=5,s1=5)

# Self-tuned Wiener
disk_size=int(1)
psf=np.ones((disk_size,disk_size))/disk_size**2
self_tuned,_=skimage.restoration.unsupervised_wiener(cropped_image,psf)

# Wiener deconvolution
disk_size=int(10)
psf=np.ones((disk_size,disk_size))/disk_size**2
wiener_filtered=skimage.restoration.wiener(cropped_image,psf,50)

# Visualizing results here

fig,axes=plt.subplots(ncols=3,nrows=2,figsize=(12,8),sharex=True,sharey=True)

for axis,img,title in zip(axes.flat,[loc_mean,median_filtered,gaussian_filtered,\
self_tuned,bilat_filtered,wiener_filtered],['Local mean','Median','Gaussian',\
'Self-tuned Restoration','Mean Bilatereral','Wiener']):
	axis.imshow(img,cmap='gray')
	axis.set_title(title)
	
fig.suptitle('Many morefilters',fontsize=18,y=1.03)
fig.tight_layout()

fig.savefig("scikit-image-15.jpg",dpi=300)
plt.show()

# edge sensitive algorithms

rice_image = io.imread('https://www.mathworks.com/content/mathworks/www\
/en/company/newsletters/articles/new-features-for-high-performance-image-processing-in-matlab\
/jcr:content/mainParsys/image_1.adapt.full.high.jpg/1469941471626.jpg')

# Convert to float: Important for subtraction later which won't work with uint8

rice_image=skimage.img_as_float(skimage.color.rgb2gray(rice_image))

fig,axis=plt.subplots()
axis.imshow(rice_image,cmap='gray')

fig.savefig("scikit-image-16.jpg",dpi=300)
plt.show()

# edge detection applications

# detecting discontinuities in material properties
# detecting discontinuities in orientation
# compressing amount of information and simplifying computations (conversion to binary)

# few popular filters

# Sobel
# Scharr
# Roberts
# Prewitt

fig,axes=plt.subplots(figsize=(8, 8),nrows=2,ncols=2,sharex=True,sharey=True)

for axis,filt_alg,title in zip(axes.flat,[filters.sobel,filters.scharr,\
filters.prewitt, filters.roberts],['Sobel','Scharr','Prewitt','Roberts']):
	axis.imshow(filt_alg(rice_image),cmap='gray')
	axis.set_title(title, fontsize=16)
	
fig.tight_layout()
fig.savefig("scikit-image-17.jpg",dpi=300)
plt.show()

# take the result from the Prewitt filter and convert it to a binary image
# 1. Apply the filter to the raw image
# 2. Normalize the data to lie within [0, 1]
# 3. Threshold the data

# step 1: apply the filter

prewitt_bin=filters.prewitt(rice_image)

# step 2: Let's normalize this data to [0, 1]

prewitt_bin=prewitt_bin-prewitt_bin.min()
prewitt_bin=prewitt_bin/prewitt_bin.max()

print("#-------------------------------------#")
print('Min: {},max: {}'.format(prewitt_bin.min(),prewitt_bin.max()))
print("#-------------------------------------#")
print("\n")

# plot the histogram of the normalized image

fig,axis=plt.subplots()
axis.hist(prewitt_bin.ravel(),bins=256,histtype='step',color='black')
axis.set_ylabel('Number of pixels',fontsize=14)
axis.set_xlabel('Pixel intensity',fontsize=14)

fig.savefig("scikit-image-18.jpg",dpi=300)
plt.show()

# pick a reasonable value to threshold with

threshold=0.3
prewitt_bin[prewitt_bin > threshold]=1
prewitt_bin[prewitt_bin <= threshold]=0

fig,axis=plt.subplots()
axis.set_title('Binary image')
axis.imshow(prewitt_bin, cmap='gray')

fig.savefig("scikit-image-19.jpg",dpi=300)
plt.show()

# Directional Edge filters

fig,axes=plt.subplots(ncols=2,figsize=(8, 4),sharey=True)
for axis,algo in zip(axes.flat,[filters.sobel_h,filters.sobel_v]):
	filtered_image=algo(rice_image)
	axis.imshow(filtered_image,cmap='bwr')
	axis.set_title(algo.__name__)
fig.tight_layout()

fig.savefig("scikit-image-20.jpg",dpi=300)
plt.show()

# thresholding

# Normalizing to [0, 255]

rice_renormalized=rice_image-rice_image.min()
rice_renormalized=255*rice_renormalized/rice_renormalized.max()

fig,axis=plt.subplots(figsize=(5,5))
im_handle=axis.imshow(rice_renormalized,cmap='jet')

cbar=plt.colorbar(im_handle,ax=axis,fraction=0.046,pad=0.04)
axis.set_title('Normalized imge')

fig.savefig("scikit-image-21.jpg",dpi=300)
plt.show()

# There are two steps to creating the binary image with the two phases
# Calculate the thresholding value
# Find all pixels in the original image greater than the threshold

fig,axes=plt.subplots(ncols=3,figsize=(12, 4),sharex=True,sharey=True)

for axis,algo in zip(axes.flat,[threshold_triangle, threshold_otsu, threshold_minimum]):
	# First calculate the threshold value
	thresh=algo(rice_renormalized)
	
	# Find all pixels whose intensity is greater than the threshold
	binary_img=rice_renormalized > thresh
	
	axis.imshow(binary_img,cmap='gray',origin='lower')
	algo_name = algo.__name__.split('_')[1]
	axis.set_title('{} - {}'.format(algo_name,np.round(thresh, 2)),fontsize=16)
	
fig.tight_layout()

fig.savefig("scikit-image-22.jpg",dpi=300)
plt.show()

# background subtraction

# From the images above, it appears that the long-range features / background\
# intensity is making it challenging to perform local operations such as thresholding

section_x = 400
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
im_handle = axes[0].imshow(rice_renormalized, cmap='jet')
cbar = plt.colorbar(im_handle, ax=axes[0], fraction=0.046, pad=0.04)
axis.set_title('Normalized imge')
axes[0].axvline(x=section_x, color='k')
axes[0].set_title('Raw image', fontsize=14)
axes[0].set_xlabel('X', fontsize=14)
axes[0].set_ylabel('Y', fontsize=14)

axes[1].plot(np.squeeze(rice_image[:, section_x]))
axes[1].set_title('Line profile with x={}'.format(section_x), fontsize=14)
axes[1].set_xlabel('Y', fontsize=14)
axes[1].set_ylabel('Intensity', fontsize=14)
fig.tight_layout()

fig.savefig("scikit-image-23.jpg",dpi=300)
plt.show()
	
# remove the background

x_pixels, y_pixels = rice_image.shape
x_edge_length = 5.0 
y_edge_length = 5.0
x_sampling = x_pixels / x_edge_length
y_sampling = y_pixels / y_edge_length
x_axis_vec = np.linspace(-x_edge_length / 2, x_edge_length / 2, x_pixels)
y_axis_vec = np.linspace(-y_edge_length / 2, y_edge_length / 2, y_pixels)
x_mat, y_mat = np.meshgrid(x_axis_vec, y_axis_vec)

u_max = x_sampling / 2
v_max = y_sampling / 2
u_axis_vec = np.linspace(-u_max / 2, u_max / 2, x_pixels)
v_axis_vec = np.linspace(-v_max / 2, v_max / 2, y_pixels)
u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
r = np.sqrt(u_mat**2+v_mat**2)

# create the filter

filter_width = 3.5
inverse_gauss_filter = 1-np.e**(-(r*filter_width)**2)

fig, axis = plt.subplots()
axis.imshow(inverse_gauss_filter, cmap=plt.cm.OrRd)
axis.set_title('background removal filter')

fig.savefig("scikit-image-24.jpg",dpi=300)
plt.show()

# apply the filter to the raw image

# take the fft of the image
fft_image_w_background = np.fft.fftshift(np.fft.fft2(rice_image))
fft_abs_image_background = np.abs(fft_image_w_background)

# apply the filter
fft_image_corrected = fft_image_w_background * inverse_gauss_filter

# perform the inverse fourier transform on the filtered data
image_corrected = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image_corrected)))

# find what was removed from the image by filtering

filtered_background = rice_image - image_corrected

fig, axes = plt.subplots(ncols=3, figsize=(12, 4), sharey=True)
for axis,img, title in zip(axes.flat,[rice_image, image_corrected, filtered_background],\
['Original', 'Corrected', 'Removed background']):
	axis.imshow(img, cmap='jet', vmin=rice_image.min(), vmax=rice_image.max())
	axis.set_title(title)
	
fig.savefig("scikit-image-25.jpg",dpi=300)
plt.show()

# Renormalizing the background-corrected data

corrected_normalized = image_corrected - image_corrected.min()
corrected_normalized = 255 * corrected_normalized / corrected_normalized.max()

fig, axis = plt.subplots(figsize=(5, 5))
im_handle = axis.imshow(corrected_normalized, cmap='jet')
cbar = plt.colorbar(im_handle, ax=axis, fraction=0.046, pad=0.04)
axis.set_title('Normalized imge')

fig.savefig("scikit-image-26.jpg",dpi=300)
plt.show()

# Trying out the thresholding algorithms again

#fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 8), sharex=True, sharey=True)
#for axis, algo in zip(axes.flat,[threshold_mean, threshold_isodata, threshold_minimum,\
#threshold_otsu, threshold_triangle, threshold_li]):
#	# First calculate the threshold value
#	thresh = algo(corrected_normalized)
#	
#	# Find all pixels whose intensity is greater than the threshold
#	binary_img = corrected_normalized > thresh
#	
#	axis.imshow(binary_img, cmap='gray', origin='lower')
#	algo_name = algo.__name__.split('_')[1]
#	axis.set_title('{} - {}'.format(algo_name, np.round(thresh, 2)), fontsize=16)
#	
#fig.suptitle('Thresholding algorithms', fontsize=18, y=1.05)
#fig.tight_layout()
#
#fig.savefig("scikit-image-27.jpg",dpi=300)
#plt.show()

# Removing the background!

thresh = threshold_otsu(corrected_normalized)
background_removed_rice = corrected_normalized.copy()
background_removed_rice[corrected_normalized < thresh] = 0

fig, axis = plt.subplots(figsize=(5, 5))
im_handle = axis.imshow(background_removed_rice, cmap='Greys', vmin=75)
cbar = plt.colorbar(im_handle, ax=axis, fraction=0.046, pad=0.04)
axis.set_title('Normalized imge')

fig.savefig("scikit-image-28.jpg",dpi=300)
plt.show()

# Blob detection for finding atoms

# Three popular techniques

# Laplacian of Gaussian (LoG)
# Difference of Gaussian (DoG)
# Determinant of Hessian (DoH)

whitened_rice=corrected_normalized[:350,:350] * (1/256)
result=blob_log(whitened_rice, min_sigma=20, max_sigma=500,\
num_sigma=10, threshold=0.05)

fig,axis=plt.subplots(figsize=(5,5))
im_handle=axis.imshow(whitened_rice,cmap='gray')

axis.scatter(result[:,1],result[:,0],s=15,c='r')
axis.set_xlim(left=0,right=whitened_rice.shape[0])
axis.set_ylim(bottom=0,top=whitened_rice.shape[1])
cbar=plt.colorbar(im_handle,ax=axis,fraction=0.046,pad=0.04)
axis.set_title('Normalized imge')

fig.savefig("scikit-image-29.jpg",dpi=300)
plt.show()

# Label each grain seprately

thresh=threshold_otsu(corrected_normalized)
grains=corrected_normalized > thresh
all_labels=measure.label(grains)

fig,axis=plt.subplots()
axis.imshow(all_labels)

fig.savefig("scikit-image-30.jpg",dpi=300)
plt.show()

map_props=measure.regionprops(all_labels)
grain=map_props[70]
type(grain)

fig,axes=plt.subplots(ncols=2,figsize=(8, 4))
axes[0].imshow(corrected_normalized,cmap='gray')

# centroid
axes[0].scatter(grain.centroid[1],grain.centroid[0],c='red')

# bounding box
axes[0].add_patch(Rectangle((grain.bbox[1], grain.bbox[0]),\
grain.bbox[2]-grain.bbox[0],grain.bbox[3]-grain.bbox[1],\
fill=False,color='yellow',linewidth=2))

axes[1].imshow(grain.image,cmap='gray')
axes[1].set_title('Area: {},equiv. diameter: {}''.'.format(grain.area,\
np.round(grain.equivalent_diameter,2)))

fig.savefig("scikit-image-31.jpg",dpi=300)
plt.show()
