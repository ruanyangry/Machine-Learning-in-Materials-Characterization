# _*_ coding:UTF-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Reference: https://github.com/ruanyangry/pycroscopy
           https://pycroscopy.github.io/pyUSID/about.html
'''

# First import modules

# Python build-in library

import sys
import os

import time 
from datetime import timedelta

from tabulate import tabulate

import operator
import itertools

# import numpy and scipy

import numpy as np
from scipy import spatial

# import matplotlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid

# import sklearn

from sklearn import decomposition
from sklearn import manifold
from sklearn import metrics
from sklearn import preprocessing

# import skimage

import skimage
import skimage.measure
from skimage import io as skio

# import pysptools
# Hyperspectral library for Python
# https://github.com/ctherien/pysptools

import pysptools.abundance_maps
import pysptools.distance
import pysptools.eea
from pysptools.abundance_maps.amaps import FCLS
from pysptools.abundance_maps.amaps import NNLS

# https://github.com/kimjingu/nonnegfac-python
# Python Toolbox for Nonnegative Matrix Factorization (NMF)

from nonnegfac import matrix_utils as mu
from nonnegfac import nnls

print("#---------------------------------")
print(" All neceaasry library imports done")
print("#---------------------------------")
print("\n")

# endmember_algoriths: ATGP FIPPI N-FINDER PPI
# abundance_maps_algoriths: ISOMAP,LLE,MLLE,HLLE,SE,LTSA,MMDS,NMDS,TSNE
#                           these methods list in sklearn.manifold 
# joint_algorithms: PCA,NMF,Joint-NMF,GR-NMF

dirname="C:\\Users\\RY\\Desktop\\CNMS_ML_in_MS_Workshop_2018-master\\sample-data\\"
data_filename="data.npz"

n_endmembers = 5

# List algoriths used in this .py file.

endmember_algorithms = ["ATGP", "FIPPI", "N-FINDR", "PPI"]
abundance_maps_algorithms = ["ISOMAP", "LLE", "MLLE", "HLLE", "SE", "LTSA", "MMDS", "NMDS", "TSNE"]
joint_algorithms = ["PCA", "NMF", "Joint-NMF", "GR-NMF"]

# n_neighbors must be greater than
# (n_endmembers * (n_endmembers + 3) / 2) for Hessian-based LLE 

n_neighbors = (n_endmembers*(n_endmembers+3)/2)+1

# whether used precomputed results?

compute="cache_only"

# Save any computed results?

save_results=True

#  Mean Structural Similarity (SSIM)

abundance_maps_comp=skimage.measure.compare_ssim

# Spectral Angle Mapper

endmember_comp=pysptools.distance.SAM

print("#---------------------------------")
print(" Parameters defined done")
print("#---------------------------------")
print("\n")

abundance_maps_dict={}
endmembers_dict={}

algorithms=endmember_algorithms+abundance_maps_algorithms+joint_algorithms

print("#---------------------------------")
print(" Global variables defined done")
print("#---------------------------------")
print("\n")

# Functions for loading DataSets

# read hsi data

def read_hsi_data(filename):
	'''
	filename: the name of the input file
	'''
	with np.load(filename) as npz_file:
		expected_fields=["data","labels"]
		unexpected_fields=[name for name in npz_file.files if name not in expected_fields]
		if unexpected_fields:
			message="ignoring unexpected fields in '{}':{}"
			message=message.format(filename,",".join(unexpected_fields))
			print(sys.stderr,message)
		
		hsi_3d=npz_file["data"]
		
		if len(hsi_3d.shape) != 3:
			raise TypeError ("Image must have size height x width x bands. That's to say 3 dimensions")
			
		if "labels" in npz_file.files:
			labels=npz_file["labels"]
			if len(labels.shape) != 2:
				raise TypeError("Labels must have size height x width")
			elif hsi_3d.shape[:-1] != labels.shape:
				message="Image and label dimensions do not match: {} {}"
				message=message.format(hsi_3d.shape,labels.shape)
				raise TypeError(message)
		else:
			labels=None
	return hsi_3d,labels
	
def read_ground_truth(filename,hsi_3d):
	'''
	filename: the name of the input data
	hsi_3d: returned by hsi_3d
	'''
	hsi_2d=np.reshape(hsi_3d,(hsi_3d.shape[0]*hsi_3d.shape[1],hsi_3d.shape[2]))
	image_dim=(hsi_3d.shape[0],hsi_3d.shape[1])
	spectrial_len=hsi_3d.shape[2]
	
	with np.load(filename) as npz_file:
		expected_fields=["abundance_maps","endmembers"]
		unexpected_fields=[name for name in npz_file.files if name not in expected_fields]
		if unexpected_fields:
			message="Ignoring unexpected fields in '{}': {}"
			message=message.format(ground_truth_filename,",".join(unexpected_fields))
			print(sys.stderr,message)
		
		if "abundance_maps" in npz_file.files or "endmembers" in npz_file.files:
			if "abundance_maps" in npz_file.files:
				abundance_maps=npz_file["abundance_maps"]
				if abundance_maps.shape[1:3] != image_dim:
					message="data image size does not match ground truth image size: {} {}"
					message=message.format(hsi_3d.shape,abundance_maps.shape)
					raise TypeError(message)
					
			if "endmembers" in npz_file.files:
				endmembers=npz_file["endmembers"]
				if endmembers.shape[1] != spectral_len:
					message="data spectrum does not match ground truth image size: {} {}"
					message=message.format(hsi_3d.shape,endmembers.shape)
					raise TypeError(message)
				
			if "abundance_maps" not in npz_file.files:
				message="ground truth abundance maps not provided, estimating with NNLS"
				print(sys.stderr,message)
				abundance_maps=NNLS(hsi_2d,endmembers)
				abundance_maps=preprocessing.normalize(abundance_maps,norm="l1")
				abundance_maps=np.reshape(abundance_maps,(hsi_3d.shape[0],hsi_3d.shape[1],\
				abundance_maps.shape[1]))
				abundance_maps=np.moveaxis(abundance_maps,2,0)
			elif "endmembers" not in npz_file.files:
				message="ground truth endmembers not provided, estimating with NNLS"
				print(sys.stderr,message)
				print(abundance_maps.shape)
				tmp_maps=np.moveaxis(abundance_maps,0,2)
				print(tmp_maps.shape)
				endmembers=NNLS(hsi_2d.transpose(),tmp_maps.transpose())
				endmembers=endmembers.transpose()
				print(endmembers.shape)
			elif abundance_maps.shape[0] != endmembers.shape[0]:
				message = "number of abundance maps ({}) does not match associated endmembers ({})"
				message=message.format(abundance_maps.shape,endmembers.shape)
				print(sys.stderr,message)
		else:
			message="No ground truth data found in '{}'"
			message=message.format(ground_truth_filename)
			raise RuntimeError(message)
		return endmembers,abundance_maps
	
print("#---------------------------------")
print(" Functions for Loading Datasets done")
print("#---------------------------------")
print("\n")
				
def read_results(filename):
	'''
	filename: the name of the input file
	'''
	with np.load(filename) as npz_file:
		unexpected_fields=[field for field in npz_file.files if field not in ["endmembers","abundance)maps"]]
		if unexcepted_fields:
			message="ignoring unexcepted fields in '{}':{}"
			message=message.format(filename,",".join(unexpected_fields))
			print(sys.stderr,message)
			
		endmembers=npz_file["endmembers"]
		abundance_maps=npz_file["abundance_maps"]
		
	return endmembers,abundance_maps
	
def write_results(filename,endmembers,abundance_maps):
	np.save(filename,endmembers=endmembers,abundance_maps=abundance_maps)
	
print("#---------------------------------")
print(" Functions for Reading / Writing Precomputed Results done")
print("#---------------------------------")
print("\n")

# This class module obtained python 2.X 版本

class PYSP_Extractor(object):
	ALGORITHM = None
	
	def __init__(self):
		pass
		
	def get_components(self,hsi_3d,n_endmembers):
		hsi_2d=np.reshape(hsi_3d,(hsi_3d.shape[0]*hsi_3d.shape[1],hsi_3d.shape[2]))
		
		start_time=time.time()
		endmembers=self.eea_algorithm.extract(hsi_3d,n_endmembers)
		elapsed=time.time()-start_time
		print(self.ALGORITHM,"Computation Time:",timedelta(seconds=elapsed))
		
		start_time=time.time()
		
		abundance_maps=NNLS(hsi_2d,endmembers)
		elapsed=time.time()-start_time
		print(slef.ALGORITHM,"Abundance Map Estimation Time: ",timedelta(secondss=elapsed))
		
		abundance_maps=preprocessing.normalize(abundance_maps,norm="l1")
		abundance_maps=np.moveaxis(abundance_maps,1,0)
		abundance_maps=np.reshape(abundance_maps,(abundance_maps.shape[0],hsi_3d.shape[0],hsi_3d.shape[1]))
		return endmembers,abundance_maps
		
# This means "class ATGP_Extractor(PYSP_Extractor)" inheritance of class
class ATGP_Extractor(PYSP_Extractor):
	ALGORITHM = "ATGP"
	def __init__(self):
		self.eea_algorithm = pysptools.eea.ATGP()
		super(ATGP_Extractor,self).__init__()
		
class FIPPI_Extractor(PYSP_Extractor):
	ALGORITHM="FIPPI"
	def __init__(self):
		self.eea_algorithm = pysptools.eea.FIPPI()
		super(FIPPI_Extractor,self).__init__()
		
class NFINDR_Extractor(PYSP_Extractor):
	ALGORITHM="N-FINDER"
	def __init__(self):
		self.eea_algorithm=pysptools.eea.NFINDR()
		super(NFINDR_Extractor,self).__init__()
		
class PPI_Extractor(PYSP_Extractor):
	ALGORITHM="PPI"
	def __init__(self):
		self.eea_algorithm=pysptools.eea.PPI()
		super(PPI_Extractor,self).__init__()
		
print("#---------------------------------")
print(" Helper Classes for Running PySptool Algorithms done")
print("#---------------------------------")
print("\n")


# All the decomposition methods list in sklearn.manifold 
class SKLEARN_Extractor(object):
	ALGORITHM=None
	
	def __init__(self):
		pass
		
	def get_components(self,hsi_3d,n_endmembers):
		hsi_2d = np.reshape(hsi_3d, (hsi_3d.shape[0]*hsi_3d.shape[1],hsi_3d.shape[2]))
		start_time=time.time()
		
		abundance_maps=self.mfld_obj.fit_transform(hsi_2d)
		elapsed=time.time()-start_time
		print(self.ALGORITHM,"Computation Time:",timedelta(seconds=elapsed))
		
		strat_time=time.time()
		endmembers=NNLS(hsi_2d.transpose(),abundance_maps.transpose())
		elapsed=time.time()-start_time
		print(self.ALGORITHM,"Abundance Map Estimation Time:",timedelta(seconds=elapsed))
		
		abundance_maps=np.moveaxis(abundance_maps,1,0)
		abundance_maps=np.reshape(abundance_maps,(abundance_maps.shape[0],hsi_3d.shape[0],hsi_3d.shape[1]))
		endmembers=endmembers.transpose()
		return endmembers,abundance_maps
		
class ISOMAP_Extractor(SKLEARN_Extractor):
	ALGORITHM="ISOMAP"
	def __init__(self,n_components,n_neighbors,n_jobs=1):
		self.mfld_obj = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=n_jobs)
		super(ISOMAP_Extractor, self).__init__()

class LLE_Extractor(SKLEARN_Extractor):
    ALGORITHM = "LLE"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method="standard", n_jobs=-1)
        super(LLE_Extractor, self).__init__()

class MLLE_Extractor(SKLEARN_Extractor):
    ALGORITHM = "MLLE"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method="modified", n_jobs=n_jobs)
        super(MLLE_Extractor, self).__init__()

class HLLE_Extractor(SKLEARN_Extractor):
    ALGORITHM = "HLLE"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method="hessian", eigen_solver="dense", n_jobs=n_jobs)
        super(HLLE_Extractor, self).__init__()

class SE_Extractor(SKLEARN_Extractor):
    ALGORITHM = "SE"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = manifold.SpectralEmbedding(n_neighbors=n_neighbors, n_components=n_components, n_jobs=n_jobs)
        super(SE_Extractor, self).__init__()

class LTSA_Extractor(SKLEARN_Extractor):
    ALGORITHM = "LTSA"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method="ltsa", n_jobs=n_jobs)
        super(LTSA_Extractor, self).__init__()

class MMDS_Extractor(SKLEARN_Extractor):
    ALGORITHM = "MMDS"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = manifold.MDS(n_components=n_components, metric=True, n_jobs=n_jobs)
        super(MMDS_Extractor, self).__init__()

class NMDS_Extractor(SKLEARN_Extractor):
    ALGORITHM = "NMDS"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = manifold.MDS(n_components=n_components, metric=False, n_jobs=n_jobs)
        super(NMDS_Extractor, self).__init__()

class TSNE_Extractor(SKLEARN_Extractor):
    ALGORITHM = "TSNE"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = manifold.TSNE(n_components=n_components, method="exact")
        super(TSNE_Extractor, self).__init__()

class PCA_Extractor(SKLEARN_Extractor):
    ALGORITHM = "PCA"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = decomposition.PCA(n_components=n_components)
        super(PCA_Extractor, self).__init__()

class NMF_Extractor(SKLEARN_Extractor):
    ALGORITHM = "NMF"
    def __init__(self, n_components, n_neighbors, n_jobs=1):
        self.mfld_obj = decomposition.NMF(n_components=n_components)
        super(NMF_Extractor, self).__init__()
        
print("#---------------------------------")
print(" Helper Classes for Running scikit-learn Algorithms done")
print("#---------------------------------")
print("\n")
		
def show_cube(data,top_band=None,top_cmap=plt.cm.jet,side_cmap=plt.cm.jet,elev=None,azim=-30):
	top_cmap=plt.get_cmap(top_cmap)
	side_cmap=plt.get_cmap(side_cmap)
	
	if top_band is  None:
		top_band=np.argmax(np.var(np.reshape(data,(-1,data.shape[2])),0))
		
	fig=plt.figure(figsize=(7,7))
	fig.suptitle("HSI Cube {}".format(data.shape),fontsize=16)
	ax=fig.gca(projection="3d")
	ax.set_aspect("equal")
	ax.set_axis_off()
	ax.view_init(elev,azim)
	
	# X constant faces
	YY,ZZ=np.mgrid[0:data.shape[1],0:data.shape[2]]
	XX=np.zeros(YY.shape)
	
	XX += data.shape[0]-1
	ax.plot_surface(XX,YY,ZZ,shade=False,facecolors=side_cmap(plt.Normalize()(data[-1,:,:])))
	
	# Y constant faces
	XX, ZZ, = np.mgrid[0:data.shape[0], 0:data.shape[2]]
	YY = np.zeros(XX.shape)
	ax.plot_surface(XX, YY, ZZ, shade=False, facecolors=side_cmap(plt.Normalize()(data[:,0,:])))
	YY += data.shape[1]-1
	#ax.plot_surface(XX, YY, ZZ, shade=False, facecolors=side_cmap(plt.Normalize()(data[:,-1,:])))
	# Z constant faces
	XX,YY= np.mgrid[0:data.shape[0], 0:data.shape[1]]
	ZZ = np.zeros(XX.shape)
	#ax.plot_surface(XX, YY, ZZ, shade=False, facecolors=side_cmap(plt.Normalize()(data[:,:,0])))
	ZZ += data.shape[2]-1
	ax.plot_surface(XX, YY, ZZ, shade=False, facecolors=top_cmap(plt.Normalize()(data[:,:,top_band])))
	# Create cubic bounding box to simulate equal aspect ratio
	max_range = np.max(data.shape)
	Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(data.shape[0])
	Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(data.shape[1])
	Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(data.shape[0])
	# Comment or uncomment following both lines to test the fake bounding box:
	for xb, yb, zb in zip(Xb, Yb, Zb):
		ax.plot([xb], [yb], [zb], 'w')
	
	plt.tight_layout(0, 0, 0)
	plt.show()
    
print("#---------------------------------")
print(" Function for Displaying a HSI Cube done")
print("#---------------------------------")
print("\n")

def display_components(endmembers, abundance_maps, alg_name=None, vmin=0.0, vmax=None, cmap="jet"):
    if endmembers is not None and abundance_maps is not None:
        if endmembers.shape[0] != abundance_maps.shape[0]:
            message = "Endmember count ({}) and abundance map count ({}) must match."
            message = message.format(endmembers.shape, abundance_maps.shape)
            raise ValueError(message)
        n_components = endmembers.shape[0]
        n_cols = 2
    else:
        if endmembers is not None:
            n_components = endmembers.shape[0]
        elif abundance_maps is not None:
            n_components = abundance_maps.shape[0]
        else:
            raise ValueError("Both endmembers and abundance_maps cannot be None.")
        n_cols = 1
    
    if endmembers is not None:
        if alg_name is not None:
            title = "{} - Endmembers".format(alg_name)
        else:
            title = "Endmembers".format(alg_name)
        plt.title(title)
        plt.xlabel("Wavelength")
        plt.ylabel("Brightness")
        lines = plt.plot(endmembers.transpose())
        plt.show()
    
    for index in np.arange(n_components):
        fig = plt.figure(figsize = (8,3))
        gs = mpl.gridspec.GridSpec(1, n_cols)
        
        if alg_name is not None:
            title = "{} - Component {}".format(alg_name, index)
        else:
            title = "Component {}".format(index)
        fig.suptitle(title, fontsize=16)
        
        if endmembers is not None:
            ax = fig.add_subplot(gs[0,0])
            ax.set_xlabel("Wavelength")
            ax.set_ylabel("Brightness")
            ax.plot(endmembers[index,:], lines[index].get_color())
        
        if abundance_maps is not None:
            ax = fig.add_subplot(gs[0,n_cols-1])
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(abundance_maps[index,:,:], vmin=vmin, vmax=vmax, cmap=cmap)
            fig.colorbar(im, ax=ax)
        
        #plt.tight_layout(rect=[0, 0.03, 1, 0.9])
        plt.show()
        
print("#---------------------------------")
print(" Function for Displaying Endmembers and Abundance Maps done")
print("#---------------------------------")
print("\n")

def _display_matrix(matrix, xlabel=None, xticks=None, ylabel=None, yticks=None, floatfmt="0.4f", cmap="binary", title=None, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    
    #labelbottom, labeltop, labelleft, labelright
    labelbottom = True
    labelleft = True
    
    if xlabel is not None:
        labelbottom=False
        ax.set_xlabel(xlabel)
    if xticks is None:
        xticks = list(range(matrix.shape[1]))
    ax.set_xticklabels([""]+xticks)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    if ylabel is not None:
        labelleft=False
        ax.set_ylabel(ylabel)
    if yticks is None:
        yticks = list(range(matrix.shape[0]))
    ax.set_yticklabels([""]+yticks)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    if title is not None:
        ax.set_title(title, y=1.08)
            
    #ax.tick_params(axis="both", which="both", labelbottom=labelbottom, labeltop=True, labelleft=labelleft, labelright=True, length=0)
    ax.tick_params(axis="both", which="both", labelbottom=True, labeltop=True, labelleft=True, labelright=True, length=0)
    
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)
    
    threshold = (matrix.max()-matrix.min()) / 2. + matrix.min()
    for ii, jj in itertools.product(np.arange(matrix.shape[0]), np.arange(matrix.shape[1])):
        text = format(matrix[ii,jj], floatfmt)
        if matrix[ii,jj] > threshold:
            color = "white"
        else:
            color = "black"
        plt.text(jj, ii, text, horizontalalignment="center", color=color)
    #plt.savefig("{}-vs-{}.mat.png".format(ylabel, xlabel), format="png")
    plt.show()
    
print("#---------------------------------")
print(" Function for Displaying a Matrix done")
print("#---------------------------------")
print("\n")

def _possible_endmember_metric(endmembers, metric):
    size = endmembers.shape[0]
    diag = np.zeros(size)
    for index in np.arange(size):
        diag[index] = metric(endmembers[index,:], endmembers[index,:])
    res = np.max( diag[np.isfinite(diag)] )
    return np.isclose(res, 0.0, atol=1e-07)

def _possible_abundance_maps_metric(abundance_maps, metric):
    size = abundance_maps.shape[0]
    diag = np.zeros(size)
    for index in np.arange(size):
        diag[index] = metric(abundance_maps[index,:,:], abundance_maps[index,:,:])
    res = np.max( diag[np.isfinite(diag)] )
    return np.isclose(res, 0.0, atol=1e-07)

def _match_best(distances, metric=True):
    if metric:
        distances[np.isnan(distances)] = np.finfo(distances.dtype).max
        fun = np.argmin
    else:
        distances[np.isnan(distances)] = np.finfo(distances.dtype).min
        fun = np.argmax
    matches = [fun(distances[ii,:]) for ii in np.arange(distances.shape[0])]
    return matches

def _pair_greedy(distances, metric=True):
    mean = np.nanmean(distances)
    distances = np.copy(distances)
    if metric:
        distances[np.isnan(distances)] = np.finfo(distances.dtype).max
        fun = np.nanargmin
        compare = operator.lt
    else:
        distances[np.isnan(distances)] = np.finfo(distances.dtype).min
        fun = np.nanargmax
        compare = operator.gt
    
    matches = [-1] * distances.shape[0]
    n_iter = np.min(distances.shape)
    for ii in np.arange(n_iter):
        loc = np.unravel_index(fun(distances), distances.shape)
        if compare(distances[loc], mean):
            matches[loc[0]] = loc[1]
        distances[loc[0],:] = np.nan
        distances[:,loc[1]] = np.nan
    
    return matches
    
print("#---------------------------------")
print(" Helper Functions for Matching Items done")
print("#---------------------------------")
print("\n")

def compare_endmembers(endmembers1, endmembers2, algorithm1="Endmembers1", algorithm2="Endmembers2", metric=pysptools.distance.SAM, \
                       matrix_vmin=0.0, matrix_vmax=None, matrix_cmap=None, floatfmt="0.4f", match_fun=_pair_greedy, is_metric=None):
    try:
        if is_metric is None:
            is_metric = _possible_endmember_metric(endmembers1, metric)
    except Exception as ex:
        print (sys.stderr, "Test for metric failed:", str(ex))
    
    distances = spatial.distance.cdist(endmembers1, endmembers2, metric=metric)
    if match_fun is None or is_metric is None:
        matches = [-1] * distances.shape[0]
    else:
        matches = match_fun(distances, is_metric)
    
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    markers = mpl.lines.Line2D.filled_markers
    c1 = [None] * distances.shape[0]
    c2 = [None] * distances.shape[1]
    for index,match in enumerate(matches):
        if match != -1:
            if c2[match] == None:
                c2[match] = {"color":"C{}".format(index), "marker":markers[index], "markevery":0.75}
            c1[index] = c2[match]
    default_args = {"color":"black", "linestyle":"dotted"}
    for index,match in enumerate(c1):
        if match is None:
            c1[index] = default_args
    for index,match in enumerate(c2):
        if match is None:
            c2[index] = default_args
        
    if matrix_cmap is None:
        if is_metric is None:
            matrix_cmap = "binary"
        elif is_metric:
            matrix_cmap = "YlOrRd"
        else:
            matrix_cmap = "Greens"
    
    columns = max(endmembers1.shape[0], endmembers2.shape[0])
    
    alg1_max = np.max(endmembers1)
    alg2_max = np.max(endmembers2)
    
    fig = plt.figure(figsize=(10,5))
    fig.suptitle("{} vs. {}".format(algorithm1, algorithm2), size=20)
    
    gs = mpl.gridspec.GridSpec(2, columns)
    for col in np.arange(columns):
        if endmembers1.shape[0] > col:
            endmember = endmembers1[col,:]
            ax = fig.add_subplot(gs[0,col])
            ax.set_title("{} - EM {}".format(algorithm1, col))
            ax.set_xticks([])
            if col != 0:
                ax.set_yticks([])
            ax.set_ylim([0.0,alg1_max])
            ax.plot(endmember, **c1[col])
        
        if endmembers2.shape[0] > col:
            endmember = endmembers2[col,:]
            ax = fig.add_subplot(gs[1,col])
            ax.set_title("{} - EM {}".format(algorithm2, col))
            ax.set_xticks([])
            if col != 0:
                ax.set_yticks([])
            ax.set_ylim([0.0,alg2_max])
            ax.plot(endmember, **c2[col])
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    #plt.savefig("{}-vs-{}.em.fig.png".format(algorithm1, algorithm2), format="png")
    plt.show()
    
    _display_matrix(distances, xlabel=algorithm2, ylabel=algorithm1, title=metric.__name__, cmap=matrix_cmap)
    print (tabulate(distances, tablefmt="grid", floatfmt=floatfmt))
    
print("#---------------------------------")
print(" Function for Comparing Endmembers done")
print("#---------------------------------")
print("\n")

def compare_abundance_maps(abundance_maps1, abundance_maps2, algorithm1="AbundanceMaps1", algorithm2="AbundanceMaps2", metric=skimage.measure.compare_ssim, abundance_cmap="jet", matrix_cmap=None, floatfmt="0.4f", match_fun=_pair_greedy, is_metric=None):
    try:
        if is_metric is None:
            is_metric = _possible_abundance_maps_metric(abundance_maps1, metric)
    except Exception as ex:
        print >> sys.stderr, "Test for metric failed:", str(ex)
    
    rows = abundance_maps1.shape[0]
    columns = abundance_maps2.shape[0]
    distances = np.zeros(shape=(rows, columns))
    for row in np.arange(rows):
        for col in np.arange(columns):
            distances[row,col] = metric(abundance_maps1[row,:,:], abundance_maps2[col,:,:])
            
    if match_fun is None or is_metric is None:
        matches = [-1] * distances.shape[0]
    else:
        matches = match_fun(distances, is_metric)
            
    # https://matplotlib.org/api/text_api.html#matplotlib.text.Text
    c1 = [None] * distances.shape[0]
    c2 = [None] * distances.shape[1]
    for index,match in enumerate(matches):
        if match != -1:
            if c2[match] == None:
                c2[match] = {"backgroundcolor":"C{}".format(index)}
            c1[index] = c2[match]
    default_args = {}
    for index,match in enumerate(c1):
        if match is None:
            c1[index] = default_args
    for index,match in enumerate(c2):
        if match is None:
            c2[index] = default_args
    
    if matrix_cmap is None:
        if is_metric is None:
            matrix_cmap = "binary"
        elif is_metric:
            matrix_cmap = "YlOrRd"
        else:
            matrix_cmap = "Greens"
            
    columns = max(abundance_maps1.shape[0], abundance_maps2.shape[0])
    
    alg1_max = np.max( abundance_maps1 )
    alg2_max = np.max( abundance_maps2 )
    
    fig = plt.figure(figsize=(15,5))
    fig.suptitle("{} vs. {}".format(algorithm1, algorithm2), size=20)
    
    #https://stackoverflow.com/questions/44837082/colorbar-for-each-row-in-imagegrid
    TopGrid = ImageGrid(fig, 211,
                nrows_ncols=(1,columns),
                axes_pad=0.5,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="10%",
                cbar_pad=0.2,
                )
    BottomGrid = ImageGrid(fig, 212,
                nrows_ncols=(1,columns),
                axes_pad=0.5,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="10%",
                cbar_pad=0.2,
                )
    
    for col in np.arange(columns):
        if abundance_maps1.shape[0] > col:
            abundance_map = abundance_maps1[col,:,:]
            
            ax = TopGrid[col]
            ax.set_title("{} - AM {}".format(algorithm1, col), **c1[col])
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(abundance_map, vmin=0.0, vmax=alg1_max, cmap=abundance_cmap)
            ax.cax.colorbar(im)
            
            """if c1 is not None and False:
                for spine in ax.spines.values():
                    spine.set_edgecolor(c1[col])
                    spine.set_position(("outward", 3))
                    spine.set_linewidth(6)"""
        
        if abundance_maps2.shape[0] > col:
            abundance_map = abundance_maps2[col,:,:]
            
            ax = BottomGrid[col]
            ax.set_title("{0} - AM {1}".format(algorithm2, col), **c2[col])
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(abundance_map, vmin=0.0, vmax=alg2_max, cmap=abundance_cmap)
            ax.cax.colorbar(im)
            
            """if c2 is not None:
                for spine in ax.spines.values():
                    spine.set_edgecolor(c2[col])
                    spine.set_position(("outward", 3))
                    spine.set_linewidth(6)"""
            
    #fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    #plt.savefig("{}-vs-{}.am.fig.png".format(algorithm1, algorithm2), format="png")
    plt.show()
    
    _display_matrix(distances, xlabel=algorithm2, ylabel=algorithm1, title=metric.__name__, cmap=matrix_cmap)
    print (tabulate(distances, tablefmt="grid", floatfmt=floatfmt))
    
print("#---------------------------------")
print(" Function for Comparing Abundance Maps done")
print("#---------------------------------")
print("\n")

class JointNMF(object):
    """
    solves min_W>=0,H>=0,H_hat>=0 ||A-WH.T||_F^2 + alpha * ||S-H_hat*H.T||_F^2 +
                                   beta * ||H_hat - H||_F^2
    Equation 8 in paper https://arxiv.org/pdf/1703.09646.pdf
    """
    
    def __init__(self, max_iter=100, alpha=0.1, beta=1):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta

    def extract(self, A, k, W=None, H=None):
        """ Run a NMF algorithm
                Parameters
                ----------
                A : numpy.array or scipy.sparse matrix, shape (m,n)
                m : Number of features
                n : Number of samples
                k : int - target lower rank
                Returns
                -------
                (W, H, rec)
                W : Obtained factor matrix, shape (m,k)
                H : Obtained coefficient matrix, shape (n,k)
        """
        if W is None and H is None:
            W = np.random.rand(A.shape[0], k)
            H = np.random.rand(A.shape[1], k)
        elif W is None:
            Sol, info = nnls.nnlsm_blockpivot(H, A.T)
            W = Sol.T
        elif H is None:
            Sol, info = nnls.nnlsm_blockpivot(W.T, A)
            H = Sol.T    
        H_hat = np.random.rand(A.shape[1], k)
        S = metrics.pairwise_kernels(A.T)
        norm_A = mu.norm_fro(A)
        for i in range(1, self.max_iter + 1):
            (W, H, H_hat) = self.iter_solver(A, S, W, H, H_hat, self.alpha, self.beta)
            rel_error = mu.norm_fro_err(A, W, H, norm_A) / norm_A
        return W, H, H_hat

    def iter_solver(self, A, S, W, H, H_hat, alpha, beta):
        # equation 9 in paper https://arxiv.org/pdf/1703.09646.pdf
        Sol, info = nnls.nnlsm_blockpivot(H, A.T, init=W.T)
        W = Sol.T
        # equation 10 in paper https://arxiv.org/pdf/1703.09646.pdf
        tmp = np.sqrt(beta) * np.identity(H.shape[1])
        lhs = np.concatenate((np.sqrt(alpha) * H, tmp))
        tmp = np.sqrt(beta)*H
        rhs = np.concatenate((np.sqrt(alpha)*S, tmp.T))
        Sol, info = nnls.nnlsm_blockpivot(lhs, rhs)
        H_hat = Sol.T
        # equation 11 in paper https://arxiv.org/pdf/1703.09646.pdf
        tmp_1 = np.sqrt(alpha) * H_hat
        tmp_2 = np.sqrt(beta) * np.identity(H.shape[1])
        lhs = np.concatenate((W, tmp_1, tmp_2))
        tmp_1 = np.sqrt(alpha)*S
        tmp_2 = np.sqrt(beta) * H_hat.T
        rhs = np.concatenate((A, tmp_1, tmp_2))
        Sol, info = nnls.nnlsm_blockpivot(lhs, rhs)
        #Sol, info = NNLS(lhs, rhs)
        H = Sol.T
        return (W, H, H_hat)
        
print("#---------------------------------")
print(" Joint NMF Class done")
print("#---------------------------------")
print("\n")

class GRNMF(object):
    """
    solves min_W>=0,H>=0 ||A-WH.T||_F^2 + lambda * Tr(H.T L H) +
                                   alpha * sum_i=1^n || h_i||_1
    where, L = D - W. Remember, if we explicity compute L, we cannot
    use in MU type of algorithms as the off diagonal entries of L are negative. 

    Equation 10 in paper https://www.hindawi.com/journals/mpe/2015/239589/
    If lambda is zero, it is sparse NMF and alpha is 0 it is deng cai
    graph regularized nmf
    """

    def extract(self, A, k, max_iter=100, lambda_reg=0.1, alpha_reg=0.1):
        """ Run a NMF algorithm
                Parameters
                ----------
                A : numpy.array or scipy.sparse matrix, shape (m,n)
                m : Number of features
                n : Number of samples
                k : int - target lower rank
                lambda_reg : Regularization constant for GRNMF
                alpha : L1 regularization constant for H matrix
                Returns
                -------
                (W, H, rec)
                W : Obtained factor matrix, shape (m,k)
                H : Obtained coefficient matrix, shape (n,k)
        """
        W = np.random.rand(A.shape[0], k)
        H = np.random.rand(A.shape[1], k)
        S = metrics.pairwise_kernels(A.T)
        # normalize the distance matrix between 0 to 1
        S = S-np.min(S)/(np.max(S)-np.min(S))
        D = np.sum(S, axis=1)
        norm_A = mu.norm_fro(A)
        for i in range(1, max_iter + 1):
            (W, H) = self.iter_solver(A, S, D, W, H, lambda_reg, alpha_reg)
            rel_error = mu.norm_fro_err(A, W, H, norm_A) / norm_A
        return W, H

    def iter_solver(self, A, S, D, W, H, lambda_reg, alpha_reg):
        # equation 11 of paper https://www.hindawi.com/journals/mpe/2015/239589/
        AtW = np.matmul(A.T, W)
        SH = np.matmul(S, H)
        WtW = np.matmul(W.T, W)
        HWtW = np.matmul(H, WtW)
        DH = np.matmul(np.diagflat(D), H)
        H_nr = 2 * (AtW + lambda_reg*SH) - alpha_reg
        H_dr = 2 * (HWtW + lambda_reg*DH)
        H = np.divide(H_nr, H_dr)
        AH = np.matmul(A, H)
        HtH = np.matmul(H.T, H)
        WHtH = np.matmul(W, HtH)
        W = np.divide(AH, WHtH)
        W = W / (W.sum(axis=1)[:,np.newaxis])
        return (W, H)
        
print("#---------------------------------")
print(" GR NMF Class done")
print("#---------------------------------")
print("\n")
