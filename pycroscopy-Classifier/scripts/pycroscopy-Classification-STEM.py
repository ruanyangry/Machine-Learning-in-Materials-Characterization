# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Purpose: pycroscopy classification methods

Classification methods:

1. Support Vector Machines
2. Decision Tree
3. Clustering

Dataset: STEM image of an oxide catalyst
'''

from __future__ import division, print_function, absolute_import, unicode_literals

# numpy

import numpy as np

# h5py

import h5py

# plot

import matplotlib.pyplot as plt

# wget

import wget

# system

import os
import sys
import subprocess

# Classification methods

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF,FastICA

# defined install function

def install(package):
	subprocess.call([sys.executable,"-m","pip","install",package])
	
try:
	import pyUSID as usid
except ImportError:
	print("pyUSID not found, Will install with pip")
	import pip
	install('pyUSID')
	import pyUSID as usid
	
try:
	import pycroscopy as px
except ImportError:
	print("pycroscopy not found. Will install with pip")
	import pip
	install('pycroscopy')
	import pycroscopy as px
	
from pycroscopy.viz import cluster_utils


# Get the data

#data_file_path=wget.download('https://ftp.ornl.gov/ftp_out/uP116H9fphkL/SuperImage2.h5')
data_file_path=r"C:\Users\RY\Desktop\SuperImage2.h5"
h5_file=h5py.File(data_file_path,mode="r+")

# print the data tree structures

print("#----------------------------------------#")
print(usid.hdf_utils.print_tree(h5_file))
print("#----------------------------------------#")
print("\n")

# Get the image

h5_meas_grp=h5_file["Measurement_000"]
h5_image=h5_meas_grp["Channel_000/Raw_Data"]

# Convert to USID Dataset

h5_image=usid.USIDataset(h5_image)

# Getting a reference to the main spectral dataset:
h5_main = usid.USIDataset(h5_meas_grp['Channel_000/Raw_Data-FFT_Window_000/FFT_Data'])

print("#----------------------------------------#")
print("Data currently of shape:",h5_main.shape)
print("#----------------------------------------#")
print("\n")

# Get the position and sepctrum size

[num_rows,num_cols]=h5_main.pos_dim_sizes
num_spect=h5_main.spec_dim_sizes[0]

print("#----------------------------------------#")
print("position sizes:",h5_main.pos_dim_sizes)
print("spectrum sizes:",h5_main.spec_dim_sizes)
print("#----------------------------------------#")
print("\n")

# Support Vector Machines
# This example separate data into two classes

# Generate some random data

X1=0.2+0.2*np.random.randn(100)
Y1=1+0.5*np.random.randn(100)

X2=1+0.3*np.random.randn(100)
Y2=-1+0.4*np.random.randn(100)

# np.vstack() v:vertical
# np.hstack() h:Horizontal

class_1=np.vstack((X1,Y1))
class_2=np.vstack((X2,Y2))

labels=np.zeros(shape=(class_1.shape[1]+class_2.shape[1]))

# Create a labels vector

labels[:class_1.shape[1]]=0
labels[class_1.shape[1]:]=1

# join all points together

all_points=np.hstack((class_1,class_2))

# Plot the raw data

fig,axes=plt.subplots()
axes.scatter(all_points[0,:],all_points[1,:],c=labels)
axes.set_title('Toy Problem')

fig.savefig("classification-1.jpg",dpi=300)
plt.show()

print("#----------------------------------------#")
print("All points shape: {}".format(all_points.shape))
print("#----------------------------------------#")
print("\n")

# train_test_split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(all_points.T,labels,test_size=0.25,shuffle=True)

# Train the SVM classifier
# SVC: support vector classifier

from sklearn.svm import SVC

C=1.0

svc=SVC(kernel="linear")

# fit the train data

svc.fit(X_train,y_train)

# test the classifier

y_pred=svc.predict(X_test)

# score it

score=svc.score(X_test,y_test)

print("#----------------------------------------#")
print("Score: "+str(score))
print("#----------------------------------------#")
print("\n")

# Plotting the Decision boundary surface

def make_meshgrid(x,y,h=0.2):
	'''
	x: data to base x-axis meshgrid on
	y: data to base y-axis meshgridon
	h: stepsize for meshgrid, optional
	
	returns:
	XX,YY:ndarray
	'''
	x_min,x_max=x.min()-1,x.max()+1
	y_min,y_max=y.min()-1,y.max()+1
	xx,yy=np.meshgrid(np.arange(x_min,x_max,h),\
	np.arange(y_min,y_max,h))
	
	return xx,yy
	
def plot_contours(ax,clf,xx,yy,**params):
	'''
	plot the decision boundaries for a classifier
	
	ax: matplotlib axes object
	clf: a classifier
	xx: meshgrid ndarray
	yy: meshgrid ndarray
	params: dictionary of params to pass to contourf, optional
	'''
	Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
	Z=Z.reshape(xx.shape)
	out=ax.contourf(xx,yy,Z,**params)
	return out
	
xx,yy=make_meshgrid(all_points[0,:],all_points[1,:])

fig,axes=plt.subplots()
plot_contours(axes,svc,xx,yy,cmap=plt.cm.coolwarm,alpha=0.8)
axes.scatter(all_points[0,:],all_points[1,:],c=labels,cmap=plt.cm.coolwarm,\
s=20,edgecolors='k')

fig.savefig("classification-2.jpg",dpi=300)
plt.show()

#fig,axes=plt.subplots(nrows=2,ncols=5)
#for C in range(1,11):
#	svc=SVC(kernel="linear")
#	svc.fit(X_train,y_train)
#	plot_contours(axes,svc,xx,yy,cmap=plt.cm.coolwarm,alpha=0.8)
#	axes.scatter(all_points[0,:],all_points[1,:],c=labels,cmap=plt.cm.coolwarm,\
#	s=20,edgecolors='k')
#	
#fig.savefig("classification-3.jpg",dpi=300)
#plt.show()

# Decision Trees

X1 = 0.2 + 0.2*np.random.randn(100)
Y1 = 1 + 0.5*np.random.randn(100)

X2 = 1 + 0.3*np.random.randn(100)
Y2 = -1 + 0.4*np.random.randn(100)

class_1 = np.vstack((X1,Y1))
class_2 = np.vstack((X2,Y2))

labels = np.zeros(shape = (class_1.shape[1]+class_2.shape[1]))

#Create a labels vector
labels[:class_1.shape[1]] = 0
labels[class_1.shape[1]:] = 1

all_points = np.hstack((class_1,class_2)) #join all the points together

#Do the test-train split again
X_train, X_test, y_train, y_test = train_test_split(all_points.T, labels, test_size=0.25, shuffle = True)

#Plot it
fig, axes = plt.subplots()
axes.scatter(all_points[0,:], all_points[1,:], c = labels)
axes.set_title('Toy Problem')

fig.savefig("classification-3.jpg",dpi=300)
plt.show()
	
# import the decision tree

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion='entropy')  

dtc.fit(all_points.T,labels)

score=dtc.score(X_test,y_test)
                  
print("#----------------------------------------#")
print("Score = :",score)
print("#----------------------------------------#")
print("\n")

# plot the decision surface

fig,axes=plt.subplots()

plot_contours(axes,dtc,xx,yy,cmap=plt.cm.coolwarm,alpha=0.8)

axes.scatter(all_points[0,:],all_points[1,:],c=labels,cmap=plt.cm.coolwarm,s=20,\
edgecolors='k')
axes.set_title("Decision Tree Surface")

fig.savefig("classification-4.jpg",dpi=300)
plt.show()

# visualize the decision tree using some viz packages

import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(dtc, out_file=None, 
                         feature_names=['X', 'Y'],  
                         class_names=['Class 0', 'Class 1'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

# KMeans
#num_clusters =4
#estimator=px.processing.Cluster(h5_main,KMeans(n_clusters=num_clusters))
#h5_kmeans_grp=estimator.computer(h5_main)
#h5_kmeans_labels=h5_kmeans_grp['Labels']
#h5_kmeans_mean_resp=h5_kmeans_grp['Mean_Response']
#fig_labels=cluster_utils.plot_cluster_labels(h5_kmeans_labels[:].reshape(num_rows,num_cols),\
#num_clusters=num_clusters,figsize=(8,8))
#fig_centroids=cluster_utils.plot_map_stack(h5_kmeans_resp[:].reshape(num_clusters,\
#h5_main.spec_dim_sizes[0],h5_main.spec_dim_sizes(1))

num_clusters = 4

estimator = px.processing.Cluster(h5_main, KMeans(n_clusters=num_clusters))

h5_kmeans_grp = estimator.compute(h5_main)
h5_kmeans_labels = h5_kmeans_grp['Labels']
h5_kmeans_mean_resp = h5_kmeans_grp['Mean_Response']

fig_labels =cluster_utils.plot_cluster_labels(h5_kmeans_labels[:].reshape(num_rows,num_cols),\
num_clusters = num_clusters, figsize = (8,8))

fig_centroids =cluster_utils.plot_map_stack(h5_kmeans_mean_resp[:].reshape(num_clusters, \
h5_main.spec_dim_sizes[0],h5_main.spec_dim_sizes[1]))

fig.savefig("classification-5.jpg",dpi=300)
plt.show()

