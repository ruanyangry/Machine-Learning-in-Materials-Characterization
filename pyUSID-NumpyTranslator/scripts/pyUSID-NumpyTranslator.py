# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Reference: https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_numpy_translator.html

Purpose: Translation and the NumpyTranslator
         numpy array to USID hdf5 format
         
Steps:

1. Inverstigating how to open the proprietary raw data file
2. Reading the metadata
3. Extracring the data
4. Writing to h5USID file

Get the raw data stored in Scanning Tunnelling Spectroscopy (STS)
obtained by Omicron Scanning Tunneling Microscope (STM) 

STS experiments result: (X,Y,current) 3 dimensions

Attention: in pycroscopy have AscTranslator function to achieved above goals
'''

# import necessary packages

from __future__ import division, print_function, absolute_import, unicode_literals

# system

import os
import subprocess
import sys
import zipfile
from warnings import warn

# define install function

def install(package):
	subprocess.call([sys.executable,"-m","pip","install",package])
	
try:
	import wget
except ImportError:
	warn('wget not found.  Will install with pip.')
	import pip
	install(wget)
	import wget
	
# numpy 

import numpy as np

# read raw data

import h5py

# plot

import matplotlib.pyplot as plt

try:
	import pyUSID as usid
except ImportError:
	warn('pyUSID not found.  Will install with pip.')
	import pip
	install("pyUSID")
	import pyUSID as usid

# Procure the Raw Data File

url='https://raw.githubusercontent.com/pycroscopy/pyUSID/master/data/STS.zip'
zip_path='STS.zip'

if os.path.exists(zip_path):
	os.remove(zip_path)
	
_=wget.download(url,zip_path,bar=None)

zip_path=os.path.abspath(zip_path)

folder_path,_=os.path.split(zip_path)
zip_ref=zipfile.ZipFile(zip_path,"r")

# unzip the file

zip_ref.extractall(folder_path)
zip_ref.close()

# delete the zip file

os.remove(zip_path)

data_file_path='STS.asc'

# Exploring the Raw Data file
# If you don't kown the format, assume is TXT file

with open(data_file_path,"r") as file_handle:
	for lin_ind in range(10):
		print(file_handle.readline())
		
# Loading the data
# STS experiments result in 3 dimensional datasets (X,Y,current)
# Extracting the raw data into memory

file_handle=open(data_file_path,"r")
string_lines=file_handle.readlines()
file_handle.close()

# Read the parameters

parm_dict=dict()

for line in string_lines[3:17]:
	line=line.replace('#','')
	line=line.replace("\n",'')
	temp=line.split('=')
	test=temp[1].strip()
	try:
		test=float(test)
		if test%1==0:
			test=int(test)
	except ValueError:
		pass
	parm_dict[temp[0].strip()]=test
	
# Print out the parameters extracted

for key in parm_dict.keys():
	print(key,":\t",parm_dict[key])
	
# Prepare to read the data

num_rows=int(parm_dict['y-pixels'])
num_cols=int(parm_dict['x-pixels'])
num_pos=num_rows*num_cols
spectra_length=int(parm_dict['z-points'])

# Read the data

# num_headers = len(string_lines)-num_pos

num_headers=403

# Extract the STS data from subsequent lines

raw_data_2d=np.zeros(shape=(num_pos,spectra_length),dtype=np.float32)

for line_ind in range(num_pos):
	this_line=string_lines[num_headers+line_ind]
	string_spectrum=this_line.split("\t")[:-1] # omitting the new line
	raw_data_2d[line_ind]=np.array(string_spectrum,dtype=np.float32)
	
# Preparing some necessary parameters

max_v=1     # This is the one parameter we are not sure about

folder_path,file_name=os.path.split(data_file_path)
file_name=file_name[:-4]+'_'

# Generate the x/voltage/spectroscopic axis:

volt_vec=np.linspace(-1*max_v,1*max_v,spectra_length)

h5_path=os.path.join(folder_path,file_name+'.h5')

sci_data_type='STS'
quantity='Current'
units='nA'

# Defining the Dimensions

pos_dims=[usid.write_utils.Dimension('X','a.u.',parm_dict['x-pixels']),\
usid.write_utils.Dimension('Y','a.u.',parm_dict['y-pixels'])]

spec_dims=usid.write_utils.Dimension('Bias','V',volt_vec)

# Calling the NumpyTranslator to create the h5USID file

tran=usid.NumpyTranslator()
h5_path=tran.translate(h5_path,sci_data_type,raw_data_2d,quantity,units,\
pos_dims,spec_dims,translator_name='Omicron_ASC_Translator',parm_dict=parm_dict)

# Verifying the newly written H5 file

with h5py.File(h5_path,mode="r") as h5_file:
	# see if a tree has been created within the hdf5 file
	usid.hdf_utils.print_tree(h5_file)
	
	h5_main=h5_file['Measurement_000/Channel_000/Raw_Data']
	usid.plot_utils.use_nice_plot_params()
	fig,axes=plt.subplots(ncols=2,figsize=(11,5))
	spat_map=np.reshape(h5_main[:,100],(100,100))
	usid.plot_utils.plot_map(axes[0],spat_map,origin='lower')
	axes[0].set_title('Spatial map')
	axes[0].set_xlabel('X')
	axes[0].set_ylabel('Y')
	axes[1].plot(np.linspace(-1.0,1.0,h5_main.shape[1]),h5_main[250])
	axes[1].set_title('IV curve at a single pixel')
	axes[1].set_xlabel('Tips bias [V]')
	axes[1].set_ylabel('Current [nA]')
	
	fig.tight_layout()
	
# remove both the original and translated files

plt.savefig('numpytranslator-1.jpg',dpi=300)
plt.show()

os.remove(h5_path)
os.remove(data_file_path)



