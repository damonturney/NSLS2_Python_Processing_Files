#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:40:35 2019

['xrfmap'] holds everything
['xrfmap/positions/name'] (the name of the motors scanned, this should be something like 'x_pos', 'y_pos' or 'hf_stage_x')
['xrfmap/positions/pos'] is an array that gives the values of the positions along the dimensions named above
['xrfmap/detsum/counts'] is the sum of ['xrfmap/det1/counts'] + ['xrfmap/det2/counts'] + ['xrfmap/det3/counts']
['xrfmap/detsum/counts'] has the shape of the NxM map x 4096 channels in the detector, each channel is an energy bin of 10 eV. For example, if you want the Mn XRF, you do np.sum(f['xrfmap/detsum/counts'][:, :, 580:600], axis=2)  +/- 100 eV of the Mn emission line at 5899 eV)
['xrfmap/scalers/name']  name of the scalers for the scan, these are the ion chambers, something like 'i0' or 'ic0')
['xrfmap/scalers/val'] the actual values that can be used for normalization. This is all for pyXRF so the user can chose a name for normalization, and then the software will know that name corresponds to the nth values in the val array)

For example, scan 27853, the map was 11x3 so that is where those dimensions come from. The 4096 is related to the detector channels.

The raw counts for each XRF channel contain lots of zeros, but if you look around 590 for the Mn signal, it should be there.

@author: damon
"""

#import pyxrf
import os
import sys
import pathlib
import time
import peakutils
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#### Set working directory ##############################
if sys.platform == 'win32':
    working_folder='C:\\Users\\EI Administrator\\Desktop\\NSLS2_Data_Processing\\20190523_NSLS2_SRX_Data\\Dexela_2DXRD_tifs_AND_xrf_counts\\'
    os.chdir(working_folder)

if sys.platform == 'linux':
    working_folder='/home/damon/Desktop/NSLS2_DataProcessing/20190523_NSLS2_SRX_Data/Dexela_2DXRD_tifs_AND_xrf_counts/'
    os.chdir(working_folder)
    
if sys.platform == 'darwin': #MACBOOK PRO
    working_folder='/Users/damon/Desktop/BACKED_UP/WorkFiles/ProjectsGrants/2015_NYSERDA_Birnessite_Project/2019_NSLS2_Synchrotron_Work/20190523_SRX_Beamtime_Work/Dexela_2DXRD_tifs_AND_xrf_counts/'
    os.chdir(working_folder)
##########################################################

### Create list of xrf h5 files ##############################
object_list_allfilesdirectories = pathlib.Path('.')
object_recursiveglob_h5files = object_list_allfilesdirectories.glob('*.h5')      
object_list_filenames_h5files = list(object_recursiveglob_h5files)
# to paste together a full pathname for a file: "object_list_filenames_tiffiles[i].parents[0].joinpath(object_list_filenames_tiffiles[i].parts[-1][0:-4]"
############################################################


def return_all_xrf_h5_file_names():
    object_list_allfilesdirectories = pathlib.Path('.')
    object_recursiveglob_h5files = object_list_allfilesdirectories.glob('*.h5')      
    object_list_filenames_h5files = list(object_recursiveglob_h5files)
    return object_list_filenames_h5files


def return_single_xrf_spectrum_from_h5(file,scan=0):
    if type(file) != str:
        file = 'scan2D_'+str(file)+'.h5'
    f = h5py.File(file,'r')
    xpositions=np.array(f['xrfmap']['positions']['pos'][0,:,:])
    ypositions=np.array(f['xrfmap']['positions']['pos'][1,:,:])
    xrf_counts=np.array(f['xrfmap']['detsum']['counts'])
    f.close()
    return xrf_counts[scan,:,:], xpositions[scan], ypositions[scan]

def return_xrf_data_from_h5(file):
    if type(file) != str:
        file = 'scan2D_'+str(file)+'.h5'
    f = h5py.File(file,'r')
    xpos=np.array(f['xrfmap']['positions']['pos'][0,:,:])
    ypos=np.array(f['xrfmap']['positions']['pos'][1,:,:])
    xrf_counts=np.array(f['xrfmap']['detsum']['counts'])
    f.close()
    return xrf_counts, xpos, ypos

def return_num_xrfscans_h5(file):
    if type(file) != str:
        file = 'scan2D_'+str(file)+'.h5'
    f = h5py.File(file,'r')
    ypositions=np.array(f['xrfmap']['positions']['pos'][1,:,:])
    num_scans=len(ypositions)
    f.close()
    return num_scans

def contourf_xrf_counts(file,element):
    xrf_counts, xpos, ypos=return_xrf_data_from_h5(file)
    element_counts = np.zeros((xrf_counts.shape[0],xrf_counts.shape[1]))
    for i in range(0,xrf_counts.shape[0]):
        for j in range(0,xrf_counts.shape[1]):
            element_counts[i,j]=return_xrf_counts_single_element(xrf_counts[i,j,:],element)
    fig1, ax2 = plt.subplots(constrained_layout=True)
    CS = ax2.contourf(xpos,ypos,element_counts, 10, cmap=plt.cm.bone, origin="lower")
    ax2.set_title(element + ' XRF map')
    ax2.set_xlabel('y position (mm)')
    ax2.set_ylabel('x position (mm)')
    cbar = fig1.colorbar(CS)
    cbar.ax.set_ylabel('counts')
    


def return_xrf_counts_single_element(spectrum,element):
    if element=='K': return(np.sum(spectrum[310:340]))
    if element=='Mn':return(np.sum(spectrum[580:600]))
    if element=='Fe':return(np.sum(spectrum[630:650]))
    if element=='Cu':return(np.sum(spectrum[683:703]))
    if element=='Ni':return(np.sum(spectrum[738:758]))
    if element=='Zn':return(np.sum(spectrum[854:874]))
    if element=='Bi':return(np.sum(spectrum[984:1184]))

    
    
