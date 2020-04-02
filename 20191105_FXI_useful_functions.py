# -*- coding: utf-8 -*-
"""
Created on Sat Mar 6 2020

"""

import numpy as np
import scipy.ndimage
import h5py
import time
import pathlib
import sys
import datetime
import os
import skimage.io
import pandas
import matplotlib.pyplot as plt
import matplotlib.animation as animation

t0 = time.monotonic()


#### Set working directory ##############################
if sys.platform == 'win32':
    working_folder='C:\\Users\\EI Administrator\\Desktop\\NSLS2_Data_Processing\\20191105_FXI_Beamtime_Work\\NSLS2_Python_Processing_Files\\'
    os.chdir(working_folder)

if sys.platform == 'linux':
    working_folder='/home/damon/Desktop/NSLS2_DataProcessing/NSLS2_Python_Processing_Files'
    os.chdir(working_folder)
    
if sys.platform == 'darwin':
    working_folder='/Users/damon/Desktop/BACKED_UP/WorkFiles/ProjectsGrants/2015_NYSERDA_Birnessite_Project/2019_NSLS2_Synchrotron_Work/NSLS2_Python_Data_Processing_Files'
    os.chdir(working_folder)
##########################################################
    

### Data Directory #######################################
data_directory='../20191105_FXI_Beamtime_Work/Data_From_Beamline/';
data_subdirectory='34563_to_34875/';
results_directory='../20191105_FXI_Beamtime_Work/Results_Of_Data_Processing/';
object_list_allfilesdirectories = pathlib.Path(data_directory+data_subdirectory)
object_recursiveglob_tiffiles = object_list_allfilesdirectories.glob('*.tif')      
object_list_filenames_tiffiles = list(object_recursiveglob_tiffiles)
# to paste together a full pathname for a file: "object_list_filenames_tiffiles[i].parents[0].joinpath(object_list_filenames_tiffiles[i].parts[-1][0:-4]"
############################################################


# location 1 TOTAL list of movie xanes files
# xanes_files = np.concatenate((range(34565,34725,2),range(34726,34875,2)))
#
# location 2 TOTAL list of movie xanes files
# xanes_files = np.concatenate((range(34565.0001,34725.0001,2),range(34726.0001,34875.0001,2)))
#
#
# scans 34565 to 34644 (scans 34645 to 34657 were collected while the potenttiostat was in OCV because it's program had prematurely ended)
# biologic_file = '20191107_Cu-Bi-Birnessite_37NaOH_more_loading_C04.mpt'
#
# scans 34655 to 34724  (34725 is a dud scan because I switched NaOH for KOH)
# biologic_file = '20191107_Cu-Bi-Birnessite_37NaOH_more_loading2_C04.mpt'
#
# scans 34726 to 34838  (scans 34840 to 34854 were during OCV while potentiostat had ended prematurely)
# biologic_file = '20191107_Cu-Bi-Birnessite_37NaOH_more_loading3_C04.mpt'
#
# scans 34856 to 34874 (after 34874 the potentiostat held CV for 3 more hours -- until 11 am)
# biologic_file = '20191107_Cu-Bi-Birnessite_37NaOH_more_loading4_C04.mpt'
#



### Workflow ##############################################
# 1) Run create_aligned_h5_file() on each Manganese multipos_2D_xanes_scan2_[]...h5 file to align the images. Use a command like for i in range(34675,34725,2): create_aligned_h5_file(i);      NOTE:  file 34675 is missing, see your beamline notes -- before 34675 the Mn files are odd numbered and after 34675 the Mn files are even numbered 
# 2) Run calculate_optical_thickness() on each of the processed_images...h5 fiels created in step 1.  Use a command like    Use a command like for i in range(34675,34725,2): calculate_optical_thickness(i);   
# 3) 
############################################################
    


#filename MUST be supplied as a number    
def internally_align_h5_file(Mn_filename,cc_search_distance):  #filename MUST be supplied as a number,    cc_search_distance is the cross correlation search distance
    Mn_filename_string="%.4f" % Mn_filename
    Mn_filename_string='multipos_2D_xanes_scan2_id_'+Mn_filename_string[0:5]+'_repeat_'+Mn_filename_string[6:8]+'_pos_'+Mn_filename_string[8:10]+'.h5'
    h5object_old = h5py.File(data_directory+data_subdirectory+Mn_filename_string, 'r')    
    h5object_new = h5py.File(data_directory+data_subdirectory+'processed_images_'+Mn_filename_string[27:-3]+'.h5', 'w')
    h5object_old.copy('scan_time',h5object_new)
    h5object_old.copy('scan_id',  h5object_new)
    h5object_old.copy('note',     h5object_new)
    buffer_edges = 150
    
    ##### Shift the 2nd Mn image to be aligned with the 1st Mn image
    beam_energies_Mn, Mn_ims = read_FXI_xanes_images(Mn_filename); 
    translation1, error = find_image_translation(Mn_ims[0,:,:],Mn_ims[1,:,:], cc_search_distance)
    im_num_rows = Mn_ims[0,:,:].shape[0]
    im_num_cols = Mn_ims[0,:,:].shape[1]
    # Add a buffer of dummy values so that we don't lose data when we shift
    Mn_im1_buffered = np.pad(Mn_ims[0,:,:], buffer_edges, 'constant', constant_values=0.1234567890123456 ) 
    Mn_im2_buffered = np.pad(Mn_ims[1,:,:], buffer_edges, 'constant', constant_values=0.1234567890123456 ) 
    # Now actually do the  shift
    Mn_im2_buffered_aligned = scipy.ndimage.shift(Mn_im2_buffered, translation1, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
    # Now reset the dummy values because the shift command used spline fitting and messed up some of the dummy values
    Mn_im2_buffered_aligned[:,0:buffer_edges + np.int(np.round(translation1[1]))] = 0.1234567890123456
    Mn_im2_buffered_aligned[:,  buffer_edges + np.int(np.round(translation1[1])) + im_num_cols:] = 0.1234567890123456
    Mn_im2_buffered_aligned[0:buffer_edges + np.int(np.round(translation1[0])),:] = 0.1234567890123456
    Mn_im2_buffered_aligned[  buffer_edges + np.int(np.round(translation1[0])) + im_num_rows:,:] = 0.1234567890123456
    
    ##### Shift the Cu images to be aligned with the 1st Mn image
    beam_energies_Cu, Cu_ims = read_FXI_xanes_images(Mn_filename+1.0);  
    translation2, error = find_image_translation(Mn_ims[0,:,:],Cu_ims[0,:,:], cc_search_distance)
    translation3, error = find_image_translation(Mn_ims[0,:,:],Cu_ims[1,:,:], cc_search_distance)
    # Add a buffer of dummy values so that we don't lose data when we shift
    Cu_im1_buffered = np.pad(Cu_ims[0,:,:], buffer_edges, 'constant', constant_values=0.1234567890123456 ) 
    Cu_im2_buffered = np.pad(Cu_ims[1,:,:], buffer_edges, 'constant', constant_values=0.1234567890123456 ) 
    # Now actually do the  shift
    Cu_im1_buffered_aligned = scipy.ndimage.shift(Cu_im1_buffered, translation2, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
    Cu_im2_buffered_aligned = scipy.ndimage.shift(Cu_im2_buffered, translation3, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)

    # Now reset the dummy values because the shift command used spline fitting and messed up some of the dummy values
    Cu_im1_buffered_aligned[:,0:buffer_edges + np.int(np.round(translation2[1]))] = 0.1234567890123456
    Cu_im1_buffered_aligned[:,  buffer_edges + np.int(np.round(translation2[1])) + im_num_cols:] = 0.1234567890123456
    Cu_im1_buffered_aligned[0:buffer_edges + np.int(np.round(translation2[0])),:] = 0.1234567890123456
    Cu_im1_buffered_aligned[  buffer_edges + np.int(np.round(translation2[0])) + im_num_rows:,:] = 0.1234567890123456
    Cu_im2_buffered_aligned[:,0:buffer_edges + np.int(np.round(translation3[1]))] = 0.1234567890123456
    Cu_im2_buffered_aligned[:,  buffer_edges + np.int(np.round(translation3[1])) + im_num_cols:] = 0.1234567890123456
    Cu_im2_buffered_aligned[0:buffer_edges + np.int(np.round(translation3[0])),:] = 0.1234567890123456
    Cu_im2_buffered_aligned[  buffer_edges + np.int(np.round(translation3[0])) + im_num_rows:,:] = 0.1234567890123456
    
    # Collect an concatenate the data into single matrices
    beam_energies = np.concatenate((beam_energies_Mn, beam_energies_Cu))
    ims_buffered_aligned = np.stack((Mn_im1_buffered,Mn_im2_buffered_aligned,Cu_im1_buffered_aligned,Cu_im2_buffered_aligned))
    
    # Stuff the data into the h5 file
    h5object_new.create_dataset('translations', shape=(3,2),   dtype=np.float64, data=np.stack((translation1,translation2,translation3)))
    h5object_new.create_dataset('beam_energies', shape=(4,1), dtype=np.float64, data=beam_energies)
    h5object_new.create_dataset('xray_images', shape=(4 , im_num_rows + 2*buffer_edges , im_num_cols + 2*buffer_edges), dtype=np.float32, data=ims_buffered_aligned)
    h5object_old.close()
    h5object_new.close()
    
  
    

#filename MUST be supplied as a numpy vector of file numbers
def align_processed_images_time_series(file_numbers):  #filename MUST be supplied as a numpy vector of file numbers
    for i in range(1,len(file_numbers)):
        
        print(file_numbers[i])
        
        filename1="%.4f" % file_numbers[i-1]
        filename1='processed_images_'+filename1[0:5]+'_repeat_'+filename1[6:8]+'_pos_'+filename1[8:10]+'.h5'
        h5object1= h5py.File(data_directory+data_subdirectory+filename1, 'r')        
        xanes_raw_ims1 = np.array(h5object1['xray_images'])
        h5object1.close()
        
        filename2="%.4f" % file_numbers[i]
        filename2='processed_images_'+filename2[0:5]+'_repeat_'+filename2[6:8]+'_pos_'+filename2[8:10]+'.h5'
        h5object2= h5py.File(data_directory+data_subdirectory+filename2, 'r+')
        xanes_raw_ims2 = np.array(h5object2['xray_images'])
        
        # This info will be use several times
        im_shape_rows = xanes_raw_ims1.shape[1]
        im_shape_cols = xanes_raw_ims1.shape[2]
        im_half_rows = np.int(im_shape_rows/2)
        im_half_cols = np.int(im_shape_cols/2)

        # Figure out how much dummy values to remove on each side.  For example: value of debuffer[0] is the maximum column number of where the dummy values extend on the LHS-side of any one of the images(xanes_raw_ims1 or xanes_raw_ims2), and likewise debuffer[2] is the maximum row that the dummy values extend on the topside of any one of the images (xanes_raw_ims1 or xanes_raw_ims2)
        debuffer = [0,0,0,0]  # debuffer[0] is how many LHS dummy columns.  debuffer[1] is how many RHS dummy columns.  debuffer[2] is how many topside dummy rows.  debuffer[3] is how many bottomside dummy rows.  
        debuffer[0] = int(np.max([ np.max(np.where(xanes_raw_ims1[:,  im_half_rows ,0:im_half_cols ]==0.1234567890123456)[1])    ,  np.max(np.where(xanes_raw_ims2[:,  im_half_rows ,0:im_half_cols ]==0.1234567890123456)[1])     ]) )
        debuffer[1] = int(np.min([ np.min(np.where(xanes_raw_ims1[:,  im_half_rows ,  im_half_cols:]==0.1234567890123456)[1])    ,  np.min(np.where(xanes_raw_ims2[:,  im_half_rows ,  im_half_cols:]==0.1234567890123456)[1])     ])  +  im_half_cols )
        debuffer[2] = int(np.max([ np.max(np.where(xanes_raw_ims1[:,0:im_half_rows ,  im_half_cols ]==0.1234567890123456)[1])    ,  np.max(np.where(xanes_raw_ims2[:,0:im_half_rows ,  im_half_cols ]==0.1234567890123456)[1])     ]) )
        debuffer[3] = int(np.max([ np.max(np.where(xanes_raw_ims1[:,  im_half_rows:,  im_half_cols ]==0.1234567890123456)[1])    ,  np.max(np.where(xanes_raw_ims2[:,  im_half_rows:,  im_half_cols ]==0.1234567890123456)[1])     ])  +  im_half_rows )
        
        #Find out how much translation to move each image
        cc_search_distance = 10
        translation1, error1 = find_image_translation( xanes_raw_ims1[0,debuffer[0]+1:debuffer[1],debuffer[2]+1:debuffer[3]] , xanes_raw_ims2[0,debuffer[0]+1:debuffer[1],debuffer[2]+1:debuffer[3]] , cc_search_distance)
        translation2, error2 = find_image_translation( xanes_raw_ims1[1,debuffer[0]+1:debuffer[1],debuffer[2]+1:debuffer[3]] , xanes_raw_ims2[1,debuffer[0]+1:debuffer[1],debuffer[2]+1:debuffer[3]] , cc_search_distance)
        translation3, error3 = find_image_translation( xanes_raw_ims1[2,debuffer[0]+1:debuffer[1],debuffer[2]+1:debuffer[3]] , xanes_raw_ims2[2,debuffer[0]+1:debuffer[1],debuffer[2]+1:debuffer[3]] , cc_search_distance)
        translation4, error4 = find_image_translation( xanes_raw_ims1[3,debuffer[0]+1:debuffer[1],debuffer[2]+1:debuffer[3]] , xanes_raw_ims2[3,debuffer[0]+1:debuffer[1],debuffer[2]+1:debuffer[3]] , cc_search_distance)

        # Calculate the averaged shift to use (weighted by how much error ocurred on each calculation of the images translation)
        sum_inverse_errors = 1/error1 + 1/error2 + 1/error3 + 1/error4
        translation = np.zeros(2)
        translation[0] = np.sum([translation1[0]*(1/error1)/sum_inverse_errors , translation2[0]*(1/error2)/sum_inverse_errors , translation3[0]*(1/error3)/sum_inverse_errors , translation4[0]*(1/error4)/sum_inverse_errors ])
        translation[1] = np.sum([translation1[1]*(1/error1)/sum_inverse_errors , translation2[1]*(1/error2)/sum_inverse_errors , translation3[1]*(1/error3)/sum_inverse_errors , translation4[1]*(1/error4)/sum_inverse_errors ])
        
        print(translation1, error1)
        print(translation2, error2)
        print(translation3, error3)
        print(translation4, error4)
        print(translation)
        
        # Now actually shift the images to be in alignment
        im2_1 = scipy.ndimage.shift(xanes_raw_ims2[0,:,:], translation, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
        im2_2 = scipy.ndimage.shift(xanes_raw_ims2[1,:,:], translation, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
        im2_3 = scipy.ndimage.shift(xanes_raw_ims2[2,:,:], translation, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
        im2_4 = scipy.ndimage.shift(xanes_raw_ims2[3,:,:], translation, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)       
        temp_matrix = h5object2['xray_images']
        temp_matrix[...] = np.stack((im2_1,im2_2,im2_3,im2_4)) # stupid ass python requires this [...] notation if you want to write data to the h5 file
        #h5object2.create_dataset('xray_images2', shape=(4,im_shape_rows,im_shape_cols), dtype=np.float32, data=np.stack((im2_1,im2_2,im2_3,im2_4)))
        h5object2.close()
    
    
    
    
    
    
#Run this on the files output from save_aligned_h5_file     #All length units in mm
def calculate_optical_thickness(filename, carbon_thickness=0.15, total_thickness=0.2): #Run this on the files output from save_aligned_h5_file
    if type(filename) != str:
        filename="%.4f" % filename
        filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object= h5py.File(data_directory+data_subdirectory+filename, 'r+')
    ims     = np.array(h5object['xray_images'])
    ims[ims<=0.0]=np.median(ims[ims>0]) #so that np.log doesn't create an error
        
    # X-ray absorption coefficients in units of 1/mm 
    a_6520_Mn = 31.573;  a_6600_Mn = 207.698; a_8970_Mn = 94.641; a_9050_Mn = 92.436;  #All length units in mm
    a_6520_Cu = 85.1;    a_6600_Cu = 82.9;  a_8970_Cu = 34; a_9050_Cu = 265;
    a_6520_Bi = 389.43;  a_6600_Bi = 377.563; a_8970_Bi = 172.32; a_9050_Bi = 168.434;
    a_6520_El = 3.254;   a_6600_El = 3.136;   a_8970_El = 1.229;  a_9050_El = 1.1966;  #1 part NaOH and 5 parts H2O (by mole ratios)
    a_6520_C  = 1.8278;  a_6600_C  = 1.759;   a_8970_C  = 0.6759; a_9050_C  = 0.6577;
    
    sum_aMn_aEl = (a_6520_Mn*a_6520_El + a_6600_Mn*a_6600_El + a_8970_Mn*a_8970_El + a_9050_Mn*a_9050_El)
    sum_aCu_aEl = (a_6520_Cu*a_6520_El + a_6600_Cu*a_6600_El + a_8970_Cu*a_8970_El + a_9050_Cu*a_9050_El)
    sum_aBi_aEl = (a_6520_Bi*a_6520_El + a_6600_Bi*a_6600_El + a_8970_Bi*a_8970_El + a_9050_Bi*a_9050_El)
    sum_aMn_aMn = (a_6520_Mn*a_6520_Mn + a_6600_Mn*a_6600_Mn + a_8970_Mn*a_8970_Mn + a_9050_Mn*a_9050_Mn)
    sum_aCu_aMn = (a_6520_Cu*a_6520_Mn + a_6600_Cu*a_6600_Mn + a_8970_Cu*a_8970_Mn + a_9050_Cu*a_9050_Mn)
    sum_aBi_aMn = (a_6520_Bi*a_6520_Mn + a_6600_Bi*a_6600_Mn + a_8970_Bi*a_8970_Mn + a_9050_Bi*a_9050_Mn)
    sum_aBi_aCu = (a_6520_Cu*a_6520_Bi + a_6600_Cu*a_6600_Bi + a_8970_Cu*a_8970_Bi + a_9050_Cu*a_9050_Bi)
    sum_aCu_aCu = (a_6520_Cu*a_6520_Cu + a_6600_Cu*a_6600_Cu + a_8970_Cu*a_8970_Cu + a_9050_Cu*a_9050_Cu)
    sum_aBi_aBi = (a_6520_Bi*a_6520_Bi + a_6600_Bi*a_6600_Bi + a_8970_Bi*a_8970_Bi + a_9050_Bi*a_9050_Bi)
    
    A = np.zeros((3,3))
    A[0,:] = [ sum_aMn_aEl - sum_aMn_aMn , sum_aMn_aEl - sum_aCu_aMn , sum_aMn_aEl - sum_aBi_aMn ]
    A[1,:] = [ sum_aCu_aEl - sum_aCu_aMn , sum_aCu_aEl - sum_aCu_aCu , sum_aCu_aEl - sum_aBi_aCu ]
    A[2,:] = [ sum_aBi_aEl - sum_aBi_aMn , sum_aBi_aEl - sum_aBi_aCu , sum_aBi_aEl - sum_aBi_aBi ]
    
    b = np.zeros((3,1))
    
    optical_thickness_Cu=np.zeros(ims[0,:,:].shape,dtype=np.float32)   #All length units in mm
    optical_thickness_Mn=np.zeros(ims[0,:,:].shape,dtype=np.float32)
    optical_thickness_Bi=np.zeros(ims[0,:,:].shape,dtype=np.float32)
    optical_thickness_El=np.zeros(ims[0,:,:].shape,dtype=np.float32)
    optical_thickness_C= np.ones(ims[0,:,:].shape,dtype=np.float32)*carbon_thickness   # in units of mm. This is set in stone, not optimized. I have the two PMMA films plus the carbon foil inside  I can't remember how thick the PMMA films are
    
    
    for m in range(0,ims[0,:,:].shape[0]):
        if np.mod(m,10)==0: sys.stdout.write('\rLeast Squares, Row: '+str(m))
        sys.stdout.flush()
        for n in range(0,ims[0,:,:].shape[1]):        
            ln_I_I0_6520 = np.log(ims[0,m,n]);  ln_I_I0_6600 = np.log(ims[1,m,n]); ln_I_I0_8970 = np.log(ims[2,m,n]); ln_I_I0_9050 = np.log(ims[3,m,n]);
            
            b[0] = (total_thickness - optical_thickness_C[m,n])*sum_aMn_aEl + (a_6520_Mn*ln_I_I0_6520 + a_6600_Mn*ln_I_I0_6600 + a_8970_Mn*ln_I_I0_8970 + a_9050_Mn*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Mn*a_6520_C + a_6600_Mn*a_6600_C + a_8970_Mn*a_8970_C + a_9050_Mn*a_9050_C)
            b[1] = (total_thickness - optical_thickness_C[m,n])*sum_aCu_aEl + (a_6520_Cu*ln_I_I0_6520 + a_6600_Cu*ln_I_I0_6600 + a_8970_Cu*ln_I_I0_8970 + a_9050_Cu*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Cu*a_6520_C + a_6600_Cu*a_6600_C + a_8970_Cu*a_8970_C + a_9050_Cu*a_9050_C)
            b[2] = (total_thickness - optical_thickness_C[m,n])*sum_aBi_aEl + (a_6520_Bi*ln_I_I0_6520 + a_6600_Bi*ln_I_I0_6600 + a_8970_Bi*ln_I_I0_8970 + a_9050_Bi*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Bi*a_6520_C + a_6600_Bi*a_6600_C + a_8970_Bi*a_8970_C + a_9050_Bi*a_9050_C)
            
            temp = np.linalg.solve(A, b)
            optical_thickness_Mn[m,n] = np.float32(temp[0])  #this produces optical thickness in mm    
            optical_thickness_Cu[m,n] = np.float32(temp[1])  #this produces optical thickness in mm
            optical_thickness_Bi[m,n] = np.float32(temp[2])  #this produces optical thickness in mm         
            optical_thickness_El[m,n] = total_thickness - optical_thickness_C[m,n] - optical_thickness_Mn[m,n] - optical_thickness_Cu[m,n] - optical_thickness_Bi[m,n]  #this produces optical thickness in mm         
            sum_square_errors1 = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu[m,n],optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El[m,n])
            
    sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
                   
    #check if the optical_thickness datasets are ALREADY in the h5object, and save data
    h5object['optical_thickness_Mn'][:]=optical_thickness_Mn if 'optical_thickness_Mn' in h5object.keys()  else h5object.create_dataset('optical_thickness_Mn', shape=(ims.shape[1],ims.shape[2]), dtype=np.float32, data=optical_thickness_Mn)
    h5object['optical_thickness_Cu'][:]=optical_thickness_Cu if 'optical_thickness_Cu' in h5object.keys()  else h5object.create_dataset('optical_thickness_Cu', shape=(ims.shape[1],ims.shape[2]), dtype=np.float32, data=optical_thickness_Cu)
    h5object['optical_thickness_Bi'][:]=optical_thickness_Bi if 'optical_thickness_Bi' in h5object.keys()  else h5object.create_dataset('optical_thickness_Bi', shape=(ims.shape[1],ims.shape[2]), dtype=np.float32, data=optical_thickness_Bi)
    h5object['optical_thickness_C' ][:]=optical_thickness_C  if 'optical_thickness_C'  in h5object.keys()  else h5object.create_dataset('optical_thickness_C' , shape=(ims.shape[1],ims.shape[2]), dtype=np.float32, data=optical_thickness_C)
    h5object['optical_thickness_El'][:]=optical_thickness_El if 'optical_thickness_El' in h5object.keys()  else h5object.create_dataset('optical_thickness_El', shape=(ims.shape[1],ims.shape[2]), dtype=np.float32, data=optical_thickness_El)
    h5object.close()
        




def make_movie_34565_to_34875(file_numbers,biologic_file, location=1):
    if location == 1:
        file_numbers = file_numbers
        
    if location == 2:
        file_numbers = file_numbers + 0.0001
        
    # Read timestamps of the images, The Biologic Computer time was 3 Minutes AHEAD of "real" time ( aka the xanes images times)
    datetime_array_xanes = np.array([])
    timestamp_array_xanes = np.array([])
    for i in file_numbers:
        datetime_xanes_file  = datetime.datetime.fromtimestamp(read_FXI_xanes_timestamp_datetime_datetime(i))
        timestamp_xanes_file = datetime_xanes_file.timestamp()
        datetime_array_xanes  = np.concatenate( (datetime_array_xanes , np.array([datetime_xanes_file]) ) )
        timestamp_array_xanes = np.concatenate( (timestamp_array_xanes , np.array([timestamp_xanes_file]) ) )
    
    # Read Biologic Potentiostat Data
    fileobject=open(data_directory+'Biologic_Files/'+biologic_file,'r',errors='ignore')
    fileobject.seek(0)
    fileobject.readline()
    second_line=fileobject.readline()
    for i in range(1,11):
        fileobject.readline()       
    thirteenth_line=fileobject.readline()
    fileobject.close()
    biologic_data=pandas.read_csv(data_directory+'Biologic_Files/20191107_Cu-Bi-Birnessite_37NaOH_more_loading_C04.mpt', sep='\t', skiprows=int(second_line[18:21])-1)
    biologic_start_time=datetime.datetime.strptime(thirteenth_line[25:44],'%m/%d/%Y %H:%M:%S')
    
    # Create the static plot axes
    fig_han, axs_han = plt.subplots(3,1)
    fig_han.set_size_inches(3.5,4.5)
    big_axes_han=plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=2)
    big_axes_han.set_ylabel('y-direction (micron)')
    #big_axes_han.set_ylim([0,40])
    big_axes_han.set_xlabel('x-direction (micron)')
    #big_axes_han.set_xlim([0,40])
    temp=read_image_from_processed_file(file_numbers[0],'xray_images')
    im=temp[0,:,:]
    #im=read_image_from_processed_file(file_numbers[0],'Mn_thickness')
    big_axes_han.imshow(im,cmap='gray',interpolation='none', extent=[0,40,0,40], vmin=-0.001, vmax=0.006)
    print('displaying img: ' + str(file_numbers[0]))
    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.35, wspace=0.01)
    small_axes_han=plt.subplot2grid((3,1),(2,0))
    small_axes_han.set_xlabel('Electrode Voltage (V)')
    small_axes_han.set_ylabel('Current (mA)')
    plt.plot(biologic_data['Ewe/V'].values[0:1500],biologic_data['<I>/mA'].values[0:1500])
    scatter_han = small_axes_han.scatter(biologic_data['Ewe/V'].values[0],biologic_data['<I>/mA'].values[0],c='r',s=25)
       
    def change_imshow(frame_num):
        time_per_iteration = 20  #One frame per 20 seconds
        frame_time = biologic_start_time + datetime.timedelta(seconds = frame_num*time_per_iteration) - datetime.timedelta(minutes = 3) #The Biologic Computer time was 3 Minutes AHEAD of "real" time ( aka the xanes images times)   
        closest_index=abs(biologic_data['time/s'].values - frame_num*time_per_iteration).argmin() #One frame per 20 seconds
        scatter_han.set_offsets([biologic_data['Ewe/V'].values[closest_index],biologic_data['<I>/mA'].values[closest_index]])
        if frame_num % 6 == 1:  
            closest_index=abs(timestamp_array_xanes - np.float64(frame_time.timestamp())).argmin()
            temp=read_image_from_processed_file(file_numbers[closest_index],'xray_images')
            image=temp[0,:,:]
            #image   = read_image_from_processed_file(file_numbers[closest_index],'Mn_thickness')
            big_axes_han.imshow(image,cmap='gray',interpolation='none', extent=[0,40,0,40], vmin=-0.001, vmax=0.8)
            print('displaying img: ' + str(file_numbers[closest_index]))
            print('elapsed time: ' + str(frame_num*time_per_iteration))

            #big_axes_han.imshow(images[int((frame_num-1)/6)],cmap='gray',interpolation='none', extent=[0,40,0,40], vmin=0.235, vmax=0.94)
        
    # It iterates through e.g. "frames=range(15)" calling the function e.g "change_imshow" , and inserts a millisecond time delay between frames of e.g. "interval=100".
    animation_handle=animation.FuncAnimation(fig_han, change_imshow, frames=range(100), blit=False, interval=100, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15)
    animation_handle.save('im.mp4', writer=writer)



def make_movie_just_TXM_raw(file_numbers):
        
    # Create the static plot axes
    ims=read_image_from_processed_file(file_numbers[0],'xray_images')
    fig_han, axs_han = plt.subplots(1)
    axs_han.imshow(ims[0,:,:],cmap='gray',interpolation='none', extent=[0,40,0,40], vmin=-0.001, vmax=0.006)
       
    def change_imshow(frame_num):
        ims=read_image_from_processed_file(file_numbers[frame_num],'xray_images')
        axs_han.imshow(ims[0,:,:],cmap='gray',interpolation='none', extent=[0,40,0,40], vmin=-0.001, vmax=0.8)
        print('displaying img: ' + str(file_numbers[frame_num]))

        
    # It iterates through e.g. "frames=range(15)" calling the function e.g "change_imshow" , and inserts a millisecond time delay between frames of e.g. "interval=100".
    animation_handle=animation.FuncAnimation(fig_han, change_imshow, frames=range(20), blit=False, interval=100, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15)
    animation_handle.save('im.mp4', writer=writer)




    
    
#filename MUST be supplied as a number
def plot_single_pixel_least_squares_data(filename,row,column):   #filename MUST be supplied as a number
    if type(filename) != str:
        filename="%.4f" % filename
        filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object= h5py.File(data_directory+data_subdirectory+filename, 'r')

    beam_energies = np.array(h5object['beam_energies'])
    xanes_ims     = np.array(h5object['xray_images'])
    xanes_ims[xanes_ims<=0.0]=np.median(xanes_ims[xanes_ims>0]) #so that np.log doesn't create an error
    optical_thickness_Mn= np.array(h5object['optical_thickness_Mn'])
    optical_thickness_Cu= np.array(h5object['optical_thickness_Cu'])
    optical_thickness_Bi= np.array(h5object['optical_thickness_Bi'])
    optical_thickness_C = np.array(h5object['optical_thickness_C'])
    optical_thickness_El= np.array(h5object['optical_thickness_El'])
    
    # X-ray absorption coefficients in units of 1/mm 
    a_6520_Mn = 31.573;  a_6600_Mn = 207.698; a_8970_Mn = 94.641; a_9050_Mn = 92.436;
    a_6520_Cu = 85.1;    a_6600_Cu = 82.9;  a_8970_Cu = 34; a_9050_Cu = 265;
    a_6520_Bi = 389.43;  a_6600_Bi = 377.563; a_8970_Bi = 172.32; a_9050_Bi = 168.434;
    a_6520_El = 3.254;   a_6600_El = 3.136;   a_8970_El = 1.229;  a_9050_El = 1.1966;  #1 part NaOH and 5 parts H2O (by mole ratios)
    a_6520_C  = 1.8278;  a_6600_C  = 1.759;   a_8970_C  = 0.6759; a_9050_C  = 0.6577;
    
    #Plot measured data as an image
    plt.scatter(beam_energies, xanes_ims[:,row,column],s=50*np.ones(4),marker='x')
    
    #Plot least-squares model results
    plt.scatter(beam_energies, [np.exp(-a_6520_Mn*optical_thickness_Mn[row,column]), np.exp(-a_6600_Mn*optical_thickness_Mn[row,column]), np.exp(-a_8970_Mn*optical_thickness_Mn[row,column]), np.exp(-a_9050_Mn*optical_thickness_Mn[row,column])],marker='o',facecolors='none',edgecolors='r')
    plt.scatter(beam_energies, [np.exp(-a_6520_Cu*optical_thickness_Cu[row,column]), np.exp(-a_6600_Cu*optical_thickness_Cu[row,column]), np.exp(-a_8970_Cu*optical_thickness_Cu[row,column]), np.exp(-a_9050_Cu*optical_thickness_Cu[row,column])],marker='o',facecolors='none',edgecolors='g')
    plt.scatter(beam_energies, [np.exp(-a_6520_Bi*optical_thickness_Bi[row,column]), np.exp(-a_6600_Bi*optical_thickness_Bi[row,column]), np.exp(-a_8970_Bi*optical_thickness_Bi[row,column]), np.exp(-a_9050_Bi*optical_thickness_Bi[row,column])],marker='o',facecolors='none',edgecolors='b')
    plt.scatter(beam_energies, [np.exp(-a_6520_C *optical_thickness_C[row,column] ), np.exp(-a_6600_C *optical_thickness_C[row,column]) , np.exp(-a_8970_C *optical_thickness_C[row,column]) , np.exp(-a_9050_C*optical_thickness_C[row,column] )], marker='o',facecolors='none',edgecolors='tab:pink')
    plt.scatter(beam_energies, [np.exp(-a_6520_El*optical_thickness_El[row,column]), np.exp(-a_6600_El*optical_thickness_El[row,column]), np.exp(-a_8970_El*optical_thickness_El[row,column]), np.exp(-a_9050_El*optical_thickness_El[row,column])],marker='o',facecolors='none',edgecolors='tab:brown')
    model_I_I0_6520=np.exp(-a_6520_Mn*optical_thickness_Mn[row,column] - a_6520_Cu*optical_thickness_Cu[row,column] - a_6520_Bi*optical_thickness_Bi[row,column] - a_6520_C*optical_thickness_C[row,column] - a_6520_El*optical_thickness_El[row,column])
    model_I_I0_6600=np.exp(-a_6600_Mn*optical_thickness_Mn[row,column] - a_6600_Cu*optical_thickness_Cu[row,column] - a_6600_Bi*optical_thickness_Bi[row,column] - a_6600_C*optical_thickness_C[row,column] - a_6600_El*optical_thickness_El[row,column])
    model_I_I0_8970=np.exp(-a_8970_Mn*optical_thickness_Mn[row,column] - a_8970_Cu*optical_thickness_Cu[row,column] - a_8970_Bi*optical_thickness_Bi[row,column] - a_8970_C*optical_thickness_C[row,column] - a_8970_El*optical_thickness_El[row,column])
    model_I_I0_9050=np.exp(-a_9050_Mn*optical_thickness_Mn[row,column] - a_9050_Cu*optical_thickness_Cu[row,column] - a_9050_Bi*optical_thickness_Bi[row,column] - a_9050_C*optical_thickness_C[row,column] - a_9050_El*optical_thickness_El[row,column])
    plt.scatter(beam_energies, [model_I_I0_6520 , model_I_I0_6600 , model_I_I0_8970 , model_I_I0_9050 ] ,marker='o',facecolors='none',edgecolors='tab:orange')
    plt.legend(['measured','model: Mn','model: Cu','model: Bi','model: C','model: El','model: Mn,Bi,Cu'])

    #plt.figure()
    #plt.scatter(['Mn', 'Cu', 'Bi', 'C', 'El' ], [optical_thickness_Mn[row,column], optical_thickness_Cu[row,column], optical_thickness_Bi[row,column], optical_thickness_C[row,column], optical_thickness_El[row,column]])
    print(optical_thickness_Mn[row,column] + optical_thickness_Cu[row,column] + optical_thickness_Bi[row,column] + optical_thickness_C[row,column] + optical_thickness_El[row,column])
    print('Mn: ' + str(optical_thickness_Mn[row,column]))
    print('Cu: ' + str(optical_thickness_Cu[row,column]))
    print('Bi: ' + str(optical_thickness_Bi[row,column]))
    print('C: ' +  str(optical_thickness_C[row,column]))
    print('El: ' + str(optical_thickness_El[row,column]))

    
    plt.figure()
    imshow_image=xanes_ims[0,:,:]
    imshow_image[row-1:row+1,column-1:column+1]=1.0
    plt.imshow(imshow_image)





def read_FXI_xanes_timestamp_datetime_datetime(filename):  #filename can be 34567.0103  to denote repeat 01, position 03
    if type(filename) != str:
        filename="%.4f" % filename
        filename='multipos_2D_xanes_scan2_id_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object    = h5py.File(data_directory+data_subdirectory+filename, 'r')
    scan_time   = np.array(h5object['scan_time'])  #scan start time in local time at NSLS2, in epoch format 
    h5object.close()
    return( scan_time)




def read_FXI_raw_h5_metadata(filename):  #filename can be 34567.0103  to denote repeat 01, position 03
    if type(filename) != str:
        filename="%.4f" % filename
        filename='multipos_2D_xanes_scan2_id_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object    = h5py.File(data_directory+data_subdirectory+filename, 'r')
    beam_energy = np.array(h5object['X_eng'])
    scan_time   = np.array(h5object['scan_time'])  #scan start time in local time at NSLS2, in epoch format
    scan_id     = np.array(h5object['scan_id'])
    notes       = np.array(h5object['note'])
    scan_start_time = datetime.datetime.fromtimestamp(scan_time)
    scan_start_time_string = datetime.datetime.strftime(scan_start_time, '%Y-%m-%d %H:%M:%S' )    
    h5object.close() 
    if 'translations' in h5object.keys():
        return(scan_start_time_string, beam_energy, scan_id, notes, translations)
    else:
        return(scan_start_time_string, beam_energy, scan_id, notes) 






def read_FXI_processed_h5_metadata(filename):  #filename can be 34567.0103  to denote repeat 01, position 03
    if type(filename) != str:
        filename="%.4f" % filename
        filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object= h5py.File(data_directory+data_subdirectory+filename, 'r+')
    beam_energy = np.array(h5object['beam_energies'])
    scan_time   = np.array(h5object['scan_time'])  #scan start time in local time at NSLS2, in epoch format
    scan_id     = np.array(h5object['scan_id'])
    notes       = np.array(h5object['note'])
    translations= np.array(h5object['translations'])
    scan_start_time = datetime.datetime.fromtimestamp(scan_time)
    scan_start_time_string = datetime.datetime.strftime(scan_start_time, '%Y-%m-%d %H:%M:%S' )    
    h5object.close() 
    return(scan_start_time_string, beam_energy, scan_id, notes, translations)





def read_FXI_xanes_images(filename):
    if type(filename) != str:
        filename="%.4f" % filename
        filename='multipos_2D_xanes_scan2_id_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    print(filename)
    h5object    = h5py.File(data_directory+data_subdirectory+filename, 'r')
    images      = np.array(h5object['img_xanes'])  
    beam_energy = np.array(h5object['X_eng'])
    h5object.close()
    return(beam_energy, images)




def read_image_from_processed_file(filename,which_image='all'):
    if type(filename) != str:
        filename="%.4f" % filename
        filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object= h5py.File(data_directory+data_subdirectory+filename, 'r')
    xray_images = np.array(h5object['xray_images'])
#    optical_thickness_Mn= np.array(h5object['optical_thickness_Mn'])
#    optical_thickness_Cu= np.array(h5object['optical_thickness_Cu'])
#    optical_thickness_Bi= np.array(h5object['optical_thickness_Bi'])
#    optical_thickness_C = np.array(h5object['optical_thickness_C'])
#    optical_thickness_El= np.array(h5object['optical_thickness_El'])
    
    if which_image == 'all':
        return(np.stack((xray_images,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)))
    
    if which_image == 'xray_images':
        return(xray_images)

    if which_image == 'Mn_raw_im1':
        return(xray_images[0,:,:])

    if which_image == 'Mn_raw_im2':
        return(xray_images[1,:,:])

    if which_image == 'Cu_raw_im1':
        return(xray_images[2,:,:])

    if which_image == 'Cu_raw_im2':
        return(xray_images[3,:,:])

    if which_image == 'all_thickness':
        return(np.stack((optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)))
        
    if which_image == 'Mn_thickness':
        return(optical_thickness_Mn)
        
    if which_image == 'Cu_thickness':
        return(optical_thickness_Mn)
        
    if which_image == 'Bi_thickness':
        return(optical_thickness_Mn)
        
    if which_image == 'C_thickness':
        return(optical_thickness_Mn)
        
    if which_image == 'El_thickness':
        return(optical_thickness_Mn)
        
        
        
        
        
        
    

    
def dS_dthickness_all(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El):
    test_sum_squares = np.zeros(8)
    
    #Calculate the test case sum of squared errors
    optical_thickness_Mn1 = optical_thickness_Mn + 0.00001
    test_sum_squares[0] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn1,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
    optical_thickness_Mn1 = optical_thickness_Mn - 0.00001
    test_sum_squares[1] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn1,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
    
    optical_thickness_Cu1 = optical_thickness_Cu + 0.00001
    test_sum_squares[2] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu1,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
    optical_thickness_Cu1 = optical_thickness_Cu - 0.00001
    test_sum_squares[3] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu1,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)

    optical_thickness_Bi1 = optical_thickness_Bi + 0.00001
    test_sum_squares[4] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi1,optical_thickness_C,optical_thickness_El)
    optical_thickness_Bi1 = optical_thickness_Bi - 0.00001
    test_sum_squares[5] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi1,optical_thickness_C,optical_thickness_El)

    optical_thickness_El1 = optical_thickness_El + 0.00001
    test_sum_squares[6] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El1)
    optical_thickness_El1 = optical_thickness_El - 0.00001
    test_sum_squares[7] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El1)

    return(test_sum_squares)
    
    
    
    
def calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El):
    #Calculate the baseline sum of squared errors
    sum_squares_6520=(ln_I_I0_6520 + a_6520_Mn*optical_thickness_Mn + a_6520_Cu*optical_thickness_Cu + a_6520_Bi*optical_thickness_Bi + a_6520_C*optical_thickness_C + a_6520_El*optical_thickness_El)**2
    sum_squares_6600=(ln_I_I0_6600 + a_6600_Mn*optical_thickness_Mn + a_6600_Cu*optical_thickness_Cu + a_6600_Bi*optical_thickness_Bi + a_6600_C*optical_thickness_C + a_6600_El*optical_thickness_El)**2
    sum_squares_8970=(ln_I_I0_8970 + a_8970_Mn*optical_thickness_Mn + a_8970_Cu*optical_thickness_Cu + a_8970_Bi*optical_thickness_Bi + a_8970_C*optical_thickness_C + a_8970_El*optical_thickness_El)**2
    sum_squares_9050=(ln_I_I0_9050 + a_9050_Mn*optical_thickness_Mn + a_9050_Cu*optical_thickness_Cu + a_9050_Bi*optical_thickness_Bi + a_9050_C*optical_thickness_C + a_9050_El*optical_thickness_El)**2
    baseline_sum_square_error = sum_squares_6520 + sum_squares_6600 + sum_squares_8970 + sum_squares_9050
    return(baseline_sum_square_error)




def show_cross_correlation_map(im1,im2,debuffer=0):
    im1_debuffered = im1[debuffer:(im1.shape[0]-debuffer),debuffer:(im1.shape[1]-debuffer)]
    im2_debuffered = im2[debuffer:(im2.shape[0]-debuffer),debuffer:(im1.shape[1]-debuffer)]
    image_product=np.fft.fft2(im1_debuffered) * np.fft.fft2(im2_debuffered).conj();
    cc_image=np.fft.fftshift(np.fft.ifft2(image_product));
    plt.imshow(cc_image)




#This function returns the number of pixels that the second image is translated (positive going [downward, to-the-right]) with respect to the first image
def find_image_translation( im1, im2, im1_reduction): 
    # debuffer is meant to elimated the SAME amount of padding on im1 and im2 BEFORE the im2_masking is applied
    # im1_reduction is the reduction in size of im2 so that it can be used for cross-correlations at different locations on top of im1
    
    # Rreduce the size of im2 so that it can be translated as a window over the top of im1
    im2_reduced = im2[im1_reduction:-im1_reduction, im1_reduction:-im1_reduction ]
    
    # Calculate the cross correlations
    cc_image=erc_R(im1, im2_reduced)   #cross correlation image

    # Figure out the translation to align im2 to im1
    max_indices=np.array(np.unravel_index(np.argmax(cc_image,axis=None), cc_image.shape))
    translation_y = np.float64(max_indices[0] - im1_reduction)
    translation_x = np.float64(max_indices[1] - im1_reduction)
    
    ## Now do subpixel resolution
    y_data = cc_image[max_indices[0]-1 : max_indices[0]+2, max_indices[1]                      ]
    x_data = cc_image[max_indices[0]                     , max_indices[1]-1 : max_indices[1]+2 ]
    print(max_indices, x_data)
    print(max_indices[1]-1,max_indices[1]+2)
    plt.imshow(cc_image)
    y_data = y_data - np.min(y_data)
    x_data = x_data - np.min(x_data)
    translation_y = translation_y + ( -1*y_data[0] + 0*y_data[1] + 1*y_data[2] ) / np.sum(y_data)
    translation_x = translation_x + ( -1*x_data[0] + 0*x_data[1] + 1*x_data[2] ) / np.sum(x_data)
    #a = np.arange(max_indices[0]-5, max_indices[0]+5.001,0.1)
    #b = np.arange(max_indices[0]-5, max_indices[0]+5.001,1.0)
    #c = np.arange(max_indices[1]-5, max_indices[1]+5.001,0.1)
    #d = np.arange(max_indices[1]-5, max_indices[1]+5.001,1.0)   
    #y_data_subpixel = np.interp(a, b, y_data )
    #x_data_subpixel = np.interp(c, d, x_data )
    #translation_y = im1_debuffered.shape[0]/2 - a[np.argmax(y_data_subpixel)] 
    #translation_x = im1_debuffered.shape[1]/2 - c[np.argmax(x_data_subpixel)] 
    
    error = 1 #This is a dummy number until I figure out how to calculate error
    
    return(np.array([translation_y, translation_x]), error)
    
    

def erc_R(im1_bigger,im2):
    #im1 should be bigger than im2, so that im2 can be used for cross-correlations at different locations on top of im1
    #    %For choosing the two sub-images to correlate:
    #    %   m is the offset, measured in pixels, of the searching window to-the-right. The searching window pans across im1_bigger
    #    %   n is the offset, measured in pixels, of the searching window down. The searching window pans across im1_bigger
    
    winsize1=im1_bigger.shape
    length_im1=len(im1_bigger[:]);
    winsize2=im2.shape;
    variance_im2=np.var(im2);
    length_im2=len(im2[:]);
    
    R   = np.ones((winsize1[0]-winsize2[0]+1,winsize1[1]-winsize2[1]+1))*0.012345678901
    im1 = np.ones(im2.shape)
    
    for m in range(0,winsize1[1]-winsize2[1]+1,3):
        for n in range(0,winsize1[0]-winsize2[0]+1,3):
            #print(m,n)
            im1[:,:]=im1_bigger[n:n+winsize2[0]-1+1,m:m+winsize2[1]-1+1];
            #%I use R[n+winsize/2,m+winsize/2] in order to keep in line with Kristof Sveen's convention on the meaning of R
            R[n,m]=np.sum((im2[:]-np.mean(im2[:])) * (im1[:]-np.mean(im1[:]))) / (length_im2-1)/np.sqrt(variance_im2*np.var(im1[:]))  
    
    max_indices = np.array(np.unravel_index(np.argmax(R,axis=None), R.shape))
    focused_indices=[]
    for m in range(0,winsize1[1]-winsize2[1]+1):
        for n in range(0,winsize1[0]-winsize2[0]+1):
            if abs(m - max_indices[1])<7 and abs(n - max_indices[0])<7 and R[n,m]==0.012345678901:
                focused_indices.append([n,m])
    
    for i in range(0,len(focused_indices)):
        n = focused_indices[i][0]
        m = focused_indices[i][1]
        im1[:,:]=im1_bigger[n:n+winsize2[0]-1+1,m:m+winsize2[1]-1+1]
        R[n,m]=np.sum((im2[:]-np.mean(im2[:])) * (im1[:]-np.mean(im1[:]))) / (length_im2-1)/np.sqrt(variance_im2*np.var(im1[:]))  
        
    return(R)




#filename MUST be supplied as a number and must be the Mn raw image file
def shift_image_integer(im_old,translation):  #filename MUST be supplied as a number and must be the Mn raw image file
    
    translation = np.int(np.round(translation))

    #Create the buffered images
    im_new = np.ones(im.shape)*0.1234567890123456
    
    #Now let's actually shift the 2nd Mn image so it's aligned with the 1st Mn image
    if translation[0]== 0  and translation[1]== 0:  im_new[  translation[0]:,  translation[1]:] = im_old[ translation[0]:, translation[1]:];       
    if translation[0] > 0  and translation[1] > 0:  im_new[:-translation[0] ,:-translation[1] ] = im_old[ translation[0]:, translation[1]:];       
    if translation[0] < 0  and translation[1]== 0:  im_new[ -translation[0]:,  translation[1]:] = im_old[:translation[0],  translation[1]:];    
    if translation[0] < 0  and translation[1] > 0:  im_new[ -translation[0]:,:-translation[1] ] = im_old[:translation[0],  translation[1]:];    
    if translation[0]== 0  and translation[1] < 0:  im_new[  translation[0]:, -translation[1]:] = im_old[ translation[0]:,:translation[1] ];   
    if translation[0] > 0  and translation[1] < 0:  im_new[:-translation[0] , -translation[1]:] = im_old[ translation[0]:,:translation[1] ];   
    if translation[0] < 0  and translation[1] < 0:  im_new[ -translation[0]:, -translation[1]:] = im_old[:translation[0], :translation[1] ];
       
    return(im_new)
    


    
    
    


####### MATLAB CODE FOR MAKING MOVIES FROM MESSINGER LAB MICROSCOPE COMPUTER USED BY BRENDAN ##############
#= (biologic_start_time - scan_start_time).total_seconds()
#
#
#print(f'{time.monotonic() - t0}')
#
#A=xlsread('C:\Users\Maccor\Desktop\Damon\movie_collection\movie_20181129\Data_For_Matlab_1stRun.xlsx');
#images_dir='images_1stRun';
#fig_han=figure('Position',[1.0000    1.0000  935.2000  781.6000]);  %[x0 y0 deltax deltay ] [646.6000 46.6000 887.2000 735.2000] %[973.8000 46.6000 560.0000 460.8000]);#
#%ax3_han = axes('Position',[0.08 0.591 0.957 0.14]);
#ax1_han = axes('Position',[0.0723    0.06744    0.9002    0.2571]); %[x0 y0 deltax deltay ]
#A(:,1)=A(:,1)-0.5; %shift cell voltage to be w.r.t. Hg/HgO (use 0.5 if converting from a Bi2O3 counter)
#A(:,2)=A(:,2)*1000; 
#A(:,4)=-A(:,4)-0.1; %shift Sync voltage to be w.r.t. Hg/HgO (use 0.1 if converting from a Cu wire).  The Sync voltage is measured between reference electrode and working electrode.
#plot_han=plot(A(:,4),A(:,2),'r',A(:,1),A(:,2),'g');  %plot current vs voltage
#ylim([-20 50])
#xlim([-1.0 0.55])
#set(get(ax1_han, 'XLabel'), 'string', 'Voltage w.r.t. Hg/HgO (V)')
#set(get(ax1_han, 'YLabel'), 'string', 'Current (mA)')
#%set(get(ax1_han, 'YLabel'), 'Rotation', 0)
#%set(get(ax1_han, 'YLabel'), 'Position', [-2.5 1.5 1])
#set(ax1_han,'XColor',[0 0 0])
#set(ax1_han,'YColor',[0 0 0])
#
#ax2_han = axes('Position',[0.11 0.33 0.85 0.68],'Visible','off');%[x0 y0 deltax deltay ]
#photos=dir([images_dir '/*.png']);
#for i = 1:length(photos)
#    photos_elapsedtime(i)=(datenum(photos(i).name(1:14),'yyyymmddHHMMSS')-datenum(photos(1).name(1:14),'yyyymmddHHMMSS'))*24*3600;
#end
#movie_times=0:10:A(end,3);
#num_frames=length(A(:,3));
#movie_frames(length(movie_times))= struct('cdata',[],'colormap',[]);
#for i=1:length(movie_times)
#    axes(ax2_han);
#    [c index] = min(abs(photos_elapsedtime-movie_times(i)));
#    image=imread([images_dir '/' photos(index).name]);
#    imshow(image(1:3400,:,:));   %(yrange, xrange, colorrange)
#    axes(ax1_han);
#    [c index] = min(abs(A(:,3)-movie_times(i)));
#    line_han=scatter(A(index,1),A(index,2),15,'r','filled');
#    movie_frames(i)=getframe(gcf);
#    line_han.delete;
#end
#v = VideoWriter([images_dir '.mp4'],'MPEG-4');
#v.FrameRate=15;
#open(v);
#writeVideo(v,movie_frames);
#close(v);
#close all
#clear all
#%movie(fig_han,movie_frames,2)
#



#  Andy's code
#    # Save the files
#    # Check if a folder exists
#    new_folder = working_directory + filename[:-3] + '/'
#    if (not os.path.isdir(new_folder)):
#        try:
#            os.mkdir(new_folder)
#        except e as Exception:
#            print('Error creating directory.')
#            print(e)
#            quit()
#    
#    # Save the files
#    skimage.io.imsave(new_folder+'xanes.tif', I)
    
    
#    Great data Robin Pelc !  Thnx.  That data partly changes my mind, and makes me partly agree with all ya'll on this topic.
#
#The WaPo study is scientifically inconclusive because it 
#
#Combining your WaPo data with the data that Hillary got THE SAME number of votes in 2016 as Obama (a man) got in 2012, makes me think the sexism in Democratic voters exists but only makes a .  So I guess Hillary would have won the 2016 primary with greater excess if she were a man.  And she likely would have gained a few million more votes in the general election against Trump if she were a man.
#
#Kevin Yelenik  I totally agree the world (on average) is sexist. I wasn't convinced yet that the Democratic Party was actually picking and choosing candidates with sexist fingerprints, but now I'm convinced that there is a slight effect from sexism, thanks.  
#
#
#
# 
    
#    
#    
#    
#    
#        #Now search for a lower sum-of-square-error by pairwise swaping of thickness (via all pairwise combinatorics) 
#    for m in range(0,ims[0,:,:].shape[0]):
#        if np.mod(m,10)==0: sys.stdout.write('\rMaximum Decent, Row: '+str(m))
#        sys.stdout.flush()
#        for n in range(0,ims[0,:,:].shape[1]):
#            if m==300 and n==900:
#                print('hi')
#            for j in range(0,1):
#                baseline_sum_square_error     =       calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
#                test_sum_squares = pairwise_thicknessswapping_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)                         
#                sum_square_error_linear_slope = test_sum_squares - baseline_sum_square_error
#                max_pair = np.argmax(abs(sum_square_error_linear_slope))
#                
#                if max_pair==0:
#                    previous_sum_square_errors = baseline_sum_square_error
#                    sum_square_errors = baseline_sum_square_error
#                    while(sum_square_errors<=previous_sum_square_errors):
#                        previous_sum_square_errors = sum_square_errors
#                        optical_thickness_Mn_test = optical_thickness_Mn[m,n] - 0.0001*np.sign(sum_square_error_linear_slope[0])
#                        optical_thickness_Cu_test = optical_thickness_Cu[m,n] + 0.0001*np.sign(sum_square_error_linear_slope[0])
#                        sum_square_errors = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
#                        if sum_square_errors<=previous_sum_square_errors:
#                            optical_thickness_Mn[m,n]=optical_thickness_Mn_test
#                            optical_thickness_Cu[m,n]=optical_thickness_Cu_test
#                if max_pair==1:
#                    previous_sum_square_errors = baseline_sum_square_error
#                    sum_square_errors = baseline_sum_square_error
#                    while(sum_square_errors<=previous_sum_square_errors):
#                        previous_sum_square_errors = sum_square_errors
#                        optical_thickness_Mn_test = optical_thickness_Mn[m,n] - 0.0001*np.sign(sum_square_error_linear_slope[1])
#                        optical_thickness_Bi_test = optical_thickness_Bi[m,n] + 0.0001*np.sign(sum_square_error_linear_slope[1])
#                        sum_square_errors = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
#                        if sum_square_errors<=previous_sum_square_errors:
#                            optical_thickness_Mn[m,n]=optical_thickness_Mn_test
#                            optical_thickness_Bi[m,n]=optical_thickness_Bi_test
#                if max_pair==2:
#                    previous_sum_square_errors = baseline_sum_square_error
#                    sum_square_errors = baseline_sum_square_error
#                    while(sum_square_errors<=previous_sum_square_errors):
#                        previous_sum_square_errors = sum_square_errors
#                        optical_thickness_Mn_test = optical_thickness_Mn[m,n] - 0.0001*np.sign(sum_square_error_linear_slope[2])
#                        optical_thickness_El_test = optical_thickness_El[m,n] + 0.0001*np.sign(sum_square_error_linear_slope[2])
#                        sum_square_errors = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
#                        if sum_square_errors<=previous_sum_square_errors:
#                            optical_thickness_Mn[m,n]=optical_thickness_Mn_test
#                            optical_thickness_El[m,n]=optical_thickness_El_test
#                if max_pair==3:
#                    previous_sum_square_errors = baseline_sum_square_error
#                    sum_square_errors = baseline_sum_square_error
#                    while(sum_square_errors<=previous_sum_square_errors):
#                        previous_sum_square_errors = sum_square_errors
#                        optical_thickness_Cu_test = optical_thickness_Cu[m,n] - 0.0001*np.sign(sum_square_error_linear_slope[3])
#                        optical_thickness_Bi_test = optical_thickness_Bi[m,n] + 0.0001*np.sign(sum_square_error_linear_slope[3])
#                        sum_square_errors = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
#                        if sum_square_errors<=previous_sum_square_errors:
#                            optical_thickness_Cu[m,n]=optical_thickness_Cu_test
#                            optical_thickness_Bi[m,n]=optical_thickness_Bi_test
#                if max_pair==4:
#                    previous_sum_square_errors = baseline_sum_square_error
#                    sum_square_errors = baseline_sum_square_error
#                    while(sum_square_errors<=previous_sum_square_errors):
#                        previous_sum_square_errors = sum_square_errors
#                        optical_thickness_Cu_test = optical_thickness_Cu[m,n] - 0.0001*np.sign(sum_square_error_linear_slope[4])
#                        optical_thickness_El_test = optical_thickness_El[m,n] + 0.0001*np.sign(sum_square_error_linear_slope[4])
#                        sum_square_errors = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
#                        if sum_square_errors<=previous_sum_square_errors:
#                            optical_thickness_Cu[m,n]=optical_thickness_Cu_test
#                            optical_thickness_El[m,n]=optical_thickness_El_test
#                if max_pair==5:
#                    previous_sum_square_errors = baseline_sum_square_error
#                    sum_square_errors = baseline_sum_square_error
#                    while(sum_square_errors<=previous_sum_square_errors):
#                        previous_sum_square_errors = sum_square_errors
#                        optical_thickness_Bi_test = optical_thickness_Bi[m,n] - 0.0001*np.sign(sum_square_error_linear_slope[5])
#                        optical_thickness_El_test = optical_thickness_El[m,n] + 0.0001*np.sign(sum_square_error_linear_slope[5])
#                        sum_square_errors = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
#                        if sum_square_errors<=previous_sum_square_errors:
#                            optical_thickness_Bi[m,n]=optical_thickness_Bi_test
#                            optical_thickness_El[m,n]=optical_thickness_El_test
#                            
#                            


##Now search for a lower sum-of-square-error by pairwise swaping of thickness (via all pairwise combinatorics) 
#for m in range(0,ims[0,:,:].shape[0]):
#    if np.mod(m,10)==0: sys.stdout.write('\rMaximum Decent, Row: '+str(m))
#    sys.stdout.flush()
#    for n in range(0,ims[0,:,:].shape[1]):
#        ln_I_I0_6520 = np.log(ims[0,m,n]);  ln_I_I0_6600 = np.log(ims[1,m,n]); ln_I_I0_8970 = np.log(ims[2,m,n]); ln_I_I0_9050 = np.log(ims[3,m,n]);
#        if m==300 and n==900:
#            print('hi')
#        for j in range(0,10):
#            baseline_sum_square_error = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu[m,n],optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El[m,n])
#            test_matrix_sum_square_errors =       dS_dthickness_all(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu[m,n],optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El[m,n])                         
#            sum_square_error_linear_slope = np.array([test_matrix_sum_square_errors[1] - test_matrix_sum_square_errors[0] , test_matrix_sum_square_errors[3] - test_matrix_sum_square_errors[2] , test_matrix_sum_square_errors[5] - test_matrix_sum_square_errors[4] , test_matrix_sum_square_errors[7] - test_matrix_sum_square_errors[6] ] )
#            max_element = np.argmax(abs(sum_square_error_linear_slope))
#            
#            if max_element==0:
#                previous_sum_square_errors = baseline_sum_square_error
#                sum_square_errors_test = baseline_sum_square_error
#                while(sum_square_errors_test<=previous_sum_square_errors):
#                    previous_sum_square_errors = sum_square_errors_test
#                    optical_thickness_Mn_test = optical_thickness_Mn[m,n] - 0.00001*np.sign(sum_square_error_linear_slope[0])
#                    sum_square_errors_test = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn_test,optical_thickness_Cu[m,n],optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El[m,n])
#                    if sum_square_errors_test<=previous_sum_square_errors:
#                        optical_thickness_Mn[m,n]=optical_thickness_Mn_test
#            if max_element==1:
#                previous_sum_square_errors = baseline_sum_square_error
#                sum_square_errors_test = baseline_sum_square_error
#                while(sum_square_errors_test<=previous_sum_square_errors):
#                    previous_sum_square_errors = sum_square_errors_test
#                    optical_thickness_Cu_test = optical_thickness_Cu[m,n] - 0.00001*np.sign(sum_square_error_linear_slope[0])
#                    sum_square_errors_test = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu_test,optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El[m,n])
#                    if sum_square_errors_test<=previous_sum_square_errors:
#                        optical_thickness_Cu[m,n]=optical_thickness_Cu_test
#            if max_element==2:
#                previous_sum_square_errors = baseline_sum_square_error
#                sum_square_errors_test = baseline_sum_square_error
#                while(sum_square_errors_test<=previous_sum_square_errors):
#                    previous_sum_square_errors = sum_square_errors_test
#                    optical_thickness_Bi_test = optical_thickness_Bi[m,n] - 0.00001*np.sign(sum_square_error_linear_slope[0])
#                    sum_square_errors_test = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu[m,n],optical_thickness_Bi_test,optical_thickness_C[m,n],optical_thickness_El[m,n])
#                    if sum_square_errors_test<=previous_sum_square_errors:
#                        optical_thickness_Bi[m,n]=optical_thickness_Bi_test
#            if max_element==3:
#                previous_sum_square_errors = baseline_sum_square_error
#                sum_square_errors_test = baseline_sum_square_error
#                while(sum_square_errors_test<=previous_sum_square_errors):
#                    previous_sum_square_errors = sum_square_errors_test
#                    optical_thickness_El_test = optical_thickness_El[m,n] - 0.00001*np.sign(sum_square_error_linear_slope[0])
#                    sum_square_errors_test = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu[m,n],optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El_test)
#                    if sum_square_errors_test<=previous_sum_square_errors:
#                        optical_thickness_El[m,n]=optical_thickness_El_test
#
       
    
#        total_thickness = total_thickness + 0.0005
#
#        b[0] = (total_thickness - optical_thickness_C[m,n])*sum_aMn_aEl + (a_6520_Mn*ln_I_I0_6520 + a_6600_Mn*ln_I_I0_6600 + a_8970_Mn*ln_I_I0_8970 + a_9050_Mn*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Mn*a_6520_C + a_6600_Mn*a_6600_C + a_8970_Mn*a_8970_C + a_9050_Mn*a_9050_C)
#        b[1] = (total_thickness - optical_thickness_C[m,n])*sum_aCu_aEl + (a_6520_Cu*ln_I_I0_6520 + a_6600_Cu*ln_I_I0_6600 + a_8970_Cu*ln_I_I0_8970 + a_9050_Cu*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Cu*a_6520_C + a_6600_Cu*a_6600_C + a_8970_Cu*a_8970_C + a_9050_Cu*a_9050_C)
#        b[2] = (total_thickness - optical_thickness_C[m,n])*sum_aBi_aEl + (a_6520_Bi*ln_I_I0_6520 + a_6600_Bi*ln_I_I0_6600 + a_8970_Bi*ln_I_I0_8970 + a_9050_Bi*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Bi*a_6520_C + a_6600_Bi*a_6600_C + a_8970_Bi*a_8970_C + a_9050_Bi*a_9050_C)
#        
#        temp = np.linalg.solve(A, b)
#        optical_thickness_Mn[m,n] = np.float32(temp[0])  #this produces optical thickness in mm    
#        optical_thickness_Cu[m,n] = np.float32(temp[1])  #this produces optical thickness in mm
#        optical_thickness_Bi[m,n] = np.float32(temp[2])  #this produces optical thickness in mm         
#        optical_thickness_El[m,n] = total_thickness - optical_thickness_C[m,n] - optical_thickness_Mn[m,n] + optical_thickness_Cu[m,n] + optical_thickness_Bi[m,n]  #this produces optical thickness in mm         
#        sum_square_errors2 = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu[m,n],optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El[m,n])
#
#        if sum_square_errors2 < sum_square_errors1:
#            sum_square_errors_previous = sum_square_errors2
#            sum_square_errors_latest = sum_square_errors2
#            while(sum_square_errors_latest <= sum_square_errors_previous):
#                total_thickness = total_thickness + 0.0005
#    
#                b[0] = (total_thickness - optical_thickness_C[m,n])*sum_aMn_aEl + (a_6520_Mn*ln_I_I0_6520 + a_6600_Mn*ln_I_I0_6600 + a_8970_Mn*ln_I_I0_8970 + a_9050_Mn*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Mn*a_6520_C + a_6600_Mn*a_6600_C + a_8970_Mn*a_8970_C + a_9050_Mn*a_9050_C)
#                b[1] = (total_thickness - optical_thickness_C[m,n])*sum_aCu_aEl + (a_6520_Cu*ln_I_I0_6520 + a_6600_Cu*ln_I_I0_6600 + a_8970_Cu*ln_I_I0_8970 + a_9050_Cu*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Cu*a_6520_C + a_6600_Cu*a_6600_C + a_8970_Cu*a_8970_C + a_9050_Cu*a_9050_C)
#                b[2] = (total_thickness - optical_thickness_C[m,n])*sum_aBi_aEl + (a_6520_Bi*ln_I_I0_6520 + a_6600_Bi*ln_I_I0_6600 + a_8970_Bi*ln_I_I0_8970 + a_9050_Bi*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Bi*a_6520_C + a_6600_Bi*a_6600_C + a_8970_Bi*a_8970_C + a_9050_Bi*a_9050_C)
#                
#                temp = np.linalg.solve(A, b)
#                optical_thickness_Mn[m,n] = np.float32(temp[0])  #this produces optical thickness in mm    
#                optical_thickness_Cu[m,n] = np.float32(temp[1])  #this produces optical thickness in mm
#                optical_thickness_Bi[m,n] = np.float32(temp[2])  #this produces optical thickness in mm         
#                optical_thickness_El[m,n] = total_thickness - optical_thickness_C[m,n] - optical_thickness_Mn[m,n] + optical_thickness_Cu[m,n] + optical_thickness_Bi[m,n]  #this produces optical thickness in mm         
#                sum_square_errors_previous = sum_square_errors_latest
#                sum_square_errors_latest = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu[m,n],optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El[m,n])
#                print(total_thickness)
#                if sum_square_errors_latest > sum_square_errors_previous:
#                    total_thickness = total_thickness - 0.0005
#                    
#        else:
#            total_thickness = total_thickness - 0.0005
#            sum_square_errors_previous = sum_square_errors1
#            sum_square_errors_latest = sum_square_errors1
#            while(sum_square_errors_latest <= sum_square_errors_previous):
#                total_thickness = total_thickness - 0.0005
#    
#                b[0] = (total_thickness - optical_thickness_C[m,n])*sum_aMn_aEl + (a_6520_Mn*ln_I_I0_6520 + a_6600_Mn*ln_I_I0_6600 + a_8970_Mn*ln_I_I0_8970 + a_9050_Mn*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Mn*a_6520_C + a_6600_Mn*a_6600_C + a_8970_Mn*a_8970_C + a_9050_Mn*a_9050_C)
#                b[1] = (total_thickness - optical_thickness_C[m,n])*sum_aCu_aEl + (a_6520_Cu*ln_I_I0_6520 + a_6600_Cu*ln_I_I0_6600 + a_8970_Cu*ln_I_I0_8970 + a_9050_Cu*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Cu*a_6520_C + a_6600_Cu*a_6600_C + a_8970_Cu*a_8970_C + a_9050_Cu*a_9050_C)
#                b[2] = (total_thickness - optical_thickness_C[m,n])*sum_aBi_aEl + (a_6520_Bi*ln_I_I0_6520 + a_6600_Bi*ln_I_I0_6600 + a_8970_Bi*ln_I_I0_8970 + a_9050_Bi*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Bi*a_6520_C + a_6600_Bi*a_6600_C + a_8970_Bi*a_8970_C + a_9050_Bi*a_9050_C)
#                
#                temp = np.linalg.solve(A, b)
#                optical_thickness_Mn[m,n] = np.float32(temp[0])  #this produces optical thickness in mm    
#                optical_thickness_Cu[m,n] = np.float32(temp[1])  #this produces optical thickness in mm
#                optical_thickness_Bi[m,n] = np.float32(temp[2])  #this produces optical thickness in mm         
#                optical_thickness_El[m,n] = total_thickness - optical_thickness_C[m,n] - optical_thickness_Mn[m,n] + optical_thickness_Cu[m,n] + optical_thickness_Bi[m,n]  #this produces optical thickness in mm         
#                sum_square_errors_previous = sum_square_errors_latest
#                sum_square_errors_latest = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu[m,n],optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El[m,n])
#                if sum_square_errors_latest > sum_square_errors_previous:
#                    total_thickness = total_thickness + 0.0005
#
#                #Now that we know the best thickness to use, let's do the final calculations for the final answer
#                b[0] = (total_thickness - optical_thickness_C[m,n])*sum_aMn_aEl + (a_6520_Mn*ln_I_I0_6520 + a_6600_Mn*ln_I_I0_6600 + a_8970_Mn*ln_I_I0_8970 + a_9050_Mn*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Mn*a_6520_C + a_6600_Mn*a_6600_C + a_8970_Mn*a_8970_C + a_9050_Mn*a_9050_C)
#                b[1] = (total_thickness - optical_thickness_C[m,n])*sum_aCu_aEl + (a_6520_Cu*ln_I_I0_6520 + a_6600_Cu*ln_I_I0_6600 + a_8970_Cu*ln_I_I0_8970 + a_9050_Cu*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Cu*a_6520_C + a_6600_Cu*a_6600_C + a_8970_Cu*a_8970_C + a_9050_Cu*a_9050_C)
#                b[2] = (total_thickness - optical_thickness_C[m,n])*sum_aBi_aEl + (a_6520_Bi*ln_I_I0_6520 + a_6600_Bi*ln_I_I0_6600 + a_8970_Bi*ln_I_I0_8970 + a_9050_Bi*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Bi*a_6520_C + a_6600_Bi*a_6600_C + a_8970_Bi*a_8970_C + a_9050_Bi*a_9050_C)
#                
#                temp = np.linalg.solve(A, b)
#                optical_thickness_Mn[m,n] = np.float32(temp[0])  #this produces optical thickness in mm    
#                optical_thickness_Cu[m,n] = np.float32(temp[1])  #this produces optical thickness in mm
#                optical_thickness_Bi[m,n] = np.float32(temp[2])  #this produces optical thickness in mm         
#                optical_thickness_El[m,n] = total_thickness - optical_thickness_C[m,n] - optical_thickness_Mn[m,n] + optical_thickness_Cu[m,n] + optical_thickness_Bi[m,n]  #this produces optical thickness in mm         
#                sum_square_errors = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn[m,n],optical_thickness_Cu[m,n],optical_thickness_Bi[m,n],optical_thickness_C[m,n],optical_thickness_El[m,n])




#    # Calculate how much shift will align the 1st and 2nd Cu images with the 1st Mn image
#    beam_energies_Cu, Cu_ims = read_FXI_xanes_images(Cu_filename);
#    Cu_translation = find_image_translation(Cu_ims[0,:,:],Cu_ims[1,:,:])  #Cu_trans  is the translation (pixel shift) of Cu image 2 wrt Cu image 1
#    Cu2_translation = find_image_translation(Mn_ims[0,:,:],Cu_ims[1,:,:]) #Cu2_trans is the translation (pixel shift) of Cu image 2 wrt Mn image 1
#    Cu1_translation = Cu2_translation - Cu_translation                            #Cu1_translation is the translation (pixel shift) of Cu image 1 wrt Mn image 1
#    Cu_ims_buffered = np.zeros((2, Cu_ims.shape[1] + 2*buffer_edges, Cu_ims.shape[2] + 2*buffer_edges), dtype=np.float32) + np.min(Cu_ims) #I add np.min(Cu_ims) so that np.log doesn't create an error
#    Cu_ims_buffered[0,:,:] = np.pad(Cu_ims[0,:,:], buffer_edges)
#    Cu_ims_buffered[1,:,:] = np.pad(Cu_ims[1,:,:], buffer_edges)
#    Cu_im1_buffered_aligned = np.zeros(Cu_ims_buffered[0,:,:].shape, dtype=np.float32) + np.min(Cu_ims) #add np.min(Cu_ims) so that np.log doesn't create an error
#    Cu_im2_buffered_aligned = np.zeros(Cu_ims_buffered[0,:,:].shape, dtype=np.float32) + np.min(Cu_ims) #add np.min(Cu_ims) so that np.log doesn't create an error 
#     
#    #Now let's actually shift the 2nd Cu image so it's aligned with the 1st Mn image
#    if Cu2_translation[0]== 0  and Cu2_translation[1]== 0:  Cu_im2_buffered_aligned[  Cu2_translation[0]:,  Cu2_translation[1]:] = Cu_ims_buffered[1, Cu2_translation[0]:, Cu2_translation[1]:];       
#    if Cu2_translation[0] > 0  and Cu2_translation[1] > 0:  Cu_im2_buffered_aligned[:-Cu2_translation[0] ,:-Cu2_translation[1] ] = Cu_ims_buffered[1, Cu2_translation[0]:, Cu2_translation[1]:];       
#    if Cu2_translation[0] < 0  and Cu2_translation[1]== 0:  Cu_im2_buffered_aligned[ -Cu2_translation[0]:,  Cu2_translation[1]:] = Cu_ims_buffered[1,:Cu2_translation[0],  Cu2_translation[1]:];    
#    if Cu2_translation[0] < 0  and Cu2_translation[1] > 0:  Cu_im2_buffered_aligned[ -Cu2_translation[0]:,:-Cu2_translation[1] ] = Cu_ims_buffered[1,:Cu2_translation[0],  Cu2_translation[1]:];    
#    if Cu2_translation[0]== 0  and Cu2_translation[1] < 0:  Cu_im2_buffered_aligned[  Cu2_translation[0]:, -Cu2_translation[1]:] = Cu_ims_buffered[1, Cu2_translation[0]:,:Cu2_translation[1] ];   
#    if Cu2_translation[0] > 0  and Cu2_translation[1] < 0:  Cu_im2_buffered_aligned[:-Cu2_translation[0] , -Cu2_translation[1]:] = Cu_ims_buffered[1, Cu2_translation[0]:,:Cu2_translation[1] ];   
#    if Cu2_translation[0] < 0  and Cu2_translation[1] < 0:  Cu_im2_buffered_aligned[ -Cu2_translation[0]:, -Cu2_translation[1]:] = Cu_ims_buffered[1,:Cu2_translation[0], :Cu2_translation[1] ];
#        
#    #Now let's actually shift the 1st Cu image so it's aligned with the 1st Mn image
#    if Cu1_translation[0]== 0  and Cu1_translation[1]== 0:  Cu_im1_buffered_aligned[  Cu1_translation[0]:,  Cu1_translation[1]:] = Cu_ims_buffered[0, Cu1_translation[0]:, Cu1_translation[1]:];       
#    if Cu1_translation[0] > 0  and Cu1_translation[1] > 0:  Cu_im1_buffered_aligned[:-Cu1_translation[0] ,:-Cu1_translation[1] ] = Cu_ims_buffered[0, Cu1_translation[0]:, Cu1_translation[1]:];       
#    if Cu1_translation[0] < 0  and Cu1_translation[1]== 0:  Cu_im1_buffered_aligned[ -Cu1_translation[0]:,  Cu1_translation[1]:] = Cu_ims_buffered[0,:Cu1_translation[0],  Cu1_translation[1]:];    
#    if Cu1_translation[0] < 0  and Cu1_translation[1] > 0:  Cu_im1_buffered_aligned[ -Cu1_translation[0]:,:-Cu1_translation[1] ] = Cu_ims_buffered[0,:Cu1_translation[0],  Cu1_translation[1]:];    
#    if Cu1_translation[0]== 0  and Cu1_translation[1] < 0:  Cu_im1_buffered_aligned[  Cu1_translation[0]:, -Cu1_translation[1]:] = Cu_ims_buffered[0, Cu1_translation[0]:,:Cu1_translation[1] ];   
#    if Cu1_translation[0] > 0  and Cu1_translation[1] < 0:  Cu_im1_buffered_aligned[:-Cu1_translation[0] , -Cu1_translation[1]:] = Cu_ims_buffered[0, Cu1_translation[0]:,:Cu1_translation[1] ];   
#    if Cu1_translation[0] < 0  and Cu1_translation[1] < 0:  Cu_im1_buffered_aligned[ -Cu1_translation[0]:, -Cu1_translation[1]:] = Cu_ims_buffered[0,:Cu1_translation[0], :Cu1_translation[1] ];
#
#    Cu_ims_buffered[0,:,:]=Cu_im1_buffered_aligned
#    Cu_ims_buffered[1,:,:]=Cu_im2_buffered_aligned
    
    
    
