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
    

### Set Data Directory ###################################
data_directory='../20191105_FXI_Beamtime_Work/Data_From_Beamline/';
data_subdirectory='34563_to_34875/';
results_directory='../20191105_FXI_Beamtime_Work/Results_Of_Data_Processing/';
object_list_allfilesdirectories = pathlib.Path(data_directory+data_subdirectory)
object_recursiveglob_tiffiles = object_list_allfilesdirectories.glob('*.tif')      
object_list_filenames_tiffiles = list(object_recursiveglob_tiffiles)
##########################################################

### Lists of XANES Scan Numbers ##########################
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
##########################################################


### List of functions in this file ########################
# make_average_image(file_numbers, which_image, output_filename ): return(average_scalar_series, std_scalar_series)
# internally_align_h5_file(Mn_filename, im2_cropping, cc_search_distance, average_dark_image_filename_Mn='none', average_dark_image_filename_Cu='none',):    #cc_search_distance is the cross correlation search distance [left, right, top, bottom]    
# align_processed_images_time_series(file_numbers,im2_cropping, cc_search_distance):  
# debuffer_multiple_image_files(file_numbers):
# eliminate_beam_flickering_time_series():
# calculate_optical_thickness(filename, carbon_thickness=0.15, total_thickness=0.2):   
# make_movie_with_potentiostat_data(txm_file_numbers,biologic_file, image_used_for_plot, movie_time_span_seconds, seconds_per_movie_frame, output_filename):
# make_movie_with_image_statistics(file_numbers, image_type_2_show, movie_filename ):
# calculate_brightness_contrast(filenumbers, image_2_display, low_end_percentile, high_end_percentile): return(np.min(low_end_all_files) , np.max(high_end_all_files))
# plot_single_pixel_least_squares_data(filename,row,column):  
# read_FXI_raw_h5_metadata(filename): return(scan_start_time_string, scan_time, beam_energies, scan_id, notes) 
# read_FXI_processed_h5_metadata(filename): return(scan_start_time_string, scan_time, beam_energy, scan_id, notes, translations)
# get_raw_image(filename,which_image):
# get_processed_image(filename,which_image):     
# dS_dthickness_all(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El): return(test_sum_squares)    
# calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El): return(baseline_sum_square_error)
# show_cross_correlation_map(im1,im2,debuffer=0):
# find_image_translation( im1, im2, im2_cropping, cc_search_distance): return(np.array([translation_y, translation_x]), error, cc_image)
# erc_R(im1, im2_orig, im2_cropping, cc_search_distance): return(R)
# shift_image_integer(im_old,translation): return(im_new)
# calculate_debuffer_multiple_images(ims): return(debuffer_all_images)
# calculate_image_debuffer_multiple_files(file_numbers):  return(debuffer_multi_file)
# combine_more_loading1_with_more_loading2():
# get_images_statistics(file_numbers, image_type_2_show ): return(images_mean, images_std) 
###########################################################


### Workflow ##############################################
#   NOTE: ALL THE COPPER H5 FILES (MULTIPOS_2D_XANES...H5 FILES) USED 2.5 SEC EXPOSURE TIME WHEREAS THE MN FILES USED 5 SECDONS!!!  ALSO, SCAN 34600 HAS A BAD DARK IMAGE SO YOU HAVE TO REMEMBER TO NOT USE IT’S DARK IMAGE!!!  
# 0) Create average darkfield image for the 5 sec exposed images (Mn) and 2.5s exposed images (Cu):  make_average_image(np.concatenate((range(34565,34725,2),range(34726,34875,2))), 'img_dark',  'ave_dark_5s_exposure_34565_34875.h5')
# 1) Run internally_align_h5_file(file_number,[50,350,300,300],[50,100,75,75],'ave_dark_5s_exposure_34565_34875.h5','ave_dark_2p5s_exposure_34565_34875.h5')  on each Manganese multipos_2D_xanes_scan2_[]...h5 file to align the images. Use a command like for i in range(34675,34725,2): create_aligned_h5_file(i);      NOTE:  file 34675 is missing, see your beamline notes -- before 34675 the Mn files are odd numbered and after 34675 the Mn files are even numbered .   the [50,350,300,300] chops off L.R.T.B. which have the copper TEM mesh which confuses the cc_image. The  [50,100,75,75] is how far to search in each direction when calculating the cross correlations
# 2) Run align_processed_images_time_series(range(34565,34646,2),[100,350,300,300],[50,100,75,75])   The im2_cropping=[100,350,200,200] is how much of the sides and top/bottom to cutoff im2.    Use a command like for i in range(34675,34725,2): calculate_optical_thickness(i);   
# 3) make_movie_with_potentiostat_data(range(34565,34725,2),'20191107_Cu-Bi-Birnessite_37NaOH_more_loading1_and2.mpt', 'Mn_raw_im1', 15500,40, '34565_34599_6520eV.mp4')
############################################################
    
def make_average_image(file_numbers, which_image, output_filename ):
    test_im = get_raw_image(file_numbers[0],'img_dark')
    average_image = test_im*0.0
    average_scalar_series = np.ones(len(file_numbers))
    std_scalar_series = np.ones(len(file_numbers))
    for i in range(len(file_numbers)):
        # Grab the image
        if which_image == 'img_bkg':
            im=get_raw_image(file_numbers[i],'img_bkg')
            im=np.mean(im,axis=0)
        if which_image == 'img_dark':
            im=get_raw_image(file_numbers[i],'img_dark')
            im=im[0,:,:]
        if which_image != 'img_bkg' and which_image != 'img_dark':
            im=get_processed_image(file_numbers[i], which_image)
        
        print('calculating img: ' + str(file_numbers[i]))

        average_image = average_image + im
        average_scalar_series[i] = np.mean(im[:])
        std_scalar_series[i] = np.std(im[:])
    
    average_image = average_image/np.float32(len(file_numbers))
    h5object_new = h5py.File(data_directory+data_subdirectory+output_filename, 'w')
    h5object_new.create_dataset('average_image', shape=(test_im.shape), dtype=np.float32, data=average_image)
    return(average_scalar_series, std_scalar_series)



# Filename MUST be supplied as a number  #cc_search_distance is [left, right, top, bottom]
def internally_align_h5_file(Mn_filename, im2_cropping, cc_search_distance, average_dark_image_filename_Mn='none', average_dark_image_filename_Cu='none',):    #cc_search_distance is the cross correlation search distance [left, right, top, bottom]
    Cu_filename=Mn_filename+1
    Mn_filename_string="%.4f" % Mn_filename
    Mn_filename_string='multipos_2D_xanes_scan2_id_'+Mn_filename_string[0:5]+'_repeat_'+Mn_filename_string[6:8]+'_pos_'+Mn_filename_string[8:10]+'.h5'
    h5object_old = h5py.File(data_directory+data_subdirectory+Mn_filename_string, 'r')    
    h5object_new = h5py.File(data_directory+data_subdirectory+'processed_images_'+Mn_filename_string[27:-3]+'.h5', 'w')
    h5object_old.copy('scan_id',  h5object_new)
    h5object_old.copy('note',     h5object_new)
    h5object_old.close()
    scan_start_time_string, scan_time1, beam_energies_Mn, scan_id, notes = read_FXI_raw_h5_metadata(Mn_filename)
    scan_start_time_string, scan_time2, beam_energies_Cu, scan_id, notes = read_FXI_raw_h5_metadata(Cu_filename)
    h5object_new.create_dataset('scan_time', shape=(1,),   dtype=np.float64, data=np.mean((scan_time1,scan_time2)))
    
    buffer_edges = 100
    
    # Shift the 2nd Mn image to be aligned with the 1st Mn image
    Mn_ims = get_raw_image(Mn_filename,'img_xanes'); 
    if average_dark_image_filename_Mn != 'none':
        temp_obj = h5py.File(data_directory+data_subdirectory+average_dark_image_filename_Mn, 'r')
        average_dark_image_Mn = temp_obj['average_image'][0,:,:]
        temp_obj.close()
        im_dark = get_raw_image(Mn_filename,'img_dark')[0,:,:]
        im_bkg  = get_raw_image(Mn_filename,'img_bkg')
        Mn_ims[0,:,:] = ((Mn_ims[0,:,:] * (im_bkg[0,:,:] - im_dark) +  im_dark ) - average_dark_image_Mn ) / (im_bkg[0,:,:] - average_dark_image_Mn)
        Mn_ims[1,:,:] = ((Mn_ims[1,:,:] * (im_bkg[1,:,:] - im_dark) +  im_dark ) - average_dark_image_Mn ) / (im_bkg[1,:,:] - average_dark_image_Mn)
    translation1, error, cc_image = find_image_translation(Mn_ims[0,:,:],Mn_ims[1,:,:], im2_cropping, cc_search_distance)
    # Add a buffer of dummy values so that we don't lose data when we shift
    Mn_im1_buffered = np.pad(Mn_ims[0,:,:], buffer_edges, 'constant', constant_values=0.1234567890123456 ) 
    Mn_im2_buffered = np.pad(Mn_ims[1,:,:], buffer_edges, 'constant', constant_values=0.1234567890123456 ) 
    # Now actually do the  shift.  Use -translation to put it back to where it should be (aligned with im1)
    Mn_im2_buffered_aligned = shift_image_integer(Mn_im2_buffered, -translation1)
    #Mn_im2_buffered_aligned = scipy.ndimage.shift(Mn_im2_buffered, -translation1, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
    # Now reset the dummy values because the shift command used spline fitting and messed up some of the dummy values
    Mn_im2_buffered_aligned[:,0:buffer_edges + np.int(np.round(-translation1[1]))] = 0.1234567890123456
    Mn_im2_buffered_aligned[:,  buffer_edges + np.int(np.round(-translation1[1])) + Mn_ims[0,:,:].shape[1]:] = 0.1234567890123456
    Mn_im2_buffered_aligned[0:buffer_edges + np.int(np.round(-translation1[0])),:] = 0.1234567890123456
    Mn_im2_buffered_aligned[  buffer_edges + np.int(np.round(-translation1[0])) + Mn_ims[0,:,:].shape[0]:,:] = 0.1234567890123456
    
    # Shift the Cu images to be aligned with the 1st Mn image
    Cu_ims = get_raw_image(Cu_filename,'img_xanes');  
    if average_dark_image_filename_Cu != 'none':
        temp_obj = h5py.File(data_directory+data_subdirectory+average_dark_image_filename_Cu, 'r')
        average_dark_image_Cu = temp_obj['average_image'][0,:,:]
        temp_obj.close()
        im_dark = get_raw_image(Cu_filename,'img_dark')[0,:,:]
        im_bkg  = get_raw_image(Cu_filename,'img_bkg')
        Cu_ims[0,:,:] = ((Cu_ims[0,:,:] * (im_bkg[0,:,:] - im_dark) +  im_dark ) - average_dark_image_Cu ) / (im_bkg[0,:,:] - average_dark_image_Cu)
        Cu_ims[1,:,:] = ((Cu_ims[1,:,:] * (im_bkg[1,:,:] - im_dark) +  im_dark ) - average_dark_image_Cu ) / (im_bkg[1,:,:] - average_dark_image_Cu)    
    translation2, error, cc_image = find_image_translation(Mn_ims[0,:,:],Cu_ims[0,:,:], im2_cropping, cc_search_distance)
    translation3, error, cc_image = find_image_translation(Mn_ims[0,:,:],Cu_ims[1,:,:], im2_cropping, cc_search_distance)
    # Add a buffer of dummy values so that we don't lose data when we shift
    Cu_im1_buffered = np.pad(Cu_ims[0,:,:], buffer_edges, 'constant', constant_values=0.1234567890123456 ) 
    Cu_im2_buffered = np.pad(Cu_ims[1,:,:], buffer_edges, 'constant', constant_values=0.1234567890123456 ) 
    # Now actually do the  shift
    Cu_im1_buffered_aligned = shift_image_integer(Cu_im1_buffered, -translation2)
    Cu_im2_buffered_aligned = shift_image_integer(Cu_im2_buffered, -translation3)
    #Cu_im1_buffered_aligned = scipy.ndimage.shift(Cu_im1_buffered, -translation2, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
    #Cu_im2_buffered_aligned = scipy.ndimage.shift(Cu_im2_buffered, -translation3, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)

    # Now reset the dummy values because the shift command used spline fitting and messed up some of the dummy values
    Cu_im1_buffered_aligned[:,0:buffer_edges + np.int(np.round(-translation2[1]))] = 0.1234567890123456
    Cu_im1_buffered_aligned[:,  buffer_edges + np.int(np.round(-translation2[1])) + Mn_ims[0,:,:].shape[1]:] = 0.1234567890123456
    Cu_im1_buffered_aligned[0:buffer_edges + np.int(np.round(-translation2[0])),:] = 0.1234567890123456
    Cu_im1_buffered_aligned[  buffer_edges + np.int(np.round(-translation2[0])) + Mn_ims[0,:,:].shape[0]:,:] = 0.1234567890123456
    Cu_im2_buffered_aligned[:,0:buffer_edges + np.int(np.round(-translation3[1]))] = 0.1234567890123456
    Cu_im2_buffered_aligned[:,  buffer_edges + np.int(np.round(-translation3[1])) + Mn_ims[0,:,:].shape[1]:] = 0.1234567890123456
    Cu_im2_buffered_aligned[0:buffer_edges + np.int(np.round(-translation3[0])),:] = 0.1234567890123456
    Cu_im2_buffered_aligned[  buffer_edges + np.int(np.round(-translation3[0])) + Mn_ims[0,:,:].shape[0]:,:] = 0.1234567890123456
    
    # Collect an concatenate the data into single matrices
    beam_energies = np.concatenate((beam_energies_Mn, beam_energies_Cu))
    ims_buffered_aligned = np.stack((Mn_im1_buffered,Mn_im2_buffered_aligned,Cu_im1_buffered_aligned,Cu_im2_buffered_aligned))
    
    # Stuff the data into the h5 file
    h5object_new.create_dataset('translations', shape=(4,2),   dtype=np.float64, data=np.stack((translation1,translation2,translation3,[-1,-1])))
    h5object_new.create_dataset('beam_energies', shape=(4,1), dtype=np.float64, data=beam_energies)
    h5object_new.create_dataset('xray_images', shape=(4 , Mn_ims[0,:,:].shape[0] + 2*buffer_edges , Mn_ims[0,:,:].shape[1] + 2*buffer_edges), dtype=np.float32, data=ims_buffered_aligned)
    h5object_old.close()
    h5object_new.close()
    
  
    

# Filename MUST be supplied as a numpy vector of file numbers
def align_processed_images_time_series(file_numbers,im2_cropping, cc_search_distance):  # Filename MUST be supplied as a numpy vector of file numbers
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
                
        # Figure out how much dummy values to remove on each side.  For example: value of debuffer[0] is the maximum column number of where the dummy values extend on the LHS-side of any one of the images(xanes_raw_ims1 or xanes_raw_ims2), and likewise debuffer[2] is the maximum row that the dummy values extend on the topside of any one of the images (xanes_raw_ims1 or xanes_raw_ims2)
        debuffer = calculate_debuffer_multiple_images(np.concatenate((xanes_raw_ims1,xanes_raw_ims2),axis=0))
        
        #Find out how much translation to move each image
        translation1, error1, cc_image = find_image_translation( xanes_raw_ims1[0,debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]] , xanes_raw_ims2[0,debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]] , im2_cropping, cc_search_distance)
        translation2, error2, cc_image = find_image_translation( xanes_raw_ims1[1,debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]] , xanes_raw_ims2[1,debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]] , im2_cropping, cc_search_distance)
        translation3, error3, cc_image = find_image_translation( xanes_raw_ims1[2,debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]] , xanes_raw_ims2[2,debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]] , im2_cropping, cc_search_distance)
        translation4, error4, cc_image = find_image_translation( xanes_raw_ims1[3,debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]] , xanes_raw_ims2[3,debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]] , im2_cropping, cc_search_distance)

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
        translations = h5object2['translations']
        translations[...][3,:] = translation

        # Now actually shift the images to be in alignment
        im2_1 = shift_image_integer(xanes_raw_ims2[0,:,:], -translation)
        im2_2 = shift_image_integer(xanes_raw_ims2[1,:,:], -translation)
        im2_3 = shift_image_integer(xanes_raw_ims2[2,:,:], -translation)
        im2_4 = shift_image_integer(xanes_raw_ims2[3,:,:], -translation)
        #im2_1 = scipy.ndimage.shift(xanes_raw_ims2[0,:,:], -translation, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
        #im2_2 = scipy.ndimage.shift(xanes_raw_ims2[1,:,:], -translation, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
        #im2_3 = scipy.ndimage.shift(xanes_raw_ims2[2,:,:], -translation, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)
        #im2_4 = scipy.ndimage.shift(xanes_raw_ims2[3,:,:], -translation, order=3, mode='constant', cval=0.1234567890123456, prefilter=True)       
        temp_matrix = h5object2['xray_images']
        temp_matrix[...] = np.stack((im2_1,im2_2,im2_3,im2_4)) # stupid ass python requires this [...] notation if you want to write data to the h5 file
        #h5object2.create_dataset('xray_images2', shape=(4,im_shape_rows,im_shape_cols), dtype=np.float32, data=np.stack((im2_1,im2_2,im2_3,im2_4)))
        h5object2.close()
    debuffer_multiple_image_files(file_numbers)
    
    

def debuffer_multiple_image_files(file_numbers):
    debuffer = calculate_image_debuffer_multiple_files(file_numbers)
    for i in range(0,len(file_numbers)):
        filename ="%.4f" % file_numbers[i]
        filename ='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
        print('debuffering '+filename)
        h5object_old = h5py.File(data_directory+data_subdirectory+filename, 'r')  
        h5object_new = h5py.File(data_directory+data_subdirectory+'temp.h5', 'w')  
        h5object_old.copy('beam_energies', h5object_new)
        h5object_old.copy('scan_id',       h5object_new)
        h5object_old.copy('scan_time',     h5object_new)
        h5object_old.copy('note',          h5object_new)
        h5object_old.copy('translations',  h5object_new)
        if 'optical_thickness_Mn' in h5object_old.keys(): h5object_old.copy('optical_thickness_Mn', h5object_new)
        if 'optical_thickness_Cu' in h5object_old.keys(): h5object_old.copy('optical_thickness_Cu', h5object_new)
        if 'optical_thickness_Bi' in h5object_old.keys(): h5object_old.copy('optical_thickness_Bi', h5object_new)
        if 'optical_thickness_C' in h5object_old.keys():  h5object_old.copy('optical_thickness_C', h5object_new)
        if 'optical_thickness_El' in h5object_old.keys(): h5object_old.copy('optical_thickness_El', h5object_new)
        xanes_ims = np.array(h5object_old['xray_images'])
        h5object_old.close()
        xanes_ims2 = np.zeros((4,      -debuffer[2]-1+debuffer[3] , -debuffer[0]-1+debuffer[1]),dtype=np.float32)
        xanes_ims2[0,:,:] = xanes_ims[0,debuffer[2]+1:debuffer[3] ,  debuffer[0]+1:debuffer[1]]
        xanes_ims2[1,:,:] = xanes_ims[1,debuffer[2]+1:debuffer[3] ,  debuffer[0]+1:debuffer[1]]
        xanes_ims2[2,:,:] = xanes_ims[2,debuffer[2]+1:debuffer[3] ,  debuffer[0]+1:debuffer[1]]
        xanes_ims2[3,:,:] = xanes_ims[3,debuffer[2]+1:debuffer[3] ,  debuffer[0]+1:debuffer[1]]
        h5object_new.create_dataset('xray_images', shape=xanes_ims2.shape,  dtype=np.float32, data=xanes_ims2)
        h5object_new.close()
        os.remove(data_directory+data_subdirectory+filename)
        os.rename(data_directory+data_subdirectory+'temp.h5',data_directory+data_subdirectory+filename)
        


  
  
def eliminate_beam_flickering_time_series(i):
    print(i)
    
   
    
    
# Run this function on the files output from save_aligned_h5_file     #All length units in mm
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
        



# The variable image_used_for_plot is something like 'Mn_raw_im1' or 'optical_thickness_Bi' et cetera
def make_movie_with_potentiostat_data(txm_file_numbers,biologic_file, image_used_for_plot, movie_time_span_seconds, seconds_per_movie_frame, output_filename):
        
    # Read timestamps of the images, The Biologic Computer time was 3 Minutes AHEAD of "real" time ( aka the xanes images times)
    datetime_array_xanes = np.array([])
    timestamp_array_xanes = np.array([])
    for i in txm_file_numbers:
        scan_start_time_string, scan_time, beam_energies, scan_id, notes = read_FXI_raw_h5_metadata(i)
        datetime_xanes_file  = datetime.datetime.fromtimestamp(scan_time)
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
    biologic_data=pandas.read_csv(data_directory+'Biologic_Files/'+biologic_file, sep='\t', skiprows=int(second_line[18:21])-1)
    biologic_start_time=datetime.datetime.strptime(thirteenth_line[25:44],'%m/%d/%Y %H:%M:%S')
    
    # Create the first plot (figure layout, axes, et cetera)
    figure_width = 10    #figsize=(       height             ,   width     )
    figure_height = 1080/1280*figure_width/1.5 
    fig_han = plt.figure(figsize=(figure_width, figure_height ))
    closest_index_txm=0
    global closest_index_txm_previous
    closest_index_txm_previous = -1
    im=get_processed_image(txm_file_numbers[closest_index_txm],image_used_for_plot)
    debuffer = calculate_image_debuffer_multiple_files(txm_file_numbers)
    im=im[debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]]
    # The new axis size:left, bottom,         width                                    ,   height
    im_axes = plt.axes([0.0,   0.0  , (im.shape[1]-1)/im.shape[0]*figure_height/figure_width,   1.0   ])
    im_axes.set_axis_off()
    zmin, zmax = calculate_brightness_contrast(txm_file_numbers, image_used_for_plot, 0.005, 0.995)
    # Make the scale bar
    im[-85:-65,-175:-48]=1.0
    im_axes.text(im.shape[1]-138,im.shape[0]-68,'5 um',fontsize=7.8)
    # Make the colorbar
    im[-45:-5,-195:-23]=1.0
    im[-45:-27,-180:-40]=np.tile(np.arange(zmin,zmax,(zmax - zmin)/140),(18,1))
    im[-45:-27,-180] = 0.0; im[-45:-27,-40] = 0.0; im[-45,-180:-40] = 0.0; im[-27,-180:-40] = 0.0;     
    im[-45:-27,-181] = 0.0; im[-45:-27,-39] = 0.0; im[-46,-180:-40] = 0.0; im[-26,-180:-40] = 0.0;     
    im_axes.text(im.shape[1]-195,im.shape[0]-7,"%.1f" % zmin + '                ' + "%.1f" % zmax,fontsize=7.5)
    # Show the whole image
    im_axes.imshow(im,cmap='gray',interpolation='none', vmin=zmin, vmax=zmax, label=False)
    # Make the Potentiostat Axis
    # The new axis size:                left                                           ,   bottom,                               width,                             , height
    iV_data_axes = plt.axes([im.shape[1]/im.shape[0]*figure_height/figure_width + 0.052,   0.085,  1.0 - im.shape[1]/im.shape[0]*figure_height/figure_width - 0.065 ,    0.9])
    iV_data_axes.tick_params(axis = 'both', which = 'major', labelsize = 8)    
    iV_data_axes.plot(biologic_data['Ewe/V'].values,biologic_data['<I>/mA'].values*1000,zorder=0)
    iV_data_axes.set_xlabel('Electrode Voltage (V)',fontsize=9,labelpad=1)
    y_range = iV_data_axes.get_ylim()[1] - iV_data_axes.get_ylim()[0]
    iV_data_axes.set_ylabel('Current (uA)', fontsize=9,labelpad=-9, position=(0,-iV_data_axes.get_ylim()[0]/y_range))
    scatter_han = iV_data_axes.scatter(biologic_data['Ewe/V'].values[0],biologic_data['<I>/mA'].values[0]*1000,c='r',s=20,zorder=1)
    # Make the Authorship label Axis
    authorship_label_axis = plt.axes([im.shape[1]/im.shape[0]*figure_height/figure_width - 0.065,   0.983,  0.05 ,    0.05])
    authorship_label_axis.set_axis_off();  #authorship_label_axis.imshow(np.ones((10,100)),cmap='gray',vmin=0,vmax=1.0)
    authorship_label_axis.text(0,0.1,'                    ',size=7.5,bbox=dict(boxstyle='square,pad=0.0',ec='none',fc='w'))
    authorship_label_axis.text(0,0.0,'D.E.Turney et al. 2020',alpha=0.5,size=7.5,bbox=dict(boxstyle='square,pad=0.0',ec='none',fc='w'))
    # Show the image number    
    im_id_text = im_axes.text(im.shape[1],im.shape[0]-5,'img: ' + str(txm_file_numbers[0]) ,fontsize=7.8)           
    
   
    def change_imshow(frame_num):
        global closest_index_txm_previous
        frame_time = biologic_start_time + datetime.timedelta(seconds = frame_num*seconds_per_movie_frame) - datetime.timedelta(minutes = 3) #The Biologic Computer time was 3 Minutes AHEAD of "real" time ( aka the xanes images times)   
        closest_index_biologic=abs(biologic_data['time/s'].values - frame_num*seconds_per_movie_frame).argmin() #One frame per 20 seconds
        scatter_han.set_offsets([biologic_data['Ewe/V'].values[closest_index_biologic],biologic_data['<I>/mA'].values[closest_index_biologic]*1000])
        closest_index_txm=abs(timestamp_array_xanes - np.float64(frame_time.timestamp())).argmin()
        if closest_index_txm != closest_index_txm_previous:
            closest_index_txm_previous=closest_index_txm
            im=get_processed_image(txm_file_numbers[closest_index_txm],image_used_for_plot)
            im=im[debuffer[2]+1:debuffer[3],debuffer[0]+1:debuffer[1]]            #im   = get_processed_image(txm_file_numbers[closest_index_txm],'Mn_thickness')
            # Make the scale bar
            im[-85:-65,-175:-48]=1.0
            im_axes.text(im.shape[1]-138,im.shape[0]-68,'5 um',fontsize=7.8)           
            #im[-40:-20,-225:-98]=1.0
            #plt.text(im.shape[1]-191,im.shape[0]-22,'5 um',fontsize=7.8)
            # Make the colorbar
            im[-45:-5,-195:-23]=1.0
            im[-45:-27,-180:-40]=np.tile(np.arange(zmin,zmax,(zmax - zmin)/140),(18,1))
            im[-45:-27,-180] = 0.0; im[-45:-27,-40] = 0.0; im[-45,-180:-40] = 0.0; im[-27,-180:-40] = 0.0;     
            im[-45:-27,-181] = 0.0; im[-45:-27,-39] = 0.0; im[-46,-180:-40] = 0.0; im[-26,-180:-40] = 0.0;     
            im_axes.text(im.shape[1]-195,im.shape[0]-7,"%.1f" % zmin + '                ' + "%.1f" % zmax,fontsize=7.5)
            # Show the whole image
            im_axes.imshow(im, cmap='gray',interpolation='none', vmin=zmin, vmax=zmax, label=False)
            # Show the image number
            im_id_text.set_text('img: ' + str(txm_file_numbers[closest_index_txm]))           
            print('displaying img: ' + str(txm_file_numbers[closest_index_txm]) + ' for time ' + str(frame_num*seconds_per_movie_frame) + ' seconds (' + datetime.datetime.strftime(frame_time, '%Y-%m-%d %H:%M:%S' ) + ')')

        #big_axes_han.imshow(images[int((frame_num-1)/6)],cmap='gray',interpolation='none', vmin=0.235, vmax=0.94)
        
    # It iterates through e.g. "frames=range(15)" calling the function e.g "change_imshow" , and inserts a millisecond time delay between frames of e.g. "interval=100".
    animation_handle=animation.FuncAnimation(fig_han, change_imshow, frames=range(int(movie_time_span_seconds/seconds_per_movie_frame)), blit=False, interval=100, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=5000)
    animation_handle.save(output_filename, writer=writer)


# file_numbers format: 34565.0000 means the first beam energy, location 0         34565.0001 means the first beam energy, location 1           34565.0101 means the second beam energy, location 1
# files = np.concatenate((range(34565,34725,2),range(34726,34875,2))) ; files2=np.ones(len(files)*4)
#for i in range(len(files)):
#    files2[4*i] = files[i]
#    files2[4*i+1] = files[i]+0.0001
#    files2[4*i+2] = files[i]+0.00001
#    files2[4*i+3] = files[i]+0.00011
def make_movie_with_image_statistics(file_numbers, image_type_2_show, movie_filename ):
    # Get the first image
    if image_type_2_show == 'img_bkg1':
        im=get_raw_image(file_numbers[0],'img_bkg')
        im=im[0,:,:]
    if image_type_2_show == 'img_bkg2':
        im=get_raw_image(file_numbers[0],'img_bkg')
        im=im[1,:,:]
    if image_type_2_show == 'img_bkg':
        im=get_raw_image(file_numbers[0],'img_bkg')
        if ("%.5f" % file_numbers[0])[10] == '0': im=im[0,:,:]
        if ("%.5f" % file_numbers[0])[10] == '1': im=im[1,:,:]
    if image_type_2_show == 'img_dark':
        im=get_raw_image(file_numbers[0],'img_dark')
        im=im[0,:,:]
    if image_type_2_show != 'img_bkg1' and image_type_2_show != 'img_bkg2' and image_type_2_show != 'img_bkg' and image_type_2_show != 'img_dark':
        im=get_processed_image(file_numbers[0],image_type_2_show)
        
    # Create the first plot (figure layout, axes, et cetera)
    figure_width = 10    #figsize=(       height             ,   width     )
    figure_height = 1080/1280*figure_width/1.5 
    fig_han = plt.figure(figsize=(figure_width, figure_height ))
    # The new axis size:left, bottom,         width                                    ,   height
    im_axes = plt.axes([0.0,   0.0  , (im.shape[1]-1)/im.shape[0]*figure_height/figure_width,   1.0   ])
    im_axes.set_axis_off()
    zmin, zmax = calculate_brightness_contrast(file_numbers, image_type_2_show, 0.005, 0.995)
    # Make the colorbar
    im[-45:-5,-195:]=1E20
    im[-45:-27,-180:-40]=np.tile(np.arange(zmin,zmax,(zmax - zmin)/140),(18,1))
    im[-45:-27,-180] = 0.0; im[-45:-27,-40] = 0.0; im[-45,-180:-40] = 0.0; im[-27,-180:-40] = 0.0;     
    im[-45:-27,-181] = 0.0; im[-45:-27,-39] = 0.0; im[-46,-180:-40] = 0.0; im[-26,-180:-40] = 0.0;     
    im_axes.text(im.shape[1]-195,im.shape[0]-7,"%.1f" % zmin + '                ' + "%.1f" % zmax,fontsize=7.5)
    # Show the whole image
    im_axes.imshow(im,cmap='gray',interpolation='none', vmin=zmin, vmax=zmax, label=False)
    # Make the Image Statistics Plot Axis
    # The new axis size:                left                                           ,   bottom,                               width,                             , height
    im_statistics = plt.axes([im.shape[1]/im.shape[0]*figure_height/figure_width + 0.052,   0.085,  1.0 - im.shape[1]/im.shape[0]*figure_height/figure_width - 0.08 ,    0.85])
    im_statistics.tick_params(axis = 'both', which = 'major', direction='in',labelsize = 7)    
    images_mean, images_std = get_images_statistics(file_numbers, image_type_2_show )
    im_statistics.plot(file_numbers,images_mean,zorder=1)
    im_statistics.set_xlabel('Image Number',fontsize=9,labelpad=1)
    im_statistics2 = im_statistics.twinx()
    im_statistics2.plot(file_numbers,images_std,'g')
    im_statistics2.tick_params(axis = 'both', which = 'major', direction='in',labelsize = 7)    
    scatter_han = im_statistics.scatter(file_numbers[0],images_mean[0],c='r',s=20,zorder=2)
    # Show the image number    
    im_id_text = im_axes.text(im.shape[1]+60,im.shape[0]-5,'img: ' + str(file_numbers[0]) ,fontsize=7.8)           
    
       
    def change_imshow(frame_num):
        print('displaying img: ' + str(file_numbers[frame_num]))
        # Grab the new image  
        if image_type_2_show == 'img_bkg1':
            im=get_raw_image(file_numbers[frame_num],'img_bkg')
            im=im[0,:,:]
        if image_type_2_show == 'img_bkg2':
            im=get_raw_image(file_numbers[frame_num],'img_bkg')
            im=im[1,:,:]
        if image_type_2_show == 'img_bkg':
            im=get_raw_image(file_numbers[frame_num],'img_bkg')
            if ("%.5f" % file_numbers[frame_num])[10] == '0': im=im[0,:,:]
            if ("%.5f" % file_numbers[frame_num])[10] == '1': im=im[1,:,:]
        if image_type_2_show == 'img_dark':
            im=get_raw_image(file_numbers[frame_num],'img_dark')
            im=im[0,:,:]
        if image_type_2_show != 'img_bkg1' and image_type_2_show != 'img_bkg2' and image_type_2_show != 'img_bkg' and image_type_2_show != 'img_dark':
            im=get_processed_image(file_numbers[frame_num],image_type_2_show)
        
        scatter_han.set_offsets((file_numbers[frame_num],images_mean[frame_num]))
        # Make the colorbar
        im[-45:-5,-195:]=1E20
        im[-45:-27,-180:-40]=np.tile(np.arange(zmin,zmax,(zmax - zmin)/140),(18,1))
        im[-45:-27,-180] = 0.0; im[-45:-27,-40] = 0.0; im[-45,-180:-40] = 0.0; im[-27,-180:-40] = 0.0;     
        im[-45:-27,-181] = 0.0; im[-45:-27,-39] = 0.0; im[-46,-180:-40] = 0.0; im[-26,-180:-40] = 0.0;     
        im_axes.imshow(im, vmin=zmin, vmax=zmax, interpolation='none', cmap='gray')
        im_axes.text(im.shape[1]-195,im.shape[0]-7,"%.1f" % zmin + '                ' + "%.1f" % zmax,fontsize=7.5)
        # Show the image number
        im_id_text.set_text('img: ' + str(file_numbers[frame_num]) + ', and index:' + str(frame_num))     
        plt.draw()
        plt.show()
        time.sleep(1)

        
    
    # It iterates through e.g. "frames=range(15)" calling the function e.g "change_imshow" , and inserts a millisecond time delay between frames of e.g. "interval=100".
    animation_handle=animation.FuncAnimation(fig_han, change_imshow, frames=range(len(file_numbers)), blit=False, interval=100, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15)
    animation_handle.save(movie_filename, writer=writer)




def calculate_brightness_contrast(filenumbers, image_2_display, low_end_percentile, high_end_percentile):
    low_end_all_files= np.ones(len(filenumbers))
    high_end_all_files=np.ones(len(filenumbers))
    for i in range(len(filenumbers)):
        if image_2_display == 'img_bkg1' or image_2_display == 'img_bkg2' or image_2_display == 'img_bkg':
            image = get_raw_image(filenumbers[i],'img_bkg')
            if image_2_display == 'img_bkg1': image = image[0,:,:]
            if image_2_display == 'img_bkg2': image = image[1,:,:]
            if image_2_display == 'img_bkg':  image = image[:,:,:]            
        if image_2_display == 'img_dark':
            image = get_raw_image(filenumbers[i],'img_dark')
            image = image[0,:,:]
        if image_2_display != 'img_bkg1' and image_2_display != 'img_bkg2' and image_2_display != 'img_dark' and image_2_display != 'img_bkg':
            image = get_processed_image(filenumbers[i],image_2_display)
        pdf,bin_edges=np.histogram(image[:],bins=200, range=(np.min(image[:]),np.max(image[:])),density=True)
        cumulative_probability_distribution = np.cumsum(pdf)
        cumulative_probability_distribution = cumulative_probability_distribution/cumulative_probability_distribution[-1]
        bin_centers = bin_edges[0:-1] + (bin_edges[1] - bin_edges[0])/2
        low_end_all_files[i] =  bin_centers[np.int(np.argmin(abs(cumulative_probability_distribution - low_end_percentile)))]
        high_end_all_files[i] = bin_centers[np.int(np.argmin(abs(cumulative_probability_distribution - high_end_percentile)))]

        
    return(np.min(low_end_all_files) , np.max(high_end_all_files))


    
    
# Filename MUST be supplied as a number
def plot_single_pixel_least_squares_data(filename,row,column):   # Filename MUST be supplied as a number
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




def read_FXI_raw_h5_metadata(filename):  #filename can be 34567.0103  to denote repeat 01, position 03
    if type(filename) != str:
        filename="%.4f" % filename
        filename='multipos_2D_xanes_scan2_id_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object    = h5py.File(data_directory+data_subdirectory+filename, 'r')
    beam_energies = np.array(h5object['X_eng'])
    scan_time   = np.array(h5object['scan_time'])  #scan start time in local time at NSLS2, in epoch format
    scan_id     = np.array(h5object['scan_id'])
    notes       = np.array(h5object['note'])
    scan_start_time = datetime.datetime.fromtimestamp(scan_time)
    scan_start_time_string = datetime.datetime.strftime(scan_start_time, '%Y-%m-%d %H:%M:%S' )    
    h5object.close() 
    return(scan_start_time_string, scan_time, beam_energies, scan_id, notes) 






def read_FXI_processed_h5_metadata(filename):  #filename can be 34567.0103  to denote repeat 01, position 03
    if type(filename) != str:
        filename="%.4f" % filename
        filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object= h5py.File(data_directory+data_subdirectory+filename, 'r')
    beam_energy = np.array(h5object['beam_energies'])
    scan_time   = np.array(h5object['scan_time'])  #scan start time in local time at NSLS2, in epoch format
    scan_id     = np.array(h5object['scan_id'])
    notes       = np.array(h5object['note'])
    translations= np.array(h5object['translations'])
    scan_start_time = datetime.datetime.fromtimestamp(scan_time)
    scan_start_time_string = datetime.datetime.strftime(scan_start_time, '%Y-%m-%d %H:%M:%S' )    
    h5object.close() 
    return(scan_start_time_string, scan_time, beam_energy, scan_id, str(notes), translations)





def get_raw_image(filename,which_image):
    if type(filename) != str:
        filename="%.4f" % filename
        filename='multipos_2D_xanes_scan2_id_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object    = h5py.File(data_directory+data_subdirectory+filename, 'r')
    images      = np.array(h5object[which_image])  
    h5object.close()
    return(images)




def get_processed_image(filename,which_image):
    if type(filename) != str:
        filename="%.4f" % filename
        filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object= h5py.File(data_directory+data_subdirectory+filename, 'r')
#    optical_thickness_Mn= np.array(h5object['optical_thickness_Mn'])
#    optical_thickness_Cu= np.array(h5object['optical_thickness_Cu'])
#    optical_thickness_Bi= np.array(h5object['optical_thickness_Bi'])
#    optical_thickness_C = np.array(h5object['optical_thickness_C'])
#    optical_thickness_El= np.array(h5object['optical_thickness_El'])
    
    if which_image == 'all':
        xray_images = np.array(h5object['xray_images'])
        optical_thickness_Mn = np.array(h5object['optical_thickness_Mn'])
        optical_thickness_Cu = np.array(h5object['optical_thickness_Cu'])
        optical_thickness_Bi = np.array(h5object['optical_thickness_Bi'])
        optical_thickness_C = np.array(h5object['optical_thickness_C'])
        optical_thickness_El = np.array(h5object['optical_thickness_El'])
        return(np.stack((xray_images,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)))
    
    if which_image == 'xray_images':
        xray_images = np.array(h5object['xray_images'])
        return(xray_images)

    if which_image == '6520' or which_image == 'Mn_raw_im1':
        xray_images = np.array(h5object['xray_images'])
        return(xray_images[0,:,:])

    if which_image == '6600' or which_image == 'Mn_raw_im2':
        xray_images = np.array(h5object['xray_images'])
        return(xray_images[1,:,:])

    if which_image == '8970' or which_image == 'Cu_raw_im1':
        xray_images = np.array(h5object['xray_images'])
        return(xray_images[2,:,:])

    if which_image == '9050' or which_image == 'Cu_raw_im2':
        xray_images = np.array(h5object['xray_images'])
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




#This function returns the number of pixels that the second image is translated [positive is going downward, positive is going to-the-right] with respect to the first image
def find_image_translation( im1, im2, im2_cropping, cc_search_distance): 
    # debuffer is meant to elimated the SAME amount of padding on im1 and im2 BEFORE the im2_masking is applied
    # im2_cropping is the reduction in size of im2 so that it can be used for cross-correlations at different locations on top of im1
    
    # Calculate the cross correlations
    cc_image=erc_R(im1, im2, im2_cropping, cc_search_distance)   #cross correlation image

    # Figure out the translation to align im2 to im1
    max_indices=np.array(np.unravel_index(np.argmax(cc_image,axis=None), cc_image.shape))
    translation_y = np.float64(im2_cropping[2] - max_indices[0])  
    translation_x = np.float64(im2_cropping[0] - max_indices[1]) 

    ## Now do subpixel resolution.  I DONT DO SUBPIXEL RESOLUTOIN NOWADAys BECAUSE IT REQUIRES THE SUBSEQUENT SHIFT OPERATIONS TO USE INTERPOLATIONS OF THE WHOLE IMAGE WHICH BLUR THE IMAGE
    y_data = np.float64(cc_image[max_indices[0]-1 : max_indices[0]+2, max_indices[1]                      ])
    x_data = np.float64(cc_image[max_indices[0]                     , max_indices[1]-1 : max_indices[1]+2 ])
    y_data = y_data - np.min(y_data)
    x_data = x_data - np.min(x_data)
    subpixel_y = np.float64( -( -1*y_data[0] + 0*y_data[1] + 1*y_data[2] ) / np.sum(y_data) )  
    subpixel_x = np.float64( -( -1*x_data[0] + 0*x_data[1] + 1*x_data[2] ) / np.sum(x_data) ) 
    #a = np.arange(max_indices[0]-5, max_indices[0]+5.001,0.1)
    #b = np.arange(max_indices[0]-5, max_indices[0]+5.001,1.0)
    #c = np.arange(max_indices[1]-5, max_indices[1]+5.001,0.1)
    #d = np.arange(max_indices[1]-5, max_indices[1]+5.001,1.0)   
    #subpixel_y = np.interp(a, b, y_data )
    #subpixel_x = np.interp(c, d, x_data )
    #translation_x = translation_x + subpixel_x
    #translation_y = translation_y + subpixel_y
    
    error = 1 #This is a dummy number until I figure out how to calculate error
    
    return(np.array([translation_y, translation_x]), error, cc_image)
    
    

def erc_R(im1, im2_orig, im2_cropping, cc_search_distance):
    
    # Rreduce the size of im2 so that it can be translated as a window over the top of im1
    im2 = im2_orig[im2_cropping[2]:-im2_cropping[3], im2_cropping[0]:-im2_cropping[1] ]
    
    #im1 should be bigger than im2, so that im2 can be used for cross-correlations at different locations on top of im1
    #    %For choosing the two sub-images to correlate:
    #    %   m is the offset, measured in pixels, of the searching window to-the-right. The searching window pans across im1
    #    %   n is the offset, measured in pixels, of the searching window down. The searching window pans across im1
    
    winsize1=im1.shape
    winsize2=im2.shape;
    variance_im2=np.var(im2);
    length_im2=len(im2[:]);
    
    R   = np.ones((winsize1[0]-winsize2[0]+1,winsize1[1]-winsize2[1]+1))*0.12345678901
    im1_subwindow = np.ones(im2.shape)
    
    for m in range(im2_cropping[0]-cc_search_distance[0],im2_cropping[0]+1+cc_search_distance[1],3):
        for n in range(im2_cropping[2]-cc_search_distance[2],im2_cropping[2]+1+cc_search_distance[3],3):
            #print(m,n)
            im1_subwindow[:,:]=im1[n:n+winsize2[0]-1+1,m:m+winsize2[1]-1+1];
            #%I use R[n+winsize/2,m+winsize/2] in order to keep in line with Kristof Sveen's convention on the meaning of R
            R[n,m]=np.sum((im2[:]-np.mean(im2[:])) * (im1_subwindow[:]-np.mean(im1_subwindow[:]))) / (length_im2-1)/np.sqrt(variance_im2*np.var(im1_subwindow[:]))  
    
    max_indices = np.array(np.unravel_index(np.argmax(R,axis=None), R.shape))
    focused_indices=[]
    for m in range(0,winsize1[1]-winsize2[1]+1):
        for n in range(0,winsize1[0]-winsize2[0]+1):
            if abs(m - max_indices[1])<7 and abs(n - max_indices[0])<7 and R[n,m]==0.12345678901:
                focused_indices.append([n,m])
    
    for i in range(0,len(focused_indices)):
        n = focused_indices[i][0]
        m = focused_indices[i][1]
        im1_subwindow[:,:]=im1[n:n+winsize2[0]-1+1,m:m+winsize2[1]-1+1]
        R[n,m]=np.sum((im2[:]-np.mean(im2[:])) * (im1_subwindow[:]-np.mean(im1_subwindow[:]))) / (length_im2-1)/np.sqrt(variance_im2*np.var(im1_subwindow[:]))  
        
    return(R)




# Filename MUST be supplied as a number and must be the Mn raw image file
def shift_image_integer(im_old,translation):  # Filename MUST be supplied as a number and must be the Mn raw image file
    
    translation = [np.int(np.round(translation[0])), np.int(np.round(translation[1])), ]

    #Create the buffered images
    im_new = np.ones(im_old.shape)*0.1234567890123456
    
    #Now let's actually shift the image 
    if translation[0]== 0  and translation[1]== 0:  im_new[  translation[0]:,  translation[1]:] = im_old[  translation[0]:,  translation[1]:];       
    if translation[0]== 0  and translation[1] > 0:  im_new[  translation[0]:,  translation[1]:] = im_old[  translation[0]:,:-translation[1]];   
    if translation[0]== 0  and translation[1] < 0:  im_new[  translation[0]:,: translation[1] ] = im_old[  translation[0]:, -translation[1]:];   
    if translation[0] > 0  and translation[1]== 0:  im_new[  translation[0]:,  translation[1]:] = im_old[:-translation[0] ,  translation[1]:];    
    if translation[0] < 0  and translation[1]== 0:  im_new[: translation[0]:,  translation[1]:] = im_old[ -translation[0]:,  translation[1]:];    
    if translation[0] < 0  and translation[1] < 0:  im_new[: translation[0] ,: translation[1] ] = im_old[ -translation[0]:, -translation[1]:];       
    if translation[0] > 0  and translation[1] < 0:  im_new[  translation[0]:,: translation[1] ] = im_old[:-translation[0] , -translation[1]:];    
    if translation[0] < 0  and translation[1] > 0:  im_new[: translation[0] ,  translation[1]:] = im_old[ -translation[0]:,:-translation[1] ];   
    if translation[0] > 0  and translation[1] > 0:  im_new[  translation[0]:,  translation[1]:] = im_old[:-translation[0] ,:-translation[1] ];
       
    return(im_new)
    


    
def calculate_debuffer_multiple_images(ims):
    debuffer = [0,0,0,0]  # debuffer[0] is how many LHS dummy columns.  debuffer[1] is how many RHS dummy columns.  debuffer[2] is how many topside dummy rows.  debuffer[3] is how many bottomside dummy rows.  
    debuffer_all_images=[100000,0,100000,0] 
    if ims.ndim == 2: ims=np.stack((ims,ims))  #In case the user gives a single image
    im_shape_rows = ims.shape[1]
    im_shape_cols = ims.shape[2]
    im_half_rows = np.int(im_shape_rows/2)
    im_half_cols = np.int(im_shape_cols/2)
    for i in range(ims.shape[0]):
        debuffer[0] = int( np.max(np.where(ims[i,  im_half_rows ,0:im_half_cols ]==0.1234567890123456)) )
        debuffer[1] = int( np.min(np.where(ims[i,  im_half_rows ,  im_half_cols:]==0.1234567890123456)) ) + im_half_cols
        debuffer[2] = int( np.max(np.where(ims[i,0:im_half_rows ,  im_half_cols ]==0.1234567890123456)) )
        debuffer[3] = int( np.min(np.where(ims[i,  im_half_rows:,  im_half_cols ]==0.1234567890123456)) ) + im_half_rows
        if debuffer[0]<debuffer_all_images[0]: debuffer_all_images[0]=debuffer[0]
        if debuffer[1]>debuffer_all_images[1]: debuffer_all_images[1]=debuffer[1]
        if debuffer[2]<debuffer_all_images[2]: debuffer_all_images[2]=debuffer[2]
        if debuffer[3]>debuffer_all_images[3]: debuffer_all_images[3]=debuffer[3]
        
    return(debuffer_all_images)
    
    
def calculate_image_debuffer_multiple_files(file_numbers):
    debuffer_multi_file = [100000,0,100000,0]  # debuffer[0] is how many LHS dummy columns.  debuffer[1] is how many RHS dummy columns.  debuffer[2] is how many topside dummy rows.  debuffer[3] is how many bottomside dummy rows.  
    
    for i in range(0,len(file_numbers)):
        filename="%.4f" % file_numbers[i]
        filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
        h5object= h5py.File(data_directory+data_subdirectory+filename, 'r')        
        xanes_raw_ims = np.array(h5object['xray_images'])
        h5object.close()
        debuffer = calculate_debuffer_multiple_images(xanes_raw_ims)
        if debuffer[0]<debuffer_multi_file[0]: debuffer_multi_file[0]=debuffer[0]
        if debuffer[1]>debuffer_multi_file[1]: debuffer_multi_file[1]=debuffer[1]
        if debuffer[2]<debuffer_multi_file[2]: debuffer_multi_file[2]=debuffer[2]
        if debuffer[3]>debuffer_multi_file[3]: debuffer_multi_file[3]=debuffer[3]
        
    return(debuffer_multi_file)

    
   


def combine_more_loading1_with_more_loading2():
    # Read Biologic Potentiostat Data
    fileobject=open(data_directory+'Biologic_Files/'+biologic_file,'r',errors='ignore')
    fileobject.seek(0)
    fileobject.readline()
    second_line=fileobject.readline()
    for i in range(1,11):
        fileobject.readline()       
    thirteenth_line=fileobject.readline()
    fileobject.close()
    biologic_data=pandas.read_csv(data_directory+'Biologic_Files/'+biologic_file, sep='\t', skiprows=int(second_line[18:21])-1)
    biologic_start_time=datetime.datetime.strptime(thirteenth_line[25:44],'%m/%d/%Y %H:%M:%S')





def get_images_statistics(file_numbers, image_type_2_show ):
    images_mean = np.ones(len(file_numbers))
    images_std = np.ones(len(file_numbers))
    for i in range(0,len(file_numbers)):
        # Get the first image
        if image_type_2_show == 'img_bkg1':
            im=get_raw_image(file_numbers[i],'img_bkg')
            im=im[0,:,:]
        if image_type_2_show == 'img_bkg2':
            im=get_raw_image(file_numbers[i],'img_bkg')
            im=im[1,:,:]
        if image_type_2_show == 'img_bkg':
            im=get_raw_image(file_numbers[i],'img_bkg')
            if ("%.5f" % file_numbers[0])[10] == '0': im=im[0,:,:]
            if ("%.5f" % file_numbers[0])[10] == '1': im=im[1,:,:]
        if image_type_2_show == 'img_dark':
            im=get_raw_image(file_numbers[i],'img_dark')
            im=im[0,:,:]
        if image_type_2_show != 'img_bkg1' and image_type_2_show != 'img_bkg2' and image_type_2_show != 'img_bkg' and image_type_2_show != 'img_dark':
            im=get_processed_image(file_numbers[i],image_type_2_show)
        
        im = im[510:-510,610:-610]
        images_mean[i] = np.mean(im[:])
        images_std[i] =  np.std(im[:])
        
    return(images_mean, images_std) 

