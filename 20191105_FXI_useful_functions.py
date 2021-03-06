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
import matplotlib.animation

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
# TOTAL list of Mn-wavelength (6520eV, 6600eV) xanes files at location 1    
# xanes_files = np.concatenate((range(34565,34725,2),range(34726,34875,2)))
#
# TOTAL list of Cu-wavelength (8970eV, 9050eV) xanes files at location 1    
# xanes_files = np.concatenate((range(34566,34725,2),range(34727,34875,2)))
#
# TOTAL list of Mn-wavelength (6520eV, 6600eV) xanes files at location 2
# xanes_files = np.concatenate((range(34565.0001,34725.0001,2),range(34726.0001,34875.0001,2)))
#
# TOTAL list of Cu-wavelength (8970eV, 9050eV) xanes files at location 2    
# xanes_files = np.concatenate((range(34566.0001,34725.0001,2),range(34727.0001,34875.0001,2)))
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
# make_average_image(scan_numbers, which_image, output_filename ): return(average_scalar_series, std_scalar_series)
# internally_align_h5_file(Mn_filename, im2_cropping, cc_search_distance, average_dark_image_filename_Mn='none', average_dark_image_filename_Cu='none',):    #cc_search_distance is the cross correlation search distance [left, right, top, bottom]    
# globally_align_images_time_series(scan_numbers,im2_cropping, cc_search_distance):  
# debuffer_multiple_image_files(scan_numbers):
# deflicker_xray_images(scan_numbers,gaussian_filter_sizes,remove_elements='no'):
# calculate_elemental_moles_per_cm2(filename, carbon_thickness=180, total_thickness=250):   
# make_movie_with_potentiostat_data(txm_scan_numbers,biologic_file, image_used_for_plot, movie_time_span_seconds, seconds_per_movie_frame, output_filename):
# make_movie_with_image_statistics(scan_numbers, image_type_2_show, movie_filename ):
# calculate_brightness_contrast(filenumbers, image_2_display, low_end_percentile, high_end_percentile): return(np.min(low_end_all_files) , np.max(high_end_all_files))
# plot_single_pixel_least_squares_data(filename,row,column):  
# read_FXI_raw_h5_metadata(filename): return(scan_start_time_string, scan_time, beam_energies, scan_id, notes) 
# read_FXI_processed_h5_metadata(filename): return(scan_start_time_string, scan_time, beam_energy, scan_id, notes, translations)
# get_raw_image(filename,which_image):
# get_processed_image(filename,which_image):  return(image)
# dS_dthickness_all(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm2,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm2): return(test_sum_squares)    
# calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm2,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm2): return(baseline_sum_square_error)
# show_cross_correlation_map(im1,im2,debuffer=0):
# find_image_translation( im1, im2, im2_cropping, cc_search_distance): return(np.array([translation_y, translation_x]), error, cc_image)
# erc_R(im1, im2_orig, im2_cropping, cc_search_distance): return(R)
# shift_image_integer(im_old,translation): return(im_new)
# calculate_debuffer_multiple_images(ims): return(debuffer_all_images)
# calculate_image_debuffer_multiple_files(scan_numbers):  return(debuffer_multi_file)
# get_images_statistics(scan_numbers, image_type_2_show ): return(images_mean, images_std) 
# deflicker_using_4_neighbors_time_series(target_scan_number , four_neigbhors , gaussian_filter_sizes ):
# deflicker_using_average_image_elements_removed(target_scan_number , averaged_image_filename , gaussian_filter_size , beam_energy,  remove_elements):
# deflicker_9050_using_8970(scan_numbers):
###########################################################


### Workflow ##############################################
# NOTE: ALL THE 8970 & 9050 H5 FILES (MULTIPOS_2D_XANES...H5 FILES) USED 2.5 SEC EXPOSURE TIME WHEREAS THE 6520 & 6600 FILES USED 5 SECDONS!!!  ALSO, SCAN 34600 HAS A BAD DARK IMAGE SO YOU HAVE TO REMEMBER TO NOT USE IT’S DARK IMAGE!!!  
# 0) make_average_image((range(34565,34725,2), 'img_dark',  'ave_dark_5s_exposure_34565_34723.h5') to create average darkfield image for the 5 sec exposed images (Mn) and make_average_image((range(34566,34725,2), 'img_dark',  'ave_dark_2p5s_exposure_34566_34724.h5') 2.5s exposed images (Cu):  
# 1) internally_align_h5_file(scan_number,[50,350,300,300],[50,100,75,75],'ave_dark_5s_exposure_34565_34873.h5','ave_dark_2p5s_exposure_34566_34724.h5')  on each Manganese xanes_scan2_[]...h5 file to align the images.     NOTE:  file 34725 is missing, see your beamline notes -- before 34725 the Mn files are odd numbered and after 34725 the Mn files are even numbered .   NOTE: the [50,350,300,300] chops off L.R.T.B. so that the copper TEM mesh doesn't confuse the cc_image. The  [50,100,75,75] says how far to search in each direction when calculating the cross correlations.  The cc_search_distance can't be larger than the im2_cropping!!! 
# 2) globally_align_images_time_series(range(34565,34725,2),[100,350,300,300],[50,100,75,75])   The im2_cropping=[100,350,200,200] is how much of the sides and top/bottom to cutoff im2.    Use a command like for i in range(34675,34725,2): calculate_elemental_moles_per_cm2(i);   
# 3) deflicker_xray_images_time_series(np.arange(34565,34725,2),[100,100,50,50],background_image='none')  where I found 6720 and 6600 eV to need a large Gaussian_filter_Size of 100 meanwhile 8970 and 9050 eV used a smaller Gaussian_Filter_Size of 50.   This step partially fixes the homogeneity problems.  I have to do further work in steps 5 and 6 to better resolve the homogeneity problems for the 8970 eV and 9050 eV data.
# 4) deflicker_9050_using_8970(scan_numbers): to deflicker the 9050 images (because they're the MOST flickery) by calculating the difference between the 8970 and 9050 images and looking for when the 9050 are brighter than it could possibly be.
# 5) calculate_elemental_moles_per_cm2(filename, carbon_thickness=180, total_thickness=250): 
# 6) make_movie_with_potentiostat_data(range(34565,34725,2),'20191107_Cu-Bi-Birnessite_37NaOH_more_loading1_and2.mpt', 'Mn_raw_im1', 15500,40, '34565_34599_6520eV.mp4')

#  MAYBE USE 7 IMAGES OF TIME SERIES FOR DEFLICKERING? 
#  MAYBE RUN THE SINGLE IMAGE DEFLICKER (WITH REMOVE ELEMENTS) ON THE OTHER BEAM ENERGIES?
#  FIGURE OUT IF THE BISMUTH GHOSTS ARE REAL OR NOT
# ?) make_average_image(np.arange(34565,34725,2), '8970', 'ave_8970_34565_34725_Bi_and_Mn_removed.h5', 'Bi')   to make a relable image so that we can fix the homogeneity of xanes data acquired at the 8970 eV beam energy (only the 8970 eV and 9050 eV had serious problems and the 8970 eV data was the only data I could find a way to fix)
# ?) deflicker_using_average_image_elements_removed(34703,'ave_8970_34565_34725_Bi_and_Mn_removed.h5', 50, '8970', 'BiCu')   Now we run deflicker again with extra information of where the elements are located, so we can remove the effect of the elements and homogenize each x-ray imagedeflicker the 8970 eV images after removing Bi and Cu from the 8970 eV images
# ?) maybe deflicker by time series again?
# ?) Calculate the location of the elements by using the simpler calculation that uses just two images for Mn, and two images for Cu.
############################################################


                                                                   #remove_elements can be 'none' or 'all' or 'Mn' or 'Cu' or 'Bi' or 'BiMn' or 'CuBi' etc...
def make_average_image(scan_numbers, which_image, output_filename , remove_elements = 'none'):           
    average_scalar_series = np.ones(len(scan_numbers))
    std_scalar_series = np.ones(len(scan_numbers))
    for i in range(len(scan_numbers)):
        # Grab the image
        if which_image == 'img_bkg':
            im=get_raw_image(scan_numbers[i],'img_bkg')
            im=np.mean(im,axis=0)
            if i==0: 
                average_image = im*0.0
        if which_image == 'img_dark':
            im=get_raw_image(scan_numbers[i],'img_dark')[0,:,:]
            if i==0: 
                average_image = im*0.0
        if which_image != 'img_bkg' and which_image != 'img_dark':
            im=get_processed_image(scan_numbers[i], which_image, remove_elements)[:,:,0]
            im[im==0.12345678] = np.median(im[im!=0.12345678])
            if i==0: 
                average_image = im*0.0

        print('calculating img: ' + str(scan_numbers[i]))

        if i>0: average_image = average_image + im
        average_scalar_series[i] = np.mean(im[im!=0.12345678])
        std_scalar_series[i] = np.std(im[im!=0.12345678])
    
    average_image = average_image/np.float32(len(scan_numbers))
    h5object_new = h5py.File(data_directory+data_subdirectory+output_filename, 'w')
    h5object_new.create_dataset('average_image', shape=(average_image.shape), dtype=np.float32, data=average_image)
    return(average_scalar_series, std_scalar_series)



# Filename MUST be supplied as a number  #cc_search_distance is [left, right, top, bottom]
def internally_align_h5_file(Mn_filenames, im2_cropping, cc_search_distance, average_dark_image_filename_Mn='none', average_dark_image_filename_Cu='none',):    #cc_search_distance is the cross correlation search distance [left, right, top, bottom]
    for i in range(0,len(Mn_filenames)):
        Mn_filename = Mn_filenames[i]
        print(Mn_filename)    
        Cu_filename=Mn_filename+1
        Mn_filename_string="%.4f" % Mn_filename
        Mn_filename_string='multipos_2D_xanes_scan2_id_'+Mn_filename_string[0:5]+'_repeat_'+Mn_filename_string[6:8]+'_pos_'+Mn_filename_string[8:10]+'.h5'
        h5object_old = h5py.File(data_directory+data_subdirectory+Mn_filename_string, 'r')    
        h5object_new = h5py.File(data_directory+data_subdirectory+'processed_images_'+Mn_filename_string[27:-3]+'.h5', 'w')
        h5object_new.create_dataset('data_processing_note1', dtype=h5py.string_dtype(), data='internally_align_h5_file(Mn_h5file,'+str(im2_cropping)+','+str(cc_search_distance)+','+str(average_dark_image_filename_Mn)+','+str(average_dark_image_filename_Cu)+')')
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
            average_dark_image_Mn = np.array(temp_obj['average_image'])[0,:,:]
            temp_obj.close()
            im_dark = get_raw_image(Mn_filename,'img_dark')[0,:,:]
            im_bkg  = get_raw_image(Mn_filename,'img_bkg')
            Mn_ims[0,:,:] = ((Mn_ims[0,:,:] * (im_bkg[0,:,:] - im_dark) +  im_dark ) - average_dark_image_Mn ) / (im_bkg[0,:,:] - average_dark_image_Mn)
            Mn_ims[1,:,:] = ((Mn_ims[1,:,:] * (im_bkg[1,:,:] - im_dark) +  im_dark ) - average_dark_image_Mn ) / (im_bkg[1,:,:] - average_dark_image_Mn)
        translation1, error, cc_image = find_image_translation(Mn_ims[0,:,:],Mn_ims[1,:,:], im2_cropping, cc_search_distance)
        # Add a buffer of dummy values so that we don't lose data when we shift
        Mn_im1_buffered = np.pad(Mn_ims[0,:,:], buffer_edges, 'constant', constant_values=0.12345678 ) 
        Mn_im2_buffered = np.pad(Mn_ims[1,:,:], buffer_edges, 'constant', constant_values=0.12345678 ) 
        # Now actually do the  shift.  Use -translation to put it back to where it should be (aligned with im1)
        Mn_im2_buffered_aligned = shift_image_integer(Mn_im2_buffered, -translation1)
        #Mn_im2_buffered_aligned = scipy.ndimage.shift(Mn_im2_buffered, -translation1, order=3, mode='constant', cval=0.12345678, prefilter=True)
        # Now reset the dummy values because the shift command used spline fitting and messed up some of the dummy values
        Mn_im2_buffered_aligned[:,0:buffer_edges + np.int(np.round(-translation1[1]))] = 0.12345678
        Mn_im2_buffered_aligned[:,  buffer_edges + np.int(np.round(-translation1[1])) + Mn_ims[0,:,:].shape[1]:] = 0.12345678
        Mn_im2_buffered_aligned[0:buffer_edges + np.int(np.round(-translation1[0])),:] = 0.12345678
        Mn_im2_buffered_aligned[  buffer_edges + np.int(np.round(-translation1[0])) + Mn_ims[0,:,:].shape[0]:,:] = 0.12345678
        
        # Shift the Cu images to be aligned with the 1st Mn image
        Cu_ims = get_raw_image(Cu_filename,'img_xanes');  
        if average_dark_image_filename_Cu != 'none':
            temp_obj = h5py.File(data_directory+data_subdirectory+average_dark_image_filename_Cu, 'r')
            average_dark_image_Cu = np.array(temp_obj['average_image'])[0,:,:]
            temp_obj.close()
            im_dark = get_raw_image(Cu_filename,'img_dark')[0,:,:]
            im_bkg  = get_raw_image(Cu_filename,'img_bkg')
            Cu_ims[0,:,:] = ((Cu_ims[0,:,:] * (im_bkg[0,:,:] - im_dark) +  im_dark ) - average_dark_image_Cu ) / (im_bkg[0,:,:] - average_dark_image_Cu)
            Cu_ims[1,:,:] = ((Cu_ims[1,:,:] * (im_bkg[1,:,:] - im_dark) +  im_dark ) - average_dark_image_Cu ) / (im_bkg[1,:,:] - average_dark_image_Cu)    
        translation2, error, cc_image = find_image_translation(Mn_ims[0,:,:],Cu_ims[0,:,:], im2_cropping, cc_search_distance)
        translation3, error, cc_image = find_image_translation(Mn_ims[0,:,:],Cu_ims[1,:,:], im2_cropping, cc_search_distance)
        # Add a buffer of dummy values so that we don't lose data when we shift
        Cu_im1_buffered = np.pad(Cu_ims[0,:,:], buffer_edges, 'constant', constant_values=0.12345678 ) 
        Cu_im2_buffered = np.pad(Cu_ims[1,:,:], buffer_edges, 'constant', constant_values=0.12345678 ) 
        # Now actually do the  shift
        Cu_im1_buffered_aligned = shift_image_integer(Cu_im1_buffered, -translation2)
        Cu_im2_buffered_aligned = shift_image_integer(Cu_im2_buffered, -translation3)
        #Cu_im1_buffered_aligned = scipy.ndimage.shift(Cu_im1_buffered, -translation2, order=3, mode='constant', cval=0.12345678, prefilter=True)
        #Cu_im2_buffered_aligned = scipy.ndimage.shift(Cu_im2_buffered, -translation3, order=3, mode='constant', cval=0.12345678, prefilter=True)
    
        # Now reset the dummy values because the shift command used spline fitting and messed up some of the dummy values
        Cu_im1_buffered_aligned[:,0:buffer_edges + np.int(np.round(-translation2[1]))] = 0.12345678
        Cu_im1_buffered_aligned[:,  buffer_edges + np.int(np.round(-translation2[1])) + Mn_ims[0,:,:].shape[1]:] = 0.12345678
        Cu_im1_buffered_aligned[0:buffer_edges + np.int(np.round(-translation2[0])),:] = 0.12345678
        Cu_im1_buffered_aligned[  buffer_edges + np.int(np.round(-translation2[0])) + Mn_ims[0,:,:].shape[0]:,:] = 0.12345678
        Cu_im2_buffered_aligned[:,0:buffer_edges + np.int(np.round(-translation3[1]))] = 0.12345678
        Cu_im2_buffered_aligned[:,  buffer_edges + np.int(np.round(-translation3[1])) + Mn_ims[0,:,:].shape[1]:] = 0.12345678
        Cu_im2_buffered_aligned[0:buffer_edges + np.int(np.round(-translation3[0])),:] = 0.12345678
        Cu_im2_buffered_aligned[  buffer_edges + np.int(np.round(-translation3[0])) + Mn_ims[0,:,:].shape[0]:,:] = 0.12345678
        
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
def globally_align_images_time_series(scan_numbers,im2_cropping, cc_search_distance):  # Filename MUST be supplied as a numpy vector of file numbers
    for i in range(1,len(scan_numbers)):
        
        print(scan_numbers[i])
        
        filename1="%.4f" % scan_numbers[i-1]
        filename1='processed_images_'+filename1[0:5]+'_repeat_'+filename1[6:8]+'_pos_'+filename1[8:10]+'.h5'
        h5object1= h5py.File(data_directory+data_subdirectory+filename1, 'r')        
        xanes_raw_ims1 = np.array(h5object1['xray_images'])
        h5object1.close()
        
        filename2="%.4f" % scan_numbers[i]
        filename2='processed_images_'+filename2[0:5]+'_repeat_'+filename2[6:8]+'_pos_'+filename2[8:10]+'.h5'
        h5object2= h5py.File(data_directory+data_subdirectory+filename2, 'r+')
        h5object2.create_dataset('data_processing_note2', dtype=h5py.string_dtype(),data='globally_align_images_time_series(scan_numbers,'+str(im2_cropping)+','+str(cc_search_distance)+')')
        xanes_raw_ims2 = np.array(h5object2['xray_images'])
                
        # Figure out how much dummy values to remove on each side.  For example: value of debuffer[0] is the maximum column number of where the dummy values extend on the LHS-side of any one of the images(xanes_raw_ims1 or xanes_raw_ims2), and likewise debuffer[2] is the maximum row that the dummy values extend on the topside of any one of the images (xanes_raw_ims1 or xanes_raw_ims2)
        debuffer = calculate_debuffer_multiple_images(np.concatenate((xanes_raw_ims1,xanes_raw_ims2),axis=0))
        
        #Find out how much translation to move each image
        translation1, error1, cc_image = find_image_translation( xanes_raw_ims1[0,debuffer[2]:debuffer[3],debuffer[0]:debuffer[1]] , xanes_raw_ims2[0,debuffer[2]:debuffer[3],debuffer[0]:debuffer[1]] , im2_cropping, cc_search_distance)
        translation2, error2, cc_image = find_image_translation( xanes_raw_ims1[1,debuffer[2]:debuffer[3],debuffer[0]:debuffer[1]] , xanes_raw_ims2[1,debuffer[2]:debuffer[3],debuffer[0]:debuffer[1]] , im2_cropping, cc_search_distance)
        translation3, error3, cc_image = find_image_translation( xanes_raw_ims1[2,debuffer[2]:debuffer[3],debuffer[0]:debuffer[1]] , xanes_raw_ims2[2,debuffer[2]:debuffer[3],debuffer[0]:debuffer[1]] , im2_cropping, cc_search_distance)
        translation4, error4, cc_image = find_image_translation( xanes_raw_ims1[3,debuffer[2]:debuffer[3],debuffer[0]:debuffer[1]] , xanes_raw_ims2[3,debuffer[2]:debuffer[3],debuffer[0]:debuffer[1]] , im2_cropping, cc_search_distance)

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
        #im2_1 = scipy.ndimage.shift(xanes_raw_ims2[0,:,:], -translation, order=3, mode='constant', cval=0.12345678, prefilter=True)
        #im2_2 = scipy.ndimage.shift(xanes_raw_ims2[1,:,:], -translation, order=3, mode='constant', cval=0.12345678, prefilter=True)
        #im2_3 = scipy.ndimage.shift(xanes_raw_ims2[2,:,:], -translation, order=3, mode='constant', cval=0.12345678, prefilter=True)
        #im2_4 = scipy.ndimage.shift(xanes_raw_ims2[3,:,:], -translation, order=3, mode='constant', cval=0.12345678, prefilter=True)       
        temp_matrix = h5object2['xray_images']
        temp_matrix[...] = np.stack((im2_1,im2_2,im2_3,im2_4)) # stupid ass python requires this [...] notation if you want to write data to the h5 file
        #h5object2.create_dataset('xray_images2', shape=(4,im_shape_rows,im_shape_cols), dtype=np.float32, data=np.stack((im2_1,im2_2,im2_3,im2_4)))
        h5object2.close()
    debuffer_multiple_image_files(scan_numbers)
    
    


  
  
def deflicker_xray_images_time_series(scan_numbers , gaussian_filter_sizes):
    i=0
    print('de-flickering ' + str(scan_numbers[i])) 
    deflicker_using_4_neighbors_time_series(scan_numbers[i],[scan_numbers[i+1], scan_numbers[i+1], scan_numbers[i+2], scan_numbers[i+3]],gaussian_filter_sizes)
    i=1
    print('de-flickering ' + str(scan_numbers[i])) 
    deflicker_using_4_neighbors_time_series(scan_numbers[i],[scan_numbers[i-1], scan_numbers[i-1], scan_numbers[i+1], scan_numbers[i+2]],gaussian_filter_sizes)
    for i in range(2,len(scan_numbers)-2):
        print('de-flickering ' + str(scan_numbers[i])) 
        deflicker_using_4_neighbors_time_series(scan_numbers[i],[scan_numbers[i-2], scan_numbers[i-1], scan_numbers[i+1], scan_numbers[i+2]],gaussian_filter_sizes)
    i=i+1
    print('de-flickering ' + str(scan_numbers[i])) 
    deflicker_using_4_neighbors_time_series(scan_numbers[i],[scan_numbers[i-2], scan_numbers[i-1], scan_numbers[i+1], scan_numbers[i+1]],gaussian_filter_sizes)
    i=i+1
    print('de-flickering ' + str(scan_numbers[i])) 
    deflicker_using_4_neighbors_time_series(scan_numbers[i],[scan_numbers[i-3], scan_numbers[i-2], scan_numbers[i-1], scan_numbers[i-1]],gaussian_filter_sizes)
    
    
    
    
# Run this function on the files output from save_aligned_h5_file     #All input length units are in microns.  All output units are micro-moles/cm2
def calculate_elemental_moles_per_cm2(filenames, carbon_thickness=175, total_thickness=250): #Run this on the files output from save_aligned_h5_file
    for i in range(0,len(filenames)):
        filename = filenames[i]
        print(filename)     
        if type(filename) != str:
            filename="%.4f" % filename
            filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
        h5object= h5py.File(data_directory+data_subdirectory+filename, 'r+')
        ims     = np.array(h5object['xray_images'])
        ims[ims<=0.0]=0.00001 #so that np.log doesn't create an error
        
        # X-ray absorption coefficients in units of cm2/micro-moles    SEE EXCEL FILE Xray_absorption_coefficients.xlsx in folder /Users/damon/Desktop/BACKED_UP/WorkFiles/ReportsPublicationsPatents/20200420_Synchrotron_CuBiBirnessite_Results
        a_6520_Mn = 3.07E-03;  a_6600_Mn = 2.44E-02; a_8970_Mn = 1.10E-02; a_9050_Mn = 1.07E-02; 
        a_6520_Cu = 5.83E-03;  a_6600_Cu = 5.66E-03; a_8970_Cu = 2.39E-03; a_9050_Cu = 1.80E-02;
        a_6520_Bi = 8.39E-02;  a_6600_Bi = 8.15E-02; a_8970_Bi = 3.65E-02; a_9050_Bi = 3.56E-02;
        a_6520_El = 3.26E-03;  a_6600_El = 3.14E-03; a_8970_El = 1.23E-03; a_9050_El = 1.20E-03; # <----- One part NaOH to 5 parts H2O (mole ratio) , 37% NaOH solution
        a_6520_C  = 1.03E-04;  a_6600_C  = 9.91E-05; a_8970_C  = 3.69E-05; a_9050_C  = 3.59E-05;
        
        sum_aMn_aEl = (a_6520_Mn*a_6520_El + a_6600_Mn*a_6600_El + a_8970_Mn*a_8970_El + a_9050_Mn*a_9050_El)
        sum_aCu_aEl = (a_6520_Cu*a_6520_El + a_6600_Cu*a_6600_El + a_8970_Cu*a_8970_El + a_9050_Cu*a_9050_El)
        sum_aBi_aEl = (a_6520_Bi*a_6520_El + a_6600_Bi*a_6600_El + a_8970_Bi*a_8970_El + a_9050_Bi*a_9050_El)
        sum_aMn_aMn = (a_6520_Mn*a_6520_Mn + a_6600_Mn*a_6600_Mn + a_8970_Mn*a_8970_Mn + a_9050_Mn*a_9050_Mn)
        sum_aCu_aMn = (a_6520_Cu*a_6520_Mn + a_6600_Cu*a_6600_Mn + a_8970_Cu*a_8970_Mn + a_9050_Cu*a_9050_Mn)
        sum_aBi_aMn = (a_6520_Bi*a_6520_Mn + a_6600_Bi*a_6600_Mn + a_8970_Bi*a_8970_Mn + a_9050_Bi*a_9050_Mn)
        sum_aBi_aCu = (a_6520_Cu*a_6520_Bi + a_6600_Cu*a_6600_Bi + a_8970_Cu*a_8970_Bi + a_9050_Cu*a_9050_Bi)
        sum_aCu_aCu = (a_6520_Cu*a_6520_Cu + a_6600_Cu*a_6600_Cu + a_8970_Cu*a_8970_Cu + a_9050_Cu*a_9050_Cu)
        sum_aBi_aBi = (a_6520_Bi*a_6520_Bi + a_6600_Bi*a_6600_Bi + a_8970_Bi*a_8970_Bi + a_9050_Bi*a_9050_Bi)
        sum_aMn_aC  = (a_6520_Mn*a_6520_C  + a_6600_Mn*a_6600_C  + a_8970_Mn*a_8970_C  + a_9050_Mn*a_9050_C )
        sum_aCu_aC  = (a_6520_Cu*a_6520_C  + a_6600_Cu*a_6600_C  + a_8970_Cu*a_8970_C  + a_9050_Cu*a_9050_C )
        sum_aBi_aC  = (a_6520_Bi*a_6520_C  + a_6600_Bi*a_6600_C  + a_8970_Bi*a_8970_C  + a_9050_Bi*a_9050_C )
        
        molar_density_Mn = 7.3/54.94  *1E6  # molar weight is 7.3    g/mole.    density is 54.94 g/cm3   end result has units micro-moles/cm3
        molar_density_Cu = 8.96/63.6  *1E6  # molar weight is 8.96   g/mole.    density is 63.55 g/cm3   end result has units micro-moles/cm3
        molar_density_Bi = 9.8/209.0  *1E6  # molar weight is 208.78 g/mole.    density is 9.78 g/cm3    end result has units micro-moles/cm3
        molar_density_C  = 2.0/12.01  *1E6  # molar weight is 12.01  g/mole.    density is 2.0 g/cm3     end result has units micro-moles/cm3
        molar_density_El = 1.35/130   *1E6  # m.w. NaH11O6 is 130    g/mole.    density is 1.35 g/cm3    end result has units micro-moles/cm3
        
        A = np.zeros((3,3))
        A[0,:] = [ molar_density_El/molar_density_Mn*sum_aMn_aEl - sum_aMn_aMn , molar_density_El/molar_density_Cu*sum_aMn_aEl - sum_aCu_aMn , molar_density_El/molar_density_Bi*sum_aMn_aEl - sum_aBi_aMn ]
        A[1,:] = [ molar_density_El/molar_density_Mn*sum_aCu_aEl - sum_aCu_aMn , molar_density_El/molar_density_Cu*sum_aCu_aEl - sum_aCu_aCu , molar_density_El/molar_density_Bi*sum_aCu_aEl - sum_aBi_aCu ]
        A[2,:] = [ molar_density_El/molar_density_Mn*sum_aBi_aEl - sum_aBi_aMn , molar_density_El/molar_density_Cu*sum_aBi_aEl - sum_aBi_aCu , molar_density_El/molar_density_Bi*sum_aBi_aEl - sum_aBi_aBi ]
        
        b = np.zeros((3,1))
        
        # Create some empty arrays for the micro-moles / cm2
        moles_Mn_per_cm2=np.ones(ims[0,:,:].shape,dtype=np.float32)
        moles_Cu_per_cm2=np.ones(ims[0,:,:].shape,dtype=np.float32)    
        moles_Bi_per_cm2=np.ones(ims[0,:,:].shape,dtype=np.float32)
        moles_El_per_cm2=np.ones(ims[0,:,:].shape,dtype=np.float32)
        moles_C_per_cm2 = carbon_thickness*1E-4*molar_density_C  # 1E4 to convert carbon_thickness from microns to cm      end result: micro-moles/cm2-surface-area
        
        # Solve the matrix equation A x = b where A is a 2D matrix of coefficients, x is a vertical array of [elemental massess],  b is a vertical array of constants
        for m in range(0,ims[0,:,:].shape[0]):
            if np.mod(m,10)==0: sys.stdout.write('\rLeast Squares, Row: '+str(m))
            sys.stdout.flush()
            for n in range(0,ims[0,:,:].shape[1]): 
                if np.where(ims[:,m,n]==0.12345678)[0].shape[0]==0:
                    ln_I_I0_6520 = np.log(ims[0,m,n]);  ln_I_I0_6600 = np.log(ims[1,m,n]); ln_I_I0_8970 = np.log(ims[2,m,n]); ln_I_I0_9050 = np.log(ims[3,m,n]);
                    
                    b[0] =   carbon_thickness*1E-4*(molar_density_C*sum_aMn_aC - molar_density_El*sum_aMn_aEl) + molar_density_El*total_thickness*1E-4*sum_aMn_aEl + (a_6520_Mn*ln_I_I0_6520 + a_6600_Mn*ln_I_I0_6600 + a_8970_Mn*ln_I_I0_8970 + a_9050_Mn*ln_I_I0_9050)
                    b[1] =   carbon_thickness*1E-4*(molar_density_C*sum_aCu_aC - molar_density_El*sum_aCu_aEl) + molar_density_El*total_thickness*1E-4*sum_aCu_aEl + (a_6520_Cu*ln_I_I0_6520 + a_6600_Cu*ln_I_I0_6600 + a_8970_Cu*ln_I_I0_8970 + a_9050_Cu*ln_I_I0_9050)
                    b[2] =   carbon_thickness*1E-4*(molar_density_C*sum_aBi_aC - molar_density_El*sum_aBi_aEl) + molar_density_El*total_thickness*1E-4*sum_aBi_aEl + (a_6520_Bi*ln_I_I0_6520 + a_6600_Bi*ln_I_I0_6600 + a_8970_Bi*ln_I_I0_8970 + a_9050_Bi*ln_I_I0_9050)                
    
                    # Solve the system of linear equations
                    temp = np.linalg.solve(A, b)
                    
                    #the calculation produces micro-moles/cm2 for each component!!!
                    moles_Mn_per_cm2[m,n] = np.float32(temp[0])
                    moles_Cu_per_cm2[m,n] = np.float32(temp[1])
                    moles_Bi_per_cm2[m,n] = np.float32(temp[2])
                    moles_El_per_cm2[m,n] = molar_density_El*( total_thickness*1E-4 - carbon_thickness*1E-4 - moles_Mn_per_cm2[m,n]/molar_density_Mn - moles_Cu_per_cm2[m,n]/molar_density_Cu - moles_Bi_per_cm2[m,n]/molar_density_Bi)  #this produces optical thickness in microns         
                else:
                    #sum_square_errors1 = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2[m,n],moles_Cu_per_cm2[m,n],moles_Bi_per_cm2[m,n],moles_C_per_cm2,moles_El_per_cm2[m,n])
                    moles_Mn_per_cm2[m,n] = 0.12345678;  
                    moles_Cu_per_cm2[m,n] = 0.12345678;  
                    moles_Bi_per_cm2[m,n] = 0.12345678;  
                    moles_El_per_cm2[m,n] = 0.12345678;  
                
        sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
                       
        #check if the elemental_moles datasets are ALREADY in the h5object, and save data
        h5object['moles_Mn_per_cm2'][:]=moles_Mn_per_cm2 if 'moles_Mn_per_cm2' in h5object.keys()  else h5object.create_dataset('moles_Mn_per_cm2', shape=(ims.shape[1],ims.shape[2]), dtype=np.float32, data=moles_Mn_per_cm2)
        h5object['moles_Cu_per_cm2'][:]=moles_Cu_per_cm2 if 'moles_Cu_per_cm2' in h5object.keys()  else h5object.create_dataset('moles_Cu_per_cm2', shape=(ims.shape[1],ims.shape[2]), dtype=np.float32, data=moles_Cu_per_cm2)
        h5object['moles_Bi_per_cm2'][:]=moles_Bi_per_cm2 if 'moles_Bi_per_cm2' in h5object.keys()  else h5object.create_dataset('moles_Bi_per_cm2', shape=(ims.shape[1],ims.shape[2]), dtype=np.float32, data=moles_Bi_per_cm2)
        h5object['moles_El_per_cm2'][:]=moles_El_per_cm2 if 'moles_El_per_cm2' in h5object.keys()  else h5object.create_dataset('moles_El_per_cm2', shape=(ims.shape[1],ims.shape[2]), dtype=np.float32, data=moles_El_per_cm2)
        h5object['moles_C_per_cm2' ][:]=moles_C_per_cm2  if 'moles_C_per_cm2'  in h5object.keys()  else h5object.create_dataset('moles_C_per_cm2' , shape=(1,), dtype=np.float32, data=moles_C_per_cm2)
        h5object['carbon_thickness'][:]=carbon_thickness if 'carbon_thickness' in h5object.keys()  else h5object.create_dataset('carbon_thickness', shape=(1,), dtype=np.float32, data=carbon_thickness)
        h5object['total_thickness' ][:]=total_thickness  if 'total_thickness'  in h5object.keys()  else h5object.create_dataset('total_thickness' , shape=(1,), dtype=np.float32, data=total_thickness)
        h5object.close() 
        
    
    


#  image_used_for_plot exmaples 'Mn_raw_im1' or 'moles_Bi_per_cm2' .      
def make_movie_with_potentiostat_data(txm_scan_numbers,biologic_file, image_used_for_plot, movie_time_span_seconds=27000, seconds_per_movie_frame=40, output_filename='test.mp4'):
        
    # Read timestamps of the images, The Biologic Computer time was 3 Minutes AHEAD of "real" time ( aka the xanes images times)
    datetime_array_xanes = np.array([])
    timestamp_array_xanes = np.array([])
    for i in txm_scan_numbers:
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
    im=get_processed_image(txm_scan_numbers[0],image_used_for_plot)
    pixels_per_inch = 300 # pixels per inch
    figure_width = im.shape[1]/pixels_per_inch * 3/2  # the 3/2 expands the width figure beyond the width of the xray_image so that blank space exists for the potentiostat data
    figure_height = (figure_width*2/3) * im.shape[0]/im.shape[1] # (figure_width*2/3) is the displayed-width of the xray_image
    fig_han = plt.figure(figsize=(figure_width, figure_height ))
    
    #debuffer = calculate_image_debuffer_multiple_files(txm_scan_numbers)
    #im=im[debuffer[2]:debuffer[3],debuffer[0]:debuffer[1]]
    im_4_show = 1.0*im
    # The new axis size:left, bottom,    width  ,  height
    im_axes = plt.axes([0.0,   0.0    ,   2/3   ,   1.0   ])
    im_axes.set_axis_off()
    # Adjust the Brightnewss & Contrast
    if image_used_for_plot == 'elemental_RGB':   # The caller can state 'elemental_RGB' as the image_used_for_plot
        zmin = np.ones(3)
        zmax = np.ones(3)
        zmin[0], zmax[0] = calculate_brightness_contrast(txm_scan_numbers, 'Mn', 0.005, 0.995)
        zmin[1], zmax[1] = calculate_brightness_contrast(txm_scan_numbers, 'Cu', 0.005, 0.995)
        zmin[2], zmax[2] = calculate_brightness_contrast(txm_scan_numbers, 'Bi', 0.005, 0.995)
    else:
        zmin, zmax = calculate_brightness_contrast(txm_scan_numbers, image_used_for_plot, 0.005, 0.995)
        zmin = [zmin]; zmax=[zmax]
    for j in range(im.shape[2]):
        temp = (im[:,:,j] - zmin[j])/zmax[j]; temp[temp<0.0]=0.0;  temp[temp>1.0]=1.0; 
        im_4_show[:,:,j] = temp    
    # Paint the scale bar onto the im_4_show image
    for j in range(im.shape[2]): im_4_show[-30:-8,48:175,j]=1.0
    im_axes.text(79,im.shape[0]-11,'5 um',fontsize=6.0)
    # Paint the colorbar onto the im_4_show image
    for j in range(im.shape[2]): 
        im_4_show[-45*(j+1)-1:-45*(j)-5,-203:-18,:]=1.0  #Paint a white background
        if image_used_for_plot == 'elemental_RGB': im_4_show[-45*(j+1)-5:-45*(j)-5,-203:-18,:]=1.0  #Paint a white background
        im_4_show[-45*(j)-43 :-45*(j)-23,-181:-39,:]=0.0  #Paint the background black behind the colorbar (important for RGB images).   THIS ALSO MAKES A BLACK BORDER.
        im_4_show[-45*(j)-42 :-45*(j)-24,-180:-40,j]=np.tile(np.arange(0,1.0,1.0/140),(18,1))  #Paint a colorbar    
        string_temp = "%.1f" % (zmin[j]); im_axes.text(im.shape[1]-194-int(len(string_temp)/2),im.shape[0]-45*j-7,string_temp,fontsize=4.5)  #print the smallest shown color of colorbar
        string_temp = "%.1f" % (zmax[j]); im_axes.text(im.shape[1]-57 -int(len(string_temp)/2),im.shape[0]-45*j-7,string_temp,fontsize=4.5)  #print the largest shown color of colorbar
    if im_4_show.shape[2] == 1: im_4_show = im_4_show[:,:,0]
    # Show the image
    im_show_handle = im_axes.imshow(im_4_show,cmap='gray',interpolation='none', vmin=0.0, vmax=1.0, label=False)
    # Make the Potentiostat Axis
    # The new axis size:                left                                           ,   bottom,                               width,                             , height
    iV_data_axes = plt.axes([im.shape[1]/im.shape[0]*figure_height/figure_width + 0.052,   0.085,  1.0 - im.shape[1]/im.shape[0]*figure_height/figure_width - 0.065 ,    0.9])
    iV_data_axes.tick_params(axis = 'both', which = 'major', labelsize = 6)    
    iV_data_axes.plot(biologic_data['Ewe/V'].values,biologic_data['<I>/mA'].values*1000)
    iV_data_axes.set_xlabel('Electrode Voltage (V)',fontsize=7,labelpad=3)
    y_range = iV_data_axes.get_ylim()[1] - iV_data_axes.get_ylim()[0]
    iV_data_axes.set_ylabel('Current (uA)', fontsize=7,labelpad=-6, position=(0,-iV_data_axes.get_ylim()[0]/y_range))
    scatter_han = iV_data_axes.scatter(biologic_data['Ewe/V'].values[0],biologic_data['<I>/mA'].values[0]*1000,c='r',s=20,zorder=1)
    # Make the Authorship label Axis
    authorship_label_axis = plt.axes([im.shape[1]/im.shape[0]*figure_height/figure_width - 0.095,   0.984,  0.05 ,    0.005],zorder=2)  # left, bottom, width, height
    authorship_label_axis.set_axis_off(); 
    authorship_label_axis.text(0,0.1,'                    ',size=7.5,bbox=dict(boxstyle='square,pad=0.0',ec='none',fc='w'))
    authorship_label_axis.text(0,-0.3,'D.E.Turney et al. 2020',alpha=0.65,size=5.5,bbox=dict(boxstyle='square,pad=0.0',ec='none',fc='w'))
    # Show the image number    
    im_id_text = im_axes.text(im.shape[1],im.shape[0]-5,'img: ' + str(txm_scan_numbers[0]) ,fontsize=6.0,alpha=0.0)           
    
    # I tried for months to use matplotlib.animation but it became obvious that high-quality videos require raw linux-command-line use of ffmpeg, so now I'm saving the images to a "video_images" temp folder and running ffmpeg on the command-line
    closest_index_txm_previous = -1
    os.system('rm video_images/*')
    for frame_num in range(int(movie_time_span_seconds/seconds_per_movie_frame)):
        frame_time = biologic_start_time + datetime.timedelta(seconds = frame_num*seconds_per_movie_frame) - datetime.timedelta(minutes = 3) #The Biologic Computer time was 3 Minutes AHEAD of "real" time ( aka the xanes images times)   
        closest_index_biologic=abs(biologic_data['time/s'].values - frame_num*seconds_per_movie_frame).argmin() #One frame per 20 seconds
        scatter_han.set_offsets([biologic_data['Ewe/V'].values[closest_index_biologic],biologic_data['<I>/mA'].values[closest_index_biologic]*1000])
        closest_index_txm=abs(timestamp_array_xanes - np.float64(frame_time.timestamp())).argmin()
        if closest_index_txm != closest_index_txm_previous:
            closest_index_txm_previous=closest_index_txm
            im=get_processed_image(txm_scan_numbers[closest_index_txm],image_used_for_plot)
            im_4_show = 1.0*im
            # Adjust the Brightnewss & Contraast
            for j in range(im.shape[2]):
                temp = (im[:,:,j] - zmin[j])/zmax[j]; temp[temp<0.0]=0.0;  temp[temp>1.0]=1.0; 
                im_4_show[:,:,j] = temp 
            # Paint the scale bar onto the im_4_show image
            for j in range(im.shape[2]): im_4_show[-30:-8,48:175,j]=1.0
            im_axes.text(79,im.shape[0]-11,'5 um',fontsize=6.0)
            # Paint the colorbar onto the im_4_show image
            for j in range(im.shape[2]): 
                im_4_show[-45*(j+1)-1:-45*(j)-5,-203:-18,:]=1.0  #Paint a white background
                if image_used_for_plot == 'elemental_RGB': im_4_show[-45*(j+1)-5:-45*(j)-5,-203:-18,:]=1.0  #Paint a white background
                im_4_show[-45*(j)-43 :-45*(j)-23,-181:-39,:]=0.0  #Paint the background black behind the colorbar (important for RGB images).   THIS ALSO MAKES A BLACK BORDER.
                im_4_show[-45*(j)-42 :-45*(j)-24,-180:-40,j]=np.tile(np.arange(0,1.0,1.0/140),(18,1))  #Paint a colorbar    
                string_temp = "%.1f" % (zmin[j]); im_axes.text(im.shape[1]-194-int(len(string_temp)/2),im.shape[0]-45*j-7,string_temp,fontsize=4.5)  #print the smallest shown color of colorbar
                string_temp = "%.1f" % (zmax[j]); im_axes.text(im.shape[1]-57 -int(len(string_temp)/2),im.shape[0]-45*j-7,string_temp,fontsize=4.5)  #print the largest shown color of colorbar
            # Show the image
            if im_4_show.shape[2] == 1: im_4_show = im_4_show[:,:,0]
            im_show_handle.set_data(im_4_show)
            # Print the image number next to the image
            im_id_text.set_text('scan: ' + str(txm_scan_numbers[closest_index_txm]))           
            print('displaying scan: ' + str(txm_scan_numbers[closest_index_txm]) + ' for time ' + str(frame_num*seconds_per_movie_frame) + ' seconds (' + datetime.datetime.strftime(frame_time, '%Y-%m-%d %H:%M:%S' ) + ')')
        #Save the image to video_images
        plt.savefig('video_images/'+("%04d" % frame_num)+'.png', dpi=pixels_per_inch)
    
    # ffmpeg -i %04d.png -vcodec libx265 -x265-params "lossless=1" -preset slow -vf format=gray,format=yuv420p testttt.mp4    
    os.system('ffmpeg -i video_images/%04d.png -vcodec libx265 -x265-params "lossless=1" -preset slow -vf format=gray,format=yuv420p '+ output_filename)




# scan_numbers format: 34565.0000 means the first beam energy, location 0         34565.0001 means the first beam energy, location 1           34565.0101 means the second beam energy, location 1
# files = np.concatenate((range(34565,34725,2),range(34726,34875,2))) ; files2=np.ones(len(files)*4)
#for i in range(len(files)):
#    files2[4*i] = files[i]
#    files2[4*i+1] = files[i]+0.0001
#    files2[4*i+2] = files[i]+0.00001
#    files2[4*i+3] = files[i]+0.00011
def make_movie_with_image_statistics(scan_numbers, image_type_2_show, movie_filename ):
    # Get the first image
    if image_type_2_show == 'img_bkg1':
        im=get_raw_image(scan_numbers[0],'img_bkg')
        im=im[0,:,:]
    if image_type_2_show == 'img_bkg2':
        im=get_raw_image(scan_numbers[0],'img_bkg')
        im=im[1,:,:]
    if image_type_2_show == 'img_bkg':
        im=get_raw_image(scan_numbers[0],'img_bkg')
        if ("%.5f" % scan_numbers[0])[10] == '0': im=im[0,:,:]
        if ("%.5f" % scan_numbers[0])[10] == '1': im=im[1,:,:]
    if image_type_2_show == 'img_dark':
        im=get_raw_image(scan_numbers[0],'img_dark')
        im=im[0,:,:]
    if image_type_2_show != 'img_bkg1' and image_type_2_show != 'img_bkg2' and image_type_2_show != 'img_bkg' and image_type_2_show != 'img_dark':
        im=get_processed_image(scan_numbers[0],image_type_2_show)
        
    # Create the first plot (figure layout, axes, et cetera)
    figure_width = 10    #figsize=(       height             ,   width     )
    figure_height = 1080/1280*figure_width/1.5 
    fig_han = plt.figure(figsize=(figure_width, figure_height ))
    # The new axis size:left, bottom,         width                                    ,   height
    im_axes = plt.axes([0.0,   0.0  , (im.shape[1]-1)/im.shape[0]*figure_height/figure_width,   1.0   ])
    im_axes.set_axis_off()
    zmin, zmax = calculate_brightness_contrast(scan_numbers, image_type_2_show, 0.005, 0.995)
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
    images_mean, images_std = get_images_statistics(scan_numbers, image_type_2_show )
    im_statistics.plot(scan_numbers,images_mean,zorder=1)
    im_statistics.set_xlabel('Image Number',fontsize=9,labelpad=1)
    im_statistics2 = im_statistics.twinx()
    im_statistics2.plot(scan_numbers,images_std,'g')
    im_statistics2.tick_params(axis = 'both', which = 'major', direction='in',labelsize = 7)    
    scatter_han = im_statistics.scatter(scan_numbers[0],images_mean[0],c='r',s=20,zorder=2)
    # Show the image number    
    im_id_text = im_axes.text(im.shape[1]+60,im.shape[0]-5,'img: ' + str(scan_numbers[0]) ,fontsize=7.8)           
    
       
    def change_imshow(frame_num):
        print('displaying img: ' + str(scan_numbers[frame_num]))
        # Grab the new image  
        if image_type_2_show == 'img_bkg1':
            im=get_raw_image(scan_numbers[frame_num],'img_bkg')
            im=im[0,:,:]
        if image_type_2_show == 'img_bkg2':
            im=get_raw_image(scan_numbers[frame_num],'img_bkg')
            im=im[1,:,:]
        if image_type_2_show == 'img_bkg':
            im=get_raw_image(scan_numbers[frame_num],'img_bkg')
            if ("%.5f" % scan_numbers[frame_num])[10] == '0': im=im[0,:,:]
            if ("%.5f" % scan_numbers[frame_num])[10] == '1': im=im[1,:,:]
        if image_type_2_show == 'img_dark':
            im=get_raw_image(scan_numbers[frame_num],'img_dark')
            im=im[0,:,:]
        if image_type_2_show != 'img_bkg1' and image_type_2_show != 'img_bkg2' and image_type_2_show != 'img_bkg' and image_type_2_show != 'img_dark':
            im=get_processed_image(scan_numbers[frame_num],image_type_2_show)
        
        scatter_han.set_offsets((scan_numbers[frame_num],images_mean[frame_num]))
        # Make the colorbar
        im[-45:-5,-195:]=1E20
        im[-45:-27,-180:-40]=np.tile(np.arange(zmin,zmax,(zmax - zmin)/140),(18,1))
        im[-45:-27,-180] = 0.0; im[-45:-27,-40] = 0.0; im[-45,-180:-40] = 0.0; im[-27,-180:-40] = 0.0;     
        im[-45:-27,-181] = 0.0; im[-45:-27,-39] = 0.0; im[-46,-180:-40] = 0.0; im[-26,-180:-40] = 0.0;     
        im_axes.imshow(im, vmin=zmin, vmax=zmax, interpolation='none', cmap='gray')
        im_axes.text(im.shape[1]-195,im.shape[0]-7,"%.1f" % zmin + '                ' + "%.1f" % zmax,fontsize=7.5)
        # Show the image number
        im_id_text.set_text('img: ' + str(scan_numbers[frame_num]) + ', and index:' + str(frame_num))     
        plt.draw()
        plt.show()
        time.sleep(1)

        
    
    # It iterates through e.g. "frames=range(15)" calling the function e.g "change_imshow" , and inserts a millisecond time delay between frames of e.g. "interval=100".
    animation_handle=animation.FuncAnimation(fig_han, change_imshow, frames=range(len(scan_numbers)), blit=False, interval=100, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15)
    animation_handle.save(movie_filename, writer=writer)
    plt.close()



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
            image = get_processed_image(filenumbers[i],image_2_display)[:,:,0]
        image[image==0.12345678] = np.median(image)
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
    # Grab the calculated thicknesses in microns
    moles_Mn_per_cm2= np.array(h5object['moles_Mn_per_cm2'])
    moles_Cu_per_cm2= np.array(h5object['moles_Cu_per_cm2'])
    moles_Bi_per_cm2= np.array(h5object['moles_Bi_per_cm2'])
    moles_C_per_cm2 = np.array(h5object['moles_C_per_cm2'])
    moles_El_per_cm2= np.array(h5object['moles_El_per_cm2'])
    
    # X-ray absorption coefficients in units of cm2/micro-moles    SEE EXCEL FILE Xray_absorption_coefficients.xlsx in folder /Users/damon/Desktop/BACKED_UP/WorkFiles/ReportsPublicationsPatents/20200420_Synchrotron_CuBiBirnessite_Results
    a_6520_Mn = 3.07E-03;  a_6600_Mn = 2.44E-02; a_8970_Mn = 1.10E-02; a_9050_Mn = 1.07E-02; 
    a_6520_Cu = 5.83E-03;  a_6600_Cu = 5.66E-03; a_8970_Cu = 2.39E-03; a_9050_Cu = 1.80E-02;
    a_6520_Bi = 8.39E-02;  a_6600_Bi = 8.15E-02; a_8970_Bi = 3.65E-02; a_9050_Bi = 3.56E-02;
    a_6520_El = 3.26E-03;  a_6600_El = 3.14E-03; a_8970_El = 1.23E-03; a_9050_El = 1.20E-03; # <----- One part NaOH to 5 parts H2O (mole ratio) , 37% NaOH solution
    a_6520_C  = 1.03E-04;  a_6600_C  = 9.91E-05; a_8970_C  = 3.69E-05; a_9050_C  = 3.59E-05;
    
    #Plot measured data as an image
    plt.scatter(beam_energies, xanes_ims[:,row,column],s=50*np.ones(4),marker='x')
    
    #Plot least-squares model results
    plt.scatter(beam_energies, [np.exp(-a_6520_Mn*moles_Mn_per_cm2[row,column]), np.exp(-a_6600_Mn*moles_Mn_per_cm2[row,column]), np.exp(-a_8970_Mn*moles_Mn_per_cm2[row,column]), np.exp(-a_9050_Mn*moles_Mn_per_cm2[row,column])],marker='o',facecolors='none',edgecolors='r')
    plt.scatter(beam_energies, [np.exp(-a_6520_Cu*moles_Cu_per_cm2[row,column]), np.exp(-a_6600_Cu*moles_Cu_per_cm2[row,column]), np.exp(-a_8970_Cu*moles_Cu_per_cm2[row,column]), np.exp(-a_9050_Cu*moles_Cu_per_cm2[row,column])],marker='o',facecolors='none',edgecolors='g')
    plt.scatter(beam_energies, [np.exp(-a_6520_Bi*moles_Bi_per_cm2[row,column]), np.exp(-a_6600_Bi*moles_Bi_per_cm2[row,column]), np.exp(-a_8970_Bi*moles_Bi_per_cm2[row,column]), np.exp(-a_9050_Bi*moles_Bi_per_cm2[row,column])],marker='o',facecolors='none',edgecolors='b')
    plt.scatter(beam_energies, [np.exp(-a_6520_C *moles_C_per_cm2)             , np.exp(-a_6600_C *moles_C_per_cm2)             , np.exp(-a_8970_C *moles_C_per_cm2)             , np.exp(-a_9050_C*moles_C_per_cm2)              ],marker='o',facecolors='none',edgecolors='tab:pink')
    plt.scatter(beam_energies, [np.exp(-a_6520_El*moles_El_per_cm2[row,column]), np.exp(-a_6600_El*moles_El_per_cm2[row,column]), np.exp(-a_8970_El*moles_El_per_cm2[row,column]), np.exp(-a_9050_El*moles_El_per_cm2[row,column])],marker='o',facecolors='none',edgecolors='tab:brown')
    model_I_I0_6520=np.exp(-a_6520_Mn*moles_Mn_per_cm2[row,column] - a_6520_Cu*moles_Cu_per_cm2[row,column] - a_6520_Bi*moles_Bi_per_cm2[row,column] - a_6520_C*moles_C_per_cm2 - a_6520_El*moles_El_per_cm2[row,column])
    model_I_I0_6600=np.exp(-a_6600_Mn*moles_Mn_per_cm2[row,column] - a_6600_Cu*moles_Cu_per_cm2[row,column] - a_6600_Bi*moles_Bi_per_cm2[row,column] - a_6600_C*moles_C_per_cm2 - a_6600_El*moles_El_per_cm2[row,column])
    model_I_I0_8970=np.exp(-a_8970_Mn*moles_Mn_per_cm2[row,column] - a_8970_Cu*moles_Cu_per_cm2[row,column] - a_8970_Bi*moles_Bi_per_cm2[row,column] - a_8970_C*moles_C_per_cm2 - a_8970_El*moles_El_per_cm2[row,column])
    model_I_I0_9050=np.exp(-a_9050_Mn*moles_Mn_per_cm2[row,column] - a_9050_Cu*moles_Cu_per_cm2[row,column] - a_9050_Bi*moles_Bi_per_cm2[row,column] - a_9050_C*moles_C_per_cm2 - a_9050_El*moles_El_per_cm2[row,column])
    plt.scatter(beam_energies, [model_I_I0_6520 , model_I_I0_6600 , model_I_I0_8970 , model_I_I0_9050 ] ,marker='o',facecolors='none',edgecolors='tab:orange')
    plt.legend(['measured','model: Mn','model: Cu','model: Bi','model: C','model: El','model: Mn,Bi,Cu'])
    plt.ylim((0,1.0))

    #plt.figure()
    #plt.scatter(['Mn', 'Cu', 'Bi', 'C', 'El' ], [moles_Mn_per_cm2[row,column], moles_Cu_per_cm2[row,column], moles_Bi_per_cm2[row,column], moles_C_per_cm2, moles_El_per_cm2[row,column]])
    print('Elemental thicknesses in microns.')
    print('Mn: ' + str(moles_Mn_per_cm2[row,column]))
    print('Cu: ' + str(moles_Cu_per_cm2[row,column]))
    print('Bi: ' + str(moles_Bi_per_cm2[row,column]))
    print('C: ' +  str(moles_C_per_cm2))
    print('El: ' + str(moles_El_per_cm2[row,column]))
    print('total: '+ str(moles_Mn_per_cm2[row,column] + moles_Cu_per_cm2[row,column] + moles_Bi_per_cm2[row,column] + moles_C_per_cm2 + moles_El_per_cm2[row,column]))

    
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
    notes       = str(np.array(h5object['note']))
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
    notes       = str(np.array(h5object['note']))
    data_processing_note1 = str(np.array(h5object['data_processing_note1'])) if 'data_processing_note1' in h5object.keys() else '' 
    data_processing_note2 = str(np.array(h5object['data_processing_note2'])) if 'data_processing_note2' in h5object.keys() else '' 
    translations= np.array(h5object['translations'])
    scan_start_time = datetime.datetime.fromtimestamp(scan_time)
    scan_start_time_string = datetime.datetime.strftime(scan_start_time, '%Y-%m-%d %H:%M:%S' )    
    h5object.close() 
    return(scan_start_time_string, scan_time, beam_energy, scan_id, notes, translations, data_processing_note1, data_processing_note2)





def get_raw_image(filename,which_image):
    if type(filename) != str:
        filename="%.4f" % filename
        filename='multipos_2D_xanes_scan2_id_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object    = h5py.File(data_directory+data_subdirectory+filename, 'r')
    images      = np.array(h5object[which_image])  
    h5object.close()
    return(images)


#mask = np.where(np.repeat(np.expand_dims(elemental_thicknesses_target_image[:,:,0],axis=0),4,axis=0)==0.12345678) #Convert the mask for where 0.12345678 is from the [1080,1280,3] shape of elemental_rgb array to the [4,1080,1280] shape of the xray_images array

                                              #remove_elements can be 'none' or 'all' or 'Mn' or 'Cu' or 'Bi' or 'BiMn' or 'CuBi' etc...
def get_processed_image(filename, which_image, remove_elements='none'):
    if type(filename) != str:
        filename="%.4f" % filename
        filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object= h5py.File(data_directory+data_subdirectory+filename, 'r')   
    
    if which_image == 'all':
        xray_images = np.array(h5object['xray_images'])
        if remove_elements != 'none': 
            xray_images = remove_elements_from_TXM_image(xray_images[0,:,:], '6520', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
            xray_images = remove_elements_from_TXM_image(xray_images[1,:,:], '6600', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
            xray_images = remove_elements_from_TXM_image(xray_images[2,:,:], '8970', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
            xray_images = remove_elements_from_TXM_image(xray_images[3,:,:], '9050', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
        moles_Mn_per_cm2 = np.array(h5object['moles_Mn_per_cm2'])
        moles_Cu_per_cm2 = np.array(h5object['moles_Cu_per_cm2'])
        moles_Bi_per_cm2 = np.array(h5object['moles_Bi_per_cm2'])
        moles_C_per_cm2  = np.array(h5object['moles_C_per_cm2'])
        moles_El_per_cm2 = np.array(h5object['moles_El_per_cm2'])
        h5object.close()
        return(np.stack((xray_images,moles_Mn_per_cm2,moles_Cu_per_cm2,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm2)))
    
    if which_image == 'xray_images':
        xray_images = np.array(h5object['xray_images'])
        if remove_elements != 'none': 
            xray_images[0,:,:] = remove_elements_from_TXM_image(xray_images[0,:,:], '6520', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
            xray_images[1,:,:] = remove_elements_from_TXM_image(xray_images[1,:,:], '6600', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
            xray_images[2,:,:] = remove_elements_from_TXM_image(xray_images[2,:,:], '8970', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
            xray_images[3,:,:] = remove_elements_from_TXM_image(xray_images[3,:,:], '9050', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
        h5object.close()
        return(xray_images)

    if which_image == '6520' or which_image == 'Mn_raw_im1':
        xray_images = np.array(h5object['xray_images'])
        if remove_elements != 'none':
            xray_images[0,:,:] = remove_elements_from_TXM_image(xray_images[0,:,:], '6520', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
        returned_image = np.ones((xray_images.shape[1],xray_images.shape[2],1),dtype=xray_images.dtype)
        returned_image[:,:,0] = xray_images[0,:,:]
        h5object.close()
        return(returned_image)

    if which_image == '6600' or which_image == 'Mn_raw_im2':
        xray_images = np.array(h5object['xray_images'])
        if remove_elements != 'none':
            xray_images[1,:,:] = remove_elements_from_TXM_image(xray_images[1,:,:], '6600', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
        returned_image = np.ones((xray_images.shape[1],xray_images.shape[2],1),dtype=xray_images.dtype)
        returned_image[:,:,0] = xray_images[1,:,:]
        h5object.close()
        return(returned_image)
    
    if which_image == '8970' or which_image == 'Cu_raw_im1':
        xray_images = np.array(h5object['xray_images'])
        if remove_elements != 'none':
            xray_images[2,:,:] = remove_elements_from_TXM_image(xray_images[2,:,:], '8970', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
        returned_image = np.ones((xray_images.shape[1],xray_images.shape[2],1),dtype=xray_images.dtype)
        returned_image[:,:,0] = xray_images[2,:,:]
        h5object.close()
        return(returned_image)
    
    if which_image == '9050' or which_image == 'Cu_raw_im2':
        xray_images = np.array(h5object['xray_images'])
        if remove_elements != 'none':
            xray_images[3,:,:] = remove_elements_from_TXM_image(xray_images[3,:,:], '9050', np.array(h5object['moles_Mn_per_cm2']), np.array(h5object['moles_Cu_per_cm2']), np.array(h5object['moles_Bi_per_cm2']), remove_elements)
        returned_image = np.ones((xray_images.shape[1],xray_images.shape[2],1),dtype=xray_images.dtype)
        returned_image[:,:,0] = xray_images[3,:,:]
        h5object.close()
        return(returned_image)
    
    if which_image == 'all_thickness':
        temp = np.stack((np.array(h5object['moles_Mn_per_cm2']),np.array(h5object['moles_Cu_per_cm2']),np.array(h5object['moles_Bi_per_cm2']),np.array(h5object['moles_C_per_cm2']),np.array(h5object['moles_El_per_cm2'])))
        h5object.close()
        return(temp)
        
    if which_image == 'Mn_thickness' or which_image == 'Mn' or which_image == 'moles_Mn_per_cm2' or which_image == 'thickness_Mn':
        im = np.array(h5object['moles_Mn_per_cm2'])
        h5object.close()
        returned_image = np.ones((im.shape[0],im.shape[1],1),dtype=im.dtype)
        returned_image[:,:,0] = im[:,:]
        return(returned_image)
        
    if which_image == 'Cu_thickness' or which_image == 'Cu' or which_image == 'moles_Cu_per_cm2' or which_image == 'thickness_Cu':
        im = np.array(h5object['moles_Cu_per_cm2'])
        h5object.close()
        returned_image = np.ones((im.shape[0],im.shape[1],1),dtype=im.dtype)
        returned_image[:,:,0] = im[:,:]
        return(returned_image)
        
    if which_image == 'Bi_thickness' or which_image == 'Bi' or which_image == 'moles_Bi_per_cm2' or which_image == 'thickness_Bi':
        im = np.array(h5object['moles_Bi_per_cm2'])
        h5object.close()
        returned_image = np.ones((im.shape[0],im.shape[1],1),dtype=im.dtype)
        returned_image[:,:,0] = im[:,:]
        return(returned_image)
        
    if which_image == 'C_thickness'  or which_image == 'C' or which_image == 'moles_C_per_cm2' or which_image == 'thickness_C':
        im = np.array(h5object['moles_C_per_cm2'])
        h5object.close()
        returned_image = np.ones((im.shape[0],im.shape[1],1),dtype=im.dtype)
        returned_image[:,:,0] = im[:,:]
        return(returned_image)
        
    if which_image == 'El_thickness' or which_image == 'El' or which_image == 'moles_El_per_cm2' or which_image == 'thickness_El':
        im = np.array(h5object['moles_El_per_cm2'])
        h5object.close()
        returned_image = np.ones((im.shape[0],im.shape[1],1),dtype=im.dtype)
        returned_image[:,:,0] = im[:,:]
        return(returned_image)
        
    if which_image == 'elemental_RGB' or which_image == 'elemental_rgb':
        im = np.array(h5object['moles_Mn_per_cm2'])
        returned_image = np.ones((im.shape[0],im.shape[1],3),dtype=im.dtype)
        returned_image[:,:,0] = im[:,:]
        im = np.array(h5object['moles_Cu_per_cm2'])
        returned_image[:,:,1] = im[:,:]
        im = np.array(h5object['moles_Bi_per_cm2'])
        returned_image[:,:,2] = im[:,:]
        h5object.close()
        return(returned_image)
        
        
    
    
    
        
def remove_elements_from_TXM_image(im, beam_energy, thicknesses_Mn, thicknesses_Cu, thicknesses_Bi, remove_elements):        

    # X-ray absorption coefficients in units of cm2/micro-moles    SEE EXCEL FILE Xray_absorption_coefficients.xlsx in folder /Users/damon/Desktop/BACKED_UP/WorkFiles/ReportsPublicationsPatents/20200420_Synchrotron_CuBiBirnessite_Results
    a_6520_Mn = 3.07E-03;  a_6600_Mn = 2.44E-02; a_8970_Mn = 1.10E-02; a_9050_Mn = 1.07E-02; 
    a_6520_Cu = 5.83E-03;  a_6600_Cu = 5.66E-03; a_8970_Cu = 2.39E-03; a_9050_Cu = 1.80E-02;
    a_6520_Bi = 8.39E-02;  a_6600_Bi = 8.15E-02; a_8970_Bi = 3.65E-02; a_9050_Bi = 3.56E-02;
    a_6520_El = 3.26E-03;  a_6600_El = 3.14E-03; a_8970_El = 1.23E-03; a_9050_El = 1.20E-03; # <----- One part NaOH to 5 parts H2O (mole ratio) , 37% NaOH solution
    a_6520_C  = 1.03E-04;  a_6600_C  = 9.91E-05; a_8970_C  = 3.69E-05; a_9050_C  = 3.59E-05;
        
    mask = np.where(thicknesses_Mn==0.12345678)
    
    if  'Mn' in remove_elements or remove_elements == 'all':
        if beam_energy=='6520': im = im/np.exp(-thicknesses_Mn*a_6520_Mn)
        if beam_energy=='6600': im = im/np.exp(-thicknesses_Mn*a_6600_Mn)
        if beam_energy=='8970': im = im/np.exp(-thicknesses_Mn*a_8970_Mn)
        if beam_energy=='9050': im = im/np.exp(-thicknesses_Mn*a_9050_Mn)
    if 'Cu' in remove_elements or remove_elements == 'all':
        if beam_energy=='6520': im = im/np.exp(-thicknesses_Mn*a_6520_Cu)
        if beam_energy=='6600': im = im/np.exp(-thicknesses_Mn*a_6600_Cu)
        if beam_energy=='8970': im = im/np.exp(-thicknesses_Mn*a_8970_Cu)
        if beam_energy=='9050': im = im/np.exp(-thicknesses_Mn*a_9050_Cu)
    if 'Bi' in remove_elements or remove_elements == 'all':
        if beam_energy=='6520': im = im/np.exp(-thicknesses_Bi*a_6520_Bi)
        if beam_energy=='6600': im = im/np.exp(-thicknesses_Bi*a_6600_Bi)
        if beam_energy=='8970': im = im/np.exp(-thicknesses_Bi*a_8970_Bi)
        if beam_energy=='9050': im = im/np.exp(-thicknesses_Bi*a_9050_Bi)        

    im[mask] = 0.12345678
    return(im)
   

        
def insert_elements_into_TXM_image(im, beam_energy, thicknesses_Mn, thicknesses_Cu, thicknesses_Bi, which_elements):        

    # X-ray absorption coefficients in units of cm2/micro-moles    SEE EXCEL FILE Xray_absorption_coefficients.xlsx in folder /Users/damon/Desktop/BACKED_UP/WorkFiles/ReportsPublicationsPatents/20200420_Synchrotron_CuBiBirnessite_Results
    a_6520_Mn = 3.07E-03;  a_6600_Mn = 2.44E-02; a_8970_Mn = 1.10E-02; a_9050_Mn = 1.07E-02; 
    a_6520_Cu = 5.83E-03;  a_6600_Cu = 5.66E-03; a_8970_Cu = 2.39E-03; a_9050_Cu = 1.80E-02;
    a_6520_Bi = 8.39E-02;  a_6600_Bi = 8.15E-02; a_8970_Bi = 3.65E-02; a_9050_Bi = 3.56E-02;
    a_6520_El = 3.26E-03;  a_6600_El = 3.14E-03; a_8970_El = 1.23E-03; a_9050_El = 1.20E-03; # <----- One part NaOH to 5 parts H2O (mole ratio) , 37% NaOH solution
    a_6520_C  = 1.03E-04;  a_6600_C  = 9.91E-05; a_8970_C  = 3.69E-05; a_9050_C  = 3.59E-05;
        
    mask = np.where(thicknesses_Mn==0.12345678)
    
    if  'Mn' in which_elements or which_elements == 'all':
        if beam_energy=='6520': im = im*np.exp(-thicknesses_Mn*a_6520_Mn)
        if beam_energy=='6600': im = im*np.exp(-thicknesses_Mn*a_6600_Mn)
        if beam_energy=='8970': im = im*np.exp(-thicknesses_Mn*a_8970_Mn)
        if beam_energy=='9050': im = im*np.exp(-thicknesses_Mn*a_9050_Mn)
    if 'Cu' in which_elements or which_elements == 'all':
        if beam_energy=='6520': im = im*np.exp(-thicknesses_Mn*a_6520_Cu)
        if beam_energy=='6600': im = im*np.exp(-thicknesses_Mn*a_6600_Cu)
        if beam_energy=='8970': im = im*np.exp(-thicknesses_Mn*a_8970_Cu)
        if beam_energy=='9050': im = im*np.exp(-thicknesses_Mn*a_9050_Cu)
    if 'Bi' in which_elements or which_elements == 'all':
        if beam_energy=='6520': im = im*np.exp(-thicknesses_Bi*a_6520_Bi)
        if beam_energy=='6600': im = im*np.exp(-thicknesses_Bi*a_6600_Bi)
        if beam_energy=='8970': im = im*np.exp(-thicknesses_Bi*a_8970_Bi)
        if beam_energy=='9050': im = im*np.exp(-thicknesses_Bi*a_9050_Bi)        

    im[mask] = 0.12345678
    return(im)




    

def dS_dthickness_all(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm2,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm2):
    test_sum_squares = np.zeros(8)
    
    #Calculate the test case sum of squared errors
    moles_Mn_per_cm21 = moles_Mn_per_cm2 + 0.00001
    test_sum_squares[0] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm21,moles_Cu_per_cm2,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm2)
    moles_Mn_per_cm21 = moles_Mn_per_cm2 - 0.00001
    test_sum_squares[1] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm21,moles_Cu_per_cm2,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm2)
    
    moles_Cu_per_cm21 = moles_Cu_per_cm2 + 0.00001
    test_sum_squares[2] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm21,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm2)
    moles_Cu_per_cm21 = moles_Cu_per_cm2 - 0.00001
    test_sum_squares[3] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm21,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm2)

    moles_Bi_per_cm21 = moles_Bi_per_cm2 + 0.00001
    test_sum_squares[4] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm2,moles_Bi_per_cm21,moles_C_per_cm2,moles_El_per_cm2)
    moles_Bi_per_cm21 = moles_Bi_per_cm2 - 0.00001
    test_sum_squares[5] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm2,moles_Bi_per_cm21,moles_C_per_cm2,moles_El_per_cm2)

    moles_El_per_cm21 = moles_El_per_cm2 + 0.00001
    test_sum_squares[6] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm2,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm21)
    moles_El_per_cm21 = moles_El_per_cm2 - 0.00001
    test_sum_squares[7] = calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm2,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm21)

    return(test_sum_squares)
    
    
    
    
def calculate_sum_square_errors(ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,moles_Mn_per_cm2,moles_Cu_per_cm2,moles_Bi_per_cm2,moles_C_per_cm2,moles_El_per_cm2):
    #Calculate the baseline sum of squared errors
    sum_squares_6520=(ln_I_I0_6520 + a_6520_Mn*moles_Mn_per_cm2 + a_6520_Cu*moles_Cu_per_cm2 + a_6520_Bi*moles_Bi_per_cm2 + a_6520_C*moles_C_per_cm2 + a_6520_El*moles_El_per_cm2)**2
    sum_squares_6600=(ln_I_I0_6600 + a_6600_Mn*moles_Mn_per_cm2 + a_6600_Cu*moles_Cu_per_cm2 + a_6600_Bi*moles_Bi_per_cm2 + a_6600_C*moles_C_per_cm2 + a_6600_El*moles_El_per_cm2)**2
    sum_squares_8970=(ln_I_I0_8970 + a_8970_Mn*moles_Mn_per_cm2 + a_8970_Cu*moles_Cu_per_cm2 + a_8970_Bi*moles_Bi_per_cm2 + a_8970_C*moles_C_per_cm2 + a_8970_El*moles_El_per_cm2)**2
    sum_squares_9050=(ln_I_I0_9050 + a_9050_Mn*moles_Mn_per_cm2 + a_9050_Cu*moles_Cu_per_cm2 + a_9050_Bi*moles_Bi_per_cm2 + a_9050_C*moles_C_per_cm2 + a_9050_El*moles_El_per_cm2)**2
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
    im_new = np.ones(im_old.shape)*0.12345678
    
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
    




def debuffer_multiple_image_files(scan_numbers):
    debuffer = calculate_image_debuffer_multiple_files(scan_numbers)
    for i in range(0,len(scan_numbers)):
        filename ="%.4f" % scan_numbers[i]
        filename ='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
        print('debuffering '+filename)
        h5object_old = h5py.File(data_directory+data_subdirectory+filename, 'r')  
        h5object_new = h5py.File(data_directory+data_subdirectory+'temp.h5', 'w')  
        h5object_old.copy('beam_energies', h5object_new)
        h5object_old.copy('scan_id',       h5object_new)
        h5object_old.copy('scan_time',     h5object_new)
        h5object_old.copy('note',          h5object_new)
        h5object_old.copy('translations',  h5object_new)
        if 'moles_Mn_per_cm2' in h5object_old.keys(): h5object_old.copy('moles_Mn_per_cm2', h5object_new)
        if 'moles_Cu_per_cm2' in h5object_old.keys(): h5object_old.copy('moles_Cu_per_cm2', h5object_new)
        if 'moles_Bi_per_cm2' in h5object_old.keys(): h5object_old.copy('moles_Bi_per_cm2', h5object_new)
        if 'moles_C_per_cm2' in h5object_old.keys():  h5object_old.copy('moles_C_per_cm2', h5object_new)
        if 'moles_El_per_cm2' in h5object_old.keys(): h5object_old.copy('moles_El_per_cm2', h5object_new)
        xanes_ims = np.array(h5object_old['xray_images'])
        h5object_old.close()
        xanes_ims2 = np.zeros((4,      -debuffer[2]+debuffer[3] , -debuffer[0]+debuffer[1]),dtype=np.float32)
        xanes_ims2[0,:,:] = xanes_ims[0,debuffer[2]:debuffer[3] ,  debuffer[0]:debuffer[1]]
        xanes_ims2[1,:,:] = xanes_ims[1,debuffer[2]:debuffer[3] ,  debuffer[0]:debuffer[1]]
        xanes_ims2[2,:,:] = xanes_ims[2,debuffer[2]:debuffer[3] ,  debuffer[0]:debuffer[1]]
        xanes_ims2[3,:,:] = xanes_ims[3,debuffer[2]:debuffer[3] ,  debuffer[0]:debuffer[1]]
        h5object_new.create_dataset('xray_images', shape=xanes_ims2.shape,  dtype=np.float32, data=xanes_ims2)
        h5object_new.close()
        os.remove(data_directory+data_subdirectory+filename)
        os.rename(data_directory+data_subdirectory+'temp.h5',data_directory+data_subdirectory+filename)
        







                                           #lossy = 'yes' means you keep ONLY the pixels for which ALL images contained data.
def calculate_debuffer_multiple_images(ims, lossy='no'):  # ims must be a n x i x j array where n is the number of images
    if ims.ndim == 2: ims=np.stack((ims,ims))  #In case the user gives a single image
    debuffer = [0,0,0,0]  # debuffer[0] is how many LHS dummy columns.  debuffer[1] is how many RHS dummy columns.  debuffer[2] is how many topside dummy rows.  debuffer[3] is how many bottomside dummy rows.  
    if lossy == 'no':
        debuffer_all_images=[100000,0,100000,0]
    else:
        debuffer_all_images=[0,100000,0,100000]
    im_shape_rows = ims.shape[1]
    im_shape_cols = ims.shape[2]
    im_half_rows = np.int(im_shape_rows/2)
    im_half_cols = np.int(im_shape_cols/2)
    for i in range(ims.shape[0]):
        debuffer[0] = int( np.max(np.append(np.where(ims[i,  im_half_rows ,0:im_half_cols ]==0.12345678)[0]+1,              0           )) )
        debuffer[1] = int( np.min(np.append(np.where(ims[i,  im_half_rows ,  im_half_cols:]==0.12345678)[0]+0,im_shape_cols-im_half_cols)) ) + im_half_cols
        debuffer[2] = int( np.max(np.append(np.where(ims[i,0:im_half_rows ,  im_half_cols ]==0.12345678)[0]+1,              0           )) )
        debuffer[3] = int( np.min(np.append(np.where(ims[i,  im_half_rows:,  im_half_cols ]==0.12345678)[0]+0,im_shape_rows-im_half_rows)) ) + im_half_rows
        if lossy == 'no':
            if debuffer[0]<debuffer_all_images[0]: debuffer_all_images[0]=debuffer[0]
            if debuffer[1]>debuffer_all_images[1]: debuffer_all_images[1]=debuffer[1]
            if debuffer[2]<debuffer_all_images[2]: debuffer_all_images[2]=debuffer[2]
            if debuffer[3]>debuffer_all_images[3]: debuffer_all_images[3]=debuffer[3]
        else:
            if debuffer[0]>debuffer_all_images[0]: debuffer_all_images[0]=debuffer[0]
            if debuffer[1]<debuffer_all_images[1]: debuffer_all_images[1]=debuffer[1]
            if debuffer[2]>debuffer_all_images[2]: debuffer_all_images[2]=debuffer[2]
            if debuffer[3]<debuffer_all_images[3]: debuffer_all_images[3]=debuffer[3]
        
    return(debuffer_all_images)


    
    
def calculate_image_debuffer_multiple_files(scan_numbers, lossy='no'):
    debuffer_multi_file = [100000,0,100000,0]  # debuffer[0] is how many LHS dummy columns.  debuffer[1] is how many RHS dummy columns.  debuffer[2] is how many topside dummy rows.  debuffer[3] is how many bottomside dummy rows.  
    
    for i in range(0,len(scan_numbers)):
        filename="%.4f" % scan_numbers[i]
        filename='processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
        h5object= h5py.File(data_directory+data_subdirectory+filename, 'r')        
        xanes_raw_ims = np.array(h5object['xray_images'])
        h5object.close()
        debuffer = calculate_debuffer_multiple_images(xanes_raw_ims,lossy)
        if debuffer[0]<debuffer_multi_file[0]: debuffer_multi_file[0]=debuffer[0]
        if debuffer[1]>debuffer_multi_file[1]: debuffer_multi_file[1]=debuffer[1]
        if debuffer[2]<debuffer_multi_file[2]: debuffer_multi_file[2]=debuffer[2]
        if debuffer[3]>debuffer_multi_file[3]: debuffer_multi_file[3]=debuffer[3]
        
    return(debuffer_multi_file)

    




def get_images_statistics(scan_numbers, image_type_2_show ):
    images_mean = np.ones(len(scan_numbers))
    images_std = np.ones(len(scan_numbers))
    for i in range(0,len(scan_numbers)):
        # Get the first image
        if image_type_2_show == 'img_bkg1':
            im=get_raw_image(scan_numbers[i],'img_bkg')
            im=im[0,:,:]
        if image_type_2_show == 'img_bkg2':
            im=get_raw_image(scan_numbers[i],'img_bkg')
            im=im[1,:,:]
        if image_type_2_show == 'img_bkg':
            im=get_raw_image(scan_numbers[i],'img_bkg')
            if ("%.5f" % scan_numbers[0])[10] == '0': im=im[0,:,:]
            if ("%.5f" % scan_numbers[0])[10] == '1': im=im[1,:,:]
        if image_type_2_show == 'img_dark':
            im=get_raw_image(scan_numbers[i],'img_dark')
            im=im[0,:,:]
        if image_type_2_show != 'img_bkg1' and image_type_2_show != 'img_bkg2' and image_type_2_show != 'img_bkg' and image_type_2_show != 'img_dark':
            im=get_processed_image(scan_numbers[i],image_type_2_show)[:,:,0]
        
        im = im[510:-510,610:-610]
        images_mean[i] = np.mean(im[:])
        images_std[i] =  np.std(im[:])
        
    return(images_mean, images_std) 





def deflicker_using_4_neighbors_time_series(target_scan_number , four_neigbhors , gaussian_filter_sizes ):
                                                                                              # beam_energy can be 'all' or '6520' or '6600' or '8970' or '9050'
    # Get the target image
    target_image = get_processed_image(target_scan_number, 'xray_images',      'none'    )
    deflickered_image = 1.0*target_image
    
    # Get the baseline image
    other_image1 = get_processed_image(four_neigbhors[0], 'xray_images', 'none')
    other_image2 = get_processed_image(four_neigbhors[1], 'xray_images', 'none')
    other_image3 = get_processed_image(four_neigbhors[2], 'xray_images', 'none')
    other_image4 = get_processed_image(four_neigbhors[3], 'xray_images', 'none')  
    baseline_image = other_image1/5 + other_image2/5 + target_image/5 + other_image3/5 + other_image4/5 
    
    # Calculate the fractional difference between the target image and the baseline image(s)
    fractional_difference = (target_image - baseline_image)/baseline_image
    fractional_difference[fractional_difference> 0.5] = 0.0   #Chop out 
    fractional_difference[fractional_difference<-0.5] = 0.0
    mask = np.where(target_image==0.12345678)
    fractional_difference[mask] = 0.0  # A value of 0.0 is benign for out-of-bounds pixels

    # Blur the fractional difference image and then use it to correct the target_image_elements_removed, and then re-insert the elements
    blurred_fractional_difference1 = scipy.ndimage.gaussian_filter(fractional_difference[0,:,:],sigma=gaussian_filter_sizes[0],mode='reflect')
    blurred_fractional_difference2 = scipy.ndimage.gaussian_filter(fractional_difference[1,:,:],sigma=gaussian_filter_sizes[1],mode='reflect')
    blurred_fractional_difference3 = scipy.ndimage.gaussian_filter(fractional_difference[2,:,:],sigma=gaussian_filter_sizes[2],mode='reflect')
    blurred_fractional_difference4 = scipy.ndimage.gaussian_filter(fractional_difference[3,:,:],sigma=gaussian_filter_sizes[3],mode='reflect')
    target_image[0,:,:] = target_image[0,:,:] / (1.0 + blurred_fractional_difference1)
    target_image[1,:,:] = target_image[1,:,:] / (1.0 + blurred_fractional_difference2)
    target_image[2,:,:] = target_image[2,:,:] / (1.0 + blurred_fractional_difference3)
    target_image[3,:,:] = target_image[3,:,:] / (1.0 + blurred_fractional_difference4)
    target_image[mask] = 0.12345678

    # Save the data back into the h5 file
    filename="%.4f" % target_scan_number
    filename_string = 'processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'    
    h5object = h5py.File(data_directory+data_subdirectory+filename_string, 'r+')
    h5object['xray_images'][...] = target_image      
    h5object.close()
    




                                                                                                                                      #remove_elements can be 'Mn' or 'BiMn'
def deflicker_using_average_image_elements_removed(target_scan_numbers , averaged_image_filename , gaussian_filter_size , beam_energy,  remove_elements):
    for i in range(0,len(target_scan_numbers)):
        target_scan_number = target_scan_numbers[i]
        print(target_scan_number)                                                                                                                       # beam_energy can be '6520' or '6600' or '8970' or '9050'
        # Get the target image
        target_image                  = get_processed_image(target_scan_number, 'xray_images',      'none'    )
        target_image_elements_removed = get_processed_image(target_scan_number, 'xray_images', remove_elements)
        
        temp_obj = h5py.File(data_directory+data_subdirectory+averaged_image_filename, 'r')
        averaged_image_elements_removed = np.array(temp_obj['average_image'])
        
        # Calculate the fractional difference between the target image and the baseline image(s)
        if beam_energy=='6520': x=0
        if beam_energy=='6600': x=1
        if beam_energy=='8970': x=2
        if beam_energy=='9050': x=3
        fractional_difference = (target_image_elements_removed[x,:,:] - averaged_image_elements_removed)/averaged_image_elements_removed
        fractional_difference[fractional_difference> 0.5] = 0.0   #Chop out 
        fractional_difference[fractional_difference<-0.5] = 0.0
        mask = np.where(target_image_elements_removed[x,:,:]==0.12345678)
        fractional_difference[mask] = 0.0  # A value of 0.0 is benign for out-of-bounds pixels
        
        # Blur the fractional difference image and then use it to correct the target_image_elements_removed, 
        blurred_fractional_difference = scipy.ndimage.gaussian_filter(fractional_difference,sigma=gaussian_filter_size,mode='reflect')
        temp = target_image_elements_removed[x,:,:] / (1.0 + blurred_fractional_difference)
        temp[mask] = 0.12345678
        
        # Re-insert the elements
        temp = insert_elements_into_TXM_image(temp, beam_energy, get_processed_image(target_scan_number, 'thickness_Mn')[:,:,0], get_processed_image(target_scan_number, 'thickness_Cu')[:,:,0], get_processed_image(target_scan_number, 'thickness_Bi')[:,:,0], remove_elements)
        target_image[x,:,:] = temp
        
        filename="%.4f" % target_scan_number
        filename_string = 'processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'    
        h5object = h5py.File(data_directory+data_subdirectory+filename_string, 'r+')
        h5object['xray_images'][...] = target_image      
        h5object.close()
    




def deflicker_9050_using_8970(scan_numbers):    
    for i in range(0,len(scan_numbers)):
        print(scan_numbers[i])
        image_8970 = get_processed_image(scan_numbers[i], '8970')[:,:,0]
        image_9050 = get_processed_image(scan_numbers[i], '9050')[:,:,0]
        image_8970[np.where(image_8970<0.000001)]=0.000001
        image_9050[np.where(image_9050<0.000001)]=0.000001
        mask1 = np.where(image_8970==0.12345678)
        mask2 = np.where(image_9050==0.12345678)
        
        #If no copper is present, the ratio ln(9050)/ln(8970) should be greater than ~0.97 (note: ln(x) between x=0 and x=1 grows larger in absolute magnitude as x->0).  If the ratio is smaller than 0.97, then the 9050 image needs fixing
        difference = np.log(image_9050)/np.log(image_8970) 
        difference = scipy.ndimage.uniform_filter(difference, size=3, mode='constant')
        difference[mask1]=1.0
        difference[mask2]=1.0  
        difference[np.where(difference<0.0)]=0.0  
        difference[np.where(difference>3.0)]=3.0  
        
        #Make an image with the pixels that have difference values that cannot be physically correct
        correction = np.ones((image_8970.shape))
        correction[np.where(difference<0.975)] = difference[np.where(difference<0.975)] 
        correction[mask1]=1.0
        correction[mask2]=1.0 
        
        #Make a blurred image of the correction
        blurred_correction = scipy.ndimage.gaussian_filter(correction,sigma=75,mode='reflect')
        blurred_correction[mask1]=1.0
        blurred_correction[mask2]=1.0
        
        image_9050 = image_9050*blurred_correction
        filename="%.4f" % scan_numbers[i]
        filename_string = 'processed_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'    
        h5object = h5py.File(data_directory+data_subdirectory+filename_string, 'r+')
        h5object['xray_images'][3,:,:] = image_9050      
        h5object.close()
        

        
        
        
        
        
        
        