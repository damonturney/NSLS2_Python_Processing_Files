# -*- coding: utf-8 -*-
"""
Created on Sat Mar 6 2020

"""

import numpy as np
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
data_directory='../20191105_FXI_Beamtime_Work/Data_From_Beamline/34563_to_34875/';
results_directory='../20191105_FXI_Beamtime_Work/Results_Of_Data_Processing/';
object_list_allfilesdirectories = pathlib.Path(data_directory)
object_recursiveglob_tiffiles = object_list_allfilesdirectories.glob('*.tif')      
object_list_filenames_tiffiles = list(object_recursiveglob_tiffiles)
# to paste together a full pathname for a file: "object_list_filenames_tiffiles[i].parents[0].joinpath(object_list_filenames_tiffiles[i].parts[-1][0:-4]"
############################################################


### Workflow ##############################################
# 1) Run create_aligned_h5_file() on each Manganese multipos_2D_xanes_scan2_[]...h5 file to align the images. Use a command like for i in range(34675,34725,2): create_aligned_h5_file(i);      NOTE:  file 34675 is missing, see your beamline notes -- before 34675 the Mn files are odd numbered and after 34675 the Mn files are even numbered 
# 2) Run calculate_optical_thickness() on each of the aligned_images...h5 fiels created in step 1.  Use a command like    Use a command like for i in range(34675,34725,2): calculate_optical_thickness(i);   
############################################################
    


#filename MUST be supplied as a number    
def create_aligned_h5_file(Mn_filename):  #filename MUST be supplied as a number
    Mn_filename_string="%.4f" % Mn_filename
    Mn_filename_string='multipos_2D_xanes_scan2_id_'+Mn_filename_string[0:5]+'_repeat_'+Mn_filename_string[6:8]+'_pos_'+Mn_filename_string[8:10]+'.h5'
    h5object_old = h5py.File(data_directory+Mn_filename_string, 'r')    
    h5object_new = h5py.File(data_directory+'aligned_images_'+Mn_filename_string[27:-3]+'.h5', 'w')
    h5object_old.copy('scan_time',h5object_new)
    h5object_old.copy('X_eng',    h5object_new)
    h5object_old.copy('scan_id',  h5object_new)
    h5object_old.copy('note',     h5object_new)
    ims=create_ims_aligned_Mn_Cu(Mn_filename)
    h5object_new.create_dataset('xray_images', shape=(4,1080,1280), dtype=np.float32, data=ims)
    h5object_old.close()
    h5object_new.close()
    
    
#Run this on the files output from save_aligned_h5_file
def calculate_optical_thickness(filename): #Run this on the files output from save_aligned_h5_file
    if type(filename) != str:
        filename="%.4f" % filename
        filename='aligned_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object= h5py.File(data_directory+filename, 'r+')
    ims     = np.array(h5object['xray_images'])
    ims[ims<=0.0]=np.median(ims[ims>0]) #so that np.log doesn't create an error
        
    # X-ray absorption coefficients in units of 1/mm 
    a_6520_Mn = 31.573;  a_6600_Mn = 207.698; a_8970_Mn = 94.641; a_9050_Mn = 92.436;
    a_6520_Cu = 80.899;  a_6600_Cu = 78.209;  a_8970_Cu = 32.903; a_9050_Cu = 249.79;
    a_6520_Bi = 389.43;  a_6600_Bi = 377.563; a_8970_Bi = 172.32; a_9050_Bi = 168.434;
    a_6520_El = 3.254;   a_6600_El = 3.136;   a_8970_El = 1.229;  a_9050_El = 1.1966;  #1 part NaOH and 5 parts H2O (by mole ratios)
    a_6520_C  = 1.8278;  a_6600_C  = 1.759;   a_8970_C  = 0.6759; a_9050_C  = 0.6577;
    
    A = np.zeros((3,3))
    b = np.zeros((3,1))
    optical_thickness_Cu=np.zeros(ims[0,:,:].shape,dtype=np.float32)
    optical_thickness_Mn=np.zeros(ims[0,:,:].shape,dtype=np.float32)
    optical_thickness_Bi=np.zeros(ims[0,:,:].shape,dtype=np.float32)
    optical_thickness_El=np.zeros(ims[0,:,:].shape,dtype=np.float32)
    optical_thickness_C= np.ones(ims[0,:,:].shape,dtype=np.float32)*0.170 #In mm. This is set in stone, not optimized. I have the two PMMA films plus the carbon foil inside  I can't remember how thick the PMMA films are
    thickness_total = 0.220  #In mm.
    
    for m in range(0,ims[0,:,:].shape[0]):
        if np.mod(m,10)==0: sys.stdout.write('\rLeast Squares, Row: '+str(m))
        sys.stdout.flush()
        for n in range(0,ims[0,:,:].shape[1]):
            ln_I_I0_6520 = np.log(ims[0,m,n]);  ln_I_I0_6600 = np.log(ims[1,m,n]); ln_I_I0_8970 = np.log(ims[2,m,n]); ln_I_I0_9050 = np.log(ims[3,m,n]);

            sum_aMn_aEl = (a_6520_Mn*a_6520_El + a_6600_Mn*a_6600_El + a_8970_Mn*a_8970_El + a_9050_Mn*a_9050_El)
            sum_aCu_aEl = (a_6520_Cu*a_6520_El + a_6600_Cu*a_6600_El + a_8970_Cu*a_8970_El + a_9050_Cu*a_9050_El)
            sum_aBi_aEl = (a_6520_Bi*a_6520_El + a_6600_Bi*a_6600_El + a_8970_Bi*a_8970_El + a_9050_Bi*a_9050_El)
            sum_aMn_aMn = (a_6520_Mn*a_6520_Mn + a_6600_Mn*a_6600_Mn + a_8970_Mn*a_8970_Mn + a_9050_Mn*a_9050_Mn)
            sum_aCu_aMn = (a_6520_Cu*a_6520_Mn + a_6600_Cu*a_6600_Mn + a_8970_Cu*a_8970_Mn + a_9050_Cu*a_9050_Mn)
            sum_aBi_aMn = (a_6520_Bi*a_6520_Mn + a_6600_Bi*a_6600_Mn + a_8970_Bi*a_8970_Mn + a_9050_Bi*a_9050_Mn)
            sum_aBi_aCu = (a_6520_Cu*a_6520_Bi + a_6600_Cu*a_6600_Bi + a_8970_Cu*a_8970_Bi + a_9050_Cu*a_9050_Bi)
            sum_aCu_aCu = (a_6520_Cu*a_6520_Cu + a_6600_Cu*a_6600_Cu + a_8970_Cu*a_8970_Cu + a_9050_Cu*a_9050_Cu)
            sum_aBi_aBi = (a_6520_Bi*a_6520_Bi + a_6600_Bi*a_6600_Bi + a_8970_Bi*a_8970_Bi + a_9050_Bi*a_9050_Bi)
            
            A[0,:] = [ sum_aMn_aEl - sum_aMn_aMn , sum_aMn_aEl - sum_aCu_aMn , sum_aMn_aEl - sum_aBi_aMn ]
            A[1,:] = [ sum_aCu_aEl - sum_aCu_aMn , sum_aCu_aEl - sum_aCu_aCu , sum_aCu_aEl - sum_aBi_aCu ]
            A[2,:] = [ sum_aBi_aEl - sum_aBi_aMn , sum_aBi_aEl - sum_aBi_aCu , sum_aBi_aEl - sum_aBi_aBi ]
            
            b[0] = (thickness_total - optical_thickness_C[m,n])*sum_aMn_aEl + (a_6520_Mn*ln_I_I0_6520 + a_6600_Mn*ln_I_I0_6600 + a_8970_Mn*ln_I_I0_8970 + a_9050_Mn*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Mn*a_6520_C + a_6600_Mn*a_6600_C + a_8970_Mn*a_8970_C + a_9050_Mn*a_9050_C)
            b[1] = (thickness_total - optical_thickness_C[m,n])*sum_aCu_aEl + (a_6520_Cu*ln_I_I0_6520 + a_6600_Cu*ln_I_I0_6600 + a_8970_Cu*ln_I_I0_8970 + a_9050_Cu*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Cu*a_6520_C + a_6600_Cu*a_6600_C + a_8970_Cu*a_8970_C + a_9050_Cu*a_9050_C)
            b[2] = (thickness_total - optical_thickness_C[m,n])*sum_aBi_aEl + (a_6520_Bi*ln_I_I0_6520 + a_6600_Bi*ln_I_I0_6600 + a_8970_Bi*ln_I_I0_8970 + a_9050_Bi*ln_I_I0_9050) + optical_thickness_C[m,n]*(a_6520_Bi*a_6520_C + a_6600_Bi*a_6600_C + a_8970_Bi*a_8970_C + a_9050_Bi*a_9050_C)
            
            temp = np.linalg.solve(A, b)
            optical_thickness_Mn[m,n] = np.float32(temp[0])  #this produces optical thickness in mm    
            optical_thickness_Cu[m,n] = np.float32(temp[1])  #this produces optical thickness in mm
            optical_thickness_Bi[m,n] = np.float32(temp[2])  #this produces optical thickness in mm         
            optical_thickness_El[m,n] = thickness_total - optical_thickness_C[m,n] - optical_thickness_Mn[m,n] + optical_thickness_Cu[m,n] + optical_thickness_Bi[m,n]  #this produces optical thickness in mm         
    
    sys.stdout.write('\n')
    sys.stdout.flush()
    sys.stdout.write('\n')
    
    #Now search for a lower sum-of-square-error by pairwise swaping of thickness (via all pairwise combinatorics) 
    for m in range(0,ims[0,:,:].shape[0]):
        if np.mod(m,10)==0: sys.stdout.write('\rMaximum Decent, Row: '+str(m))
        sys.stdout.flush()
        for n in range(0,ims[0,:,:].shape[1]):
            for m in range(0,10):
                baseline_sum_squares =           calculate_baseline_sum_squares(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
                test_sum_squares = pairwise_thicknessswapping_sum_square_errors(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)                         
                sum_squares_gradient = test_sum_squares - baseline_sum_squares
                max_pair = np.argmax(abs(sum_squares_gradient))
                
                if max_pair==0:
                    previous_sum_square_errors = baseline_sum_squares
                    sum_square_errors = baseline_sum_squares
                    while(sum_square_errors<=previous_sum_square_errors):
                        previous_sum_square_errors = sum_square_errors
                        optical_thickness_Mn_test = optical_thickness_Mn[m,n] - 0.0001*np.sign(sum_squares_gradient[0])
                        optical_thickness_Cu_test = optical_thickness_Cu[m,n] + 0.0001*np.sign(sum_squares_gradient[0])
                        sum_square_errors = calculate_baseline_sum_squares(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
                        if sum_square_errors<=previous_sum_square_errors:
                            optical_thickness_Mn[m,n]=optical_thickness_Mn_test
                            optical_thickness_Cu[m,n]=optical_thickness_Cu_test
                if max_pair==1:
                    previous_sum_square_errors = baseline_sum_squares
                    sum_square_errors = baseline_sum_squares
                    while(sum_square_errors<=previous_sum_square_errors):
                        previous_sum_square_errors = sum_square_errors
                        optical_thickness_Mn_test = optical_thickness_Mn[m,n] - 0.0001*np.sign(sum_squares_gradient[1])
                        optical_thickness_Bi_test = optical_thickness_Bi[m,n] + 0.0001*np.sign(sum_squares_gradient[1])
                        sum_square_errors = calculate_baseline_sum_squares(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
                        if sum_square_errors<=previous_sum_square_errors:
                            optical_thickness_Mn[m,n]=optical_thickness_Mn_test
                            optical_thickness_Bi[m,n]=optical_thickness_Bi_test
                if max_pair==2:
                    previous_sum_square_errors = baseline_sum_squares
                    sum_square_errors = baseline_sum_squares
                    while(sum_square_errors<=previous_sum_square_errors):
                        previous_sum_square_errors = sum_square_errors
                        optical_thickness_Mn_test = optical_thickness_Mn[m,n] - 0.0001*np.sign(sum_squares_gradient[2])
                        optical_thickness_El_test = optical_thickness_El[m,n] + 0.0001*np.sign(sum_squares_gradient[2])
                        sum_square_errors = calculate_baseline_sum_squares(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
                        if sum_square_errors<=previous_sum_square_errors:
                            optical_thickness_Mn[m,n]=optical_thickness_Mn_test
                            optical_thickness_El[m,n]=optical_thickness_El_test
                if max_pair==3:
                    previous_sum_square_errors = baseline_sum_squares
                    sum_square_errors = baseline_sum_squares
                    while(sum_square_errors<=previous_sum_square_errors):
                        previous_sum_square_errors = sum_square_errors
                        optical_thickness_Cu_test = optical_thickness_Cu[m,n] - 0.0001*np.sign(sum_squares_gradient[3])
                        optical_thickness_Bi_test = optical_thickness_Bi[m,n] + 0.0001*np.sign(sum_squares_gradient[3])
                        sum_square_errors = calculate_baseline_sum_squares(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
                        if sum_square_errors<=previous_sum_square_errors:
                            optical_thickness_Cu[m,n]=optical_thickness_Cu_test
                            optical_thickness_Bi[m,n]=optical_thickness_Bi_test
                if max_pair==4:
                    previous_sum_square_errors = baseline_sum_squares
                    sum_square_errors = baseline_sum_squares
                    while(sum_square_errors<=previous_sum_square_errors):
                        previous_sum_square_errors = sum_square_errors
                        optical_thickness_Cu_test = optical_thickness_Cu[m,n] - 0.0001*np.sign(sum_squares_gradient[4])
                        optical_thickness_El_test = optical_thickness_El[m,n] + 0.0001*np.sign(sum_squares_gradient[4])
                        sum_square_errors = calculate_baseline_sum_squares(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
                        if sum_square_errors<=previous_sum_square_errors:
                            optical_thickness_Cu[m,n]=optical_thickness_Cu_test
                            optical_thickness_El[m,n]=optical_thickness_El_test
                if max_pair==5:
                    previous_sum_square_errors = baseline_sum_squares
                    sum_square_errors = baseline_sum_squares
                    while(sum_square_errors<=previous_sum_square_errors):
                        previous_sum_square_errors = sum_square_errors
                        optical_thickness_Bi_test = optical_thickness_Bi[m,n] - 0.0001*np.sign(sum_squares_gradient[5])
                        optical_thickness_El_test = optical_thickness_El[m,n] + 0.0001*np.sign(sum_squares_gradient[5])
                        sum_square_errors = calculate_baseline_sum_squares(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El)
                        if sum_square_errors<=previous_sum_square_errors:
                            optical_thickness_Bi[m,n]=optical_thickness_Bi_test
                            optical_thickness_El[m,n]=optical_thickness_El_test

                        
     #check if the optical_thickness datasets are ALREADY in the h5object, and save data
    h5object['optical_thickness_Mn'][:]=optical_thickness_Mn if 'optical_thickness_Mn' in h5object.keys()  else h5object.create_dataset('optical_thickness_Mn', shape=(1080,1280), dtype=np.float32, data=optical_thickness_Mn)
    h5object['optical_thickness_Cu'][:]=optical_thickness_Cu if 'optical_thickness_Cu' in h5object.keys()  else h5object.create_dataset('optical_thickness_Cu', shape=(1080,1280), dtype=np.float32, data=optical_thickness_Cu)
    h5object['optical_thickness_Bi'][:]=optical_thickness_Bi if 'optical_thickness_Bi' in h5object.keys()  else h5object.create_dataset('optical_thickness_Bi', shape=(1080,1280), dtype=np.float32, data=optical_thickness_Bi)
    h5object['optical_thickness_C' ][:]=optical_thickness_C  if 'optical_thickness_C'  in h5object.keys()  else h5object.create_dataset('optical_thickness_C' , shape=(1080,1280), dtype=np.float32, data=optical_thickness_C)
    h5object['optical_thickness_El'][:]=optical_thickness_El if 'optical_thickness_El' in h5object.keys()  else h5object.create_dataset('optical_thickness_El', shape=(1080,1280), dtype=np.float32, data=optical_thickness_El)
    h5object.close()
    

    
    
    
#filename MUST be supplied as a number
def plot_single_pixel_least_squares_data(filename,row,column):   #filename MUST be supplied as a number
    
    # X-ray absorption coefficients in units of 1/mm 
    a_6520_Mn = 31.573;  a_6600_Mn = 207.698; a_8970_Mn = 94.641; a_9050_Mn = 92.436;
    a_6520_Cu = 80.899;  a_6600_Cu = 78.209;  a_8970_Cu = 32.903; a_9050_Cu = 249.79;
    a_6520_Bi = 389.43;  a_6600_Bi = 377.563; a_8970_Bi = 172.32; a_9050_Bi = 168.434;
    a_6520_El = 3.254;   a_6600_El = 3.136;   a_8970_El = 1.229;  a_9050_El = 1.1966;  #1 part NaOH and 5 parts H2O (by mole ratios)
    a_6520_C  = 1.8278;  a_6600_C  = 1.759;   a_8970_C  = 0.6759; a_9050_C  = 0.6577;
        
    images_Mn = read_FXI_xanes_images(filename)
    scan_start_time_string, beam_energy, scan_id, notes = read_FXI_xanes_metadata(filename)
    beam_energy_Mn=beam_energy
    
    images_Cu = read_FXI_xanes_images(filename+1.0)
    scan_start_time_string, beam_energy, scan_id, notes = read_FXI_xanes_metadata(filename+1.0)
    beam_energy_Cu=beam_energy
 
    filename="%.4f" % filename
    filename='aligned_images_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object2= h5py.File(data_directory+filename, 'r') 
    optical_thickness_Mn= np.array(h5object2['optical_thickness_Mn'])
    optical_thickness_Cu= np.array(h5object2['optical_thickness_Cu'])
    optical_thickness_Bi= np.array(h5object2['optical_thickness_Bi'])
    optical_thickness_C = np.array(h5object2['optical_thickness_C'])
    optical_thickness_El= np.array(h5object2['optical_thickness_El'])
    
    #Plot measured data
    beam_energies = np.concatenate((beam_energy_Mn,beam_energy_Cu))
    plt.scatter(beam_energies, np.concatenate((images_Mn[:,row,column],images_Cu[:,row,column])),s=50*np.ones(4),marker='x')
    
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

    plt.figure()
    plt.scatter(['Mn', 'Cu', 'Bi', 'C', 'El' ], [optical_thickness_Mn[row,column], optical_thickness_Cu[row,column], optical_thickness_Bi[row,column], optical_thickness_C[row,column], optical_thickness_El[row,column]])
    

    plt.figure()
    imshow_image=images_Mn[0,:,:]
    imshow_image[row,column]=1.0
    plt.imshow(imshow_image)





def read_FXI_xanes_metadata(filename):  #filename can be 34567.0103  to denote repeat 01, position 03
    if type(filename) != str:
        filename="%.4f" % filename
        filename='multipos_2D_xanes_scan2_id_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    h5object    = h5py.File(data_directory+filename, 'r')
    beam_energy = np.array(h5object['X_eng'])
    scan_time   = np.array(h5object['scan_time'])  #scan start time in local time at NSLS2, in epoch format
    scan_id     = np.array(h5object['scan_id'])
    notes       = np.array(h5object['note'])
    scan_start_time = datetime.datetime.fromtimestamp(scan_time)
    scan_start_time_string = datetime.datetime.strftime(scan_start_time, '%Y-%m-%d %H:%M:%S' )    
    notes = np.array(h5object['note'])   
    h5object.close()
    return( scan_start_time_string, beam_energy, scan_id, notes)


def read_FXI_xanes_images(filename):
    if type(filename) != str:
        filename="%.4f" % filename
        filename='multipos_2D_xanes_scan2_id_'+filename[0:5]+'_repeat_'+filename[6:8]+'_pos_'+filename[8:10]+'.h5'
    print(filename)
    h5object    = h5py.File(data_directory+filename, 'r')
    images      = np.array(h5object['img_xanes'])  
    h5object.close()
    return(images)



#This function returns the number of pixels that the second image is translated (positive going [downward, to-the-right]) with respect to the first image
def find_image_translation(im1,im2):  
    image_product=np.fft.fft2(im1) * np.fft.fft2(im2).conj();
    cc_image=np.fft.fftshift(np.fft.ifft2(image_product));
    cc_image_min=np.min(cc_image.real[:])
    cc_image.real[570:,:]=cc_image_min
    cc_image.real[:510,:]=cc_image_min
    cc_image.real[:,690:]=cc_image_min
    cc_image.real[:,:610]=cc_image_min
    #
    max_indices=np.array(np.unravel_index(np.argmax(cc_image.real,axis=None), cc_image.real.shape))
    #
    #Now let's test to see if the max indices is a false peak at dead center
    if (max_indices[0]==im1.shape[0]/2 and max_indices[1]==im1.shape[1]/2):
        cc_image.real[max_indices]=cc_image_min
        secondary_max_indices=np.array(np.unravel_index(np.argmax(cc_image.real,axis=None), cc_image.real.shape))
        #
        distance_btwn_peaks=np.linalg.norm(max_indices-secondary_max_indices)
        if distance_btwn_peaks>2:
            max_indices=secondary_max_indices
    #    
    translation_y = im1.shape[0]/2 - max_indices[0] 
    translation_x = im1.shape[1]/2 - max_indices[1]
    #
    return(np.array([translation_y, translation_x]).astype('int'))
    
    
    
def pairwise_thicknessswapping_sum_square_errors(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El):
    test_sum_squares = np.zeros(6)
    
    #Calculate the test case sum of squared errors
    test_thickness1 = optical_thickness_Mn[m,n] + 0.0001
    test_thickness2 = optical_thickness_Cu[m,n] - 0.0001
    sum_squares_6520=(ln_I_I0_6520 + a_6520_Mn*test_thickness1 + a_6520_Cu*test_thickness2 + a_6520_Bi*optical_thickness_Bi[m,n] + a_6520_C*optical_thickness_C[m,n] + a_6520_El*optical_thickness_El[m,n])**2
    sum_squares_6600=(ln_I_I0_6600 + a_6600_Mn*test_thickness1 + a_6600_Cu*test_thickness2 + a_6600_Bi*optical_thickness_Bi[m,n] + a_6600_C*optical_thickness_C[m,n] + a_6600_El*optical_thickness_El[m,n])**2
    sum_squares_8970=(ln_I_I0_8970 + a_8970_Mn*test_thickness1 + a_8970_Cu*test_thickness2 + a_8970_Bi*optical_thickness_Bi[m,n] + a_8970_C*optical_thickness_C[m,n] + a_8970_El*optical_thickness_El[m,n])**2
    sum_squares_9050=(ln_I_I0_9050 + a_9050_Mn*test_thickness1 + a_9050_Cu*test_thickness2 + a_9050_Bi*optical_thickness_Bi[m,n] + a_9050_C*optical_thickness_C[m,n] + a_9050_El*optical_thickness_El[m,n])**2
    test_sum_squares[0] = sum_squares_6520 + sum_squares_6600 + sum_squares_8970 + sum_squares_9050
    
    test_thickness1 = optical_thickness_Mn[m,n] + 0.0001
    test_thickness2 = optical_thickness_Bi[m,n] - 0.0001
    sum_squares_6520=(ln_I_I0_6520 + a_6520_Mn*test_thickness1 + a_6520_Cu*optical_thickness_Cu[m,n] + a_6520_Bi*test_thickness2 + a_6520_C*optical_thickness_C[m,n] + a_6520_El*optical_thickness_El[m,n])**2
    sum_squares_6600=(ln_I_I0_6600 + a_6600_Mn*test_thickness1 + a_6600_Cu*optical_thickness_Cu[m,n] + a_6600_Bi*test_thickness2 + a_6600_C*optical_thickness_C[m,n] + a_6600_El*optical_thickness_El[m,n])**2
    sum_squares_8970=(ln_I_I0_8970 + a_8970_Mn*test_thickness1 + a_8970_Cu*optical_thickness_Cu[m,n] + a_8970_Bi*test_thickness2 + a_8970_C*optical_thickness_C[m,n] + a_8970_El*optical_thickness_El[m,n])**2
    sum_squares_9050=(ln_I_I0_9050 + a_9050_Mn*test_thickness1 + a_9050_Cu*optical_thickness_Cu[m,n] + a_9050_Bi*test_thickness2 + a_9050_C*optical_thickness_C[m,n] + a_9050_El*optical_thickness_El[m,n])**2
    test_sum_squares[1] = sum_squares_6520 + sum_squares_6600 + sum_squares_8970 + sum_squares_9050  
    
    test_thickness1 = optical_thickness_Mn[m,n] + 0.0001
    test_thickness2 = optical_thickness_El[m,n] - 0.0001
    sum_squares_6520=(ln_I_I0_6520 + a_6520_Mn*test_thickness1 + a_6520_Cu*optical_thickness_Cu[m,n] + a_6520_Bi*optical_thickness_Bi[m,n] + a_6520_C*optical_thickness_C[m,n] + a_6520_El*test_thickness2)**2
    sum_squares_6600=(ln_I_I0_6600 + a_6600_Mn*test_thickness1 + a_6600_Cu*optical_thickness_Cu[m,n] + a_6600_Bi*optical_thickness_Bi[m,n] + a_6600_C*optical_thickness_C[m,n] + a_6600_El*test_thickness2)**2
    sum_squares_8970=(ln_I_I0_8970 + a_8970_Mn*test_thickness1 + a_8970_Cu*optical_thickness_Cu[m,n] + a_8970_Bi*optical_thickness_Bi[m,n] + a_8970_C*optical_thickness_C[m,n] + a_8970_El*test_thickness2)**2
    sum_squares_9050=(ln_I_I0_9050 + a_9050_Mn*test_thickness1 + a_9050_Cu*optical_thickness_Cu[m,n] + a_9050_Bi*optical_thickness_Bi[m,n] + a_9050_C*optical_thickness_C[m,n] + a_9050_El*test_thickness2)**2
    test_sum_squares[2] = sum_squares_6520 + sum_squares_6600 + sum_squares_8970 + sum_squares_9050                
    
    test_thickness1 = optical_thickness_Cu[m,n] + 0.0001
    test_thickness2 = optical_thickness_Bi[m,n] - 0.0001
    sum_squares_6520=(ln_I_I0_6520 + a_6520_Mn*optical_thickness_Mn[m,n] + a_6520_Cu*test_thickness1 + a_6520_Bi*test_thickness2 + a_6520_C*optical_thickness_C[m,n] + a_6520_El*optical_thickness_El[m,n])**2
    sum_squares_6600=(ln_I_I0_6600 + a_6600_Mn*optical_thickness_Mn[m,n] + a_6600_Cu*test_thickness1 + a_6600_Bi*test_thickness2 + a_6600_C*optical_thickness_C[m,n] + a_6600_El*optical_thickness_El[m,n])**2
    sum_squares_8970=(ln_I_I0_8970 + a_8970_Mn*optical_thickness_Mn[m,n] + a_8970_Cu*test_thickness1 + a_8970_Bi*test_thickness2 + a_8970_C*optical_thickness_C[m,n] + a_8970_El*optical_thickness_El[m,n])**2
    sum_squares_9050=(ln_I_I0_9050 + a_9050_Mn*optical_thickness_Mn[m,n] + a_9050_Cu*test_thickness1 + a_9050_Bi*test_thickness2 + a_9050_C*optical_thickness_C[m,n] + a_9050_El*optical_thickness_El[m,n])**2
    test_sum_squares[3] = sum_squares_6520 + sum_squares_6600 + sum_squares_8970 + sum_squares_9050
    
    test_thickness1 = optical_thickness_Cu[m,n] + 0.0001
    test_thickness2 = optical_thickness_El[m,n] - 0.0001
    sum_squares_6520=(ln_I_I0_6520 + a_6520_Mn*optical_thickness_Mn[m,n] + a_6520_Cu*test_thickness1 + a_6520_Bi*optical_thickness_Bi[m,n] + a_6520_C*optical_thickness_C[m,n] + a_6520_El*test_thickness2)**2
    sum_squares_6600=(ln_I_I0_6600 + a_6600_Mn*optical_thickness_Mn[m,n] + a_6600_Cu*test_thickness1 + a_6600_Bi*optical_thickness_Bi[m,n] + a_6600_C*optical_thickness_C[m,n] + a_6600_El*test_thickness2)**2
    sum_squares_8970=(ln_I_I0_8970 + a_8970_Mn*optical_thickness_Mn[m,n] + a_8970_Cu*test_thickness1 + a_8970_Bi*optical_thickness_Bi[m,n] + a_8970_C*optical_thickness_C[m,n] + a_8970_El*test_thickness2)**2
    sum_squares_9050=(ln_I_I0_9050 + a_9050_Mn*optical_thickness_Mn[m,n] + a_9050_Cu*test_thickness1 + a_9050_Bi*optical_thickness_Bi[m,n] + a_9050_C*optical_thickness_C[m,n] + a_9050_El*test_thickness2)**2
    test_sum_squares[4] = sum_squares_6520 + sum_squares_6600 + sum_squares_8970 + sum_squares_9050
    
    test_thickness1 = optical_thickness_Bi[m,n] + 0.0001
    test_thickness2 = optical_thickness_El[m,n] - 0.0001
    sum_squares_6520=(ln_I_I0_6520 + a_6520_Mn*optical_thickness_Mn[m,n] + a_6520_Cu*optical_thickness_Cu[m,n] + a_6520_Bi*test_thickness1 + a_6520_C*optical_thickness_C[m,n] + a_6520_El*test_thickness2)**2
    sum_squares_6600=(ln_I_I0_6600 + a_6600_Mn*optical_thickness_Mn[m,n] + a_6600_Cu*optical_thickness_Cu[m,n] + a_6600_Bi*test_thickness1 + a_6600_C*optical_thickness_C[m,n] + a_6600_El*test_thickness2)**2
    sum_squares_8970=(ln_I_I0_8970 + a_8970_Mn*optical_thickness_Mn[m,n] + a_8970_Cu*optical_thickness_Cu[m,n] + a_8970_Bi*test_thickness1 + a_8970_C*optical_thickness_C[m,n] + a_8970_El*test_thickness2)**2
    sum_squares_9050=(ln_I_I0_9050 + a_9050_Mn*optical_thickness_Mn[m,n] + a_9050_Cu*optical_thickness_Cu[m,n] + a_9050_Bi*test_thickness1 + a_9050_C*optical_thickness_C[m,n] + a_9050_El*test_thickness2)**2
    test_sum_squares[5] = sum_squares_6520 + sum_squares_6600 + sum_squares_8970 + sum_squares_9050

    return(test_sum_squares)
    
    
    
    
def calculate_baseline_sum_squares(m,n,ln_I_I0_6520,ln_I_I0_6600,ln_I_I0_8970,ln_I_I0_9050,a_6520_Mn,a_6600_Mn,a_8970_Mn,a_9050_Mn,a_6520_Cu,a_6600_Cu,a_8970_Cu,a_9050_Cu,a_6520_Bi,a_6600_Bi,a_8970_Bi,a_9050_Bi,a_6520_C,a_6600_C,a_8970_C,a_9050_C,a_6520_El,a_6600_El,a_8970_El,a_9050_El,optical_thickness_Mn,optical_thickness_Cu,optical_thickness_Bi,optical_thickness_C,optical_thickness_El):
    #Calculate the baseline sum of squared errors
    sum_squares_6520=(ln_I_I0_6520 + a_6520_Mn*optical_thickness_Mn[m,n] + a_6520_Cu*optical_thickness_Cu[m,n] + a_6520_Bi*optical_thickness_Bi[m,n] + a_6520_C*optical_thickness_C[m,n] + a_6520_El*optical_thickness_El[m,n])**2
    sum_squares_6600=(ln_I_I0_6600 + a_6600_Mn*optical_thickness_Mn[m,n] + a_6600_Cu*optical_thickness_Cu[m,n] + a_6600_Bi*optical_thickness_Bi[m,n] + a_6600_C*optical_thickness_C[m,n] + a_6600_El*optical_thickness_El[m,n])**2
    sum_squares_8970=(ln_I_I0_8970 + a_8970_Mn*optical_thickness_Mn[m,n] + a_8970_Cu*optical_thickness_Cu[m,n] + a_8970_Bi*optical_thickness_Bi[m,n] + a_8970_C*optical_thickness_C[m,n] + a_8970_El*optical_thickness_El[m,n])**2
    sum_squares_9050=(ln_I_I0_9050 + a_9050_Mn*optical_thickness_Mn[m,n] + a_9050_Cu*optical_thickness_Cu[m,n] + a_9050_Bi*optical_thickness_Bi[m,n] + a_9050_C*optical_thickness_C[m,n] + a_9050_El*optical_thickness_El[m,n])**2
    baseline_sum_squares = sum_squares_6520 + sum_squares_6600 + sum_squares_8970 + sum_squares_9050
    return(baseline_sum_squares)




def cross_correlation_map(im1,im2):
    image_product=np.fft.fft2(im1) * np.fft.fft2(im2).conj();
    cc_image=np.fft.fftshift(np.fft.ifft2(image_product));
    return(cc_image.real)

def matrix_all_translations():
    translations_all=[0.,0.,0.,0.,0.,0.]
    for i in np.arange(34565,34875,2): 
        translations=Mn_Cu_image_translation_values(i)
        print(translations)
        translations_all=np.vstack((translations_all,translations))

#This function requires Mn_filename as a number, e.g., 34678.0001 where .0001 means position 1
def Mn_Cu_image_translation_values(Mn_filename):
    Mn_metadata=read_FXI_xanes_metadata(Mn_filename);
    if Mn_metadata[1][0] < 6.7:
        Mn_ims = read_FXI_xanes_images(Mn_filename);
        Mn_trans = find_image_translation(Mn_ims[0,:,:],Mn_ims[1,:,:])
        #
        Cu_filename=Mn_filename+1.0
        Cu_ims = read_FXI_xanes_images(Cu_filename);
        Cu_trans = find_image_translation(Cu_ims[0,:,:],Cu_ims[1,:,:])
        #
        Mn_2_Cu_trans = find_image_translation(Mn_ims[0,:,:],Cu_ims[1,:,:])
        return(np.array(np.hstack((Mn_trans,Cu_trans,Mn_2_Cu_trans))).astype('int'))
    else:
        return('error')
    
    
    
    
#filename MUST be supplied as a number    
def create_ims_aligned_Mn_Cu(Mn_filename):  #filename MUST be supplied as a number
    Cu_filename=Mn_filename+1.0
    Mn_ims = read_FXI_xanes_images(Mn_filename);
    Mn_trans = find_image_translation(Mn_ims[0,:,:],Mn_ims[1,:,:])
    Mn_im2_aligned = np.zeros(Mn_ims[0,:,:].shape) + np.min(Mn_ims) #add np.min(Mn_ims) so that np.log doesn't create an error
    #Now let's actually shift the 2nd Mn image so it's aligned with the 1st Mn image
    if Mn_trans[0]== 0  and Mn_trans[1]== 0:  Mn_im2_aligned[  Mn_trans[0]:,  Mn_trans[1]:] = Mn_ims[1, Mn_trans[0]:, Mn_trans[1]:];       
    if Mn_trans[0] > 0  and Mn_trans[1] > 0:  Mn_im2_aligned[:-Mn_trans[0] ,:-Mn_trans[1] ] = Mn_ims[1, Mn_trans[0]:, Mn_trans[1]:];       
    if Mn_trans[0] < 0  and Mn_trans[1]== 0:  Mn_im2_aligned[ -Mn_trans[0]:,  Mn_trans[1]:] = Mn_ims[1,:Mn_trans[0],  Mn_trans[1]:];    
    if Mn_trans[0] < 0  and Mn_trans[1] > 0:  Mn_im2_aligned[ -Mn_trans[0]:,:-Mn_trans[1] ] = Mn_ims[1,:Mn_trans[0],  Mn_trans[1]:];    
    if Mn_trans[0]== 0  and Mn_trans[1] < 0:  Mn_im2_aligned[  Mn_trans[0]:, -Mn_trans[1]:] = Mn_ims[1, Mn_trans[0]:,:Mn_trans[1] ];   
    if Mn_trans[0] > 0  and Mn_trans[1] < 0:  Mn_im2_aligned[:-Mn_trans[0] , -Mn_trans[1]:] = Mn_ims[1, Mn_trans[0]:,:Mn_trans[1] ];   
    if Mn_trans[0] < 0  and Mn_trans[1] < 0:  Mn_im2_aligned[ -Mn_trans[0]:, -Mn_trans[1]:] = Mn_ims[1,:Mn_trans[0], :Mn_trans[1] ];
    Mn_ims[1,:,:]=Mn_im2_aligned
       
    
    Cu_ims = read_FXI_xanes_images(Cu_filename);
    Cu_trans = find_image_translation(Cu_ims[0,:,:],Cu_ims[1,:,:])  #Cu_trans  is the translation (pixel shift) of Cu image 2 wrt Cu image 1
    Cu2_trans = find_image_translation(Mn_ims[0,:,:],Cu_ims[1,:,:]) #Cu2_trans is the translation (pixel shift) of Cu image 2 wrt Mn image 1
    Cu1_trans = Cu2_trans - Cu_trans                            #Cu1_trans is the translation (pixel shift) of Cu image 1 wrt Mn image 1
    Cu_im1_aligned = np.zeros(Cu_ims[0,:,:].shape) + np.min(Mn_ims) #add np.min(Mn_ims) so that np.log doesn't create an error
    Cu_im2_aligned = np.zeros(Cu_ims[0,:,:].shape) + np.min(Mn_ims) #add np.min(Mn_ims) so that np.log doesn't create an error      
    #Now let's actually shift the 2nd Cu image so it's aligned with the 1st Mn image
    if Cu2_trans[0]== 0  and Cu2_trans[1]== 0:  Cu_im2_aligned[  Cu2_trans[0]:,  Cu2_trans[1]:] = Cu_ims[1, Cu2_trans[0]:, Cu2_trans[1]:];       
    if Cu2_trans[0] > 0  and Cu2_trans[1] > 0:  Cu_im2_aligned[:-Cu2_trans[0] ,:-Cu2_trans[1] ] = Cu_ims[1, Cu2_trans[0]:, Cu2_trans[1]:];       
    if Cu2_trans[0] < 0  and Cu2_trans[1]== 0:  Cu_im2_aligned[ -Cu2_trans[0]:,  Cu2_trans[1]:] = Cu_ims[1,:Cu2_trans[0],  Cu2_trans[1]:];    
    if Cu2_trans[0] < 0  and Cu2_trans[1] > 0:  Cu_im2_aligned[ -Cu2_trans[0]:,:-Cu2_trans[1] ] = Cu_ims[1,:Cu2_trans[0],  Cu2_trans[1]:];    
    if Cu2_trans[0]== 0  and Cu2_trans[1] < 0:  Cu_im2_aligned[  Cu2_trans[0]:, -Cu2_trans[1]:] = Cu_ims[1, Cu2_trans[0]:,:Cu2_trans[1] ];   
    if Cu2_trans[0] > 0  and Cu2_trans[1] < 0:  Cu_im2_aligned[:-Cu2_trans[0] , -Cu2_trans[1]:] = Cu_ims[1, Cu2_trans[0]:,:Cu2_trans[1] ];   
    if Cu2_trans[0] < 0  and Cu2_trans[1] < 0:  Cu_im2_aligned[ -Cu2_trans[0]:, -Cu2_trans[1]:] = Cu_ims[1,:Cu2_trans[0], :Cu2_trans[1] ];
        
    #Now let's actually shift the 1st Cu image so it's aligned with the 1st Mn image
    if Cu1_trans[0]== 0  and Cu1_trans[1]== 0:  Cu_im1_aligned[  Cu1_trans[0]:,  Cu1_trans[1]:] = Cu_ims[0, Cu1_trans[0]:, Cu1_trans[1]:];       
    if Cu1_trans[0] > 0  and Cu1_trans[1] > 0:  Cu_im1_aligned[:-Cu1_trans[0] ,:-Cu1_trans[1] ] = Cu_ims[0, Cu1_trans[0]:, Cu1_trans[1]:];       
    if Cu1_trans[0] < 0  and Cu1_trans[1]== 0:  Cu_im1_aligned[ -Cu1_trans[0]:,  Cu1_trans[1]:] = Cu_ims[0,:Cu1_trans[0],  Cu1_trans[1]:];    
    if Cu1_trans[0] < 0  and Cu1_trans[1] > 0:  Cu_im1_aligned[ -Cu1_trans[0]:,:-Cu1_trans[1] ] = Cu_ims[0,:Cu1_trans[0],  Cu1_trans[1]:];    
    if Cu1_trans[0]== 0  and Cu1_trans[1] < 0:  Cu_im1_aligned[  Cu1_trans[0]:, -Cu1_trans[1]:] = Cu_ims[0, Cu1_trans[0]:,:Cu1_trans[1] ];   
    if Cu1_trans[0] > 0  and Cu1_trans[1] < 0:  Cu_im1_aligned[:-Cu1_trans[0] , -Cu1_trans[1]:] = Cu_ims[0, Cu1_trans[0]:,:Cu1_trans[1] ];   
    if Cu1_trans[0] < 0  and Cu1_trans[1] < 0:  Cu_im1_aligned[ -Cu1_trans[0]:, -Cu1_trans[1]:] = Cu_ims[0,:Cu1_trans[0], :Cu1_trans[1] ];

    Cu_ims[0,:,:]=Cu_im1_aligned
    Cu_ims[1,:,:]=Cu_im2_aligned
    
    return(np.concatenate((Mn_ims,Cu_ims),axis=0))
    



    
    
def old_stuff():
    # Read beamline x-ray data
    filename = 'xanes_scan2_id_13161.h5'
    h5object = h5py.File(filename, 'r')
    beam_energy = np.array(h5object['X_eng'])
    images = np.array(h5object['img_xanes'])
    dark_images = np.array(h5object['img_dark'])
    scan_start_time_epoch = np.array(h5object['scan_time'])  #Local time in New York, in epoch format
    scan_start_time = datetime.datetime.fromtimestamp(scan_start_time_epoch)
    scan_start_time_string = datetime.datetime.strftime(scan_start_time, '%Y-%m-%d %H:%M:%S' )    
    notes = np.array(h5object['note'])
    
    # Read Biologic Potentiostat Data
    fileobject=open('BiologicData/20190302_2AM_StartVirginCell_C01.mpt','r',errors='ignore')
    fileobject.seek(0)
    fileobject.readline()
    second_line=fileobject.readline()
    for i in range(1,9):
        fileobject.readline()
    eleventh_line=fileobject.readline()
    fileobject.close()
    biologic_data=pandas.read_csv('BiologicData/20190302_2AM_StartVirginCell_C01.mpt', sep='\t', skiprows=int(second_line[18:21])-1)
    biologic_start_time=datetime.datetime.strptime(eleventh_line[25:44],'%m/%d/%Y %H:%M:%S')
    
    
    fig_han, axs_han = plt.subplots(3,1)
    fig_han.set_size_inches(7,9)
    big_axes_han=plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=2)
    big_axes_han.set_ylabel('y-direction (micron)')
    #big_axes_han.set_ylim([0,40])
    big_axes_han.set_xlabel('x-direction (micron)')
    #big_axes_han.set_xlim([0,40])
    big_axes_han.imshow(images[0],cmap='gray',interpolation='none', extent=[0,40,0,40], vmin=0.235, vmax=0.94)
    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.35, wspace=0.01)
    small_axes_han=plt.subplot2grid((3,1),(2,0))
    small_axes_han.set_xlabel('Electrode Voltage (V)')
    small_axes_han.set_ylabel('Current (mA)')
    plt.plot(biologic_data['Ewe/V'].values[0:1500],biologic_data['<I>/mA'].values[0:1500])
    scatter_han = small_axes_han.scatter(biologic_data['Ewe/V'].values[0],biologic_data['<I>/mA'].values[0],c='r',s=25)
    
       
    def change_imshow(frame_num):
        #One frame per 100 seconds
        print(frame_num)
        closest_index=abs(biologic_data['time/s'].values-frame_num*100).argmin()
        scatter_han.set_offsets([biologic_data['Ewe/V'].values[closest_index],biologic_data['<I>/mA'].values[closest_index]])
        if frame_num % 6 == 1:  
            big_axes_han.imshow(images[int((frame_num-1)/6)],cmap='gray',interpolation='none', extent=[0,40,0,40], vmin=0.235, vmax=0.94)
        
    
    animation_handle=animation.FuncAnimation(fig_han, change_imshow, frames=300, blit=False, interval=100, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15)
    animation_handle.save('im.mp4', writer=writer)
    



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