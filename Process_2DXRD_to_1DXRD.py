import os
import sys
import pathlib
import peakutils
import exifread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyFAI  ## IF A PYFAI COMMAND FAILS, TRY TYPING export PYOPENCL_NO_CACHE=1 INTO A TERMINAL BEFORE LAUNCHING PYTHON OR SPYDER.  IF YOURE LAUNCHING VIA ANACONDA-NAVIGATOR THEN OPEN A TERMINAL IN ANACONDA-NAVIGATOR AND TYPE export PYOPENCL_NO_CACHE=1 then type spyder
import scipy 
import tifffile

#### Set working directory ##############################
if sys.platform == 'win32':
    working_folder='C:\\Users\\EI Administrator\\Desktop\\NSLS2_Data_Processing\\20190523_NSLS2_SRX_Data\\Dexela_2DXRD_tifs_AND_xrf_counts\\'
    os.chdir(working_folder)

if sys.platform == 'linux':
    working_folder='/home/damon/Desktop/NSLS2_DataProcessing/20190523_NSLS2_SRX_Data/Dexela_2DXRD_tifs_AND_xrf_counts/'
    os.chdir(working_folder)
    
if sys.platform == 'darwin':
    working_folder='/Users/damon/Desktop/BACKED_UP/WorkFiles/ProjectsGrants/2015_NYSERDA_Birnessite_Project/2019_NSLS2_Synchrotron_Work/20190523_SRX_Beamtime_Work/Dexela_2DXRD_tifs_AND_xrf_counts/'
    working_folder='/Users/damon/Desktop/BACKED_UP/WorkFiles/ProjectsGrants/2015_NYSERDA_Birnessite_Project/2019_NSLS2_Synchrotron_Work/20191014_SRX_Beamtime_Work/Brendans_ZnO_Work/XRD_Data/'
    os.chdir(working_folder)
##########################################################

### Create list of tif files ##############################
object_list_allfilesdirectories = pathlib.Path('.')
object_recursiveglob_tiffiles = object_list_allfilesdirectories.glob('*.tif')      
object_list_filenames_tiffiles = list(object_recursiveglob_tiffiles)
# to paste together a full pathname for a file: "object_list_filenames_tiffiles[i].parents[0].joinpath(object_list_filenames_tiffiles[i].parts[-1][0:-4]"
############################################################


########## NOTES  NOTES   NOTES   NOTES   NOTES   NOTES ##########
pyfai_poni_file='../20190524_1600_ZnO_15keV.poni'
pyfai_poni_file='../../20191014_ZnO_std_64274.poni'
non_peak_std_estimate=1.4/(1+np.exp(0.01*(np.arange(0,3072)-250)))+0.3 #This one is best for the 20190523 experiments
non_peak_std_estimate=1.5/(1+np.exp(0.01*(np.arange(0,3072)-100)))+0.8 #This one is best for the 20191014 experiments
#
# darkfield background dexela xrd images:  
#                              5 seconds: scan2D_27762.tif
#                             10 seconds: scan2D_27873.tif
#                             20 seconds: scan2D_27821.tif and scan2D_27822.tif
############################################################
#
# DATA EXLPLANATION
# 
#
# TYPICAL WORKFLOW FOR PROCESSING THE 2D XRD DATA
# 1) Run pyfai_all_images_in_tif_file(tif_filename_or_number,background_filename='background_filename')
#         If you want to use a tif file for the above command that was an average of many tif files, just do something like pyfai_all_images_in_tif_file('avescan64274.tif',background_filename='avescan64274.tif') 
# 2) Run auto_find_and_model_peaks_all_images_in_tif_file(tif_filename_or_number,window_average_size=3)
# 3) Run manually_pick_peak_indexes(27917,) to write down with pencil-and-paper the indices where you want a manual peak id to be remembered
# 4) Run edit_manual_peaks(27917,'add peak indexes',[1901,2611]) to add or delete peaks manually, and edit_manual_peaks('add known materials'), known_material_filename='/Zinc-Oxide-ZnO/ZnO-9011662.txt') to add xrd patterns that you know should be present
# 5) Run model_manual_peaks(27917) to fit all the manual peaks with a height and fwhm, and to create a model spectrum
# 6) Run convert_spectrum_to_ASCII_XY(27917,image_number=[]) where image_number= int or 'manual' for the manual modeled spectrum, so you can take the spectrum to Marshak's ICCD database and identify known materials using the peaks
# 7) Walk over to Marshak and identify the peaks to known materials, then download the cif file for each known material and use VESTA to create text files of the peaks and d spacings, then use 
# 8) Run edit_manual_peaks(27917, 'add known material',known_material_filename='') to include the materials you found at Marshak's ICCD database 
# 8) Run plot_1D_xrd_and_save_plots_all_images_in_tif_file(27917,with_known_materials=True) to plot the spectrum with peaks with a known material
#
# TO INSPECT THE DATABASE, RUN e.g. edit_manual_peaks(27917,'show')

def pyfai_all_images_in_tif_file(filename,background_filename=[]):
    if type(filename) != str:
        filename = 'scan2D_'+str(filename)+'.tif'
    if (len(background_filename) != 0) and (type(background_filename) != str):
        background_filename = 'scan2D_'+str(background_filename)+'.tif'  
    num_images=num_images_in_tif_file(filename)
    if os.path.exists(filename[0:-4]) == False:
        os.mkdir(filename[0:-4])
    if os.path.exists(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') == True:
        print('The npz files already exist. Delete them before running pyfai again.')
        return
    for i in range(0,int(num_images)):
        print('Running PYFAI on '+filename[0:-4]+'/image_'+f'{i:04d}')
        if len(background_filename) != 0:
            xrd_1D_mean_counts, xrd_1D_baseline_std, xrd_1D_d_spacing, warped_xrd_image = pyfai_single_image_in_tif_file(filename,background_filename,i)
        else:
            xrd_1D_mean_counts, xrd_1D_baseline_std, xrd_1D_d_spacing, warped_xrd_image = pyfai_single_image_in_tif_file(filename,[],i)
        #Now create the data arrays for saving in numpy npz files
        autodetected_peak_indices=    np.array(0,dtype=int)
        all_autodetected_peak_indices=np.array(0,dtype=int)
        autodetected_peak_heights=    np.zeros(0) 
        autodetected_peaks_fwhm=      np.zeros(0) 
        autodetected_model_spectrum=  np.zeros(0)
        manual_peak_indexes=          np.zeros(0,dtype=int) # Thus user selects the location of these peaks
        manual_peak_indexes_local=    np.zeros(0,dtype=int)  # The algorithm slightly shifts the manual_peak_indexes to find the shifted location of the peak
        manual_peak_heights=          np.zeros(0) 
        manual_peak_heights_local=    np.zeros(0) 
        manual_peaks_fwhm=            np.zeros(0) 
        manual_model_spectrum=        np.zeros(0)
        known_materials_names=        np.array(['Material Names:'],dtype='<U100')         #Never delete this first entry "names"
        known_materials_d_spacing_indexes= np.array(np.ones(20)*-1,dtype=int) #Never delete this row dummy array of -1s, we will only keep the first 20 peaks, padded by -1s if less than 20
        known_materials_heights=           np.array(np.ones(20)*-1)           #Never delete this row dummy array of -1s, we will only keep the first 20 peaks, padded by -1s if less than 20
        np.savez(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz',xrd_1D_d_spacing=xrd_1D_d_spacing,xrd_1D_mean_counts=xrd_1D_mean_counts, xrd_1D_baseline_std=xrd_1D_baseline_std, autodetected_peak_indices=autodetected_peak_indices,all_autodetected_peak_indices=all_autodetected_peak_indices,autodetected_peak_heights=autodetected_peak_heights, autodetected_peaks_fwhm=autodetected_peaks_fwhm, autodetected_model_spectrum=autodetected_model_spectrum,manual_peak_indexes_local=manual_peak_indexes_local,manual_peak_heights_local=manual_peak_heights_local, manual_peak_indexes=manual_peak_indexes,manual_peak_heights=manual_peak_heights, manual_peaks_fwhm=manual_peaks_fwhm, manual_model_spectrum=manual_model_spectrum,known_materials_names=known_materials_names, known_materials_d_spacing_indexes=known_materials_d_spacing_indexes,known_materials_heights=known_materials_heights  )
    # Now create an AVERAGE spectra from all the individual spectra
    num_images=num_images_in_tif_file(filename)
    average_xrd_1D_mean_counts=np.zeros(len(xrd_1D_mean_counts))
    for i in range(0,num_images):
        prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
        xrd_1D_d_spacing=  prev_data['xrd_1D_d_spacing']
        average_xrd_1D_mean_counts= average_xrd_1D_mean_counts + prev_data['xrd_1D_mean_counts']
    average_xrd_1D_mean_counts = average_xrd_1D_mean_counts/num_images
    #Save auto detected peaks into the AVERAGE file only
    autodetected_peak_indices, autodetected_peak_heights, autodetected_peaks_fwhm, autodetected_model_spectrum = find_and_model_peaks_single_spectrum(xrd_1D_d_spacing,average_xrd_1D_mean_counts,non_peak_std_estimate,window_average_size=3)
    np.savez(  filename[0:-4]+'/'+filename[0:-4]+'_image_average.npz'           ,xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=autodetected_peak_indices,              all_autodetected_peak_indices=all_autodetected_peak_indices, autodetected_peak_heights=autodetected_peak_heights,              autodetected_peaks_fwhm=autodetected_peaks_fwhm,              autodetected_model_spectrum=autodetected_model_spectrum,             manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=prev_data['manual_peak_indexes'],manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=prev_data['known_materials_names'], known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes'],known_materials_heights=prev_data['known_materials_heights'] )

      
        
    
def auto_find_and_model_peaks_all_images_in_tif_file(filename,window_average_size=1):
    if type(filename) != str:
        filename = 'scan2D_'+str(filename)+'.tif'
    if os.path.exists(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') == True:
        print('Autodetecting and modeling peaks.....')
        all_autodetected_peak_indices=np.ones(0,dtype=int)
        num_spectra=num_images_in_tif_file(filename)
        for i in range(0,num_spectra):
            prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
            xrd_1D_d_spacing=  prev_data['xrd_1D_d_spacing']
            xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts']
            autodetected_peak_indices, autodetected_peak_heights, autodetected_peaks_fwhm, autodetected_model_spectrum = find_and_model_peaks_single_spectrum(xrd_1D_d_spacing,xrd_1D_mean_counts,non_peak_std_estimate,window_average_size=3)
            all_autodetected_peak_indices=np.append(all_autodetected_peak_indices,autodetected_peak_indices)
            temp=np.ones(0,dtype=int)
            for j in range(0,len(xrd_1D_d_spacing)):
                if np.where(all_autodetected_peak_indices==j)[0].size>0:
                    temp=np.append(temp,np.int(j))
            all_autodetected_peak_indices=temp
            np.savez(  filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz',xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=autodetected_peak_indices,              all_autodetected_peak_indices=all_autodetected_peak_indices, autodetected_peak_heights=autodetected_peak_heights,              autodetected_peaks_fwhm=autodetected_peaks_fwhm,              autodetected_model_spectrum=autodetected_model_spectrum,             manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=prev_data['manual_peak_indexes'],manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=prev_data['known_materials_names'], known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes'],known_materials_heights=prev_data['known_materials_heights'] )
        #Now save all_autodetected_peak_indices to all the npz files
        for i in range(0,num_spectra):
            prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
            np.savez(  filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz',xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=all_autodetected_peak_indices, autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=prev_data['manual_peak_indexes'],manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=prev_data['known_materials_names'], known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes'],known_materials_heights=prev_data['known_materials_heights'] )
    else:
        print('You need to run pyfai_all_images_in_tif_file first to create the 1D spectra')


    

def manually_pick_peak_indexes(filename,average_of_all_images=False,xlim=[],ylim=[]):
    global break_while_loop_switch
    break_while_loop_switch=True
    if type(filename) != str:
        filename = 'scan2D_'+str(filename)+'.tif'
    if os.path.exists(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') == True:
        #Collect all autodetected peaks from all locations in the tif file
        num_spectra=num_images_in_tif_file(filename)
        prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') # prev_data stands for previous_data
        all_autodetected_peak_indices=     prev_data['all_autodetected_peak_indices']
        manual_peak_indexes=           prev_data['manual_peak_indexes']
        xrd_1D_d_spacing=           prev_data['xrd_1D_d_spacing']
        xrd_1D_mean_counts=         prev_data['xrd_1D_mean_counts']
        plot_figure_han = plt.figure()
        plot_figure_han.canvas.mpl_connect('close_event', break_while_loop)
        #plt.get_current_fig_manager().window.setGeometry(0,649,645,430)
        plot_figure_han.tight_layout()
        plot_axis_han=plot_figure_han.add_subplot(1,1,1)
        ylim=plot_axis_han.get_ylim() if ylim==[] else plot_axis_han.set_ylim(ylim)
        xlim=plot_axis_han.get_xlim() if xlim==[] else plot_axis_han.set_xlim(xlim)
        #line_handle=plot_axis_han.plot(xrd_1D_d_spacing,xrd_1D_mean_counts,'k')  
        plot_axis_han.set_xlabel(r"d spacing  (${\rm \AA}$)")
        plot_axis_han.set_ylabel('xrd counts') 
        #Plot the manual peaks
        if manual_peak_indexes.size>0: 
            for m in range(0,len(manual_peak_indexes)):
                plot_axis_han.plot([xrd_1D_d_spacing[manual_peak_indexes[m]],xrd_1D_d_spacing[manual_peak_indexes[m]]],[-1,-3] ,linewidth=1,color='r',label='manual id peaks')
        #Plot the autodetected peaks
        for m in range(0,len(all_autodetected_peak_indices)):
            plot_axis_han.plot([xrd_1D_d_spacing[all_autodetected_peak_indices[m]],xrd_1D_d_spacing[all_autodetected_peak_indices[m]]],[-4,-6] ,linewidth=1,color='b',label='manual id peaks')
        #Put text annotation next to each peak, saying the index number so you can edit and keep track of all the peaks
        for m in range(0,len(xrd_1D_d_spacing)):
            if np.any(np.append(all_autodetected_peak_indices,manual_peak_indexes)==m):
                plot_axis_han.annotate(str(m),[xrd_1D_d_spacing[m],-6-0.01*(ylim[1]-ylim[0])],rotation=270,fontsize=6,horizontalalignment='right',verticalalignment='top',linespacing=0.1)
        if average_of_all_images==False:
            i=0
            print('Enter q to quit:')
            while break_while_loop_switch==True:
            #for i in range(0,num_spectra):
                prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
                xrd_1D_d_spacing=           prev_data['xrd_1D_d_spacing']
                xrd_1D_mean_counts=         prev_data['xrd_1D_mean_counts']
                #Plot the XRD line
                line_handle=plot_axis_han.plot(xrd_1D_d_spacing,xrd_1D_mean_counts,'k')  
                plt.title('image '+str(i)+' of '+str(num_spectra))
                plt.pause(.3)
                l=line_handle.pop(0)
                l.remove()
                del l
                i=i+1
                if i>=num_spectra: i=0
        if average_of_all_images==True:
            prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_average.npz') # prev_data stands for previous_data
            xrd_1D_d_spacing=           prev_data['xrd_1D_d_spacing']
            xrd_1D_mean_counts=         prev_data['xrd_1D_mean_counts']
            #Plot the XRD line
            line_handle=plot_axis_han.plot(xrd_1D_d_spacing,xrd_1D_mean_counts,'k')  
            plt.title('average of all images in tif file')            
    else:
        print('You need to run pyfai_all_images_in_tif_file and then autodetect_find_and_model_peaks_all_images_in_tif_file')
    break_while_loop_switch=True



#This global variable enables the matplotlib plot to be eliminated when the user presses "q"
break_while_loop_switch=True
#This function enables the matplotlib plot to be eliminated when the user presses "q"
def break_while_loop(event):
    global break_while_loop_switch
    break_while_loop_switch=False
    


def edit_manual_peaks(tif_filename,action,indexes=[],known_material_filename=[]):  # action can be: 1) add, 2) delete, 3) show
    if type(tif_filename) != str:
        tif_filename = 'scan2D_'+str(tif_filename)+'.tif'
    num_spectra=num_images_in_tif_file(tif_filename)
    if os.path.exists(tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_0000.npz') == True:
        prev_data=np.load(tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_0000.npz') # prev_data stands for previous_data
        xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing']
        manual_peak_indexes=np.int16(prev_data['manual_peak_indexes'])
        if (action == 'delete manual peak indexes') and (len(indexes)>0):
            print('Editing file '+tif_filename)
            if indexes == 'all':
                manual_peak_indexes=np.int16([])
            else:
                indexes=np.int16(indexes)
                if len(manual_peak_indexes)==0:
                    print('manual_peak_indexes is already empty')
                else:
                    manual_peak_indexes=np.delete(manual_peak_indexes,np.where(manual_peak_indexes==indexes))
            for i in range(0,num_spectra):
                prev_data=np.load(tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
                np.savez(  tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_'+f'{i:04d}'+'.npz',xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=manual_peak_indexes,manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=prev_data['known_materials_names'], known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes'],known_materials_heights=prev_data['known_materials_heights'] )
            #np.savez(      tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_average.npz'       ,xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=manual_peak_indexes,manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=prev_data['known_materials_names'], known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes'],known_materials_heights=prev_data['known_materials_heights'] )
        elif action == 'add manual peak indexes':
            print('Editing file '+tif_filename)
            already_there=np.zeros(len(indexes))
            indexes=np.int16(indexes)
            for j in indexes:
                if np.any(manual_peak_indexes==j): indexes=np.delete(indexes,np.where(indexes==j))
            manual_peak_indexes=np.append(manual_peak_indexes,indexes)
            for i in range(0,num_spectra):
                prev_data=np.load(tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
                np.savez(  tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_'+f'{i:04d}'+'.npz',xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=manual_peak_indexes,manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=prev_data['known_materials_names'], known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes'],known_materials_heights=prev_data['known_materials_heights'] )
            prev_data=np.load(tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_average.npz') # prev_data stands for previous_data
            np.savez(      tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_average.npz'       ,xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=manual_peak_indexes,manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=prev_data['known_materials_names'], known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes'],known_materials_heights=prev_data['known_materials_heights'] )
        elif action == 'show':
            print('Showing file '+tif_filename)
            print('manual_peak_indexes')
            print(prev_data['manual_peak_indexes'])
            print('known_materials_names')
            print(prev_data['known_materials_names'])
        elif action == 'add known material':
            print('Editing file '+tif_filename)
            known_material_abbreviated_name,known_material_d_spacings,known_material_peak_heights = return_d_spacing_and_peak_heights_from_txt_file(known_material_filename)
            d_spacing_indices_known_material=  [np.abs(d_spacing-xrd_1D_d_spacing).argmin() for d_spacing in known_material_d_spacings]; 
            #Create arrays that have the top 20 peaks only, pad empty with -1
            known_material_peak_heights_top20=      np.ones(20)*-1
            known_material_d_spacings_indexes_top20=np.ones(20,dtype=int)*-1
            for m in range(0,min(20,len(known_material_peak_heights))):
                known_material_peak_heights_top20[m]=   known_material_peak_heights[known_material_peak_heights.argmax()]
                known_material_d_spacings_indexes_top20[m]= d_spacing_indices_known_material[known_material_peak_heights.argmax()]
                known_material_peak_heights[known_material_peak_heights.argmax()]=-1
            #If this material isn't already in the npz files, then append it into the npz files
            if np.where(prev_data['known_materials_names'] == known_material_abbreviated_name)[0].size == 0:
                known_materials_names=            np.append(prev_data['known_materials_names'][:],             known_material_abbreviated_name)
                known_materials_d_spacing_indexes=np.vstack((prev_data['known_materials_d_spacing_indexes'][:],known_material_d_spacings_indexes_top20))
                known_materials_heights=          np.vstack((prev_data['known_materials_heights'] [:],         known_material_peak_heights_top20))
                for i in range(0,num_spectra):
                    prev_data=np.load(tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
                    np.savez(  tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_'+f'{i:04d}'+'.npz',xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=prev_data['manual_peak_indexes'],manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=known_materials_names, known_materials_d_spacing_indexes=known_materials_d_spacing_indexes,known_materials_heights=known_materials_heights )
                prev_data=np.load(tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_average.npz') # prev_data stands for previous_data
                np.savez(      tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_average.npz'       ,xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=prev_data['manual_peak_indexes'],manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=known_materials_names, known_materials_d_spacing_indexes=known_materials_d_spacing_indexes,known_materials_heights=known_materials_heights )
            else:
                print('This known material is already entered into the npz files!')
        elif (action == 'delete known material') or (action == 'remove known material'):
            if np.where(prev_data['known_materials_names'] == known_material_filename)[0].size > 0:
                print('Editing file '+tif_filename)
                index_to_delete=np.where(prev_data['known_materials_names'] == known_material_filename)[0][0]
                known_materials_names=            np.delete(prev_data['known_materials_names'],            index_to_delete)
                known_materials_d_spacing_indexes=np.delete(prev_data['known_materials_d_spacing_indexes'],index_to_delete,0)
                known_materials_heights=          np.delete(prev_data['known_materials_heights'],          index_to_delete,0)
                for i in range(0,num_spectra):
                    prev_data=np.load(tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
                    np.savez(  tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_'+f'{i:04d}'+'.npz',xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=prev_data['manual_peak_indexes'],manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=known_materials_names, known_materials_d_spacing_indexes=known_materials_d_spacing_indexes,known_materials_heights=known_materials_heights )
                prev_data=np.load(tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_average.npz') # prev_data stands for previous_data
                np.savez(  tif_filename[0:-4]+'/'+tif_filename[0:-4]+'_image_average.npz'           ,xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=prev_data['manual_peak_indexes'],manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=known_materials_names, known_materials_d_spacing_indexes=known_materials_d_spacing_indexes,known_materials_heights=known_materials_heights )
            else:
                print('Could not find that known material!  Try again.')
        else:
            print('Not a valid action.')
    else:
        print('You need to run pyfai_all_images_in_tif_file and then autodetect_find_and_model_peaks_all_images_in_tif_file')







def model_manually_picked_peaks(filename,baseline_std_noise,smoothing_window_size=7):   #smoothing_window_size must be an odd number
    if type(filename) != str:
        filename = 'scan2D_'+str(filename)+'.tif'
    if os.path.exists(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') == True:
        num_images=num_images_in_tif_file(filename)
        for i in range(0,num_images): 
            print('Modeling file '+filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz')
            prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
            xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing']
            xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts']
            manual_peak_indexes=prev_data['manual_peak_indexes']
            manual_peak_indexes_local=manual_peak_indexes*1.0
            manual_peak_heights=prev_data['manual_peak_heights']
            manual_peak_heights_local=manual_peak_heights*1.0
            peak_indices, peak_heights, peaks_fwhm, model_spectrum = find_and_model_peaks_single_spectrum(xrd_1D_d_spacing,xrd_1D_mean_counts,non_peak_std_estimate=non_peak_std_estimate/2,window_average_size=smoothing_window_size)
            for j in range(0,len(manual_peak_indexes)):
                manual_peak_indexes_local[j] =  peak_indices[np.abs(manual_peak_indexes[j]-peak_indices).argmin()]
            manual_peak_indexes_local=peakutils.interpolate(np.array(range(0,len(xrd_1D_d_spacing))),window_average(xrd_1D_mean_counts,smoothing_window_size),ind=list(np.int16(manual_peak_indexes_local)))
            manual_peak_heights_local=xrd_1D_mean_counts[np.int16(manual_peak_indexes_local)]
            np.savez(  filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz',xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=manual_peak_indexes_local,manual_peak_heights_local=manual_peak_heights_local, manual_peak_indexes=prev_data['manual_peak_indexes'],manual_peak_heights=prev_data['manual_peak_heights'], manual_peaks_fwhm=prev_data['manual_peaks_fwhm'], manual_model_spectrum=prev_data['manual_model_spectrum'],known_materials_names=prev_data['known_materials_names'], known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes'],known_materials_heights=prev_data['known_materials_heights'] )



#            manual_peak_heights_single_spectrum=xrd_1D_mean_counts[manual_peak_indexes]
#            higher_peaks=np.where(manual_peak_heights_single_spectrum>manual_peak_heights)[0]
#            higher_peak_indexes=manual_peak_indexes[higher_peaks]
#            manual_peak_heights[higher_peaks]=xrd_1D_mean_counts[higher_peak_indexes]
#        manual_peaks_fwhm=    manual_peak_heights*1E-4              #This is just a first guess.  We will refine this later
#        #Now assemble the manual model spectrum by adding all the individual peaks together
#        manual_model_spectrum=np.zeros(len(xrd_1D_mean_counts))
#        for i in range(0,len(manual_peak_indexes)):
#            manual_model_spectrum=manual_model_spectrum + manual_peak_heights[i]/(1+((xrd_1D_d_spacing[manual_peak_indexes[i]]-xrd_1D_d_spacing)/(0.7*manual_peaks_fwhm[i]/2))**2)  #This is a Lorentzian shaped peak
#        for i in range(0,num_images):
#            prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
#            np.savez(  filename[0:-4]+'/'+filename[0:-4]+f'_image_{i:04d}'+'.npz',xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing'],xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts'], xrd_1D_baseline_std=prev_data['xrd_1D_baseline_std'], autodetected_peak_indices=prev_data['autodetected_peak_indices'], all_autodetected_peak_indices=prev_data['all_autodetected_peak_indices'],autodetected_peak_heights=prev_data['autodetected_peak_heights'], autodetected_peaks_fwhm=prev_data['autodetected_peaks_fwhm'], autodetected_model_spectrum=prev_data['autodetected_model_spectrum'],manual_peak_indexes_local=prev_data['manual_peak_indexes_local'],manual_peak_heights_local=prev_data['manual_peak_heights_local'], manual_peak_indexes=prev_data['manual_peak_indexes'],manual_peak_heights=manual_peak_heights, manual_peaks_fwhm=manual_peaks_fwhm, manual_model_spectrum=manual_model_spectrum,known_materials_names=prev_data['known_materials_names'], known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes'],known_materials_heights=prev_data['known_materials_heights'] )
#



def plot_1D_xrd_and_save_plots_all_images_in_tif_file(filename,xlim=[],ylim=[]): 
    if (len(filename) > 0) and (type(filename) != str):
        filename = 'scan2D_'+str(filename)+'.tif'       
    num_images=num_images_in_tif_file(filename)
    if os.path.exists(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') == True:
        num_images=num_images_in_tif_file(filename)
        prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') # prev_data stands for previous_data
        known_materials_names=prev_data['known_materials_names']
        if len(ylim)==0:
            ylim=[-len(known_materials_names)-5,0]
            for i in range(0,int(num_images)): 
                prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
                ylim[1]=np.max([ylim[1],np.max(prev_data['xrd_1D_mean_counts'])])
        for i in range(0,int(num_images)): 
            prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
            xrd_1D_d_spacing=     prev_data['xrd_1D_d_spacing']
            xrd_1D_mean_counts=   prev_data['xrd_1D_mean_counts']
            xrd_1D_baseline_std=           prev_data['xrd_1D_baseline_std']
            manual_peak_indexes_local=  prev_data['manual_peak_indexes_local'] 
            manual_peak_heights=  prev_data['manual_peak_heights'] 
            manual_peaks_fwhm=    prev_data['manual_peaks_fwhm']
            manual_model_spectrum=prev_data['manual_model_spectrum']
            plot_figure_han, plot_axis_han, line_handle, legend_figure_han = plot_1D_XRD_single_spectrum(filename,i,xlim=xlim,ylim=ylim)
            plot_axis_han.title.set_text('Scan:'+filename[-9:-4]+'  Location:'+f'{i:04d}')
            plot_figure_han.savefig(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.png')
            legend_figure_han.savefig(filename[0:-4]+'/'+'legend.png')
            plt.close(fig=plot_figure_han)
            plt.close(fig=legend_figure_han)
    else:
        print('You need to run pyfai_all_images_in_tif_file first to create the 1D spectra')



#d_spacing_angstroms = d_spacing_angstroms - 0.0014 #comparing cif card copper peaks to the experimental data suggests this is wise
def plot_1D_XRD_single_spectrum(tiffilename,image_number=0,with_known_materials=True,xlim=[],ylim=[]):
    if type(tiffilename) != str:
        tiffilename = 'scan2D_'+str(tiffilename)+'.tif' 
    if type(image_number)!=str:
        if os.path.exists(tiffilename[0:-4]+'/'+tiffilename[0:-4]+'_image_'+f'{image_number:04d}'+'.npz') == True:
            prev_data=np.load(tiffilename[0:-4]+'/'+tiffilename[0:-4]+'_image_'+f'{image_number:04d}'+'.npz') # prev_data stands for previous_data
        else:
            print('You need to create the npz files first.')
    else:
        if os.path.exists(tiffilename[0:-4]+'/'+tiffilename[0:-4]+'_image_average.npz') == True:
            prev_data=np.load(tiffilename[0:-4]+'/'+tiffilename[0:-4]+'_image_average.npz') # prev_data stands for previous_data
        else:
            print('You need to create the average npz file first.') 
    xrd_1D_d_spacing=     prev_data['xrd_1D_d_spacing']
    xrd_1D_mean_counts=   prev_data['xrd_1D_mean_counts']
    xrd_1D_baseline_std=  prev_data['xrd_1D_baseline_std']
    manual_peak_indexes=  prev_data['manual_peak_indexes'] 
    manual_peak_indexes_local=  prev_data['manual_peak_indexes_local']
    known_materials_names=            prev_data['known_materials_names']
    known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes']
    #Create the figure and set it's properties
    plot_figure_han =   plt.figure()
    legend_figure_han = plt.figure()
    #plt.get_current_fig_manager().window.setGeometry(0,649,645,430)
    plot_figure_han.tight_layout()
    legend_figure_han.tight_layout()
    plot_axis_han=plot_figure_han.add_subplot(1,1,1)
    legend_axis_han=legend_figure_han.add_subplot(1,1,1)
    line_handle=plot_axis_han.plot(xrd_1D_d_spacing,xrd_1D_mean_counts,'k')
    plot_axis_han.set_xlabel(r"d spacing  (${\rm \AA}$)")
    plot_axis_han.set_ylabel('xrd counts')
    if len(ylim)!=0: plot_axis_han.set_ylim(ylim)
    if len(xlim)!=0: plot_axis_han.set_xlim(xlim)
    legend_axis_han.set_ylim((0,1))
    legend_axis_han.set_xlim((0,1))
    #Calculate the qualitative polycrystallinity (i.e. if the 2D XRD image had a Bragg Ring or Laue Spots)
    polycrystallinity=return_polycrystallinity(xrd_1D_mean_counts,xrd_1D_baseline_std)
    plot_axis2_han = plot_axis_han.twinx()
    plot_axis2_han.scatter(xrd_1D_d_spacing[manual_peak_indexes],polycrystallinity[manual_peak_indexes],marker='_',color='k',alpha=0.5)
    plot_axis2_han.set_ylabel("polycrystalline  |  monocrystalline -->                     ")
    plot_axis2_han.set_ylim((-1,8))
    plot_axis2_han.set_yticks([0,2,4,6,8])
    #Plot the manual peak positions
    if manual_peak_indexes.size>0: 
        for m in range(0,len(manual_peak_indexes)):
            plot_axis_han.plot([xrd_1D_d_spacing[manual_peak_indexes[m]],xrd_1D_d_spacing[manual_peak_indexes[m]]],[0,-1] ,linewidth=1,color='r',label='manual id peaks')
    legend_line_han=legend_axis_han.plot([0.1,0.4],[1-1/(len(known_materials_names)+2),1-1/(len(known_materials_names)+2)],color='r')
    legend_axis_han.text(0.4,1-1/(len(known_materials_names)+2), 'manually id\'d peaks')
    # Plot the manual peak positions for this particular local image 
    if manual_peak_indexes_local.size>0: 
        for m in range(0,len(manual_peak_indexes_local)):
            plot_axis_han.plot([xrd_1D_d_spacing[np.int16(manual_peak_indexes_local[m])],xrd_1D_d_spacing[np.int16(manual_peak_indexes_local[m])]],[0,-1] ,linewidth=1,color='b',label='manual id peaks')
    legend_line_han=legend_axis_han.plot([0.1,0.4],[1-2/(len(known_materials_names)+2),1-2/(len(known_materials_names)+2)],color='b')
    legend_axis_han.text(0.4,1-2/(len(known_materials_names)+2), 'manual local peaks')
    #Mark locations with known material peaks
    for i in range(1,len(known_materials_names)):
        #Plot just the first tick because you want to grab the color, then use the same color for all the ticks of that same material
        tick_handle=plot_axis_han.plot([xrd_1D_d_spacing[known_materials_d_spacing_indexes[i,1]],xrd_1D_d_spacing[known_materials_d_spacing_indexes[i,1]]],[-i,-i-1] ,linewidth=1,label='manual id peaks')
        legend_line_han=legend_axis_han.plot([0.1,0.4],[1-(i+2)/(len(known_materials_names)+2),1-(i+2)/(len(known_materials_names)+2)])
        tick_color=legend_line_han[0].get_color()
        legend_axis_han.text(0.4,1-(i+2)/(len(known_materials_names)+2), known_materials_names[i])
        for m in range(0,len(known_materials_d_spacing_indexes[i,:])):
            plot_axis_han.plot([xrd_1D_d_spacing[known_materials_d_spacing_indexes[i,m]],xrd_1D_d_spacing[known_materials_d_spacing_indexes[i,m]]],[-i,-i-1] ,color=tick_color,linewidth=1,label='manual id peaks')
    #plt.scatter(d_spacing_zinc.values[:,3],50*np.ones(len(d_spacing_zinc.values[:,3])))
    return plot_figure_han, plot_axis_han, line_handle, legend_figure_han
        
        
    
def convert_all_npz_to_XY_files(tif_data_foldername,mode='d spacing'):
    if type(tif_data_foldername) != str:
        tif_data_foldername = 'scan2D_'+str(tif_data_foldername)
    object_list_allfilesdirectories = pathlib.Path(tif_data_foldername)
    object_recursiveglob_npz_files = object_list_allfilesdirectories.glob('*.npz')      
    object_recursiveglob_npz_files = list(object_recursiveglob_npz_files)
    num_npz_files=len(object_recursiveglob_npz_files)
    for i in range(0,int(num_npz_files)):
        convert_single_spectrum_to_XY_file(object_recursiveglob_npz_files[i].as_posix())

        

def convert_single_spectrum_to_XY_file(npz_filename,image=[],mode='d spacing'):  #use image='manual_model' to convert the manual model
    if type(npz_filename) != str:
        npz_filename = 'scan2D_'+str(npz_filename)+'/'+'scan2D_'+str(npz_filename)+'_image_'+f'{i:04d}'+'.npz'
    if os.path.exists(npz_filename) == True:
        prev_data=np.load(npz_filename) # prev_data stands for previous_data
        print('Converting files in folder: '+npz_filename[0:13])
        if mode=='CuKalpha':
            charge_of_an_electron=1.602E-19 #Coulombs
            Cu_Kalpha_beam_energy_ev=8040
            beam_energy_joules=charge_of_an_electron*Cu_Kalpha_beam_energy_ev  #Joules
            planks_h=6.626E-34 #Joule second
            speed_of_light=299792458 #m/s
            beam_lambda=planks_h*speed_of_light/beam_energy_joules
            Cu_Kalpha_two_theta=2*360/(2*np.pi)*np.arcsin(beam_lambda/(2*prev_data['xrd_1D_d_spacing']*1E-10))
            with open(npz_filename[0:-4]+'_CuKalpha_2theta.asc','w+') as asc_file:
                for i in range(0,len(prev_data['xrd_1D_d_spacing'])):
                    print(str(Cu_Kalpha_two_theta[i])+" "+str(prev_data['xrd_1D_mean_counts'][i]),file=asc_file) 
        if mode=='d spacing':
            with open(npz_filename[13:-4]+'.asc','w+') as asc_file:
                for i in range(0,len(prev_data['xrd_1D_d_spacing'])):
                    print(str(prev_data['xrd_1D_d_spacing'][i])+" "+str(prev_data['xrd_1D_mean_counts'][i]),file=asc_file) 
    else:
        print('NPZ file doesn\'t exist')



def output_d_spacings_all_files_to_csv(filename):
    if type(filename) != str:
        filename = 'scan2D_'+str(filename)+'.tif'
    if os.path.exists(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') == True:
        num_spectra=num_images_in_tif_file(filename)
        for i in range(0,num_spectra):    
            prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
            print('Opening file: '+filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz')
            xrd_1D_d_spacing=prev_data['xrd_1D_d_spacing']
            xrd_1D_mean_counts=prev_data['xrd_1D_mean_counts']
            manual_peak_indexes_local=prev_data['manual_peak_indexes_local']
            d_spacing_of_manual_local_peaks=np.interp(manual_peak_indexes_local,np.array(range(0,len(xrd_1D_d_spacing))),xrd_1D_d_spacing)
            csv_string_data=''
            for j in range(0,len(d_spacing_of_manual_local_peaks)):
                csv_string_data=csv_string_data+str(d_spacing_of_manual_local_peaks[j])+','
            csv_string_data=csv_string_data[0:-1]
            with open('file.csv','a+') as csv_file:
                print(filename+',image:'+f'{i:04d}'+','+csv_string_data,file=csv_file)




#d_spacing_angstroms = d_spacing_angstroms - 0.0014 #comparing cif card copper peaks to the experimental data suggests this is wise
def plot_all_1D_XRD_spectra_while_loop(filename,with_known_materials=True,xlim=[],ylim=[]):
    global break_while_loop_switch
    if (len(filename) > 0) and (type(filename) != str):
        filename = 'scan2D_'+str(filename)+'.tif' 
    num_spectra=num_images_in_tif_file(filename)
    if os.path.exists(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') == True:
        prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_0000.npz') # prev_data stands for previous_data
        xrd_1D_d_spacing=     prev_data['xrd_1D_d_spacing']
        xrd_1D_mean_counts=   prev_data['xrd_1D_mean_counts']
        xrd_1D_baseline_std=           prev_data['xrd_1D_baseline_std']
        manual_peak_indexes=     prev_data['manual_peak_indexes'] 
        #Create the figure and set it's properties
        plot_figure_han =   plt.figure()
        plot_figure_han.canvas.mpl_connect('close_event', break_while_loop)
        legend_figure_han = plt.figure()
        #plt.get_current_fig_manager().window.setGeometry(0,649,645,430)
        plot_figure_han.tight_layout()
        legend_figure_han.tight_layout()
        plot_axis_han=plot_figure_han.add_subplot(1,1,1)
        legend_axis_han=legend_figure_han.add_subplot(1,1,1)
        line_handle=plot_axis_han.plot(xrd_1D_d_spacing,xrd_1D_mean_counts,'k')
        plot_axis_han.set_xlabel(r"d spacing  (${\rm \AA}$)")
        plot_axis_han.set_ylabel('xrd counts')
        if xlim!=[]: plot_axis_han.set_ylim(ylim)
        if xlim!=[]: plot_axis_han.set_xlim(xlim)
        legend_axis_han.set_ylim((0,1))
        legend_axis_han.set_xlim((0,1))
        #Calculate the qualitative polycrystallinity (i.e. if the 2D XRD image had a Bragg Ring or Laue Spots)
        polycrystallinity=return_polycrystallinity(xrd_1D_mean_counts,xrd_1D_baseline_std)
        plot_axis2_han = plot_axis_han.twinx()
        plot_axis2_han.scatter(xrd_1D_d_spacing[manual_peak_indexes],polycrystallinity[manual_peak_indexes],marker='_',color='k',alpha=0.5)
        plot_axis2_han.set_ylabel("polycrystalline  |  monocrystalline -->                     ")
        plot_axis2_han.set_ylim((-1,8))
        plot_axis2_han.set_yticks([0,2,4,6,8])
        #Plot the manual peak positions
        if manual_peak_indexes.size>0: 
            for m in range(0,len(manual_peak_indexes)):
                plot_axis_han.plot([xrd_1D_d_spacing[manual_peak_indexes[m]],xrd_1D_d_spacing[manual_peak_indexes[m]]],[0,-1] ,linewidth=1,color='r',label='manual id peaks')
        #If you've got the data, mark locations with known material peaks
        known_materials_names=            prev_data['known_materials_names']
        known_materials_d_spacing_indexes=prev_data['known_materials_d_spacing_indexes']
        #known_materials_heights=          prev_data['known_materials_heights']
        for i in range(1,len(known_materials_names)):
            #Plot just the first tick because you want to grab the color, then use the same color for all the ticks of that same material
            tick_handle=plot_axis_han.plot([xrd_1D_d_spacing[known_materials_d_spacing_indexes[i,1]],xrd_1D_d_spacing[known_materials_d_spacing_indexes[i,1]]],[-i,-i-1] ,linewidth=1,label='manual id peaks')
            legend_line_han=legend_axis_han.plot([0.1,0.4],[i/len(known_materials_names),i/len(known_materials_names)])
            tick_color=legend_line_han[i-1].get_color()
            legend_axis_han.text(0.4,i/len(known_materials_names), known_materials_names[i])
            for m in range(1,len(known_materials_d_spacing_indexes[i,:])):
                plot_axis_han.plot([xrd_1D_d_spacing[known_materials_d_spacing_indexes[i,m]],xrd_1D_d_spacing[known_materials_d_spacing_indexes[i,m]]],[-i,-i-1] ,color=tick_color,linewidth=1,label='manual id peaks')
        #plt.scatter(d_spacing_zinc.values[:,3],50*np.ones(len(d_spacing_zinc.values[:,3])))
        plot_figure_han.show()
        
        print('Enter q to quit:')
        i=0
        while break_while_loop_switch==True:
            prev_data=np.load(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.npz') # prev_data stands for previous_data
            xrd_1D_d_spacing=           prev_data['xrd_1D_d_spacing']
            xrd_1D_mean_counts=         prev_data['xrd_1D_mean_counts']
            #Plot the XRD line
            plot_figure_han.show()
            line_handle=plot_axis_han.plot(xrd_1D_d_spacing,xrd_1D_mean_counts,'k')  
            
            plt.title('image '+str(i)+' of '+str(num_spectra))
            plt.pause(.3)
            l=line_handle.pop(0)
            l.remove()
            del l
            i=i+1
            if i>=num_spectra: i=0
    else:
        print('You need to create the npz files first.')
    break_while_loop_switch=True


   
def calculate_qualitative_amounts_of_known_materials_spatial_distribution(filename):
    if (len(filename) > 0) and (type(filename) != str):
        filename = 'scan2D_'+str(filename)+'.tif' 
    



#            plot_axis2_han = plot_axis_han.twiny()
#            plot_axis_han.xaxis.set_label_position('top')
#            plot_axis_han.xaxis.tick_top()
#            plot_axis2_han.set_xlim(xlim)
#            plot_axis2_han.set_xticks(np.linspace(xlim[0],xlim[1],10))
#            plot_axis2_han.set_xlabel('hi')
#            plot_axis2_han.xaxis.set_label_position('bottom')
#            plot_axis2_han.xaxis.tick_bottom()
            #plot_axis2_han.set_xlim(plot_axis_han.get_xlim())
            #line_handle2=plot_axis2_han.plot(xrd_1D_d_spacing,xrd_1D_mean_counts,'k')
            #if ylim!=[]: plot_axis2_han.set_ylim(ylim)
            #if xlim!=[]: plot_axis2_han.set_xlim(xlim)
            #scatter_handle=plot_axis_han.scatter(xrd_1D_d_spacing[peak_indices],xrd_1D_mean_counts[peak_indices])
            #model_handle=plot_axis_han.plot(xrd_1D_d_spacing,model_spectrum,'g')
            #If you've got the data, calculate the qualitative polycrystallinity (i.e. if the 2D XRD image had a Bragg Ring or Laue Spots)
            #polycrystallinity=return_polycrystallinity(xrd_1D_mean_counts,xrd_1D_baseline_std)
            #plot_axis2_han = plot_axis_han.twinx()
            #plot_axis2_han.scatter(xrd_1D_d_spacing[peak_indices],polycrystallinity[peak_indices],marker='_',color='k',alpha=0.5)
            #plot_axis2_han.set_ylabel("polycrystalline  |  monocrystalline -->                     ")
            #plot_axis2_han.set_ylim((0,8))
            #plt.savefig(filename[0:-4]+'/'+filename[0:-4]+'_image_'+f'{i:04d}'+'.jpg',dpi=500,quality=95)
            #plt.close(fig=plot_figure_han)

def return_d_spacing_and_peak_heights_from_txt_file(known_material_filename):
    if sys.platform == 'win32':
        xrd_data_folder=''
        location_of_forward_slash=known_material_filename[::-1].find('\\')
    if sys.platform == 'linux':
        xrd_data_folder=''
        location_of_forward_slash=known_material_filename[::-1].find('/')
    if sys.platform == 'darwin':
        xrd_data_folder='/Users/damon/Desktop/BACKED_UP/WorkFiles/OthersScienceLiterature/X-ray_Diffraction_Data_Files/'
        location_of_forward_slash=known_material_filename[::-1].find('/') 
    abbreviated_name=known_material_filename[-location_of_forward_slash:-4]    
    cif_card_data=pd.read_csv(xrd_data_folder + known_material_filename,delim_whitespace=True);         
    d_spacings=       cif_card_data.values[:,3];
    peak_heights=     cif_card_data.values[:,8];
    d_spacings_abbreviated=np.array([])
    peak_heights_abbreviated=np.array([])
    for i in range(0,len(d_spacings)):
        if len(np.where(d_spacings_abbreviated==d_spacings[i])[0])==0:
            d_spacings_abbreviated=np.append(d_spacings_abbreviated,d_spacings[i])
            peak_heights_abbreviated=np.append(peak_heights_abbreviated,peak_heights[i])
        else:
            index=np.where(d_spacings_abbreviated==d_spacings[i])[0][0]
            peak_heights_abbreviated[index]=peak_heights_abbreviated[index]+peak_heights[i]
    return(abbreviated_name,d_spacings_abbreviated,peak_heights_abbreviated)

    
 

#non_peak_std_estimate=1.4/(1+np.exp(0.01*(np.arange(0,3072)-250)))+0.3
def find_and_model_peaks_single_spectrum(xrd_1D_d_spacing,xrd_1D_mean_counts,non_peak_std_estimate=np.ones(0),window_average_size=1):  #BASELINE MUST BE ALREADY REMOVED!!
    # Setup information on the size of background noise
    if non_peak_std_estimate.size==0:
        non_peak_std_estimate=1.4/(1+np.exp(0.01*(np.arange(0,3072)-250)))+0.3 
    # Smooth the data
    smoothed_xrd_1D_mean_counts=window_average(xrd_1D_mean_counts, window_average_size)
    # Find the peaks that are at least 3 times higher than the background std noise
    peak_indices=peakutils.indexes(smoothed_xrd_1D_mean_counts,3*non_peak_std_estimate[int(len(xrd_1D_d_spacing)/2)],min_dist=3,thres_abs=True).astype(int)
    #Get rid of peaks near the boundaries
    peak_indices=np.delete(peak_indices,np.where(peak_indices<15))
    peak_indices=np.delete(peak_indices,np.where(peak_indices>len(xrd_1D_d_spacing)-15))
#    # Test each peak for it's second derivative, from http://web.media.mit.edu/~crtaylor/calculator.html
#    second_derivative=np.zeros(len(peak_indices))
#    f=smoothed_xrd_1D_mean_counts
#    h=1
#    for j in range(0,len(peak_indices)):
#        i=peak_indices[j]
#        second_derivative[j]=(-31752*f[i-10]+784000*f[i-9]-9426375*f[i-8]+73872000*f[i-7]-427329000*f[i-6]+1969132032*f[i-5]-7691922000*f[i-4]+27349056000*f[i-3]-99994986000*f[i-2]+533306592000*f[i-1]-909151481810*f[i+0]+533306592000*f[i+1]-99994986000*f[i+2]+27349056000*f[i+3]-7691922000*f[i+4]+1969132032*f[i+5]-427329000*f[i+6]+73872000*f[i+7]-9426375*f[i+8]+784000*f[i+9]-31752*f[i+10])/(293318625600*1.0*h**2)
#    peak_indices=np.delete(peak_indices,np.where(second_derivative>-1))
    # Calculate how high the signal is above background noise
    significance=np.zeros(len(xrd_1D_mean_counts))
    for i in range(3,len(xrd_1D_mean_counts)-3):
        significance[i]=xrd_1D_mean_counts[i]/non_peak_std_estimate[i]   #The scatter of the baseline is the std of the background measurement
    # Delete indices that are not high enough above the background noise (aka background std)
    for i in range(0,len(peak_indices)):
        if significance[peak_indices[i]]<3:
            peak_indices[i]=0
#    peak_indices=np.delete(peak_indices,np.where(peak_indices==0))
    # Now create a model spectrum using ideal Lorentzian peaks
    model_spectrum=np.zeros(len(xrd_1D_mean_counts))
    peaks_fwhm=xrd_1D_mean_counts[peak_indices]*1E-3
    for i in range(0,len(peak_indices)):
        #Below we use a Lorentzian shaped peak to model each peak
        model_spectrum=model_spectrum + (xrd_1D_mean_counts[peak_indices[i]]-non_peak_std_estimate*2)/(1+((xrd_1D_d_spacing[peak_indices[i]]-xrd_1D_d_spacing)/(0.7*peaks_fwhm[i]/2))**2) #This is a Lorentzian shaped peak
    model_spectrum = model_spectrum + non_peak_std_estimate*2
    #Now re-assess the significance with the new information you have about neighboring peaks.
    peak_heights=np.zeros(len(peak_indices))
#    model_spectrum=np.zeros(len(xrd_1D_mean_counts))
#    for i in range(0,len(peak_indices)):
#        model_spectrum=model_spectrum + peak_heights[i]/(1+((xrd_1D_d_spacing[peak_indices[i]]-xrd_1D_d_spacing)/(0.7*peaks_fwhm[i]/2))**2)  #This is a Lorentzian shaped peak
#    model_spectrum = model_spectrum + non_peak_std_estimate*2
    return peak_indices, peak_heights, peaks_fwhm, model_spectrum;

    


#    peaks_fwhm_left= np.zeros(len(peak_indices))
#    peaks_fwhm_right=np.zeros(len(peak_indices))
#    peaks_fwhm=      np.zeros(len(peak_indices))
#    for i in range(0,len(peak_indices)):
#        height_threshold=1.0
#        j=peak_indices[i]
#        while (height_threshold > 0.5) & (j<3071):
#            height_threshold=xrd_1D_mean_counts[j]/xrd_1D_mean_counts[peak_indices[i]]
#            j=j+1
#        peaks_fwhm_left[i]=(xrd_1D_d_spacing[peak_indices[i]] - xrd_1D_d_spacing[j])
#        height_threshold=1.0
#        j=peak_indices[i]
#        while (height_threshold > 0.5) & (j>1):
#            height_threshold=xrd_1D_mean_counts[j]/xrd_1D_mean_counts[peak_indices[i]]
#            j=j-1
#        peaks_fwhm_right[i]=(xrd_1D_d_spacing[j] - xrd_1D_d_spacing[peak_indices[i]])
#        peaks_fwhm[i]=peaks_fwhm_right[i] + peaks_fwhm_left[i]
#        if peaks_fwhm[i]/xrd_1D_mean_counts[peak_indices[i]] > 1E-3: 
#            peaks_fwhm[i]=xrd_1D_mean_counts[peak_indices[i]]*1E-3
#        #Below we use a Lorentzian shaped peak to model each peak
#        model_spectrum=model_spectrum + (xrd_1D_mean_counts[peak_indices[i]]-non_peak_std_estimate*2)/(1+((xrd_1D_d_spacing[peak_indices[i]]-xrd_1D_d_spacing)/(0.7*peaks_fwhm[i]/2))**2) #This is a Lorentzian shaped peak
#    model_spectrum = model_spectrum + non_peak_std_estimate*2
#    #Now re-assess the significance with the new information you have about neighboring peaks.
#    peak_heights=np.zeros(len(peak_indices))
##    for i in range(0,len(peak_indices)):
##        #Below we use a Lorentzian shaped peak to model each peak
##        test_model=model_spectrum -  (xrd_1D_mean_counts[peak_indices[i]]-non_peak_std_estimate*2)/(1+((xrd_1D_d_spacing[peak_indices[i]]-xrd_1D_d_spacing)/(0.7*peaks_fwhm[i]/2))**2)
##        peak_heights[i]=xrd_1D_mean_counts[peak_indices[i]] - test_model[peak_indices[i]]
##        significance[peak_indices[i]] = (peak_heights[i] + non_peak_std_estimate[i]*2) /non_peak_std_estimate[peak_indices[i]]  #I don't really need to add the non_peak_std_estimate*2 but I do it anyway in order to stay consistent with the first time I calculate significance
##        if significance[peak_indices[i]]<10:   
##            peak_indices[i]=0
##    peaks_fwhm        =np.delete(peaks_fwhm,        np.where(peak_indices==0))
##    peak_heights =np.delete(peak_heights, np.where(peak_indices==0))
##    peak_indices=np.delete(peak_indices,np.where(peak_indices==0))
#    #Reconstruct the spectrum -- some of the peaks have been deleted  


    
    
def metadata_from_tif_file(filename):
    #grab the tif image data
    if (len(filename) > 0) and (type(filename) != str):
        f = open('scan2D_'+str(filename)+'.tif', 'rb')
    else:
        f = open(filename, 'rb')
    tif_dict= exifread.process_file(f)
    print(tif_dict["Image StripOffsets"].values)
    f.close()
    return tif_dict



def imshow_and_histogram(tif_filename,image_number=0,background_image=[],vmin=0.1,vmax=0.1):
    if type(tif_filename) != str:
        tif_filename = 'scan2D_'+str(tif_filename)+'.tif' 
    image_data=image_from_tif_file(tif_filename,image_number)
    if background_image != []:
        if type(background_image) != str:
            background_image = 'scan2D_'+str(background_image)+'.tif' 
        background_image_data=image_from_tif_file(background_image)
        image_data=image_data-background_image_data
    fig_han_imshow=plt.figure()  # you can find the size and position of a plot with plt.get_current_fig_manager().window.geometry().getRect()
    plt.get_current_fig_manager().window.setGeometry(0, 57, 649, 557)
    fig_han_imshow.tight_layout()
    if vmin==0.1:
        vmin=np.min(image_data[:])
    if vmax==0.1:
        vmax=np.max(image_data[:])
    plt.imshow(image_data,cmap='gray',vmin=vmin,vmax=vmax)
    fig_han_hist=plt.figure()    # you can find the size and position of a plot with plt.get_current_fig_manager().window.geometry().getRect()
    plt.get_current_fig_manager().window.setGeometry(0,649,645,430)
    fig_han_hist.tight_layout()
    axis_han_hist=fig_han_hist.add_subplot(1,1,1)
    plot_han_hist=axis_han_hist.hist(np.reshape(image_data,(1944*3072,1)),bins=1000,)
    axis_han_hist.set_xlim(0)
    axis_han_hist.set_xlabel('xrd counts')
    axis_han_hist.set_ylabel('histogram counts')
    
    



    
    
def image_from_tif_file(filename,image_number=0):
    #grab the tif image data
    if (len(filename) > 0) and (type(filename) != str):
        f = open('scan2D_'+str(filename)+'.tif', 'rb')
    else:
        f = open(filename, 'rb')
    tif_dict= exifread.process_file(f)
    num_images=num_images_in_tif_file(filename)
    if num_images > len(tif_dict["Image StripOffsets"].values) and image_number>1:
        strip_offset=tif_dict['IFD '+str(image_number)+' StripOffsets'].values
        f.seek(int(strip_offset[0]))
        image_byte_counts=tif_dict['IFD '+str(image_number)+' StripByteCounts'].values
        imagedata = np.fromfile(f,count=int(image_byte_counts[0]/2),dtype=np.dtype('uint16'))
    if num_images > len(tif_dict["Image StripOffsets"].values) and image_number==1:
        strip_offset=tif_dict['Thumbnail StripOffsets'].values
        f.seek(int(strip_offset[0]))
        image_byte_counts=tif_dict['Thumbnail StripByteCounts'].values
        imagedata = np.fromfile(f,count=int(image_byte_counts[0]/2),dtype=np.dtype('uint16'))
    if num_images == len(tif_dict["Image StripOffsets"].values) or image_number==0:
        f.seek(tif_dict["Image StripOffsets"].values[image_number])                              #/2 because uint16 is 2 bytes
        imagedata = np.fromfile(f,count=int(tif_dict["Image StripByteCounts"].values[image_number]/2),dtype=np.dtype('uint16'))
    imagedata = np.int16(np.reshape(imagedata,(1944,3072)))
    f.close()
    return imagedata





def average_all_images_in_a_tif_file(filename):
    if (len(filename) > 0) and (type(filename) != str):
        filename = 'scan2D_'+str(filename)+'.tif' 
    num_images=num_images_in_tif_file(filename)
    image=image_from_tif_file(filename)*0.0;
    for i in range(0,num_images):
        image=image+image_from_tif_file(filename)
    image=np.int16(image/num_images)
    image_for_tif_file=np.int16(np.zeros((1,image.shape[0],image.shape[1])))
    image_for_tif_file[0,:,:]=image*1.0
    tifffile.imsave('avescan'+filename[-9:-4]+'.tif',image_for_tif_file)
    
    
    
def window_average(input_array,window_size):  #window_size must be an odd number
    if np.mod(window_size,2)==0:
        print('You must use an odd number for window_size otherwise the window averaging will shift the spectrum left or right')
        return(1)
    output=np.convolve(input_array, np.ones((window_size,))/window_size, mode='same')
    return output



def list_all_tif_files():
    object_list_allfilesdirectories = pathlib.Path('.')
    object_recursiveglob_tiffiles = object_list_allfilesdirectories.glob('*.tif')  
    list_of_filename_objects = list(object_recursiveglob_tiffiles)
    list_filenames_tiffiles=[]
    for i in range(0,len(list_of_filename_objects)):
        list_filenames_tiffiles.append(list_of_filename_objects[i].as_posix())
    return list_filenames_tiffiles



def list_all_folders():
    object_list_allfilesdirectories = pathlib.Path('.')
    object_recursiveglob_all = object_list_allfilesdirectories.glob('*')  
    list_of_all_objects = list(object_recursiveglob_all)
    list_of_folders=[]
    for i in range(0,len(list_of_all_objects)):
        if list_of_all_objects[i].is_dir()==True:
            list_of_folders.append(list_of_all_objects[i].as_posix())
    return list_of_folders




def num_images_in_tif_file(filename):
    if (len(filename) > 0) and (type(filename) != str):
        f = open('scan2D_'+str(filename)+'.tif', 'rb')
    else:
        f = open(filename, 'rb')
    tif_dict= exifread.process_file(f)
    description=tif_dict['Image ImageDescription'].values
    length_description=len(description)
    tif_dimensions=np.fromstring(description[11:length_description-2],sep=',')
    f.close()
    
    if np.size(tif_dimensions)==0:
        num_images=1
    else:
        num_images=tif_dimensions[0]
    return np.int(num_images)

        


        
def pyfai_single_raw_image(raw_image,background_image=[]):
    pyfai_object = pyFAI.load(pyfai_poni_file)  # IF THIS COMMAND FAILS, TRY TYPING export PYOPENCL_NO_CACHE=1 INTO A TERMINAL BEFORE LAUNCHING PYTHON OR SPYDER
    warped_xrd_object = pyfai_object.integrate2d(raw_image,raw_image.shape[1],raw_image.shape[0],unit='q_A^-1')
    warped_xrd_image=warped_xrd_object.intensity
    print('IGNORE PYFAI WARNING ABOUT INTEGRATION METHOD!')
    if len(background_image) != 0:
        warped_background_object = pyfai_object.integrate2d(background_image,raw_image.shape[1],raw_image.shape[0],unit='q_A^-1',method='')
        warped_background_image=warped_background_object.intensity
        warped_xrd_image=warped_xrd_image-warped_background_image
    warped_xrd_image[warped_xrd_image>np.mean(warped_xrd_image)+20*np.std(warped_xrd_image)]=0.001
    warped_xrd_image[warped_xrd_image<np.mean(warped_xrd_image)-20*np.std(warped_xrd_image)]=0.001
    warped_xrd_image[warped_xrd_image==0.00000]='nan' 
    xrd_1D_mean_counts  =np.nanmean(warped_xrd_image,axis=0);
    xrd_1D_mean_counts  =np.nan_to_num(xrd_1D_mean_counts)
    temp = warped_xrd_image
    with np.errstate(invalid='ignore'): temp[temp<5]='nan'
    xrd_1D_baseline_std   =np.nanstd(temp,axis=0);             #If the slice is ALL NANs then a warning is printed RuntimeWarning: Degrees of freedom <= 0 for slice. Just ignore this warning. It works. It ain't broke, don't fix it.
    xrd_1D_baseline_std   =np.nan_to_num(xrd_1D_baseline_std)  #If the slice is ALL NANs then a warning is printed RuntimeWarning: Degrees of freedom <= 0 for slice. Just ignore this warning. It works. It ain't broke, don't fix it.
    xrd_1D_x_axis=warped_xrd_object.radial
    xrd_1D_d_spacing=2*3.14159/xrd_1D_x_axis
    return xrd_1D_mean_counts, xrd_1D_baseline_std, xrd_1D_d_spacing, warped_xrd_image




def pyfai_single_image_in_tif_file(filename,background_filename=[],image_number=0):
    if (len(filename) > 0) and (type(filename) != str):
        filename = 'scan2D_'+str(filename)+'.tif'
    raw_image=image_from_tif_file(filename,image_number)
    if len(background_filename) != 0:
        if type(background_filename) != str:
            background_filename = 'scan2D_'+str(background_filename)+'.tif'   
        background_image=image_from_tif_file(background_filename,0)
        raw_image=raw_image-background_image
        raw_image[raw_image>np.mean(raw_image)+12*np.std(raw_image)]=np.int16(0.)
        raw_image[raw_image<0]=np.int16(0.)
    xrd_1D_mean_counts, xrd_1D_baseline_std, xrd_1D_d_spacing, warped_xrd_image = pyfai_single_raw_image(raw_image)
    #Now remove the baseline
    xrd_1D_mean_counts = remove_baseline(xrd_1D_mean_counts)
    return xrd_1D_mean_counts, xrd_1D_baseline_std, xrd_1D_d_spacing, warped_xrd_image
  
    

def remove_baseline(y):
    #Snip off the ends because data quality is so poor there
    y[0:5]=np.mean(y[0:8])
    y[-4:]=np.mean(y[-7:])
    #Use rubberband correction of concavity
    y=y-baseline_rubberband(y)
    yy=y*1.0
    yy[yy>5]=5
    y=y-peakutils.baseline(yy,8)*np.arange(1,0,-1/len(y))
    y[y<0]=0.5
    yy=y*1.0
    yy[yy>5]=5
    y=y-np.flip(peakutils.baseline(np.flip(yy),8))*np.arange(1,0,-1/len(y))
    y[y<0]=0.5
    yy=y*1.0
    yy[yy>5]=5
    y=y-peakutils.baseline(yy,9)*np.arange(1,0,-1/len(y))
    y[y<0]=0.5
    yy=y*1.0
    yy[yy>5]=5
    y=y-np.flip(peakutils.baseline(np.flip(yy),9))*np.arange(1,0,-1/len(y))
    y[y<0]=0.5
    yy=y*1.0
    yy[yy>4]=4
    y=y-peakutils.baseline(yy,7)*np.arange(1,0,-1/len(y))
    y[y<0]=0.5
    yy=y*1.0
    yy[yy>4]=4
    y=y-np.flip(peakutils.baseline(np.flip(yy),7))*np.arange(1,0,-1/len(y))
    return y




def return_polycrystallinity(xrd_1D_mean_counts,xrd_1D_baseline_std):
    polycrystallinity=xrd_1D_baseline_std/xrd_1D_mean_counts
    return polycrystallinity



        
def return_xrd_counts_std_d_spacting_from_csv_file(filename):
    data=pd.read_csv(filename,delim_whitespace=False)
    counts_column   =np.where(data.columns=='counts')[0][0]
    x_axis_column =np.where(data.columns=='d A')[0][0]
    std_column    =np.where(data.columns=='std')[0][0]
    xrd_1D_d_spacing =data.values[:,x_axis_column]
    xrd_1D_mean_counts   =data.values[:,counts_column]
    xrd_1D_baseline_std    =data.values[:,std_column]
    return xrd_1D_mean_counts, xrd_1D_baseline_std, xrd_1D_d_spacing



def baseline_rubberband(y):
    # Find the convex hull
    x=np.arange(0,len(y))
    v = scipy.spatial.ConvexHull(np.stack((x,y),axis=1)).vertices
    # Rotate convex hull vertices until they start from the lowest one
    v = np.roll(v, -v.argmin())
    # Leave only the ascending part
    v = v[:v.argmax()]
    # Create baseline using linear interpolation between vertices
    return np.interp(x, x[v], y[v])



def baseline_als2(y, lam, p, niter=10):
  L = len(y)
  D = scipy.sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = scipy.sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = scipy.sparse.linalg.spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z


#def process_all_tif_files():
    #for i in range(0,len(object_list_filenames)):
        #batch_file_path_and_filename = str(object_list_filenames[i].parents[0].joinpath(object_list_filenames[i].parts[-1][0:-4]+'_datasqueeze.txt'))
        #filename=object_list_filenames[i].name[0:-4]
        
        

#def get_filenumber(filename):
#    for i in range(0,len(object_list_filenames_tiffiles)):
#        if object_list_filenames_tiffiles[i].name[:18] == filename[:18]:
#            return i


def baseline_als(y, lam, p, niter=10):  # I don't use this anymore
    #This is from an algorithm called "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005. The paper is free and you can find it on google.
    #two parameters: p for asymmetry and λ for smoothness. Both have to be tuned to the data at hand. 
    #We found that generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) and 10^2 ≤ λ ≤ 10^9 , but exceptions may occur.
    L = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = scipy.sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z




def calculate_xrd_compound_confidence(xrd_intensity,trial_peak_indices):
    #xrd_peaks=d_spacing_angstroms[peakutils.peak.indexes(x,thres=5, min_dist=100, thres_abs=True)]
    confidence=0
    
    for trial_peak_index in trial_peak_indices:
        print(trial_peak_index)
        maxheight=xrd_intensity[trial_peak_index]
        stop=False
        index_peaks_fwhm_right=trial_peak_index
        while stop==False:
            index_peaks_fwhm_right=index_peaks_fwhm_right+1
            if xrd_intensity[index_peaks_fwhm_right]<maxheight/2:
                stop=True
        stop=False
        index_peaks_fwhm_left=trial_peak_index
        while stop==False:
            index_peaks_fwhm_left=index_peaks_fwhm_left-1
            if xrd_intensity[index_peaks_fwhm_left]<maxheight/2:
                stop=True            
        
        confidence=confidence+xrd_intensity[trial_peak_index]/(xrd_intensity[index_peaks_fwhm_right]/2+xrd_intensity[index_peaks_fwhm_left]/2)
        print('hi')
    return confidence



def calculate_xrd_compound_presence(xrd_intensity,trial_peak_indices,peaks_expected_height):
    #xrd_peaks=d_spacing_angstroms[peakutils.peak.indexes(x,thres=5, min_dist=100, thres_abs=True)]
    presence=0
    for i in range(0,len(trial_peak_indices)):
        presence=presence+xrd_intensity[trial_peak_indices[i]]/2/(peaks_expected_height[i]/100)
    return presence

    
def twoD_xrd_plot():
    d_spacing_angstroms,xrd_intensity_baseline_subtracted=grab_xrd_1D_data(15)
    
    ##Zinc
    zinc_cif_file=pd.read_csv('/Users/damon/Desktop/BACKED_UP/WorkFiles/OthersScienceLiterature/X-ray_Diffraction_Data_Files/Copper-metallic/Copper_7101269.txt',delim_whitespace=True);         
    d_spacing_zinc=zinc_cif_file.values[:,3];
    peaks_expected_height_zinc = zinc_cif_file.values[d_spacing_zinc>1.0242,8];
    d_spacing_zinc = d_spacing_zinc[d_spacing_zinc>1.0242]; 
    d_spacing_zinc=d_spacing_zinc[peaks_expected_height_zinc>0.5] 
    peaks_expected_height_zinc=peaks_expected_height_zinc[peaks_expected_height_zinc>0.5]                                      
    d_indices_zinc=[np.abs(d_spacing_angstroms-d_spacing).argmin() for d_spacing in d_spacing_zinc];                          
    ##Bi2O3
    Bi2O3_cif_file=pd.read_csv('../xrd_d_spacings/alpha_Bi2O3_1526458.txt',delim_whitespace=True);                               
    d_spacing_Bi2O3=Bi2O3_cif_file.values[:,3];  
    peaks_expected_height_Bi2O3 =  Bi2O3_cif_file.values[d_spacing_Bi2O3>1.0242,8];  
    d_spacing_Bi2O3 = d_spacing_Bi2O3[d_spacing_Bi2O3>1.0242];                      
    d_spacing_Bi2O3=d_spacing_Bi2O3[peaks_expected_height_Bi2O3>0.5] 
    peaks_expected_height_Bi2O3=peaks_expected_height_Bi2O3[peaks_expected_height_Bi2O3>0.5]
    d_indices_Bi2O3=[np.abs(d_spacing_angstroms-d_spacing).argmin() for d_spacing in d_spacing_Bi2O3];
    ##Copper
    copper_cif_file         =pd.read_csv('../xrd_d_spacings/Copper_7101269.txt',delim_whitespace=True);         
    d_spacing_copper=copper_cif_file.values[:,3];  
    peaks_expected_height_copper = copper_cif_file.values[d_spacing_copper>1.0242,8];                      
    d_spacing_copper = d_spacing_copper[d_spacing_copper>1.0242];
    d_spacing_copper=d_spacing_copper[peaks_expected_height_copper>0.5] 
    peaks_expected_height_copper=peaks_expected_height_copper[peaks_expected_height_copper>0.5]
    d_indices_copper=[np.abs(d_spacing_angstroms-d_spacing).argmin() for d_spacing in d_spacing_copper];
    #Nickel
    nickel_cif_file         =pd.read_csv('../xrd_d_spacings/Ni-9012968.txt',delim_whitespace=True);             
    d_spacing_nickel=nickel_cif_file.values[:,3];  
    peaks_expected_height_nickel = nickel_cif_file.values[d_spacing_nickel>1.0242,8];                      
    d_spacing_nickel = d_spacing_nickel[d_spacing_nickel>1.0242];                                   
    d_spacing_nickel=d_spacing_nickel[peaks_expected_height_nickel>0.5] 
    peaks_expected_height_nickel=peaks_expected_height_nickel[peaks_expected_height_nickel>0.5]
    d_indices_nickel=[np.abs(d_spacing_angstroms-d_spacing).argmin() for d_spacing in d_spacing_nickel];
    ##MnOH2_pyrochroite
    MnOH2_pyrochroite_cif_file=pd.read_csv('../xrd_d_spacings/pyrochroite_1548809.txt',delim_whitespace=True);  
    d_spacing_MnOH2_pyrochroite=MnOH2_pyrochroite_cif_file.values[:,3];  
    peaks_expected_height_MnOH2_pyrochroite =MnOH2_pyrochroite_cif_file.values[d_spacing_MnOH2_pyrochroite>1.0242,8];
    d_spacing_MnOH2_pyrochroite = d_spacing_MnOH2_pyrochroite[d_spacing_MnOH2_pyrochroite>1.0242];  
    d_spacing_MnOH2_pyrochroite=d_spacing_MnOH2_pyrochroite[peaks_expected_height_MnOH2_pyrochroite>0.5] 
    peaks_expected_height_MnOH2_pyrochroite=peaks_expected_height_MnOH2_pyrochroite[peaks_expected_height_MnOH2_pyrochroite>0.5]
    d_indices_MnOH2_pyrochroite=[np.abs(d_spacing_angstroms-d_spacing).argmin() for d_spacing in d_spacing_MnOH2_pyrochroite];
    ##Mn3O4_hausmenite
    Mn3O4_hausmenite_cif_file=pd.read_csv('../xrd_d_spacings/hausmenite_1514115.txt',delim_whitespace=True);    
    d_spacing_Mn3O4_hausmenite=Mn3O4_hausmenite_cif_file.values[:,3];  
    peaks_expected_height_Mn3O4_hausmenite = Mn3O4_hausmenite_cif_file.values[d_spacing_Mn3O4_hausmenite>1.0242,8];  
    d_spacing_Mn3O4_hausmenite = d_spacing_Mn3O4_hausmenite[d_spacing_Mn3O4_hausmenite>1.0242];     
    d_spacing_Mn3O4_hausmenite=d_spacing_Mn3O4_hausmenite[peaks_expected_height_Mn3O4_hausmenite>0.5] 
    peaks_expected_height_Mn3O4_hausmenite=peaks_expected_height_Mn3O4_hausmenite[peaks_expected_height_Mn3O4_hausmenite>0.5]
    d_indices_Mn3O4_hausmenite=[np.abs(d_spacing_angstroms-d_spacing).argmin() for d_spacing in d_spacing_Mn3O4_hausmenite];
    ##MnO2_ramsdelite
    MnO2_ramsdelite_cif_file=pd.read_csv('../xrd_d_spacings/ramsdelite_1514109.txt',delim_whitespace=True);     
    d_spacing_MnO2_ramsdelite=MnO2_ramsdelite_cif_file.values[:,3];  
    peaks_expected_height_MnO2_ramsdelite = MnO2_ramsdelite_cif_file.values[d_spacing_MnO2_ramsdelite>1.0242,8];    
    d_spacing_MnO2_ramsdelite = d_spacing_MnO2_ramsdelite[d_spacing_MnO2_ramsdelite>1.0242];        
    d_spacing_MnO2_ramsdelite=d_spacing_MnO2_ramsdelite[peaks_expected_height_MnO2_ramsdelite>0.5] 
    peaks_expected_height_MnO2_ramsdelite=peaks_expected_height_MnO2_ramsdelite[peaks_expected_height_MnO2_ramsdelite>0.5]
    d_indices_MnO2_ramsdelite=[np.abs(d_spacing_angstroms-d_spacing).argmin() for d_spacing in d_spacing_MnO2_ramsdelite];
    ##MnO2_pyrolusite
    MnO2_pyrolusite_cif_file=pd.read_csv('../xrd_d_spacings/pyrolusite_1514117.txt',delim_whitespace=True);     
    d_spacing_MnO2_pyrolusite=MnO2_pyrolusite_cif_file.values[:,3];  
    peaks_expected_height_MnO2_pyrolusite =  MnO2_pyrolusite_cif_file.values[d_spacing_MnO2_pyrolusite>1.0242,8];    
    d_spacing_MnO2_pyrolusite = d_spacing_MnO2_pyrolusite[d_spacing_MnO2_pyrolusite>1.0242];
    d_spacing_MnO2_pyrolusite=d_spacing_MnO2_pyrolusite[peaks_expected_height_MnO2_pyrolusite>0.5] 
    peaks_expected_height_MnO2_pyrolusite=peaks_expected_height_MnO2_pyrolusite[peaks_expected_height_MnO2_pyrolusite>0.5]            
    d_indices_MnO2_pyrolusite=[np.abs(d_spacing_angstroms-d_spacing).argmin() for d_spacing in d_spacing_MnO2_pyrolusite];
    #_cif_file         =pd.read_csv('../xrd_d_spacings/Ni-9012968.txt',delim_whitespace=True);     d_spacing_=_cif_file.values[:,3];   d_spacing_ = d_spacing_[d_spacing_>1.0242]; 
    #_cif_file         =pd.read_csv('../xrd_d_spacings/Ni-9012968.txt',delim_whitespace=True);     d_spacing_=_cif_file.values[:,3];   d_spacing_ = d_spacing_[d_spacing_>1.0242]; 
    #_cif_file         =pd.read_csv('../xrd_d_spacings/Ni-9012968.txt',delim_whitespace=True);     d_spacing_=_cif_file.values[:,3];   d_spacing_ = d_spacing_[d_spacing_>1.0242]; 
    CNTs_cif_file=1 
    d_spacing_sillenite=1
    xlocs=(3810.0-40*2)+np.arange(0,60,2)
    ylocs=(20.053-30*0.002)+np.arange(0,80,2) 
    copper_presence=np.zeros((len(ylocs),len(xlocs)))
    nickel_presence=np.zeros((len(ylocs),len(xlocs)))
    MnOH2_pyrochroite_presence=np.zeros((len(ylocs),len(xlocs)))
    MnO2_ramsdelite_presence=np.zeros((len(ylocs),len(xlocs)))
    MnO2_pyrolusite_presence=np.zeros((len(ylocs),len(xlocs)))
    Mn3O4_hausmenite_presence=np.zeros((len(ylocs),len(xlocs)))
    CNTs_presence=np.zeros((len(ylocs),len(xlocs)))
    Bi2O3_presence=np.zeros((len(ylocs),len(xlocs)))
    for i in range(0,len(xlocs)):
        for j in range(0,len(ylocs)):
            #print(i*40+j,i,j)
            d_spacing_angstroms,xrd_intensity = grab_xrd_1D_data(i*40+j)
            copper_presence[j,i]=calculate_xrd_compound_presence(xrd_intensity,d_indices_copper,peaks_expected_height_copper)
            nickel_presence[j,i]=calculate_xrd_compound_presence(xrd_intensity,d_indices_nickel,peaks_expected_height_nickel)
            Bi2O3_presence[j,i] =calculate_xrd_compound_presence(xrd_intensity,d_indices_Bi2O3 ,peaks_expected_height_Bi2O3)
            MnOH2_pyrochroite_presence[j,i] =calculate_xrd_compound_presence(xrd_intensity,d_indices_MnOH2_pyrochroite ,peaks_expected_height_MnOH2_pyrochroite)
            MnO2_pyrolusite_presence[j,i] =calculate_xrd_compound_presence(xrd_intensity,d_indices_MnO2_pyrolusite ,peaks_expected_height_MnO2_pyrolusite)
            MnO2_ramsdelite_presence[j,i] =calculate_xrd_compound_presence(xrd_intensity,d_indices_MnO2_ramsdelite ,peaks_expected_height_MnO2_ramsdelite)
            Mn3O4_hausmenite_presence[j,i] =calculate_xrd_compound_presence(xrd_intensity,d_indices_Mn3O4_hausmenite ,peaks_expected_height_Mn3O4_hausmenite)
            
    plt.figure()
    plt.imshow(copper_presence)
    plt.figure()
    plt.imshow(nickel_presence)
    plt.figure()
    plt.imshow(Bi2O3_presence)
    plt.figure()
    plt.imshow(MnOH2_pyrochroite_presence)
    plt.figure()
    plt.imshow(Mn3O4_hausmenite_presence)
    plt.figure()
    plt.imshow(MnO2_ramsdelite_presence)
    plt.figure()
    plt.imshow(MnO2_pyrolusite_presence)    
    
    #plt.figure()
    #plt.imshow(nickel_presence)
    #plt.figure()
    #plt.imshow(Bi2O3_presence)    
    #colors = np.random.rand(N)
    #area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
    #plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    
    
def sqlite_to_pandas(sqlite_filename):
    db = sqlite3.connect(sqlite_filename)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        table.to_csv(table_name + '.csv', index_label='index')
    cursor.close()
    db.close()