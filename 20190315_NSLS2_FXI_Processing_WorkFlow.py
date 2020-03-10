# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:08:57 2019

"""

import numpy as np
import h5py
import time
import os
import skimage.io
import pandas


def add_IE_plot_below_movie():
    t0 = time.monotonic()
    
    os.chdir('/Users/damon/Desktop/BACKED_UP/WorkFiles/ProjectsGrants/2015_NYSERDA_Birnessite_Project/2019_NSLS2_Synchrotron_Work/20190315_FXI_TXM_Beamline_Work/Results')
    filename = 'xanes_scan2_id_13161.h5'
    
    h5object = h5py.File(filename, 'r')
    
    beam_energy = np.array(h5object['X_eng'])
    images = np.array(h5object['img_xanes'])
    dark_images = np.array(h5object['img_dark'])
    scan_time = np.array(h5object['scan_time'])
    datetime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(scan_times))
    notes = np.array(h5object['note'])


    # Save the files
    # Check if a folder exists
    new_folder = working_directory + filename[:-3] + '/'
    if (not os.path.isdir(new_folder)):
        try:
            os.mkdir(new_folder)
        except e as Exception:
            print('Error creating directory.')
            print(e)
            quit()
    
    # Save the files
    skimage.io.imsave(new_folder+'xanes.tif', I)
    
    print(f'{time.monotonic() - t0}')

    close all
    clear all
    A=xlsread('C:\Users\Maccor\Desktop\Damon\movie_collection\movie_20181129\Data_For_Matlab_1stRun.xlsx');
    images_dir='images_1stRun';
    fig_han=figure('Position',[1.0000    1.0000  935.2000  781.6000]);  %[x0 y0 deltax deltay ] [646.6000 46.6000 887.2000 735.2000] %[973.8000 46.6000 560.0000 460.8000]);#
    %ax3_han = axes('Position',[0.08 0.591 0.957 0.14]);
    ax1_han = axes('Position',[0.0723    0.06744    0.9002    0.2571]); %[x0 y0 deltax deltay ]
    A(:,1)=A(:,1)-0.5; %shift cell voltage to be w.r.t. Hg/HgO (use 0.5 if converting from a Bi2O3 counter)
    A(:,2)=A(:,2)*1000; 
    A(:,4)=-A(:,4)-0.1; %shift Sync voltage to be w.r.t. Hg/HgO (use 0.1 if converting from a Cu wire).  The Sync voltage is measured between reference electrode and working electrode.
    plot_han=plot(A(:,4),A(:,2),'r',A(:,1),A(:,2),'g');  %plot current vs voltage
    ylim([-20 50])
    xlim([-1.0 0.55])
    set(get(ax1_han, 'XLabel'), 'string', 'Voltage w.r.t. Hg/HgO (V)')
    set(get(ax1_han, 'YLabel'), 'string', 'Current (mA)')
    %set(get(ax1_han, 'YLabel'), 'Rotation', 0)
    %set(get(ax1_han, 'YLabel'), 'Position', [-2.5 1.5 1])
    set(ax1_han,'XColor',[0 0 0])
    set(ax1_han,'YColor',[0 0 0])
    
    hold on
    ax2_han = axes('Position',[0.11 0.33 0.85 0.68],'Visible','off');%[x0 y0 deltax deltay ]
    photos=dir([images_dir '/*.png']);
    for i = 1:length(photos)
        photos_elapsedtime(i)=(datenum(photos(i).name(1:14),'yyyymmddHHMMSS')-datenum(photos(1).name(1:14),'yyyymmddHHMMSS'))*24*3600;
    end
    movie_times=0:10:A(end,3);
    num_frames=length(A(:,3));
    movie_frames(length(movie_times))= struct('cdata',[],'colormap',[]);
    for i=1:length(movie_times)
        axes(ax2_han);
        [c index] = min(abs(photos_elapsedtime-movie_times(i)));
        image=imread([images_dir '/' photos(index).name]);
        imshow(image(1:3400,:,:));   %(yrange, xrange, colorrange)
        axes(ax1_han);
        [c index] = min(abs(A(:,3)-movie_times(i)));
        line_han=scatter(A(index,1),A(index,2),15,'r','filled');
        movie_frames(i)=getframe(gcf);
        line_han.delete;
    end
    v = VideoWriter([images_dir '.mp4'],'MPEG-4');
    v.FrameRate=15;
    open(v);
    writeVideo(v,movie_frames);
    close(v);
    close all
    clear all
    %movie(fig_han,movie_frames,2)
