#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:17:15 2020

@author: damon
"""

from numpy import var
from numpy import ones
from numpy import mean
from numpy import sum
from numpy import sqrt
from numpy import argmax

            

def erc_R(im1,im2_bigger):
    winsize1=im1.shape
    length_im1=len(im1[:]);
    winsize2=im2_bigger.shape;
    variance_im1=var(im1);
    
    R   = ones((winsize2[0]-winsize1[0]+1,winsize2[1]-winsize1[1]+1))*0.012345678901
    im2 = ones(im1.shape)
    
    #    %For choosing the two sub-images to correlate:
    #    %   m is the offset, measured in pixels, of the searching window to-the-right. The searching window pans across im2_bigger
    #    %   n is the offset, measured in pixels, of the searching window down. The searching window pans across im2_bigger
    for m in range(0,winsize2[1]-winsize1[1]+1,3):
        for n in range(0,winsize2[0]-winsize1[0]+1,3):
            #print(m,n)
            im2[:,:]=im2_bigger[n:n+winsize1[0]-1+1,m:m+winsize1[1]-1+1];
            #%I use R[n+winsize/2,m+winsize/2] in order to keep in line with Kristof Sveen's convention on the meaning of R
            R[n,m]=sum((im1[:]-mean(im1[:])) * (im2[:]-mean(im2[:]))) / (length_im1-1)/sqrt(variance_im1*var(im2[:]))  
    
    max_indices = argmax(R)
    focused_indices=[]
    for m in range(0,winsize2[1]-winsize1[1]+1):
        for n in range(0,winsize2[0]-winsize1[0]+1):
            if abs(m - max_indices[1])<5 and abs(n - max_indices[0])<5 and R[n,m]==0.012345678901:
                focused_indices.append([n,m])
    
    for i in range(0,len(focused_indices)):
        n = focused_indices[i][0]
        m = focused_indices[i][1]
        im2[:,:]=im2_bigger[n:n+winsize1[0]-1+1,m:m+winsize1[1]-1+1]
        R[n,m]=sum((im1[:]-mean(im1[:])) * (im2[:]-mean(im2[:]))) / (length_im1-1)/sqrt(variance_im1*var(im2[:]))  
        
    return(R)