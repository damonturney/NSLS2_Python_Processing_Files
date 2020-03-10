# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:08:57 2019

@author: akiss
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import os
import scipy.ndimage.interpolation
import skimage.io
from skimage.feature import register_translation
import skimage.filters

t0 = time.monotonic()

# root = r'C:\Users\akiss\Documents\SRX\collaborations\Turney\FXI\\'
root = '/media/usb0/BNL_FXI_2019_03_01/'
fn = 'xanes_scan2_id_13179.h5'

f = h5py.File(root+fn, 'r')

E = np.array(f['X_eng'])
I = np.array(f['img_xanes'])

# %% 
def norm_image_01(I):
    I_min = np.amin(I, axis=(0, 1))
    I_max = np.amax(I, axis=(0, 1))
    return (I - I_min) / (I_max - I_min)


def calc_shift(I_ref, I, roi=None, blur=2, plot=False, filter_std=1):
    if (roi == None):
        roi = (0, I.shape[1], 0, I.shape[2])
    I_ref = np.copy(I_ref[:, roi[0]:roi[1], roi[2]:roi[3]])
    I = np.copy(I[:, roi[0]:roi[1], roi[2]:roi[3]])
    
    shift = []
    for i in range(I.shape[0]):
        try:
            I0 = norm_image_01(I_ref[i, :, :])
        except:
            I0 = norm_image_01(I_ref[i-1, :, :])
        I1 = norm_image_01(I[i, :, :])
        if (blur > 0):
            I0 = skimage.filters.gaussian(I0, sigma=blur)
            I1 = skimage.filters.gaussian(I1, sigma=blur)
    
        tmp_shift, _, _ = register_translation(I0, I1, upsample_factor=100)
        shift.append(tmp_shift)

    shift = np.array(shift)
    
    if plot:
        plt.figure(dpi=300)
        plt.plot(shift[:, 1], label='X shift')
        plt.plot(shift[:, 0], label='Y shift')

    # Filter results
    if (filter_std != 0):
        x_mean = np.mean(shift[:, 0])
        x_std = np.std(shift[:, 0])
        y_mean = np.mean(shift[:, 1])
        y_std = np.std(shift[:, 1])
        for i in range(I.shape[0]):
            x_star = (shift[i, 0] - x_mean) / x_std
            y_star = (shift[i, 1] - y_mean) / y_std
            if (x_star > filter_std):
                shift[i, 0] = x_mean
            if (y_star > filter_std):
                shift[i, 1] = y_mean
    
    if plot:
        plt.plot(shift[:, 1], label='X shift filter')
        plt.plot(shift[:, 0], label='Y shift filter')
        plt.legend()

    return shift


def shift_images(I, shift):
    I = np.copy(I)
    for i in range(I.shape[0]):
        I[i, :, :] = scipy.ndimage.interpolation.shift(I[i, :, :], shift[i, :])
    
    return I


# %%
Mn_below = np.array(I[0::4, :, :])
Mn_above = np.array(I[1::4, :, :])
Cu_below = np.array(I[2::4, :, :])
Cu_above = np.array(I[3::4, :, :])

N = Mn_above.shape[0]

# %% Align Mn images
# Calculate the shift from below to above
Mn_below2above = calc_shift(Mn_above, Mn_below, plot=True, filter_std=1)

# Shift by an average value
shift_x = np.mean(Mn_below2above[:, 1])
shift_y = np.mean(Mn_below2above[:, 0])
Mn_below2above_mean = np.copy(Mn_below2above)
for i in range(len(Mn_below2above)):
    Mn_below2above_mean[i,:] = [shift_y, shift_x]

# Shift the images
Mn_below_reg = shift_images(Mn_below, Mn_below2above)

# Display improvement
plt.figure(dpi=300)
plt.subplot(121)
plt.imshow(Mn_above[0, :, :] - Mn_below[0, :, :], cmap='gray')
plt.subplot(122)
plt.imshow(Mn_above[0, :, :] - Mn_below_reg[0, :, :], cmap='gray')


# %% Align Cu images
# Calculate the shift from below to above
Cu_below2above = calc_shift(Cu_above, Cu_below, plot=True, filter_std=1)

# Shift by an average value
shift_x = np.mean(Cu_below2above[:, 1])
shift_y = np.mean(Cu_below2above[:, 0])
Cu_below2above_mean = np.copy(Cu_below2above)
for i in range(len(Cu_below2above)):
    Cu_below2above_mean[i,:] = [shift_y, shift_x]

# Shift the images
Cu_below_reg = shift_images(Cu_below, Cu_below2above)

# Display improvement
plt.figure(dpi=300)
plt.subplot(121)
plt.imshow(Cu_above[0, :, :] - Cu_below[0, :, :], cmap='gray')
plt.subplot(122)
plt.imshow(Cu_above[0, :, :] - Cu_below_reg[0, :, :], cmap='gray')


# %% Align Cu and Mn images
# Isolate only Cu
Cu = Cu_above - Cu_below_reg

# Calculate the shift from Cu to Mn
Cu2Mn = calc_shift(Mn_below_reg, Cu, roi=(50, -50, 50, -50), plot=True, filter_std=1)

# Shift by an average value
shift_x = np.mean(Cu2Mn[:, 1])
shift_y = np.mean(Cu2Mn[:, 0])
Cu2Mn_mean = np.copy(Cu2Mn)
for i in range(len(Cu2Mn)):
    Cu2Mn_mean[i,:] = [shift_y, shift_x]

# Shift the images
Cu_below_reg = shift_images(Cu_below_reg, Cu2Mn)
Cu_above_reg = shift_images(Cu_above, Cu2Mn)

# Display improvement
plt.figure(dpi=300)
plt.subplot(121)
plt.imshow(Mn_above[0, :, :] - Cu_above[0, :, :], cmap='gray')
plt.subplot(122)
plt.imshow(Mn_above[0, :, :] - Cu_above_reg[0, :, :], cmap='gray')

# %% Save the files
# Check if a folder exists
root = root + fn[:-3] + '/'
if (not os.path.isdir(root)):
    try:
        os.mkdir(root)
    except e as Exception:
        print('Error creating directory.')
        print(e)
        quit()

# Save the files
skimage.io.imsave(root+'Mn_below_reg.tif', Mn_below_reg)
skimage.io.imsave(root+'Mn_above_reg.tif', Mn_above)
skimage.io.imsave(root+'Cu_below_reg.tif', Cu_below_reg)
skimage.io.imsave(root+'Cu_above_reg.tif', Cu_above_reg)

# %% Isolate elements and save files
img_Cu = Cu_above_reg - Cu_below_reg
img_Mn = Mn_above - Mn_below_reg
img_Bi_Cu = Cu_above_reg - img_Cu
img_Bi_Mn = Mn_above - img_Mn

skimage.io.imsave(root+'Mn.tif', img_Mn)
skimage.io.imsave(root+'Cu.tif', img_Cu)
skimage.io.imsave(root+'Bi_from_Mn.tif', img_Bi_Mn)
skimage.io.imsave(root+'Bi_from_Cu.tif', img_Bi_Cu)


print(f'{time.monotonic() - t0}')
