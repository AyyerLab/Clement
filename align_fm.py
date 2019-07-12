import sys
import glob
import numpy as np
import scipy.signal as sc
import pyqtgraph as pg
from matplotlib import pyplot as plt
from PIL import Image
import os
import scipy.ndimage as ndi

def calc_shift(data):
    plt.ion()
    coordinates = []
    for i in range(1,len(data)):
        img = data[i]
        img_max = ndi.maximum_filter(img,size=3,mode='reflect')
        maxima = (img == img_max)
        img_min = ndi.minimum_filter(img,size=3,mode='reflect')

        threshold = int(np.mean(data[i])) + int(np.mean(data[i])//3)
        diff = ((img_max - img_min) > threshold)
        maxima[diff==0] = 0

        labeled, num_objects = ndi.label(maxima)
        c_i = np.array(ndi.center_of_mass(img,labeled,range(1,num_objects+1)))
        coordinates.append(c_i)

        print('Number of peaks found in channel {}: '.format(flist_order[i]),len(c_i))

    coordinates = [np.array(k).astype(np.int16) for k in coordinates]
    #coordinates_unsorted = np.copy(coordinates)
    #coordinates.sort(key=len)
    
    #minimum_counts = len(coordinates[0]) 
    #min_pos = 0
    #for i in range(1,len(coordinates)):
    #    if len(coordinates[i]) < minimum_counts:
    #        minimum_counts = len(coordinates[i])
    #        min_pos = i

    matches = []
    counter = 0
    max_shift = 10
    shift = []
    diff_list = []
    for i in range(1,len(coordinates)):
        tmp_list_match = []
        tmp_list_diff = []
        for k in range(len(coordinates[0])):   
            for l in range(len(coordinates[i])):
                diff_norm = np.linalg.norm(coordinates[0][k]-coordinates[i][l])
                if diff_norm < max_shift and diff_norm != 0:
                    tmp_list_diff.append(coordinates[0][k]-coordinates[i][l])
                    tmp_list_match.append(coordinates[0][k])
        print(tmp_list_diff)   
        matches.append(tmp_list_match)
        diff_list.append(tmp_list_diff)
    
    print(len(diff_list))
    print(diff_list)
    if len(diff_list[0]) != 0: 
        shift1_arr = np.array(diff_list[0])
        shift1 = (np.median(shift1_arr[:,0],axis=0),np.median(shift1_arr[:,1],axis=0))
    else:
        shift1 = np.zeros((2))
    if len(diff_list[1]) != 0:
        shift2_arr = np.array(diff_list[1])
        shift2 = (np.median(shift2_arr[:,0],axis=0),np.median(shift2_arr[:,1],axis=0))
    else:
        shift2 = np.zeros((2))
    return data[1],ndi.shift(data[2],shift1), ndi.shift(data[2],shift2), coordinates


if __name__=='__main__':
    
    flist = glob.glob('/home/wittetam/mount/clem/pascale/c06*.tif')
    if len(flist) == 0:
         flist = glob.glob('/home/wittetam/maxwell_mount/clem/pascale/c06*.tif')

    flist.sort()
    my_order = [2,1,3,0]
    flist_order = [flist[i] for i in my_order]
    print(flist_order)

    my_data = [np.array(Image.open(i)) for i in flist_order]
    #pg.show(data)
    [print(image.shape) for image in my_data]
    
    ref,shifted_img1, shifted_img2, coor = calc_shift(my_data)

    for i in range(1,len(my_data)):
        img = my_data[i]
        fig2 = plt.figure(figsize=(8,16))
        ax1 = fig2.add_subplot(121)
        ax2 = fig2.add_subplot(122)
        ax1.imshow(img,cmap='gray')
        ax2.imshow(img,cmap='gray')
        ax2.scatter(coor[i-1][:,1],coor[i-1][:,0],50,facecolors='none',edgecolors='r')
        plt.show()

   
    plt.figure()
    plt.imshow(my_data[3],vmax=120,cmap='gray')
    plt.figure()
    plt.imshow(shifted_img1,vmax=500,cmap='gray')
    plt.show()
