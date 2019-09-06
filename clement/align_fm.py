import sys
import glob
import numpy as np
import scipy.signal as sc
import pyqtgraph as pg
from matplotlib import pyplot as plt
from PIL import Image
import os
import scipy.ndimage as ndi
import os

def calc_shift(flist,data):
    
    print('flist: ',flist)
    flist.sort()
    names = [x for x in os.listdir('../pascale') if x.endswith('.tif')]
    my_order = [2,1,3,0]
    my_order2 = [0,3,2,1]
    flist_order = [flist[i] for i in my_order]
    names_order = [names[i] for i in my_order2]
    print('flist_order: ',flist_order)
    print('names: ',names)
    print('names_order: ', names_order)
    data = [data[i] for i in my_order]
    #pg.show(data)
    [print(image.shape) for image in data]
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
    
    print(shift1)
    print(shift2)
    data_shifted = np.zeros_like(data[1:])
    data_shifted[0] = data[1]
    data_shifted[1] = ndi.shift(data[2],shift1)
    data_shifted[2] = ndi.shift(data[3],shift2)
    
    if not os.path.isdir('../pascale/shifted_data'):
        os.mkdir('../pascale/shifted_data')

    name_list = []
    for i in range(data_shifted.shape[0]):
        if os.path.isfile('../pascale/shifted_data/'+names_order[1+i][:-4]+'_shifted.tif'):
            os.remove('../pascale/shifted_data/'+names_order[1+i][:-4]+'_shifted.tif')
        print(i)
        img = Image.fromarray(data_shifted[i])
        img.save('../pascale/shifted_data/'+names_order[1+i][:-4]+'_shifted.tif')
        name_list.append('../pascale/shifted_data/'+names_order[1+i][:-4]+'_shifted.tif')
        
    print('Done')
    
    for i in range(data_shifted.shape[0]):
        print(data_shifted[i].shape)

    return name_list
    #return data[1],ndi.shift(data[2],shift1), ndi.shift(data[2],shift2), coordinates


if __name__=='__main__':
    
    flist = glob.glob('/home/wittetam/mount/clem/pascale/c06*.tif')
    if len(flist) == 0:
         flist = glob.glob('/home/wittetam/maxwell_mount/clem/pascale/c06*.tif')

    ref,shifted_img1, shifted_img2, coor = calc_shift(flist)

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
