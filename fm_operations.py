import sys
import glob
import os
import numpy as np
import scipy.signal as sc
import javabridge
import bioformats
import pyqtgraph as pg
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
matplotlib.use('QT5Agg')


class FM_ops():
    def __init__(self):
        self.data = None
        self.flipv = False
        self.fliph = False
        self.trans = False
        self.rot = False
        self.coordinates = []
        self.threshold = 0
        self.max_shift = 10
        self.matches = []
        self.diff_list = []
        self.shift = []
    
    def parse(self, fname):
        javabridge.start_vm(class_path=bioformats.JARS)
        self.data = np.transpose(bioformats.load_image(fname),(2,1,0))
        javabridge.kill_vm()

    def flip_horizontal(self):
        self.data= np.flip(self.data,axis=1)
        self.fliph = not self.fliph

    def flip_vertical(self):
        self.data = np.flip(self.data,axis=2)
        self.flipv = not self.flipv

    def transpose(self):
        self.data = np.transpose(self.data,(0,2,1))
        self.trans = not self.trans

    def rotate_clockwise(self):
        self.data = np.rot90(self.data,axes=(1,2))
        self.rot = not self.rot

    def rotate_counterclock(self):
        self.data = np.rot90(self.data,axes=(2,1))
        self.rot = not self.rot
    
    def peak_finding(self):
        for i in range(1,len(data)):
            img = self.data[i]
            img_max = ndi.maximum_filter(img,size=3,mode='reflect')
            maxima = (img == img_max)
            img_min = ndi.minimum_filter(img,size=3,mode='reflect')

            self.threshold = int(np.mean(data[i])) + int(np.mean(data[i])//3)
            diff = ((img_max - img_min) > self.threshold)
            maxima[diff==0] = 0

            labeled, num_objects = ndi.label(maxima)
            c_i = np.array(ndi.center_of_mass(img,labeled,range(1,num_objects+1)))
            self.coordinates.append(c_i)

            print('Number of peaks found in channel {}: '.format(flist_order[i]),len(c_i))

        self.coordinates = [np.array(k).astype(np.int16) for k in coordinates]

        counter = 0
        for i in range(1,len(coordinates)):
            tmp_list_match = []
            tmp_list_diff = []
            for k in range(len(self.coordinates[0])):
                for l in range(len(self.coordinates[i])):
                    diff_norm = np.linalg.norm(coordinates[0][k]-coordinates[i][l])
                    if diff_norm < max_shift and diff_norm != 0:
                        tmp_list_diff.append(coordinates[0][k]-coordinates[i][l])
                        tmp_list_match.append(coordinates[0][k])
            print(tmp_list_diff)
            matches.append(tmp_list_match)
            diff_list.append(tmp_list_diff)


    def align(self):


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
      
    
    def transform(self):
        pass






