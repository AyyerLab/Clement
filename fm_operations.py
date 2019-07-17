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
import scipy.ndimage as ndi
from skimage import transform as tf
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
        self.transformed_data = None
        self.shift = []
        self.transform_shift = 0
    
    def parse(self, fname):
        javabridge.start_vm(class_path=bioformats.JARS)
        self.data = bioformats.load_image(fname)
        self.data /= self.data.mean((0,1))
        self.data = np.transpose(self.data)
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
        for i in range(1,len(self.data)):
            img = self.data[i]
            img_max = ndi.maximum_filter(img,size=3,mode='reflect')
            maxima = (img == img_max)
            img_min = ndi.minimum_filter(img,size=3,mode='reflect')

            self.threshold = int(np.mean(self.data[i])) + int(np.mean(self.data[i])//3)
            diff = ((img_max - img_min) > self.threshold)
            maxima[diff==0] = 0

            labeled, num_objects = ndi.label(maxima)
            c_i = np.array(ndi.center_of_mass(img,labeled,range(1,num_objects+1)))
            self.coordinates.append(c_i)

            print('Number of peaks found in channel {}: '.format(i),len(c_i))

        self.coordinates = [np.array(k).astype(np.int16) for k in self.coordinates]
        if len(self.coordinates[0])<1000:
            counter = 0
            for i in range(1,len(self.coordinates)):
                tmp_list_match = []
                tmp_list_diff = []
                for k in range(len(self.coordinates[0])):
                    for l in range(len(self.coordinates[i])):
                        diff_norm = np.linalg.norm(self.coordinates[0][k]-self.coordinates[i][l])
                        if diff_norm < self.max_shift and diff_norm != 0:
                            tmp_list_diff.append(self.coordinates[0][k]-self.coordinates[i][l])
                            tmp_list_match.append(self.coordinates[0][k])
                print(tmp_list_diff)
                self.matches.append(tmp_list_match)
                self.diff_list.append(tmp_list_diff)
        else:
            pass

    def align(self):
        if len(self.diff_list[0]) != 0:
            shift1_arr = np.array(self.diff_list[0])
            shift1 = (np.median(shift1_arr[:,0],axis=0),np.median(shift1_arr[:,1],axis=0))
        else:
            shift1 = np.zeros((2))

        if len(self.diff_list[1]) != 0:
            shift2_arr = np.array(self.diff_list[1])
            shift2 = (np.median(shift2_arr[:,0],axis=0),np.median(shift2_arr[:,1],axis=0))
        else:
            shift2 = np.zeros((2))

        print(shift1)
        print(shift2)
        shift.append(shift1)
        shift.append(shift2)
        data_shifted = np.zeros_like(self.data[1:])
        data_shifted[0] = self.data[1]
        data_shifted[1] = ndi.shift(self.data[2],np.array(self.shift[0]))
        data_shifted[2] = ndi.shift(self.data[3],np.array(self.shift[1]))

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
      
    def affine_transform(self,my_points):
        print('Input points:\n', my_points)
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))

        side_length = np.mean(side_list)
        print('ROI side length:', side_length, '\xb1', side_list.std())

        cen = my_points.mean(0) - np.ones(2)*side_length/2.
        new_points = np.zeros_like(my_points)
        new_points[0] = cen + (0, 0)
        new_points[1] = cen + (side_length,0)
        new_points[2] = cen + (side_length,side_length)
        new_points[3] = cen + (0,side_length)
        
        matrix = tf.estimate_transform('affine', my_points[:4], new_points).params
        
        nx, ny = self.data.shape[1:]
        corners = np.array([[0,0,1], [nx,0,1], [nx,ny,1], [0,ny,1]]).T
        tr_corners = np.dot(matrix, corners)
        output_shape = tuple([int(i) for i in (tr_corners.max(1) - tr_corners.min(1))[:2]])
        matrix[:2,2] -= tr_corners.min(1)[:2]
        print('Transform matrix:\n', matrix)
        
        self.transformed_data = [] 
        for i in range(self.data.shape[0]):
            sys.stderr.write('\r%d'%i)
            self.transformed_data.append(ndi.affine_transform(self.data[i], np.linalg.inv(matrix), order=1, output_shape=output_shape))
        self.transformed_data = np.array(self.transformed_data)
        print('\r', self.transformed_data.shape)
        self.transform_shift = -tr_corners.min(1)[:2]
