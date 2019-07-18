import sys
import glob
import os
import numpy as np
import scipy.signal as sc
import mrcfile as mrc
#import pyqtgraph as pg
import matplotlib
import scipy.ndimage as ndi
from skimage import transform as tf
from matplotlib import pyplot as plt
matplotlib.use('QT5Agg')

class Assembler():
    def __init__(self, step=100):
        self.step = int(np.sqrt(step))
        self.data = None
        self.backup = None
        self.transformed_data = None   
    
    def parse(self, fname):
        with mrc.open(fname, 'r', permissive=True) as f:
            try:
                self._orig_data = f.data[:,::self.step,::self.step]
            except IndexError:
                self._orig_data = f.data
            self._h = f.header
            self._eh = np.frombuffer(f.extended_header, dtype='i2')
            
    def assemble(self):
        dimensions = self._orig_data.shape
        
        if len(dimensions) == 3:
            pos_x = self._eh[1:10*dimensions[0]:10] // self.step
            pos_y = self._eh[2:10*dimensions[0]:10] // self.step
            pos_z = self._eh[3:10*dimensions[0]:10]

            cy, cx = np.indices(dimensions[1:3])

            self.data = np.zeros((np.max(pos_x)+dimensions[2],np.max(pos_y)+dimensions[1]), dtype='f4')
            #sys.stderr.write(self.data.shape)

            self.mcounts = np.zeros_like(self.data)
            for i in range(dimensions[0]):
                sys.stderr.write('\rMerge for image {}'.format(i))
                np.add.at(self.mcounts, (cx+pos_x[i], cy+pos_y[i]), 1)
                np.add.at(self.data, (cx+pos_x[i], cy+pos_y[i]), self._orig_data[i])
            sys.stderr.write('\n')

            self.data[self.mcounts>0] /= self.mcounts[self.mcounts>0]
            self.backup = np.copy(self.data)
        else:
            pass

    def save_merge(self, fname):
        with mrc.new(fname, overwrite=True) as f:
            f.set_data(self.data)
            f.update_header_stats()

    def toggle_original(self, transformed=None):
        if self.transformed_data is None:
            print('Need to transform data first')
            return
        if transformed is None:
            self.data = np.copy(self.transformed_data if self.transformed else self.backup)
            self.transformed = not self.transformed
        else:
            self.transformed = transformed
            self.data = np.copy(self.transformed_data if self.transformed else self.backup)

    def affine_transform(self, my_points):
        print('Input points:\n', my_points)
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))

        side_length = np.mean(side_list)
        print('ROI side length:', side_length, '\xb1', side_list.std())

        cen = my_points.mean(0) - np.ones(2)*side_length/2.
        new_points = np.zeros_like(my_points)
        new_points[0] = cen + (0, 0)
        new_points[1] = cen + (side_length, 0)
        new_points[2] = cen + (side_length, side_length)
        new_points[3] = cen + (0, side_length)

        matrix = tf.estimate_transform('affine', my_points[:4], new_points).params

        nx, ny = self.data.shape
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        tr_corners = np.dot(matrix, corners)
        output_shape = tuple([int(i) for i in (tr_corners.max(1) - tr_corners.min(1))[:2]])
        matrix[:2, 2] -= tr_corners.min(1)[:2]
        print('Transform matrix:\n', matrix)

        #self.transformed_data = []
        #sys.stderr.write('\r%d'%i)
        self.transformed_data = ndi.affine_transform(self.data, np.linalg.inv(matrix), order=1, output_shape=output_shape)
        #self.transformed_data = np.array(self.transformed_data)
        print('\r', self.transformed_data.shape)
        self.transform_shift = -tr_corners.min(1)[:2]
        self.transformed = True
        self.data = np.copy(self.transformed_data)

if __name__=='__main__':
    path = '../gs.mrc'
    assembler = Assembler()
    assembler.parse(path)
    assembler.assemble()
    #pg.show(merged.T)
