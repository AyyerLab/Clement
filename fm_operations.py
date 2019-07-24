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
        self.transp = False
        self.rot = False
        self.transformed = False
        self.coordinates = []
        self.threshold = 0
        self.max_shift = 10
        self.matches = []
        self.diff_list = []
        self._tf_data = None
        self.old_fname = None
        self.new_points = None
        self.side_length = None
        self.orig_points = None
        self.shift = []
        self.transform_shift = 0
        self.tf_matrix = np.identity(3)
        self.no_shear = False

        javabridge.start_vm(class_path=bioformats.JARS)

    def parse(self, fname, z):
        ''' Parses file

        Saves parsed file in self._orig_data
        self.data is the array to be displayed
        '''

        if fname != self.old_fname:
            self.reader = bioformats.ImageReader(fname)
            self.old_fname = fname
            self.num_channels = self.reader.rdr.getSizeZ()
        self._orig_data = self.reader.read(z=z)
        self._orig_data /= self._orig_data.mean((0, 1))
        self.data = np.copy(self._orig_data)
        self.color_data = np.copy(self._orig_data)
        if self.transformed:
            self.apply_transform()
            self._update_data()

    def __del__(self):
        javabridge.kill_vm()

    def _update_data(self):

        
        if self.transformed and self._tf_data is not None:
            self.data = np.copy(self._tf_data)
            self.points = np.copy(self.new_points)
        else:
            self.data = np.copy(self._orig_data)
            self.points = np.copy(self.orig_points) if self.orig_points is not None else None

        if self.fliph:
            self.data = np.flip(self.data, axis=0)
        if self.flipv:
            self.data = np.flip(self.data, axis=1)
        if self.transp:
            self.data = np.transpose(self.data, (1, 0, 2))
        if self.rot:
            self.data = np.rot90(self.data, axes=(0, 1))

        if self.points is not None:
            self._update_points()

    def _update_points(self):
        if self.fliph:
            self.points[:,0] = self.data.shape[0] - self.points[:,0]
        if self.flipv:
            self.points[:,1] = self.data.shape[1] - self.points[:,1]
        if self.transp:
            self.points = self.points[:,::-1]
        if self.rot:
            temp = self.data.shape[1] - self.points[:,1]
            self.points[:,1] = self.points[:,0]
            self.points[:,0] = temp
        print('Updating points', self.points[0])

    def flip_horizontal(self, do_flip):
        self.fliph = do_flip
        self._update_data()

    def flip_vertical(self, do_flip):
        self.flipv = do_flip
        self._update_data()

    def transpose(self, do_transp):
        self.transp = do_transp
        self._update_data()

    def rotate_clockwise(self, do_rot):
        self.rot = do_rot
        self._update_data()

    def toggle_original(self, transformed=None):
        if self._tf_data is None:
            print('Need to transform data first')
            return

        if transformed is None:
            self.transformed = not self.transformed
        else:
            self.transformed = transformed
        self._update_data()

    def peak_finding(self):
        for i in range(1, len(self.data)):
            img = self.data[i]
            img_max = ndi.maximum_filter(img, size=3, mode='reflect')
            maxima = (img == img_max)
            img_min = ndi.minimum_filter(img, size=3, mode='reflect')

            self.threshold = int(np.mean(self.data[i])) + int(np.mean(self.data[i])//3)
            diff = ((img_max - img_min) > self.threshold)
            maxima[diff==0] = 0

            labeled, num_objects = ndi.label(maxima)
            c_i = np.array(ndi.center_of_mass(img, labeled, range(1, num_objects+1)))
            self.coordinates.append(c_i)

            print('Number of peaks found in channel {}: '.format(i), len(c_i))

        self.coordinates = [np.array(k).astype(np.int16) for k in self.coordinates]
        if len(self.coordinates[0])<1000:
            counter = 0
            for i in range(1, len(self.coordinates)):
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
            shift1 = (np.median(shift1_arr[:, 0], axis=0), np.median(shift1_arr[:, 1], axis=0))
        else:
            shift1 = np.zeros((2))

        if len(self.diff_list[1]) != 0:
            shift2_arr = np.array(self.diff_list[1])
            shift2 = (np.median(shift2_arr[:, 0], axis=0), np.median(shift2_arr[:, 1], axis=0))
        else:
            shift2 = np.zeros((2))

        print(shift1)
        print(shift2)
        shift.append(shift1)
        shift.append(shift2)
        data_shifted = np.zeros_like(self.data[1:])
        data_shifted[0] = self.data[1]
        data_shifted[1] = ndi.shift(self.data[2], np.array(self.shift[0]))
        data_shifted[2] = ndi.shift(self.data[3], np.array(self.shift[1]))

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
        #return data[1], ndi.shift(data[2], shift1), ndi.shift(data[2], shift2), coordinates

    def calc_transform(self, my_points):
        print('Input points:\n', my_points)
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))

        self.side_length = np.mean(side_list)
        print('ROI side length:', self.side_length, '\xb1', side_list.std())

        cen = my_points.mean(0) - np.ones(2)*self.side_length/2.
        self.new_points = np.zeros_like(my_points)
        self.new_points[0] = cen + (0, 0)
        self.new_points[1] = cen + (self.side_length, 0)
        self.new_points[2] = cen + (self.side_length, self.side_length)
        self.new_points[3] = cen + (0, self.side_length)

        if self.no_shear:
            self.tf_matrix = self.calc_rot_transform(my_points)
        else:
            self.tf_matrix = tf.estimate_transform('affine', my_points, self.new_points).params

        nx, ny = self.data.shape[:-1]
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]
        print('Transform matrix:\n', self.tf_matrix)
        self.orig_points = np.copy(np.array(my_points))
        self.apply_transform()

    def calc_rot_transform(self,pts):
            sides = np.zeros_like(pts)
            sides[:3] = np.diff(pts,axis=0)
            sides[3] = pts[0]-pts[-1]
            dst_sides = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
            print(sides)
            angles = []
            for i in range(len(pts)):
                angles.append(np.arccos(np.dot(sides[i],dst_sides[i])/(np.linalg.norm(sides[i])*np.linalg.norm(dst_sides[i]))))
            angles_deg = [angle * 180/np.pi for angle in angles]
            angles_deg = [np.abs(angle-180) if angle > 90 else angle for angle in angles_deg] 
            print('angles_deg: ', angles_deg)
            theta = -(np.pi/180*np.mean(angles_deg))
            tf_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

            return tf_matrix

    def apply_transform(self):
        if self.tf_matrix is None:
            print('Calculate transform matrix first')
            return
        self._tf_data = np.empty(self._tf_shape+(self.data.shape[-1],))
        for i in range(self.data.shape[-1]):
            self._tf_data[:,:,i] = ndi.affine_transform(self.data[:,:,i], np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
            sys.stderr.write('\r%d'%i)
        print('\r', self._tf_data.shape)
        self.transform_shift = -self.tf_corners.min(1)[:2]
        print(self.transform_shift)
        self.transformed = True
        self.data = np.copy(self._tf_data)
        self.new_points = np.array([point + self.transform_shift for point in self.new_points])

    @classmethod
    def get_transform(self, source, dest):
        if len(source) != len(dest):
            print('Point length do not match')
            return
        return tf.estimate_transform('affine', source, dest).params
