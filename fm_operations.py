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
import multiprocessing as mp
matplotlib.use('QT5Agg')

class FM_ops():
    def __init__(self):
        self.num_channels = None
        self.data = None
        self.flipv = False
        self.fliph = False
        self.transp = False
        self.rot = False
        self.flips = [self.fliph, self.flipv, self.transp, self.rot]
        self.transformed = False
        self.coordinates = []
        self.threshold = 0
        self.max_shift = 10
        self.matches = []
        self.diff_list = []
        self._tf_data = None
        self.old_fname = None
        self.points = None
        self.new_points = None
        self.side_length = None
        self.orig_points = None
        self.shift = []
        self.transform_shift = 0
        self.tf_matrix = np.identity(3)
        self.no_shear = False
        self.show_max_proj = False
        self.max_proj_data = None
        self.tf_max_proj_data = None
        self.data_slices = []
        self.counter_clockwise = False
        self.rotated = False
        self.refine_matrix = None
        self.refine_points = None
        self.merged = None
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
        self.data_slices.append(self.data)

        if self.transformed:
            self.apply_transform()
            self._update_data() 
        
    def parse_slices(self):
        for i in range(self.num_channels):
            data = self.reader.read(z=i)
            data /= data.mean((0,1))
            self.data_slices.append(data)
        print('Done')
        
    def __del__(self):
        javabridge.kill_vm()

    def _update_data(self):
        if self.transformed and self._tf_data is not None:
            if self.show_max_proj and self.tf_max_proj_data is not None:
                self.data = np.copy(self.tf_max_proj_data)
            else:
                self.data = np.copy(self._tf_data)
            self.points = np.copy(self.new_points)
        else:
            if self.show_max_proj and self.data_slices is not None:
                self.data = np.copy(self.max_proj_data)
            else:
                self.data = np.copy(self._orig_data)
            self.points = np.copy(self.orig_points) if self.orig_points is not None else None

        if self.fliph:
            if self.transp and not self.rot:
                self.data = np.flip(self.data, axis=1)
            elif self.rot and not self.transp:
                self.data = np.flip(self.data, axis=1)
            else:
                self.data = np.flip(self.data, axis=0)
        if self.flipv:
            if self.transp and not self.rot:
                self.data = np.flip(self.data, axis=0)
            elif self.rot and not self.transp:
                self.data = np.flip(self.data, axis=0)
            else:
                self.data = np.flip(self.data, axis=1)
        if self.transp:
            self.data = np.transpose(self.data, (1, 0, 2))
        if self.rot:
            self.data = np.rot90(self.data, axes=(0, 1)) 
    
        if self.points is not None:
            self.points = self.update_points(self.points)
    
    def update_points(self,points):
        if self.fliph:
            if self.transp and not self.rot:
                points[:,1] = self.data.shape[0] - points[:,1]
            elif self.rot and not self.transp:
                points[:,1] = self.data.shape[0] - points[:,1]
            else:
                points[:,0] = self.data.shape[0] - points[:,0]
        if self.flipv:
            if self.transp and not self.rot:
                points[:,0] = self.data.shape[1] - points[:,0]
            elif self.rot and not self.transp:
                points[:,0] = self.data.shape[1] - points[:,0]
            else:
                points[:,1] = self.data.shape[1] - points[:,1]
        if self.transp:
            points = np.flip(self.points,axis=1)
        if self.rot:
            temp = self.data.shape[0] - points[:,1]
            points[:,1] = points[:,0]
            points[:,0] = temp
        print('Updating points \n', points)

        return points

    def flip_horizontal(self, do_flip):
        self.fliph = do_flip
        self.flips[0] = self.fliph
        self._update_data()

    def flip_vertical(self, do_flip):
        self.flipv = do_flip
        self.flips[1] = self.flipv
        self._update_data()

    def transpose(self, do_transp):
        self.transp = do_transp
        self.flips[2] = self.transp
        self._update_data()

    def rotate_clockwise(self, do_rot):
        self.rot = do_rot
        self.flips[3] = self.rot 
        self._update_data()

    def toggle_original(self):
        if self._tf_data is None:
            print('Need to transform data first')
            return
        
        print(self.transformed)
        self._update_data()

    def calc_max_projection(self):
        self.show_max_proj = not self.show_max_proj
        print(self.show_max_proj)
        if self.show_max_proj:
            if self.max_proj_data is None:
                self.parse_slices()
                self.max_proj_data = np.max(np.array(self.data_slices),axis=0)
                if self.transformed:
                    self.apply_transform() 

        self._update_data(self.first_flips)

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
        line.addWidget(self.fm_fname, stretch=1)

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
   
    def calc_affine_transform(self, my_points):
        self.first_flips = list(np.copy([True if flip else False for flip in self.flips]))
        if True in self.first_flips:
            if not self.transformed:
                self._orig_points = np.copy(self.points)
                self._orig_data = np.copy(self.data)
        print('First flips: ',self.first_flips) 
        my__points = self.calc_orientation(my_points)
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

        self.tf_matrix = tf.estimate_transform('affine', my_points, self.new_points).params

        nx, ny = self.data.shape[:-1]
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]
        print('Transform matrix:\n', self.tf_matrix)
        print('Shift: ', -self.tf_corners.min(1)[:2])
        
        if not self.transformed:
            self.orig_points = np.copy(np.array(my_points))
        self.apply_transform()
        self.points = np.copy(self.new_points)
        print('New points: \n', self.new_points)

    def calc_rot_transform(self, my_points):
        self.refine_matrix = None
        #my_points = self.calc_orientation(my_points)
        print('Input points:\n', my_points)

        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))
        self.side_length = np.mean(side_list)
        print('ROI side length:', self.side_length, '\xb1', side_list.std())
        
        my_points_sorted = np.array(sorted(my_points, key=lambda k: [k[0],k[1]]))
        print('Sorted points:\n',my_points_sorted)
        
        self.tf_matrix = self.calc_rot_matrix(my_points_sorted)
        nx, ny = self.data.shape[:-1]
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]
        print('Tf: ', self.tf_matrix)
        
        cen = my_points.mean(0) + self.tf_corners.min(1)[:2] + (0,self.side_length/2)
        self.new_points = np.zeros_like(my_points)
        self.new_points[0] = cen + (0, 0)
        self.new_points[1] = cen + (self.side_length, 0)
        self.new_points[2] = cen + (self.side_length, self.side_length)
        self.new_points[3] = cen + (0,self.side_length)

        self.orig_points = np.copy(np.array(my_points))
        self.apply_transform()
        self.points = np.copy(self.new_points)
        self.rotated = True

    def calc_orientation(self,points):
        my_list = []
        for i in range(1,len(points)):
            my_list.append((points[i][0]-points[i-1][0])*(points[i][1]+points[i-1][1]))
        my_list.append((points[0][0]-points[-1][0])*(points[0][1]+points[-1][1]))
        my_sum = np.sum(my_list)
        print(my_list)
        print(my_sum)
        if my_sum > 0:
            print('counter-clockwise')
            self.counter_clockwise = True
            return points
        else:
            print('clockwise --> transpose points')
            order = [0,3,2,1]
            return points[order] 

    def calc_rot_matrix(self,pts):
            sides = np.zeros_like(pts)
            sides[:3] = np.diff(pts,axis=0)
            sides[3] = pts[0]-pts[-1]
            dst_sides = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
            print(sides)
            angles = []
            for i in range(len(pts)):
                angles.append(np.arccos(np.dot(sides[i],dst_sides[i])/(np.linalg.norm(sides[i])*np.linalg.norm(dst_sides[i]))))
            angles_deg = [angle * 180/np.pi for angle in angles]
            if not self.rotated:
                angles_deg = [np.abs(angle-180) if angle > 90 else angle for angle in angles_deg] 
            print('angles_deg: ', angles_deg)
            theta = -(np.pi/180*np.mean(angles_deg))
            tf_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

            return tf_matrix

    def apply_transform(self):
        if self.tf_matrix is None:
            print('Calculate transform matrix first')
            return
        
        self.transformed = True
        if self.max_proj_data is None:
            self._tf_data = np.empty(self._tf_shape+(self.data.shape[-1],))
            for i in range(self.data.shape[-1]):
                self._tf_data[:,:,i] = ndi.affine_transform(self.data[:,:,i], np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
                sys.stderr.write('\r%d'%i)
            print('\r', self._tf_data.shape)

        elif self.tf_max_proj_data is None and self.transformed:
            self.tf_max_proj_data  = np.empty(self._tf_shape+(self.data.shape[-1],))
            for i in range(self.data.shape[-1]):
                self.tf_max_proj_data[:,:,i] = ndi.affine_transform(self.max_proj_data[:,:,i], np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
                sys.stderr.write('\r%d'%i)
            print('\r', self._tf_data.shape)
        
        else:
            manager = mp.Manager()
            dict1 = manager.dict()
            dict2 = manager.dict()
            p1 = mp.Process(target=self.apply_transform_mp,args=(0,self.max_proj_data,dict1))
            p2 = mp.Process(target=self.apply_transform_mp,args=(1,self._orig_data,dict2))
            p1.start()
            p2.start()
            p1.join()
            p2.join()
        
            self.tf_max_proj_data = np.array(dict1[0])
            self._tf_data = np.array(dict2[0])
            print('\r', self._tf_data.shape)

        self.transform_shift = -self.tf_corners.min(1)[:2]
        print(self.transform_shift) 
        self.data = np.copy(self._tf_data)
        self.new_points = np.array([point + self.transform_shift for point in self.new_points])

    def apply_transform_mp(self,k,data,return_dict):
        channel_list = []
        for i in range(data.shape[-1]):
            channel_list.append(ndi.affine_transform(data[:,:,i], np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape))
            sys.stderr.write('\r%d'%i)
        print('k: ', k)
        return_dict[0] = np.transpose(np.array(channel_list),(1,2,0))
    
    def refine(self, source, dst):
        if self._tf_data is not None:
            self.refine_matrix = tf.estimate_transform('affine',source,dst).params
            nx, ny = self._tf_data.shape[:-1]
            corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
            self.refine_corners = np.dot(self.refine_matrix, corners)
            self._refine_shape = tuple([int(i) for i in (self.refine_corners.max(1) - self.refine_corners.min(1))[:2]])
            self.refine_matrix[:2, 2] -= self.refine_corners.min(1)[:2]
            tmp = np.copy(self._tf_data)
            self._tf_data = np.empty(self._refine_shape+(self._tf_data.shape[-1],))
            for i in range(self._tf_data.shape[-1]):
                self._tf_data[:,:,i] = ndi.affine_transform(tmp[:,:,i], np.linalg.inv(self.refine_matrix), order=1, output_shape=self._refine_shape)
                sys.stderr.write('\r%d'%i)               
    
    def refine_grid(self, src, dst):
        if self._tf_data is not None:
            self.rg_matrix = tf.estimate_transform('affine',src,dst).params
            for i in range(len(self.points)):
                points = [src[i][0],src[i][1],1]
                self.new_points[i] = (self.rg_matrix @ points)[:2]

            self.refine_shift = -self.refine_corners.min(1)[:2]
            print('Refinement shift: ', self.refine_shift)
            self.new_points = np.array([point + self.refine_shift for point in self.new_points])
            self._update_data()

    def merge(self, em_data,em_points):        
        fm_origin = self.points[0]
        em_origin = em_points[0]

        shift = (fm_origin - em_origin).astype(np.int)
       
        x_shape = np.max([em_data.shape[0],np.abs(shift[0])+self.data.shape[0]])
        y_shape = np.max([em_data.shape[1],np.abs(shift[1])+self.data.shape[1]])

        print(fm_origin)
        print(em_origin)
        print(shift)
        
        self.merged = np.zeros((x_shape,y_shape,self.data.shape[-1]+1))
        print(self.merged.shape)
        self.merged[np.abs(shift[0]):self.data.shape[0]+np.abs(shift[0]),np.abs(shift[1]):self.data.shape[1]+np.abs(shift[1]),:-1] = self.data
        self.merged[:em_data.shape[0],:em_data.shape[1],-1] = em_data/np.max(em_data)*np.max(self.data)

    @classmethod
    def get_transform(self, source, dest):
        if len(source) != len(dest):
            print('Point length do not match')
            return
        return tf.estimate_transform('affine', source, dest).params
