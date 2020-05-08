import sys
import glob
import os
import numpy as np
import copy
import mrcfile as mrc
from scipy import ndimage as ndi
from scipy import signal as sc
from skimage import transform as tf
from skimage import io, measure, feature
import tifffile
from . ransac import Ransac
import time

class EM_ops():
    def __init__(self):
        self._orig_points = None
        self._grid_points_tmp = None
        self._transformed = False
        self._tf_points = None
        self._orig_points_region = None
        self._tf_points_region = None
        self._refine_matrix = None
        self._total_shift = None

        self.h = None
        self.eh = None
        self.orig_data = None
        self.tf_data = None
        self.pixel_size = None #should be in nanometer
        self.old_fname = None
        self.data = None
        self.stacked_data = False
        self.orig_region = None
        self.selected_region = None
        self.tf_region = None
        self.data_backup = None
        self.transformed_data = None
        self.pos_x = None
        self.pos_y = None
        self.pos_z = None
        self.grid_points = []
        self.tf_grid_points = []
        self.side_length = None
        self.mcounts = None
        self.tf_mcounts = None
        self.count_map = None
        self.tf_count_map = None
        self.tf_matrix = np.identity(3)
        self.clockwise = False
        self.rot_angle = None
        self.first_rotation = False
        self.rotated = False
        self.tf_prev = np.identity(3)

        self.stage_origin = None

        self.points = None
        self.assembled = True
        self.cum_matrix = None
        self.dimension = None

        self.fib_matrix = None
        self.fib_shift = None
        self.fib_angle = None #angle relative to xy-plane
        self.transposed = False

    def parse_2d(self, fname):
        if '.tif' in fname or '.tiff' in fname:
            self.data = np.array(io.imread(fname))
            self.dimensions = self.data.shape
            self.old_fname = fname
            try:
                md = tifffile.TiffFile(fname).fei_metadata
                self.pixel_size = np.array([md['Scan']['PixelWidth'], md['Scan']['PixelHeight']]) * 1e9
            except KeyError:
                print('No pixel size found! This might cause the program to crash at some point...')
        else:
            f = mrc.open(fname, 'r', permissive=True)
            if fname != self.old_fname:
                self.h = f.header
                self.eh = np.frombuffer(f.extended_header, dtype='i2')
                self.old_fname = fname
                self.pixel_size = np.array([f.voxel_size.x, f.voxel_size.y, f.voxel_size.y]) / 10
            self.dimensions = np.array(f.data.shape) # (dim_z, dim_y, dim_x)
            if len(self.dimensions) == 2:
                self.stacked_data = False
                self.data = np.copy(f.data)

        self.orig_data = np.copy(self.data)
        print('Pixel size: ', self.pixel_size)

    def parse_3d(self, step, fname):
        f = mrc.open(fname, 'r', permissive=True)
        if len(self.dimensions) == 3 and self.dimensions[0] > 1:
            self.stacked_data = True
            self.dimensions[1] = int(np.ceil(self.dimensions[1] / step))
            self.dimensions[2] = int(np.ceil(self.dimensions[2] / step))
            self.pos_x = self.eh[1:10*self.dimensions[0]:10] // step
            self.pos_y = self.eh[2:10*self.dimensions[0]:10] // step
            self.pos_x -= self.pos_x.min()
            self.pos_y -= self.pos_y.min()
            self.pos_z = self.eh[3:10*self.dimensions[0]:10]
            self.grid_points = []
            for i in range(len(self.pos_x)):
                point = np.array((self.pos_x[i], self.pos_y[i], 1))
                box_points = [point + (0, 0, 0),
                              point + (self.dimensions[2], 0, 0),
                              point + (self.dimensions[2], self.dimensions[1], 0),
                              point + (0, self.dimensions[1], 0)]
                self.grid_points.append(box_points)

            cy, cx = np.indices(self.dimensions[1:3])

            self.data = np.zeros((self.pos_x.max() + self.dimensions[2], self.pos_y.max() + self.dimensions[1]), dtype='f4')
            self.mcounts = np.zeros_like(self.data)
            self.count_map = np.zeros_like(self.data)
            sys.stdout.write('Assembling images into %s-shaped array...'% (self.data.shape,))
            for i in range(self.dimensions[0]):
                np.add.at(self.mcounts, (cx+self.pos_x[i], cy+self.pos_y[i]), 1)
                np.add.at(self.data, (cx+self.pos_x[i], cy+self.pos_y[i]), f.data[i, ::step, ::step])
                np.add.at(self.count_map, (cx+self.pos_x[i], cy+self.pos_y[i]), i)
            sys.stdout.write('done\n')
            self.data[self.mcounts>0] /= self.mcounts[self.mcounts>0]
            self.count_map[self.mcounts>1] = 0

        f.close()
        print(self.data.shape)
        self.orig_data = np.copy(self.data)

    def save_merge(self, fname):
        with mrc.new(fname, overwrite=True) as f:
            f.set_data(self.data)
            f.update_header_stats()

    def transpose(self):
        self.transposed = True
        self.data = self.data.T
        print('TRANSPOSE')
        print(self.points)
        if self.points is not None:
            self.points = np.array([np.flip(point)for point in self.points])
        print(self.points)

    def toggle_original(self):
        if self.assembled:
            if self._transformed:
                if self.tf_data is not None:
                    self.data = copy.copy(self.tf_data)
                else:
                    self.data = copy.copy(self.orig_data)
                    self._transformed = False
                self.points = copy.copy(self._tf_points)
            if not self._transformed:
                self.data = copy.copy(self.orig_data)
                self.points = copy.copy(self._orig_points)
        else:
            if self._transformed:
                if self.tf_region is not None:
                    self.data = copy.copy(self.tf_region)
                else:
                    self.data = copy.copy(self.orig_region)
                    self._transformed = False
                if self._tf_points_region is not None:
                    self.points = copy.copy(self._tf_points_region)
                else:
                    self.points = None
            else:
                self.data = np.copy(self.orig_region)
                self.points = copy.copy(self._orig_points_region)

    def toggle_region(self):
        self.toggle_original()

    def calc_affine_transform(self, my_points):
        my_points = self.calc_orientation(my_points)
        print('Input points:\n', my_points)
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))
        self.side_length = np.mean(side_list)
        print('ROI side length:', self.side_length, '\xb1', side_list.std())

        cen = my_points.mean(0) - np.ones(2)*self.side_length/2.
        points_tmp = np.zeros_like(my_points)
        points_tmp[0] = cen + (0, 0)
        points_tmp[1] = cen + (self.side_length, 0)
        points_tmp[2] = cen + (self.side_length, self.side_length)
        points_tmp[3] = cen + (0, self.side_length)

        self.tf_matrix = tf.estimate_transform('affine', my_points, points_tmp).params

        nx, ny = self.data.shape
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]
        print('Transform matrix:\n', self.tf_matrix)
        print('Shift: ', -self.tf_corners.min(1)[:2])
        print('Assembled? ', self.assembled)
        if not self._transformed:
            if self.assembled:
                self._orig_points = np.copy(my_points)
            else:
                self._orig_points_region = np.copy(my_points)
        self.apply_transform(points_tmp)

    def calc_rot_transform(self, my_points):
        print('Input points:\n', my_points)
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))
        self.side_length = np.mean(side_list)
        print('ROI side length:', self.side_length, '\xb1', side_list.std())

        self.tf_matrix = self.calc_rot_matrix(my_points)

        center = np.mean(my_points,axis=0)
        tf_center = (self.tf_matrix @ np.array([center[0],center[1],1]))[:2]

        nx, ny = self.data.shape
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]
        print('Tf: ', self.tf_matrix)

        points_tmp  = np.zeros_like(my_points)
        points_tmp[0] = tf_center + (-self.side_length/2, -self.side_length/2)
        points_tmp[1] = tf_center + (self.side_length/2, -self.side_length/2)
        points_tmp[2] = tf_center + (self.side_length/2, self.side_length/2)
        points_tmp[3] = tf_center + (-self.side_length/2,self.side_length/2)

        if not self._transformed:
            if self.assembled:
                self._orig_points = np.copy(my_points)
            else:
                self._orig_points_region = np.copy(my_points)
        self.apply_transform(points_tmp)
        self.rotated = True
        print('New points: \n', self._tf_points)

    def calc_orientation(self,points):
        my_list = []
        for i in range(1,len(points)):
            my_list.append((points[i][0]-points[i-1][0])*(points[i][1]+points[i-1][1]))
        my_list.append((points[0][0]-points[-1][0])*(points[0][1]+points[-1][1]))
        my_sum = np.sum(my_list)
        if my_sum > 0:
            print('counter-clockwise')
            return points
        else:
            print('clockwise --> transpose points')
            order = [0,3,2,1]
            return points[order]

    def calc_rot_matrix(self,pts):
            sides = np.zeros_like(pts)
            sides[:3] = np.diff(pts,axis=0)
            sides[3] = pts[0]-pts[-1]
            sides = np.array(sorted(sides, key=lambda k: np.cos(30*np.pi/180*k[0]+k[1])))

            dst_sides = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
            dst_sides = np.array(sorted(dst_sides, key=lambda k: np.cos(30*np.pi/180*k[0]+k[1])))

            angles = []
            for i in range(len(pts)):
                angles.append(np.arccos(np.dot(sides[i],dst_sides[i])/(np.linalg.norm(sides[i])*np.linalg.norm(dst_sides[i]))))
            angles_deg = [angle * 180/np.pi for angle in angles]

            angles_deg = [np.min([angle,np.abs((angle%90)-90),np.abs(angle-90)]) for angle in angles_deg]
            print('angles_deg: ', angles_deg)
            if self._transformed:
                theta = (np.pi/180*np.mean(angles_deg))
            else:
                theta = -(np.pi/180*np.mean(angles_deg))
            tf_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            return tf_matrix

    def apply_transform(self,pts):
        if self.tf_matrix is None:
            print('Calculate transform matrix first')
            return
        self._transformed = True

        if len(self.dimensions) == 3 and self.dimensions[0] > 1:
            self.tf_mcounts = ndi.affine_transform(self.mcounts, np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
            self.tf_count_map = ndi.affine_transform(self.count_map, np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)

        if self.assembled:
            self.tf_data = ndi.affine_transform(self.data, np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
        else:
            self.tf_region = ndi.affine_transform(self.data, np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)

        self.transform_shift = -self.tf_corners.min(1)[:2]
        pts = np.array([point + self.transform_shift for point in pts])
        if self.assembled:
            self._tf_points = np.copy(pts)
            self.tf_grid_points = []
            for i in range(len(self.grid_points)):
                tf_box_points = []
                for point in self.grid_points[i]:
                    x_i, y_i, z_i = self.tf_matrix @ point
                    tf_box_points.append(np.array([x_i,y_i,z_i]))
                self.tf_grid_points.append(tf_box_points)
        else:
            self._tf_points_region = np.copy(pts)
        self.toggle_original()

    def calc_fib_transform(self, sigma_angle, sem_shape, sem_pixel_size):
        #rotate by 90 degrees in plane
        phi = 90 * np.pi / 180
        self.fib_matrix = np.array([[np.cos(phi), -np.sin(phi), 0,0],
                                    [np.sin(phi), np.cos(phi), 0, 0],
                                    [0,0,1,0],
                                    [0,0,0,1]])

        #flip and scale
        scale = sem_pixel_size / self.pixel_size
        print('SEM shape: ', sem_shape)
        print('FIB shape: ', self.data.shape)
        print('Scale: ', scale)
        self.fib_matrix = np.array([[-scale[0], 0, 0, 0],
                                    [0, scale[1], 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]) @ self.fib_matrix

        #rotate by 77 degrees
        self.fib_angle = sigma_angle - 7
        total_angle = (90-self.fib_angle) * np.pi / 180
        self.fib_matrix = np.array([[1,0,0,0],
                                    [0, np.cos(total_angle), -np.sin(total_angle), 0],
                                    [0, np.sin(total_angle), np.cos(total_angle), 0],
                                    [0,0,0,1]]) @ self.fib_matrix

        nx, ny = sem_shape
        corners = np.array([[0, 0, 0, 1], [nx, 0, 0, 1], [nx, ny, 0, 1], [0, ny, 0, 1]]).T
        fib_corners = np.dot(self.fib_matrix, corners)
        self.fib_shift = -fib_corners.min(1)[:3]
        self.fib_matrix[:3, 3] += self.fib_shift
        print('Shifted fib matrix: ', self.fib_matrix)

    def apply_fib_transform(self, points, num_slices, scaling=1):
        print('Num slices: ', num_slices)
        if num_slices is None:
            num_slices = 0
        else:
            num_slices *= scaling
        src = np.zeros((points.shape[0], 4))
        dst = np.zeros_like(src)
        for i in range(points.shape[0]):
            src[i,:] = [points[i,0], points[i,1], int(num_slices/2), 1]
            dst[i,:] = self.fib_matrix @ src[i,:]
        self.points = np.array(dst[:,:2])
        if self._orig_points is None:
            self._orig_points = np.copy(self.points)

    def get_selected_region(self, coordinate, transformed):
        coordinate = coordinate.astype(int)
        try:
            if not self._transformed:
                if (0 <= coordinate[0] < self.mcounts.shape[0]) and (0 <= coordinate[1] < self.mcounts.shape[1]):
                    if self.mcounts[coordinate[0],coordinate[1]] > 1:
                        print('Selected region ambiguous. Try again!')
                        return
                    else:
                        counter = 0
                        my_bool = False
                        while not my_bool:
                            x_range = np.arange(self.pos_x[counter],self.pos_x[counter]+self.dimensions[2])
                            y_range = np.arange(self.pos_y[counter],self.pos_y[counter]+self.dimensions[1])
                            if coordinate[0] in x_range and coordinate[1] in y_range:
                                my_bool = True
                            counter += 1
                        print('Selected region: ', counter-1)
                        return counter - 1
            else:
                if (0 <= coordinate[0] < self.tf_count_map.shape[0]) and (0 <= coordinate[1] < self.tf_count_map.shape[1]):
                    if self.tf_count_map[coordinate[0],coordinate[1]] == 0:
                        print('Selected region ambiguous. Try again!')
                        return
                    else:
                        counter = int(self.tf_count_map[coordinate[0],coordinate[1]])
                        print('Selected region: ', counter)
                        return counter
        except(IndexError):
            print('Watch out, index error. Try again!')
            print(coordinate)
            print(x_range.min(),x_range.max())
            print(y_range.min(),y_range.max())

    def select_region(self,coordinate,transformed):
        self.assembled = False
        self.selected_region = self.get_selected_region(coordinate, transformed)
        self._transformed = False
        if self.selected_region is None:
            return
        else:
            with mrc.open(self.old_fname, 'r', permissive=True) as f:
                self.orig_region = np.copy(f.data[self.selected_region].T)

    def calc_stage_positions(self, clicked_points, downsampling):
        if self.eh is not None:
            stage_x = self.eh[4:10*self.dimensions[0]:10]
            stage_y = self.eh[5:10*self.dimensions[0]:10]
            if self.assembled or self.selected_region is None:
                curr_region = 0
            else:
                curr_region = self.selected_region
            self.stage_origin = np.array([stage_x[curr_region],stage_y[curr_region]])
        else:
            self.stage_origin = 0
            self.pixel_size = np.array([1,1])
        inverse_matrix = np.linalg.inv(self.tf_matrix)
        stage_positions = []
        for i in range(len(clicked_points)):
            point = np.array([clicked_points[i][0],clicked_points[i][1],1])
            coordinate_angstrom = (inverse_matrix @ point)[:2] * self.pixel_size[:2] * downsampling
            coordinate_microns = coordinate_angstrom * 10**-4
            stage_positions.append(coordinate_microns + self.stage_origin)       #stage position in microns
        print(stage_positions)
        return stage_positions

    def calc_grid_shift(self, shift_x, shift_y):
        shift = np.array([shift_x, shift_y])
        if self._total_shift is None:
            self._total_shift = shift
        else:
            self._total_shift += shift
        self.points += shift
        self._orig_points = np.copy(self.points)
        self.fib_matrix[:2, 3] = shift

    def calc_refine_matrix(self, src, dst):
        print('Source: \n', src)
        print('Dest: \n', dst)
        refine_matrix = tf.estimate_transform('affine', src, dst).params
        print('Refine matrix i: \n', refine_matrix)

        if self._refine_matrix is None:
            self._refine_matrix = refine_matrix
        else:
            self._refine_matrix = refine_matrix @ self._refine_matrix

        print('Refine matrix: \n', self._refine_matrix)

    def apply_refinement(self, points=None):
        update_points = False
        if points is None:
            points = np.copy(self._orig_points)
            update_points = True
        for i in range(points.shape[0]):
            point = np.array([points[i,0], points[i,1], 1])
            points[i] = (self._refine_matrix @ point)[:2]
        if update_points:
            self.points = points

    def fit_circles(self, points, bead_size):
        points_model = []
        successfull = False
        roi_size = int(np.round(bead_size * 1000 / self.pixel_size[0] + bead_size * 1000 / (2 * self.pixel_size[0])) / 2)
        for i in range(len(points)):
            x = int(np.round(points[i, 0]))
            y = int(np.round(points[i, 1]))
            roi = self.data[x - roi_size:x + roi_size, y - roi_size:y + roi_size]
            np.save('roi_{}.npy'.format(i), roi)
            edges = feature.canny(roi, 3).astype(np.float32)
            coor_x, coor_y = np.where(edges!=0)
            if len(coor_x) != 0:
                rad = bead_size * 1e3 / self.pixel_size[0] / 2 #bead size is supposed to be in microns
                ransac = Ransac(coor_x, coor_y, 100, rad)
                counter = 0
                while True:
                    try:
                        ransac.run()
                        successfull = True
                        break
                    except np.linalg.LinAlgError:
                        counter += 1
                        if counter % 1 == 0:
                            print('\r LinAlgError: ', counter)
                        if counter == 100:
                            break
            if successfull:
                cx, cy = ransac.best_fit[0], ransac.best_fit[1]
                coor = np.array([cx, cy]) + np.array([x, y]).T - np.array([roi_size, roi_size])
                points_model.append(coor)
            else:
                print('Unable to fit bead #{}! Used original coordinate instead!'.format(i))
                points_model.append(np.array([x,y]))
        return np.array(points_model)

    @classmethod
    def get_transform(self, source, dest):
        if len(source) != len(dest):
            print('Point length do not match')
            return
        return tf.estimate_transform('affine', source, dest).params

    def get_fib_transform(self, source, dest, sem_transform):
        fm_to_sem = tf.estimate_transform('affine', source, dest).params
        inv_tf_sem = np.linalg.inv(sem_transform)
        print('FM_to_SEM: \n', fm_to_sem)
        print('inv_tf_sem: \n', inv_tf_sem)
        return inv_tf_sem @ fm_to_sem

