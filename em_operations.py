import sys
import glob
import os
import numpy as np
import scipy.signal as sc
import mrcfile as mrc
import multiprocessing as mp
import matplotlib
import scipy.ndimage as ndi
from skimage import transform as tf
from matplotlib import pyplot as plt
matplotlib.use('QT5Agg')

class EM_ops():
    def __init__(self, step=10):
        self.data_highres = None
        self.step = int(step)
        self.data = None
        self.stacked_data = None
        self.region = None
        self.data_backup = None
        self.transformed = False
        self.pos_x = None
        self.pos_y = None
        self.pos_z = None
        self.grid_points = []
        self.tr_grid_points = []      
        self._tf_data = None
        self._orig_data = None
        self.side_length = None
        self.mcounts = None
        self.tr_mcounts = None
        self.count_map = None
        self.tr_count_map = None
        self.tf_matrix = np.identity(3)
        self.no_shear = False
        self.clockwise = False
        self.rot_angle = None
        self.first_rotation = False
        self.rotated = False
        self.tf_prev = np.identity(3)

    def parse(self, fname):
        with mrc.open(fname, 'r', permissive=True) as f:
            try:
                self.data_highres = f.data
                self.stacked_data = self.data_highres[:,::self.step,::self.step]
            except IndexError:
                self.stacked_data = f.data
            self._h = f.header
            self._eh = np.frombuffer(f.extended_header, dtype='i2')

    def assemble(self):
        dimensions = self.stacked_data.shape
        
        if len(dimensions) == 3:
            self.pos_x = self._eh[1:10*dimensions[0]:10] // self.step
            self.pos_y = self._eh[2:10*dimensions[0]:10] // self.step
            self.pos_z = self._eh[3:10*dimensions[0]:10]
            if len(self.grid_points) != 0:
                self.grid_points = []
            for i in range(len(self.pos_x)):
                point = np.array((self.pos_x[i],self.pos_y[i],1))
                box_points = [point,point+(self.stacked_data.shape[1],0,0),point+(self.stacked_data.shape[1],self.stacked_data.shape[2],0),point+(0,self.stacked_data.shape[2],0)]
                self.grid_points.append(box_points)

            cy, cx = np.indices(dimensions[1:3])

            self.data = np.zeros((np.max(self.pos_x)+dimensions[2],np.max(self.pos_y)+dimensions[1]), dtype='f4')
            #sys.stderr.write(self.data.shape)

            self.mcounts = np.zeros_like(self.data)
            self.count_map = np.zeros_like(self.data)
            for i in range(dimensions[0]):
                sys.stderr.write('\rMerge for image {}'.format(i))
                np.add.at(self.mcounts, (cx+self.pos_x[i], cy+self.pos_y[i]), 1)    
                np.add.at(self.data, (cx+self.pos_x[i], cy+self.pos_y[i]), self.stacked_data[i])
                np.add.at(self.count_map, (cx+self.pos_x[i], cy+self.pos_y[i]), i)
            sys.stderr.write('\n')
            self.data[self.mcounts>0] /= self.mcounts[self.mcounts>0]
            self.count_map[self.mcounts>1] = 0
        else:
            self.data = np.copy(self._stack_data)

        self._orig_data = np.copy(self.data)

    def save_merge(self, fname):
        with mrc.new(fname, overwrite=True) as f:
            f.set_data(self.data)
            f.update_header_stats()

    def toggle_original(self, transformed=None):
        if self._tf_data is None:
            print('Need to transform data first')
            return
        if transformed is None:
            self.data = np.copy(self._tf_data if self.transformed else self._orig_data)
            self.transformed = not self.transformed
        else:
            self.transformed = transformed
            self.data = np.copy(self._tf_data if self.transformed else self._orig_data)
    
    def toggle_region(self,transformed=True,assembled=False):
        if assembled:
                if transformed:
                    self.data = self._tf_data
                else:
                    self.data = self._orig_data
        else:
            self.data = np.copy(self.region)
    
    def calc_affine_transform(self, my_points):
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

        nx, ny = self.data.shape
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]
        print('Transform matrix:\n', self.tf_matrix)
        print('Shift: ', -self.tf_corners.min(1)[:2])
        self.orig_points = np.copy(np.array(my_points))
        self.points = np.copy(self.new_points)
        self.apply_transform()

    def calc_rot_transform(self, my_points):
        #my_points = self.calc_orientation(my_points)
        print('Input points:\n', my_points)

        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))
        self.side_length = np.mean(side_list)
        print('ROI side length:', self.side_length, '\xb1', side_list.std())

        my_points_sorted = np.array(sorted(my_points, key=lambda k: [k[0],k[1]]))
        print('Sorted points:\n',my_points_sorted)

        self.tf_matrix = self.calc_rot_matrix(my_points_sorted)
        nx, ny = self.data.shape
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
        self.points = np.copy(self.new_points)
        self.apply_transform()
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

        manager = mp.Manager()
        dict1 = manager.dict()
        dict2 = manager.dict()
        dict3 = manager.dict()
        p1 = mp.Process(target=self.apply_transform_mp,args=(self.data,dict1))
        p2 = mp.Process(target=self.apply_transform_mp,args=(self.mcounts,dict2))
        p3 = mp.Process(target=self.apply_transform_mp,args=(self.count_map,dict3))
        p1.start()
        p2.start()
        p3.start()
        p1.join()
        p2.join()
        p3.join()
        self._tf_data = np.array(dict1[0])
        self.tr_mcounts = np.array(dict2[0])
        self.tr_count_map = np.array(dict3[0])
        
        #self._tf_data = ndi.affine_transform(self.data, np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
        self.transform_shift = -self.tf_corners.min(1)[:2]
        print(self._tf_data.shape, self.transform_shift)
        self.transformed = True
        self.data = np.copy(self._tf_data)
        self.new_points = np.array([point + self.transform_shift for point in self.new_points])
        self.points = np.copy(self.new_points)
        
        for i in range(len(self.grid_points)):
            tr_box_points = []
            for point in self.grid_points[i]:
                x_i, y_i, z_i = self.tf_matrix @ (self.tf_prev @ point)
                tr_box_points.append(np.array([x_i,y_i,z_i]))
            self.tr_grid_points.append(tr_box_points)
        self.tf_prev = np.copy(self.tf_matrix @ self.tf_prev) 
    

    def apply_transform_mp(self,data,return_dict):
        return_dict[0] = ndi.affine_transform(data, np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
        
    @classmethod
    def get_transform(self, source, dest):
        if len(source) != len(dest):
            print('Point length do not match')
            return
        return tf.estimate_transform('affine', source, dest).params

    def get_selected_region(self, coordinate, transformed):
        coordinate = coordinate.astype(int)
        if not self.transformed:
            if self.mcounts[coordinate[0],coordinate[1]] > 1:
                print('Selected region ambiguous. Try again!')
                return
            else:
                counter = 0
                my_bool = False
                while not my_bool:
                    x_range = np.arange(self.pos_x[counter],self.pos_x[counter]+self.stacked_data.shape[1])
                    y_range = np.arange(self.pos_y[counter],self.pos_y[counter]+self.stacked_data.shape[2])
                    counter += 1
                    if coordinate[0] in x_range and coordinate[1] in y_range:
                        my_bool = True

                print('Selected region: ', counter-1)
                return counter - 1
        else:
            if self.tr_count_map[coordinate[0],coordinate[1]] == 0:
                print('Selected region ambiguous. Try again!')
                return
            else:
                counter = int(self.tr_count_map[coordinate[0],coordinate[1]])
                print('Selected region: ', counter)
                return counter

    def select_region(self,coordinate,transformed):
        counter = self.get_selected_region(coordinate, transformed)
        if counter is None:
            return
        self.region = (self.data_highres[counter]).T
        if not self.transformed:
            self.data = np.copy(self.region)
        else:
            self.toggle_region(transformed=transformed, assembled=False)

if __name__=='__main__':
    path = '../gs.mrc'
    assembler = EM_ops()
    assembler.parse(path)
    assembler.assemble()
    #pg.show(merged.T)
