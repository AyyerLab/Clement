import sys
import glob
import os
import numpy as np
import copy
import scipy.signal as sc
import mrcfile as mrc
import multiprocessing as mp
import scipy.ndimage as ndi
from skimage import transform as tf

class EM_ops():
    def __init__(self, step=10):
        self.data_highres = None
        self.step = int(step)
        self.data = None
        self.stacked_data = None
        self.orig_region = None
        self.tf_region = None
        self.data_backup = None
        self.transformed = False
        self.transformed_data = None   
        self.pos_x = None
        self.pos_y = None
        self.pos_z = None
        self.grid_points = []
        self.tf_grid_points = []      
        self._tf_data = None
        self._orig_data = None
        self.side_length = None
        self.mcounts = None
        self.tf_mcounts = None
        self.count_map = None
        self.tf_count_map = None
        self.tf_matrix = np.identity(3)
        self.no_shear = False
        self.clockwise = False
        self.rot_angle = None
        self.first_rotation = False
        self.rotated = False
        self.tf_prev = np.identity(3)
        self.orig_points = None
        self.new_points = None
        self.orig_points_region = None
        self.tf_points_region = None
        self.selected_region = None
        self.stage_origin = None
        self.points = None
        self.assembled = True
        self.cum_matrix = None
        self.history = [np.identity(3)]

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
            print(self.data.shape)
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

    def toggle_original(self):
        if self.assembled:
            if self.transformed:
                if self._tf_data is not None:
                    self.data = copy.copy(self._tf_data)
                else:
                    self.data = copy.copy(self._orig_data)
                    self.transformed = False
                self.points = copy.copy(self.new_points)
            if not self.transformed:
                self.data = copy.copy(self._orig_data)
                self.points = copy.copy(self.orig_points)
        else:
            if self.transformed:
                if self.tf_region is not None:
                    self.data = copy.copy(self.tf_region)
                else:
                    self.data = copy.copy(self.orig_region)
                    self.transformed = False
                if self.tf_points_region is not None:
                    self.points = copy.copy(self.tf_points_region)
                else:
                    self.points = None 
            else:
                self.data = np.copy(self.orig_region)
                self.points = copy.copy(self.orig_points_region)

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
        if not self.transformed:
            if self.assembled:
                self.orig_points = np.copy(my_points)
            else:
                self.orig_points_region = np.copy(my_points)
        self.apply_transform(points_tmp)

    def calc_rot_transform(self, my_points):
        print('Input points:\n', my_points)
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))
        self.side_length = np.mean(side_list)
        print('ROI side length:', self.side_length, '\xb1', side_list.std())

        #my_points_sorted = np.array(sorted(my_points, key=lambda k: np.cos(30*np.pi/180*k[0]+k[1]))
        #print('Sorted points:\n',my_points_sorted)

        self.tf_matrix = self.calc_rot_matrix(my_points)
        nx, ny = self.data.shape
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]
        print('Tf: ', self.tf_matrix)

        cen = my_points.mean(0) # + self.tf_corners.min(1)[:2] #+ (0,self.side_length/2)

        points_tmp  = np.zeros_like(my_points)
        points_tmp[0] = cen + (0, 0)
        points_tmp[1] = cen + (self.side_length, 0)
        points_tmp[2] = cen + (self.side_length, self.side_length)
        points_tmp[3] = cen + (0,self.side_length)

        if not self.transformed:
            if self.assembled:
                self.orig_points = np.copy(my_points)
            else:
                self.orig_points_region = np.copy(my_points)
        self.apply_transform(points_tmp) 
        self.rotated = True

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
            
            #for i in range(len(angles_deg)):
            angles_deg = [np.min([angle,np.abs((angle%90)-90),np.abs(angle-90)]) for angle in angles_deg] 
            print('angles_deg: ', angles_deg)
            if self.transformed:
                theta = (np.pi/180*np.mean(angles_deg))
            else:
                theta = -(np.pi/180*np.mean(angles_deg))
            tf_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]) 
            return tf_matrix

    def apply_transform(self,pts):
        if self.tf_matrix is None:
            print('Calculate transform matrix first')
            return 
        self.transformed = True

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
        if self.assembled:
            self._tf_data = np.array(dict1[0])
            self.tf_mcounts = np.array(dict2[0])
            self.tf_count_map = np.array(dict3[0])
        else: 
            self.tf_region = np.array(dict1[0])
            self.tf_mcounts = np.array(dict2[0])
            self.tf_count_map = np.array(dict3[0])

        self.transform_shift = -self.tf_corners.min(1)[:2]
        pts = np.array([point + self.transform_shift for point in pts]) 
        if self.assembled:
            self.new_points = np.copy(pts)
        else:
            self.tf_points_region = np.copy(pts)  
        self.toggle_original()
        self.tf_grid_points = []
        
        self.cum_matrix = self.history[0]
        for i in range(1,len(self.history)):
            self.cum_matrix = self.history[i] @ self.cum_matrix

        for i in range(len(self.grid_points)):
            tf_box_points = []
            for point in self.grid_points[i]:
                #x_i, y_i, z_i = self.tf_matrix @ (self.tf_prev @ point)
                x_i, y_i, z_i = self.cum_matrix @ point
                tf_box_points.append(np.array([x_i,y_i,z_i]))
            self.tf_grid_points.append(tf_box_points)
        #self.tf_prev = np.copy(self.tf_matrix @ self.tf_prev) 
        self.history.append(self.tf_matrix)

    def apply_transform_mp(self,data,return_dict):
        return_dict[0] = ndi.affine_transform(data, np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
        
    def get_selected_region(self, coordinate, transformed):
        coordinate = coordinate.astype(int)
        if not self.transformed:
            if (coordinate[0] < self.mcounts.shape[0]) and (coordinate[1] < self.mcounts.shape[1]):
                if self.mcounts[coordinate[0],coordinate[1]] > 1:
                    print('Selected region ambiguous. Try again!')
                    return
                else:
                    counter = 0
                    my_bool = False
                    while not my_bool:
                        x_range = np.arange(self.pos_x[counter],self.pos_x[counter]+self.stacked_data.shape[1])
                        y_range = np.arange(self.pos_y[counter],self.pos_y[counter]+self.stacked_data.shape[2])
                        #counter += 1
                        if coordinate[0] in x_range and coordinate[1] in y_range:
                            my_bool = True
                        counter += 1
                    print('Selected region: ', counter-1)
                    return counter - 1
        else:
            if (coordinate[0] < self.tf_count_map.shape[0]) and (coordinate[1] < self.tf_count_map.shape[1]):
                if self.tf_count_map[coordinate[0],coordinate[1]] == 0:
                    print('Selected region ambiguous. Try again!')
                    return
                else:
                    counter = int(self.tf_count_map[coordinate[0],coordinate[1]])
                    print('Selected region: ', counter)
                    return counter

    def select_region(self,coordinate,transformed):
        self.assembled = False
        self.selected_region = self.get_selected_region(coordinate, transformed)
        self.transformed = False
        if self.selected_region is None:
            return
        else:
            self.orig_region  = np.copy(self.data_highres[self.selected_region].T)
        self.stage_origin = np.array((self.pos_x[self.selected_region]*self.step,self.pos_y[self.selected_region]*self.step))

    def calc_stage_positions(self, clicked_points):
        inverse_matrix = np.linalg.inv(self.history[-1] @ self.cum_matrix)
        stage_positions = []
        for i in range(len(clicked_points)):
            point = np.array([clicked_points[i][0],clicked_points[i][1],1])
            stage_positions.append((inverse_matrix @ point)[:2] + self.stage_origin)
        print(stage_positions)
        return stage_positions

    @classmethod
    def get_transform(self, source, dest):
        if len(source) != len(dest):
            print('Point length do not match')
            return
        return tf.estimate_transform('affine', source, dest).params
