import sys
import os
import multiprocessing as mp
import numpy as np
import pyqtgraph as pg
from scipy import signal as sc
from scipy import ndimage as ndi
from scipy import interpolate
from skimage import transform as tf
from skimage import measure, morphology, io
import read_lif
from .peak_finding import Peak_finding

class FM_ops(Peak_finding):
    def __init__(self):
        super(FM_ops, self).__init__()
        self._show_max_proj = False
        self._orig_points = None
        self._transformed = False
        self._tf_points = None
        self._show_mapping = False
        self._show_no_tilt = False
        self._refine_matrix = None
        self._refine_shape = None

        self.reader = None
        self.voxel_size = None
        self.tif_data = None
        self.orig_data = None
        self.channel = None
        self.tf_data = None
        self.max_proj_data = None
        self.selected_slice = None
        self.data = None
        self.flipv = False
        self.fliph = False
        self.transp = False
        self.rot = False
        self.flipv_matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]])
        self.fliph_matrix = np.array([[-1,0,0],[0,1,0],[0,0,1]])
        self.transp_matrix = np.array([[0,1,0],[1,0,0],[0,0,1]])
        self.rot_matrix = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        self.flip_matrix = np.identity(3)
        self.threshold = 0
        self.max_shift = 10
        self.matches = []
        self.diff_list = []
        self.old_fname = None
        self.points = None
        self.side_length = None
        self.shift = []
        self.transform_shift = 0
        self.tf_matrix = np.identity(3)
        self.tf_max_proj_data = None
        self.cmap = None
        self.hsv_map = None
        self.tf_hsv_map = None
        self.hsv_map_no_tilt = None
        self.tf_hsv_map_no_tilt = None
        self.max_proj_status = False #status of max_projection before doing the mapping
        self.counter_clockwise = False
        self.rotated = False
        self.corr_matrix = None
        self.refine_points = None
        self.refine_history = []
        self.refined = False
        self.refined_max_proj = None
        self.refined_data = None
        self.merged = None

    def parse(self, fname, z, series=None, reopen=True):
        ''' Parses file

        Saves parsed file in self.orig_data
        self.data is the array to be displayed
        '''
        if '.tif' in fname or '.tiff' in fname:
            self.tif_data = np.array(io.imread(fname)).astype('f4')
            self.num_slices = self.tif_data.shape[0]

            self.orig_data = self.tif_data[z,:,:,:]
            self.orig_data = self.orig_data
            self.orig_data /= self.orig_data.mean((0,1))
            print(self.orig_data.shape)
            self.data = np.copy(self.orig_data)
            self.old_fname = fname
            self.num_channels = self.orig_data.shape[-1]
            self.selected_slice = z
        else:
            if reopen:
                base_reader = read_lif.Reader(fname)
                if len(base_reader.getSeries()) == 1:
                    self.reader = base_reader.getSeries()[0]
                elif series is not None:
                    self.reader = base_reader.getSeries()[series]
                else:
                    return [s.getName() for s in base_reader.getSeries()]

                self.num_slices = self.reader.getFrameShape()[0]
                self.num_channels = len(self.reader.getChannels())
                md = self.reader.getMetadata()
                self.voxel_size = np.array([md['voxel_size_x'], md['voxel_size_y'], md['voxel_size_z']]) * 1e-6
                self.old_fname = fname

            # TODO: Look into modifying read_lif to get
            # a single Z-slice with all channels rather than all slices for a single channel
            self.orig_data = np.array([self.reader.getFrame(channel=i, dtype='u2')[:,:,z].astype('f4')
                                        for i in range(self.num_channels)])
            self.orig_data = self.orig_data.transpose(1,2,0)
            self.orig_data /= self.orig_data.mean((0, 1))
            self.data = np.copy(self.orig_data)
            self.selected_slice = z


        if self.refined:
            refined_tmp = True
            self.refined = False
        else:
            refined_tmp = False

        if self._transformed:
            self.apply_transform(shift_points=False)
            self._update_data()

        print('Refined?: ', self.refined)
        if refined_tmp and self._transformed:
            for i in range(len(self.refine_history)):
                self._refine_matrix = self.refine_history[i]
                self.apply_refinement()
        if refined_tmp:
            self.refined = True

    def _update_data(self,update=True,update_points=True):
        if self._transformed and (self.tf_data is not None or self.tf_max_proj_data is not None or self.tf_hsv_map is not None or self.tf_hsv_map_no_tilt is not None):
            if self._show_mapping:
                if self._show_no_tilt:
                    self.data = np.copy(self.tf_hsv_map_no_tilt)
                else:
                    self.data = np.copy(self.tf_hsv_map)
            elif self._show_max_proj:
                if self.refined:
                    self.data = np.copy(self.refined_max_proj)
                else:
                    self.data = np.copy(self.tf_max_proj_data)
            else:
                if self.refined:
                    self.data = np.copy(self.refined_data)
                else:
                    self.data = np.copy(self.tf_data)
            self.points = np.copy(self._tf_points)
        else:
            if self._show_mapping and self.hsv_map is not None:
                if self._show_no_tilt:
                    self.data = np.copy(self.hsv_map_no_tilt)
                else:
                    self.data = np.copy(self.hsv_map)
            elif self._show_max_proj and self.max_proj_data is not None:
                self.data = np.copy(self.max_proj_data)
            else:
                self.data = np.copy(self.orig_data)
            self.points = np.copy(self._orig_points) if self._orig_points is not None else None

        if update:
            if self._transformed:
                fliph = self.fliph
                flipv = self.flipv
                transp = self.transp
                rot = self.rot
            else:
                fliph = False
                flipv = False
                transp = False
                rot = False

            if transp:
                self.data = np.transpose(self.data, (1, 0, 2))
            if rot:
                self.data = np.rot90(self.data, axes=(0,1))
            if fliph:
                self.data = np.flip(self.data, axis=0)
            if flipv:
                self.data = np.flip(self.data, axis=1)
            if self.points is not None:
                if update_points:
                    self.points = self.update_points(self.points)

    def update_points(self,points):
        if self._transformed:
            fliph = self.fliph
            flipv = self.flipv
            transp = self.transp
            rot = self.rot
        else:
            fliph = False
            flipv = False
            transp = False
            rot = False

        if transp:
            points = np.array([np.flip(point) for point in points])
        if rot:
            temp = self.data.shape[0] - 1 - points[:,1]
            points[:,1] = points[:,0]
            points[:,0] = temp
        if fliph:
            points[:,0] = self.data.shape[0] - points[:,0]
        if flipv:
            points[:,1] = self.data.shape[1] - points[:,1]
        print('Updating points \n', points)
        return points

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

    def toggle_original(self,update=True):
        self._update_data(update=update)

    def calc_max_projection(self):
        self._show_max_proj = not self._show_max_proj
        if self.refined:
            refined_tmp = True
            self.refined = False
        else:
            refined_tmp = False

        if self.max_proj_data is None:
            self.calc_max_proj_data()
        if self._transformed:
            if self.tf_max_proj_data is None:
                self.apply_transform()
            elif self.refined:
                self.apply_transform()
        self._update_data()
        if refined_tmp and self._transformed:
        #if self.refined:
            for i in range(len(self.refine_history)):
                self._refine_matrix = self.refine_history[i]
                self.apply_refinement()
        if refined_tmp:
            self.refined = True

    def calc_max_proj_data(self):
        if self.reader is None:
            self.max_proj_data = self.tif_data.max(0)
            self.max_proj_data /= self.max_proj_data.mean((0, 1))
        else:
            self.max_proj_data = np.array([self.reader.getFrame(channel=i, dtype='u2').max(2)
                                           for i in range(self.num_channels)]).transpose(1, 2, 0).astype('f4')
            self.max_proj_data /= self.max_proj_data.mean((0, 1))

    def colorize2d(self, brightness, zvals, cmap_funcs):
        hfunc, sfunc, vfunc = cmap_funcs
        nb = (brightness-brightness.min()) / (brightness.max()-brightness.min())
        nb2 = (brightness-brightness.min()) * (0.8-0.2) / (brightness.max()-brightness.min()) + 0.2
        nz = (zvals-zvals.min()) / (zvals.max() - zvals.min())
        hsv = np.zeros(brightness.shape + (3,))
        hsv[:,:,0] = np.fmod(hfunc(np.clip(nb2, 0, 1), nz), 1.)
        hsv[:,:,1] = np.clip(sfunc(nb, nz), 0., 1.)
        hsv[:,:,2] = vfunc(nb, nz)
        return hsv

    def create_cmaps(self, rot=0.):
        points = np.array([[0,0],[1,0],[0,0.5],[1,0.5],[0,1],[1,1]])
        hue = (points[:,1]*1.5+3)/6. + rot
        sat = np.array([1., 1., 1./3, 1./3, 1., 1.])
        val = points[:,0]
        hfunc = interpolate.LinearNDInterpolator(points, hue)
        sfunc = interpolate.CloughTocher2DInterpolator(points, sat)
        vfunc = interpolate.LinearNDInterpolator(points, val)
        return hfunc, sfunc, vfunc

    def calc_mapping(self):
        self._show_mapping = not self._show_mapping
        if self.refined:
            refined_tmp = True
            self.refined = False
        else:
            refined_tmp = False
        if self.hsv_map is None:
            if self.max_proj_data is None:
               self.calc_max_proj_data()
            argmax_map = np.argmax(self.reader.getFrame(channel=3, dtype='u2'), axis=2).astype('f4')
            self.cmap = self.create_cmaps(rot=1./2)
            hsv_data = np.array([self.max_proj_data[:,:,-1], argmax_map, self.cmap])
            np.save('hsv_data.npy', hsv_data)
            self.hsv_map = self.colorize2d(self.max_proj_data[:,:,-1], argmax_map, self.cmap)

        if self._transformed:
            if self.tf_hsv_map is None:
                self.apply_transform()
            elif self.refined:
                self.apply_transform()

        self._update_data()

        if refined_tmp and self._transformed:
            # if self.refined:
            for i in range(len(self.refine_history)):
                self._refine_matrix = self.refine_history[i]
                self.apply_refinement()
        if refined_tmp:
            self.refined = True

    def remove_tilt(self, remove_tilt):
        self._show_no_tilt = remove_tilt
        if self.refined:
            refined_tmp = True
            self.refined = False
        else:
            refined_tmp = False
        if self.hsv_map_no_tilt is None:
            if self.peak_slices is None or self.peak_slices[-1] == 0:
                self.peak_finding(self.max_proj_data[:,:,-1], transformed=False)
            red_channel = np.array(self.reader.getFrame(channel=3, dtype='u2').astype('f4'))
            self.fit_z(red_channel, transformed=False)
            # fit plane to peaks with least squares to remove tilt
            peaks_2d = self.peak_slices[-1]

            A = np.array([peaks_2d[:,0], peaks_2d[:,1], np.ones_like(peaks_2d[:,0])]).T #matrix
            beta = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), np.expand_dims(self.peaks_3d[:,2], axis=1)) #params

            point = np.array([0.0, 0.0, beta[2]]) #point on plane
            normal = np.array(np.cross([1, 0, beta[0][0]], [0, 1, beta[1][0]])) #normal vector of plane
            d = -point.dot(normal) #distance to origin

            x, y = np.indices((2048, 2048))
            z_plane = -(normal[0] * x + normal[1] * y + d) / normal[2] #z values of plane for whole image
            z_max_all = np.argmax(red_channel, axis=2)

            argmax_map_no_tilt = z_max_all - z_plane
            #argmax_map_no_tilt /= argmax_map_no_tilt.max() * 2
            #self.hsv_map_no_tilt = np.array([argmax_map_no_tilt, np.ones_like(argmax_map_no_tilt), self.max_proj_data[:,:,-1]]).transpose(1,2,0)
            self.hsv_map_no_tilt = self.colorize2d(self.max_proj_data[:,:,-1], argmax_map_no_tilt, self.cmap)

        if self._transformed:
            if self.tf_hsv_map_no_tilt is None:
                self.apply_transform()
                #if self.tf_peak_slices is None or self.tf_peak_slices[-1] is None:
                #    flip_list = [self.transp, self.rot, self.fliph, self.flipv]
                #self.calc_transformed_coordinates(self.tf_matrix, flip_list, self.data.shape[:-1])
            elif self.refined:
                self.apply_transform()

        self._update_data()

        if refined_tmp and self._transformed:
            # if self.refined:
            for i in range(len(self.refine_history)):
                self._refine_matrix = self.refine_history[i]
                self.apply_refinement()
        if refined_tmp:
            self.refined = True

    def calc_z(self, ind, pos): #size
        z = None
        if self._transformed:
            if self.tf_peaks_3d is not None:
                if ind is not None:  # do not replace by else-statement!!!
                    z = self.tf_peaks_3d[ind, 2]
            if z is not None:
                print('Index found! ind, z:', ind, z)
        else:
            if self.peaks_3d is not None:
                if ind is not None:  # do not replace by else-statement!!!
                    z = self.peaks_3d[ind, 2]
        if z is None:
            print('Index not found. Calculate local z position!')
            flip_list = [self.transp, self.rot, self.fliph, self.flipv]
            z = self.calc_local_z(self.channel, np.array((pos.x(), pos.y())), self._transformed,
                                      self.tf_matrix, flip_list, self.data.shape[:-1])

        if z is None:
            print('Oops, something went wrong. Try again!')
            return None
        #scale z slice number by voxel size ratio
        z = 30 - z
        scaling = self.voxel_size[0] / self.voxel_size[2]
        print('Z value before scaling: ', z)
        #z *= 1/scaling * 2
        z *= 1/scaling
        print('Scaling: ', scaling)
        return z

    def load_channel(self, ind):
        self.channel = np.array(self.reader.getFrame(channel=ind, dtype='u2').astype('f4'))
        print('Load channel {}'.format(ind))

    def clear_channel(self):
        self.channel = None

    def align(self):
        '''
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
        '''
        pass

    def calc_affine_transform(self, my_points):
        my_points = self.calc_orientation(my_points)
        print('Input points:\n', my_points)

        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))

        self.side_length = np.mean(side_list)
        print('ROI side length:', self.side_length, '\xb1', side_list.std())

        cen = my_points.mean(0) - np.ones(2)*self.side_length/2.
        self._tf_points = np.zeros_like(my_points)
        self._tf_points[0] = cen + (0, 0)
        self._tf_points[1] = cen + (self.side_length, 0)
        self._tf_points[2] = cen + (self.side_length, self.side_length)
        self._tf_points[3] = cen + (0, self.side_length)

        self.tf_matrix = tf.estimate_transform('affine', my_points, self._tf_points).params

        nx, ny = self.data.shape[:-1]
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.corners = np.copy(corners)
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]
        print('Transform matrix:\n', self.tf_matrix)

        self.apply_transform()
        self.points = np.copy(self._tf_points)
        print('New points: \n', self._tf_points)

    def calc_rot_transform(self, my_points):
        print('Input points:\n', my_points)
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))
        self.side_length = np.mean(side_list)
        print('ROI side length:', self.side_length, '\xb1', side_list.std())

        self.tf_matrix = self.calc_rot_matrix(my_points)

        center = np.mean(my_points,axis=0)
        tf_center = (self.tf_matrix @ np.array([center[0],center[1],1]))[:2]

        nx, ny = self.data.shape[:-1]
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] += -self.tf_corners.min(1)[:2]
        print('Tf: ', self.tf_matrix)

        self._tf_points = np.zeros_like(my_points)
        self._tf_points[0] = tf_center + (-self.side_length/2, -self.side_length/2)
        self._tf_points[1] = tf_center + (self.side_length/2, -self.side_length/2)
        self._tf_points[2] = tf_center + (self.side_length/2, self.side_length/2)
        self._tf_points[3] = tf_center + (-self.side_length/2,self.side_length/2)

        if not self._transformed:
            self._orig_points = np.copy(my_points)
        self.apply_transform()
        self.points = np.copy(self._tf_points)
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

    def apply_transform(self,shift_points=True):
        if not self._transformed:
            self.fliph = False
            self.transp = False
            self.rot = False
            self.flipv = False
            self.refined = False

        if self.tf_matrix is None:
            print('Calculate transform matrix first')
            return

        # Calculate transform_shift for point transforms
        self.transform_shift = -self.tf_corners.min(1)[:2]

        if self._show_mapping:
            if self._show_no_tilt:
                self.tf_hsv_map_no_tilt = np.empty(self._tf_shape + (self.hsv_map_no_tilt.shape[-1],))
                for i in range(self.tf_hsv_map_no_tilt.shape[-1]):
                    self.tf_hsv_map_no_tilt[:, :, i] = ndi.affine_transform(self.hsv_map_no_tilt[:, :, i],
                                                                    np.linalg.inv(self.tf_matrix), order=1,
                                                                    output_shape=self._tf_shape)
                    sys.stderr.write('\r%d' % i)
                self._update_data(update_points=False)
                print(self.tf_hsv_map_no_tilt.shape)

            else:
                self.tf_hsv_map = np.empty(self._tf_shape+(self.hsv_map.shape[-1],))
                for i in range(self.tf_hsv_map.shape[-1]):
                    self.tf_hsv_map[:,:,i] = ndi.affine_transform(self.hsv_map[:,:,i], np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
                    sys.stderr.write('\r%d'%i)
                self._update_data(update_points=False)
            if shift_points and not self._transformed:
                self._tf_points = np.array([point + self.transform_shift for point in self._tf_points])
        else:
            if not self._show_max_proj and self.max_proj_data is None:
                # If max_projection has not yet been selected
                self.tf_data = np.empty(self._tf_shape+(self.data.shape[-1],))
                for i in range(self.data.shape[-1]):
                    self.tf_data[:,:,i] = ndi.affine_transform(self.orig_data[:,:,i], np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
                    sys.stderr.write('\r%d'%i)
                print('\n', self.tf_data.shape)
                if shift_points:
                    self._tf_points = np.array([point + self.transform_shift for point in self._tf_points])
            elif self._show_max_proj and self._transformed:
                # If showing max_projection with image already transformed (???)
                self.tf_max_proj_data  = np.empty(self._tf_shape+(self.data.shape[-1],))
                for i in range(self.data.shape[-1]):
                    self.tf_max_proj_data[:,:,i] = ndi.affine_transform(self.max_proj_data[:,:,i], np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
                    sys.stderr.write('\r%d'%i)
                self._update_data(update_points=False)
            else:
                self.tf_data = np.empty(self._tf_shape+(self.data.shape[-1],))
                self.tf_max_proj_data  = np.empty(self._tf_shape+(self.data.shape[-1],))
                for i in range(self.data.shape[-1]):
                    self.tf_data[:,:,i] = ndi.affine_transform(self.orig_data[:,:,i], np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
                    self.tf_max_proj_data[:,:,i] = ndi.affine_transform(self.max_proj_data[:,:,i], np.linalg.inv(self.tf_matrix), order=1, output_shape=self._tf_shape)
                    sys.stderr.write('\r%d'%i)
                print('\n', self.tf_data.shape)
                print(self.max_proj_data.shape)
                if shift_points:
                    self._tf_points = np.array([point + self.transform_shift for point in self._tf_points])

        if self._show_mapping:
            if self._show_no_tilt:
                self.data = np.copy(self.tf_hsv_map_no_tilt)
            else:
                self.data = np.copy(self.tf_hsv_map)
        elif self._show_max_proj:
            self.data = np.copy(self.tf_max_proj_data)
        else:
            self.data = np.copy(self.tf_data)

        self._transformed = True

    def calc_refine_matrix(self, src, dst):
        if self.tf_data is not None:
            self._refine_matrix = tf.estimate_transform('affine', src, dst).params
            nx, ny = self.data.shape[:-1]
            corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
            refine_corners = np.dot(self._refine_matrix, corners)
            self._refine_shape = tuple([int(i) for i in (refine_corners.max(1) - refine_corners.min(1))[:2]])
            self.refine_history.append(self._refine_matrix)
            self._refine_matrix[:2, 2] -= refine_corners.min(1)[:2]
            print(self._refine_matrix)

    def apply_refinement(self):
        data_tmp = np.copy(self.data)
        self.data = np.empty(self._refine_shape+(self.data.shape[-1],))
        for i in range(self.data.shape[-1]):
            self.data[:,:,i] = ndi.affine_transform(data_tmp[:,:,i], np.linalg.inv(self._refine_matrix), order=1, output_shape=self._refine_shape)
            sys.stderr.write('\r%d'%i)
        if self._show_max_proj:
            self.refined_max_proj = np.copy(self.data)
        else:
            self.refined_data = np.copy(self.data)
        print('\r', self.data.shape)
        self.refined = True

    def refine_grid(self, src, dst, em_points):
        print('refine grid')
        corr_matrix_new = tf.estimate_transform('affine',src,dst).params
        grid_matrix = self._refine_matrix @ np.linalg.inv(corr_matrix_new)
        self._tf_points = np.array([(grid_matrix @ np.array([point[0],point[1],1]))[:2] for point in em_points])
        self.points = np.copy(self._tf_points)

    def optimize(self, fm_max, em_img, fm_points, em_points):
        def preprocessing(img, points, size=15, em=False):
            roi = img[points[0]-size:points[0]+size, points[1]-size:points[1]+size]
            if em:
                roi_filtered = ndi.gaussian_filter(roi, sigma=1)
                inverse = 1/roi_filtered
                inverse[inverse<0.5*inverse.max()] = 0
                return inverse
            else:
                roi[roi<0.5*np.max(roi)] = 0
                return roi

        em_roi_list = []
        fm_roi_list = []
        size = 15 # actual roi_size = 2*size
        for i in range(len(em_points)):
            em_roi_list.append(preprocessing(em_img, em_points[i], em=True))
            fm_roi_list.append(preprocessing(fm_max, fm_points[i]))

        fm_coor_list = []

        em_coor_list = []

        for i in range(len(em_roi_list)):
            fm_coor = self.peak_finding(fm_roi_list[i], None, roi=True)
            print(fm_coor)
            fm_coor_list.append(list(fm_coor + np.array(fm_points[i] - size)))

            em_coor = self.peak_finding(em_roi_list[i], None, roi=True)
            em_coor_list.append(list(em_coor + np.array(em_points[i] - size)))

        return fm_coor_list, em_coor_list



    def calc_merge_matrix(self, em_data, em_points):
        src = np.array(sorted(self.points, key=lambda k: [np.cos(30*np.pi/180)*k[0] + k[1]]))
        dst = np.array(sorted(em_points, key=lambda k: [np.cos(30*np.pi/180)*k[0] + k[1]]))
        self.merge_matrix = tf.estimate_transform('affine', src, dst).params
        '''
        nx, ny = self.data.shape[:-1]
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.merge_fm_corners = np.dot(self.merge_matrix, corners)
        self._merge_fm_shape = tuple([int(i) for i in (self.merge_fm_corners.max(1) - self.merge_fm_corners.min(1))[:2]])
        shift = self.merge_fm_corners.min(1)[:2]
        self.merge_matrix[:2, 2] -= shift

        self.fm_shift = np.zeros(2).astype(int)
        self.em_shift = np.zeros(2).astype(int)
        if shift[0] < 0:
            self.em_shift[0] = np.abs(int(shift[0]))
        else:
            self.fm_shift[0] = int(shift[0])
        if shift[1] < 0:
            self.em_shift[1] = np.abs(int(shift[1]))
        else:
            self.fm_shift[1] = int(shift[1])

        self.merge_fm_data = np.empty(self._merge_fm_shape+(self.data.shape[-1],))
        x_shape = np.max([em_data.shape[0]+self.em_shift[0],self.merge_fm_data.shape[0]+self.fm_shift[0]]).astype(int)
        y_shape = np.max([em_data.shape[1]+self.em_shift[1],self.merge_fm_data.shape[1]+self.fm_shift[1]]).astype(int)
        self.merged = np.zeros((x_shape,y_shape,self.data.shape[-1]+1))
        self.merged[self.em_shift[0]:em_data.shape[0]+self.em_shift[0],self.em_shift[1]:em_data.shape[1]+self.em_shift[1],-1] = em_data/np.max(em_data)*np.max(self.data)
        '''

        self.merged = np.zeros(em_data.shape + (self.data.shape[-1]+1,))
        self.merged[:,:,-1] = em_data / em_data.max() * self.data.max()
        self.apply_merge()

    def apply_merge(self):
        for i in range(self.data.shape[-1]):
            self.merged[:,:,i] = ndi.affine_transform(self.data[:,:,i], np.linalg.inv(self.merge_matrix), order=1, output_shape=self.merged.shape[:2])
            sys.stderr.write('\r%d'%i)

        print('Merged.shape: ', self.merged.shape)
        #self.merged[self.fm_shift[0]:self._merge_fm_shape[0]+self.fm_shift[0],self.fm_shift[1]:self._merge_fm_shape[1]+self.fm_shift[1],:-1] = self.merge_fm_data

    def get_transform(self, source, dest):
        if len(source) != len(dest):
            print('Point length do not match')
            return
        self.corr_matrix = tf.estimate_transform('affine', source, dest).params
        return self.corr_matrix
