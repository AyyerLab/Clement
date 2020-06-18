import sys
import os
import multiprocessing as mp
import numpy as np
import pyqtgraph as pg
from scipy import signal as sc
from scipy import ndimage as ndi
from scipy import interpolate
from skimage import transform as tf
from skimage import measure, morphology, io, feature
import read_lif
from .ransac import Ransac
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
        self._color_matrices = []
        self._aligned_channels = []
        self._peak_reference = None
        self._point_reference = None

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
        self.flipv_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.fliph_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.transp_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        self.rot_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
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
        self.max_proj_status = False  # status of max_projection before doing the mapping
        self.counter_clockwise = False
        self.corr_matrix = None
        self.merged_2d = None
        self.merged_3d = None
        self.merge_shift = None

    def parse(self, fname, z, series=None, reopen=True):
        ''' Parses file

        Saves parsed file in self.orig_data
        self.data is the array to be displayed
        '''
        if '.tif' in fname or '.tiff' in fname:
            self.tif_data = np.array(io.imread(fname)).astype('f4')
            self.num_slices = self.tif_data.shape[0]

            self.orig_data = self.tif_data[z, :, :, :]
            self.orig_data = self.orig_data
            self.orig_data /= self.orig_data.mean((0, 1))
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
                print(self.voxel_size)
                self.old_fname = fname

            # TODO: Look into modifying read_lif to get
            # a single Z-slice with all channels rather than all slices for a single channel
            self.orig_data = np.array([self.reader.getFrame(channel=i, dtype='u2')[z, :, :].astype('f4')
                                       for i in range(self.num_channels)])
            self.orig_data = self.orig_data.transpose(2, 1, 0)
            self.orig_data /= self.orig_data.mean((0, 1))
            self.data = np.copy(self.orig_data)
            self.selected_slice = z
            [self._aligned_channels.append(False) for i in range(self.num_channels)]
            [self._color_matrices.append(np.identity(3)) for i in range(self.num_channels)]

        if self._transformed:
            self.apply_transform(shift_points=False)
            self._update_data()

    def _update_data(self, update=True, update_points=True):
        if self._transformed and (
                self.tf_data is not None or self.tf_max_proj_data is not None or self.tf_hsv_map is not None or self.tf_hsv_map_no_tilt is not None):
            if self._show_mapping:
                if self._show_no_tilt:
                    self.data = np.copy(self.tf_hsv_map_no_tilt)
                else:
                    self.data = np.copy(self.tf_hsv_map)
            elif self._show_max_proj:
                self.data = np.copy(self.tf_max_proj_data)
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

        if True in self._aligned_channels and not self._show_mapping:
            self.apply_alignment()

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
                self.data = np.rot90(self.data, axes=(0, 1))
            if fliph:
                self.data = np.flip(self.data, axis=0)
            if flipv:
                self.data = np.flip(self.data, axis=1)
            if self.points is not None:
                if update_points:
                    self.points = self.update_points(self.points)

    def update_points(self, points):
        print('Updating points \n', points)
        peaks = None
        if self._transformed:
            if self.tf_peak_slices is not None:
                if self._show_max_proj:
                    peaks = self.orig_tf_peak_slices[-1]
                else:
                    peaks = self.orig_tf_peak_slices[self.selected_slice]
                if peaks is not None:
                    peaks = np.array(peaks)
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
            temp = self.data.shape[0] - 1 - points[:, 1]
            points[:, 1] = points[:, 0]
            points[:, 0] = temp
        if fliph:
            points[:, 0] = self.data.shape[0] - points[:, 0]
        if flipv:
            points[:, 1] = self.data.shape[1] - points[:, 1]

        if peaks is not None and self._transformed:
            if transp:
                peaks = np.array([np.flip(point) for point in peaks])
            if rot:
                temp = self.data.shape[0] - 1 - peaks[:, 1]
                peaks[:, 1] = peaks[:, 0]
                peaks[:, 0] = temp
            if fliph:
                peaks[:, 0] = self.data.shape[0] - peaks[:, 0]
            if flipv:
                peaks[:, 1] = self.data.shape[1] - peaks[:, 1]

            if self._show_max_proj:
                self.tf_peak_slices[-1] = np.copy(peaks)
            else:
                self.tf_peak_slices[self.selected_slice] = np.copy(peaks)
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

    def toggle_original(self, update=True):
        self._update_data(update=update)

    def calc_max_projection(self):
        self._show_max_proj = not self._show_max_proj
        if self.max_proj_data is None:
            self.calc_max_proj_data()
        if self._transformed:
            if self.tf_max_proj_data is None:
                self.apply_transform()
        self._update_data()

    def calc_max_proj_data(self):
        if self.reader is None:
            self.max_proj_data = self.tif_data.max(0)
            self.max_proj_data /= self.max_proj_data.mean((0, 1))
        else:
            self.max_proj_data = np.array([self.reader.getFrame(channel=i, dtype='u2').max(0)
                                           for i in range(self.num_channels)]).transpose(2, 1, 0).astype('f4')
            self.max_proj_data /= self.max_proj_data.mean((0, 1))

    def colorize2d(self, brightness, zvals, cmap_funcs):
        hfunc, sfunc, vfunc = cmap_funcs
        nb = (brightness - brightness.min()) / (brightness.max() - brightness.min())
        nb2 = (brightness - brightness.min()) / (brightness.max() - brightness.min())
        nz = (zvals - zvals.min()) / (zvals.max() - zvals.min())
        hsv = np.zeros(brightness.shape + (3,))
        hsv[:, :, 0] = np.fmod(hfunc(np.clip(nb2, 0, 1), nz), 1.)
        hsv[:, :, 1] = np.clip(sfunc(nb, nz), 0., 1.)
        hsv[:, :, 2] = vfunc(nb, nz)
        return hsv

    def create_cmaps(self, rot=0.):
        points = np.array([[0, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [1, 1]])
        hue = (points[:, 1] * 1.5 + 3) / 6. + rot
        sat = np.array([1., 1., 1. / 3, 1. / 3, 1., 1.])
        val = points[:, 0]
        hfunc = interpolate.LinearNDInterpolator(points, hue)
        sfunc = interpolate.CloughTocher2DInterpolator(points, sat)
        vfunc = interpolate.LinearNDInterpolator(points, val)
        return hfunc, sfunc, vfunc

    def calc_mapping(self):
        self._show_mapping = not self._show_mapping
        if self.hsv_map is None:
            if self.max_proj_data is None:
                self.calc_max_proj_data()
            argmax_map = np.argmax(self.reader.getFrame(channel=self._peak_reference, dtype='u2'), axis=0).astype('f4').transpose((1, 0))
            self.cmap = self.create_cmaps(rot=1. / 2)
            self.hsv_map = self.colorize2d(self.max_proj_data[:, :, self._peak_reference], argmax_map, self.cmap)

        if self._transformed:
            if self.tf_hsv_map is None:
                self.apply_transform()

        self._update_data()

    def remove_tilt(self, remove_tilt):
        self._show_no_tilt = remove_tilt
        if self.hsv_map_no_tilt is None:
            if self.peak_slices is None or self.peak_slices[-1] is None:
                self.peak_finding(self.max_proj_data[:, :, self._peak_reference], transformed=False)
            ref = np.array(self.reader.getFrame(channel=self._peak_reference, dtype='u2').astype('f4')).transpose((2, 1, 0))
            self.fit_z(ref, transformed=False)
            # fit plane to peaks with least squares to remove tilt
            peaks_2d = self.peak_slices[-1]

            A = np.array([peaks_2d[:, 0], peaks_2d[:, 1], np.ones_like(peaks_2d[:, 0])]).T  # matrix
            beta = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), np.expand_dims(self.peaks_z, axis=1))  # params
            point = np.array([0.0, 0.0, beta[2]])  # point on plane
            normal = np.array(np.cross([1, 0, beta[0][0]], [0, 1, beta[1][0]]))  # normal vector of plane
            d = -point.dot(normal)  # distance to origin
            x, y = np.indices((2048, 2048))
            z_plane = -(normal[0] * x + normal[1] * y + d) / normal[2]  # z values of plane for whole image
            z_max_all = np.argmax(ref, axis=2)

            argmax_map_no_tilt = z_max_all - z_plane
            self.hsv_map_no_tilt = self.colorize2d(self.max_proj_data[:, :, self._peak_reference], argmax_map_no_tilt, self.cmap)

        if self._transformed:
            if self.tf_hsv_map_no_tilt is None:
                self.apply_transform()

        self._update_data()

    def calc_z(self, ind, pos, channel=None):
        z = None
        if channel is None:
            channel = self._peak_reference
        if ind is not None:
            z = self.tf_peaks_z[ind]
        else:
            print('Index not found. Calculate local z position!')
            flip_list = [self.transp, self.rot, self.fliph, self.flipv]
            point = np.array((pos[0], pos[1]))
            tf_aligned = self.tf_matrix @ self._color_matrices[channel]
            z = self.calc_local_z(self.channel, point, tf_aligned, flip_list, self.data.shape[:-1])
        if z is None:
            print('Oops, something went wrong. Try somewhere else!')
            return None
        # flip z axis and scale by voxel size ratio
        z = self.num_slices - 1 - z
        scaling = self.voxel_size[2] / self.voxel_size[0]
        z *= scaling
        return z

    def load_channel(self, ind):
        self.channel = np.array(self.reader.getFrame(channel=ind, dtype='u2').astype('f4')).transpose((2, 1, 0))
        print('Load channel {}'.format(ind))

    def clear_channel(self):
        self.channel = None

    def estimate_alignment(self, peaks_2d, idx):
        roi_size = 20
        tmp = []
        ref = []
        for i in range(len(peaks_2d)):
            roi_min_0 = int(peaks_2d[i][0] - roi_size // 2) if int(peaks_2d[i][0] - roi_size // 2) > 0 else 0
            roi_min_1 = int(peaks_2d[i][1] - roi_size // 2) if int(peaks_2d[i][1] - roi_size // 2) > 0 else 0
            roi_max_0 = int(peaks_2d[i][0] + roi_size // 2) if int(peaks_2d[i][0] + roi_size // 2) < self.data.shape[
                0] else self.data.shape[0]
            roi_max_1 = int(peaks_2d[i][1] + roi_size // 2) if int(peaks_2d[i][1] + roi_size // 2) < self.data.shape[
                1] else self.data.shape[1]
            tmp_coor_i = self.peak_finding(self.max_proj_data[:, :, idx][roi_min_0:roi_max_0, roi_min_1:roi_max_1],
                                             transformed=False, roi=True)
            if tmp_coor_i is not None:
                tmp_coor_0 = tmp_coor_i[0] + peaks_2d[i][0] - roi_size // 2 \
                                if tmp_coor_i[0] + peaks_2d[i][0] - roi_size // 2 > 0 else tmp_coor_i[0]
                tmp_coor_1 = tmp_coor_i[1] + peaks_2d[i][1] - roi_size // 2 \
                                if tmp_coor_i[1] + peaks_2d[i][1] - roi_size // 2 > 0 else tmp_coor_i[1]
                tmp_i = np.array((tmp_coor_0, tmp_coor_1))
                diff = np.array(tmp_i - peaks_2d[i])
                if np.linalg.norm(diff) < roi_size:
                    tmp.append(tmp_i)
                    ref.append(peaks_2d[i])
        if len(ref) != 0 and len(tmp) != 0:
            self._color_matrices[idx] = tf.estimate_transform('affine', np.array(tmp), np.array(ref)).params
            self._aligned_channels[idx] = True
        else:
            print('Unable to align channels. Be sure you select a fluorescence channel!')

    def apply_alignment(self):
        for i in range(self.num_channels):
            if self._aligned_channels[i]:
                if self._transformed:
                    color_matrix = self.tf_matrix @ self._color_matrices[i] @ np.linalg.inv(self.tf_matrix)
                else:
                    color_matrix = self._color_matrices[i]
                print(color_matrix)
                self.data[:,:,i] = np.copy(ndi.affine_transform(self.data[:, :, i], np.linalg.inv(color_matrix), order=1))

    def calc_affine_transform(self, my_points):
        my_points = self.calc_orientation(my_points)
        print('Input points:\n', my_points)

        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))

        self.side_length = np.mean(side_list)
        print('ROI side length:', self.side_length, '\xb1', side_list.std())

        cen = my_points.mean(0) - np.ones(2) * self.side_length / 2.
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
        print('Tf shape: ', self._tf_shape)

        self.apply_transform()
        self.points = np.copy(self._tf_points)
        print('New points: \n', self._tf_points)

    def calc_rot_transform(self, my_points):
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))
        self.side_length = np.mean(side_list)
        self.tf_matrix = self.calc_rot_matrix(my_points)
        center = np.mean(my_points, axis=0)
        tf_center = (self.tf_matrix @ np.array([center[0], center[1], 1]))[:2]

        self._tf_points = np.zeros_like(my_points)
        self._tf_points[0] = tf_center + (-self.side_length / 2, -self.side_length / 2)
        self._tf_points[1] = tf_center + (self.side_length / 2, -self.side_length / 2)
        self._tf_points[2] = tf_center + (self.side_length / 2, self.side_length / 2)
        self._tf_points[3] = tf_center + (-self.side_length / 2, self.side_length / 2)

        nx, ny = self.data.shape[:-1]
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] += -self.tf_corners.min(1)[:2]
        print('Tf: ', self.tf_matrix)
        if not self._transformed:
            self._orig_points = np.copy(my_points)
        self.apply_transform()
        self.points = np.copy(self._tf_points)

    def calc_rot_matrix(self, pts):
        angles = []
        ref = np.array([-1, 0])
        for i in range(1, len(pts) + 1):
            if i < len(pts):
                side = pts[i] - pts[i - 1]
            else:
                side = pts[-1] - pts[0]
            if side[1] > 0:
                side *= -1
            angle = np.arctan2(ref[0] * side[1] - ref[1] * side[0], ref[0] * side[0] + ref[1] * side[1])
            if angle < 0:
                angle = angle * -1 + np.pi
            angles.append(angle)

        theta = np.min(angles)
        theta = np.pi / 2 - theta
        tf_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])
        return tf_matrix

    def calc_orientation(self, points):
        my_list = []
        for i in range(1, len(points)):
            my_list.append((points[i][0] - points[i - 1][0]) * (points[i][1] + points[i - 1][1]))
        my_list.append((points[0][0] - points[-1][0]) * (points[0][1] + points[-1][1]))
        my_sum = np.sum(my_list)
        if my_sum > 0:
            print('counter-clockwise')
            self.counter_clockwise = True
            return points
        else:
            print('clockwise --> transpose points')
            order = [0, 3, 2, 1]
            return points[order]
            return tf_matrix

    def apply_transform(self, shift_points=True):
        if not self._transformed:
            self.fliph = False
            self.transp = False
            self.rot = False
            self.flipv = False

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
                self.tf_hsv_map = np.empty(self._tf_shape + (self.hsv_map.shape[-1],))
                for i in range(self.tf_hsv_map.shape[-1]):
                    self.tf_hsv_map[:, :, i] = ndi.affine_transform(self.hsv_map[:, :, i],
                                                                    np.linalg.inv(self.tf_matrix), order=1,
                                                                    output_shape=self._tf_shape)
                    sys.stderr.write('\r%d' % i)
                self._update_data(update_points=False)
            if shift_points and not self._transformed:
                self._tf_points = np.array([point + self.transform_shift for point in self._tf_points])
        else:
            if not self._show_max_proj and self.max_proj_data is None:
                # If max_projection has not yet been selected
                self.tf_data = np.empty(self._tf_shape + (self.data.shape[-1],))
                for i in range(self.data.shape[-1]):
                    self.tf_data[:, :, i] = ndi.affine_transform(self.orig_data[:, :, i], np.linalg.inv(self.tf_matrix),
                                                                 order=1, output_shape=self._tf_shape)
                    sys.stderr.write('\r%d' % i)
                print('\n', self.tf_data.shape)
                if shift_points:
                    self._tf_points = np.array([point + self.transform_shift for point in self._tf_points])
            elif self._show_max_proj and self._transformed:
                # If showing max_projection with image already transformed (???)
                self.tf_max_proj_data = np.empty(self._tf_shape + (self.data.shape[-1],))
                for i in range(self.data.shape[-1]):
                    self.tf_max_proj_data[:, :, i] = ndi.affine_transform(self.max_proj_data[:, :, i],
                                                                          np.linalg.inv(self.tf_matrix), order=1,
                                                                          output_shape=self._tf_shape)
                    sys.stderr.write('\r%d' % i)
                self._update_data(update_points=False)
            else:
                self.tf_data = np.empty(self._tf_shape + (self.data.shape[-1],))
                self.tf_max_proj_data = np.empty(self._tf_shape + (self.data.shape[-1],))
                for i in range(self.data.shape[-1]):
                    self.tf_data[:, :, i] = ndi.affine_transform(self.orig_data[:, :, i], np.linalg.inv(self.tf_matrix),
                                                                 order=1, output_shape=self._tf_shape)
                    self.tf_max_proj_data[:, :, i] = ndi.affine_transform(self.max_proj_data[:, :, i],
                                                                          np.linalg.inv(self.tf_matrix), order=1,
                                                                          output_shape=self._tf_shape)
                    sys.stderr.write('\r%d' % i)
                print('\n', self.tf_data.shape)
                print(self.max_proj_data.shape)
                if shift_points:
                    self._tf_points = np.array([point + self.transform_shift for point in self._tf_points])

#        if self._show_mapping:
#            if self._show_no_tilt:
#                self.data = np.copy(self.tf_hsv_map_no_tilt)
#            else:
#                self.data = np.copy(self.tf_hsv_map)
#        elif self._show_max_proj:
#            self.data = np.copy(self.tf_max_proj_data)
#        else:
#            self.data = np.copy(self.tf_data)

        self._transformed = True
        self._update_data()

    def optimize(self, fm_max, em_img, fm_points, em_points):
        def preprocessing(img, points, size=15, em=False):
            roi = img[points[0] - size:points[0] + size, points[1] - size:points[1] + size]
            if em:
                roi_filtered = ndi.gaussian_filter(roi, sigma=1)
                inverse = 1 / roi_filtered
                inverse[inverse < 0.5 * inverse.max()] = 0
                return inverse
            else:
                roi[roi < 0.5 * np.max(roi)] = 0
                return roi

        em_roi_list = []
        fm_roi_list = []
        size = 15  # actual roi_size = 2*size
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

    def fit_circles(self, points, bead_size):
        points_model = []
        successfull = False
        roi_size = int(
            np.round(bead_size * 1e-6 / self.voxel_size[0] + bead_size * 1e-6 / (2 * self.voxel_size[0])) / 2)
        for i in range(len(points)):
            x = int(np.round(points[i, 0]))
            y = int(np.round(points[i, 1]))
            x_min = (x - roi_size) if (x - roi_size) > 0 else 0
            x_max = (x + roi_size) if (x + roi_size) < self.data.shape[0] else self.data.shape[0]
            y_min = (y - roi_size) if (y - roi_size) > 0 else 0
            y_max = (y + roi_size) if (y + roi_size) < self.data.shape[1] else self.data.shape[1]

            roi = self.data[x_min:x_max, y_min:y_max, -1]
            edges = feature.canny(roi, 3).astype(np.float32)
            coor_x, coor_y = np.where(edges != 0)
            if len(coor_x) != 0:
                rad = bead_size * 1e-6 / self.voxel_size[0] / 2  # bead size is supposed to be in microns
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
                            print('\rLinAlgError: ', counter)
                        if counter == 100:
                            break
            if successfull:
                if ransac.best_fit is not None:  # This should not happen, but it happens sometimes...
                    cx, cy = ransac.best_fit[0], ransac.best_fit[1]
                    coor = np.array([cx, cy]) + np.array([x, y]).T - np.array([roi_size, roi_size])
                    points_model.append(coor)
                # else:
                #    print('Unable to fit bead #{}! Used original coordinate instead!'.format(i))
                #    points_model.append(np.array([x, y]))
            else:
                print('Unable to fit bead #{}! Used original coordinate instead!'.format(i))
                points_model.append(np.array([x, y]))
        return np.array(points_model)

    def apply_merge_2d(self, em_data, em_points, channel):
        if channel == 0:
            src = np.array(sorted(self.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
            dst = np.array(sorted(em_points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
            self.merge_matrix = tf.estimate_transform('affine', src, dst).params
            self.merged_2d = np.zeros(em_data.shape + (self.data.shape[-1] + 1,))
            self.merged_2d[:, :, -1] = em_data / em_data.max() * self.data.max()

        self.merged_2d[:, :, channel] = ndi.affine_transform(self.data[:, :, channel], np.linalg.inv(self.merge_matrix),
                                                             order=1, output_shape=self.merged_2d.shape[:2])
        print('Merged.shape: ', self.merged_2d.shape)

    def apply_merge_3d(self, corr_matrix, fib_matrix, refine_matrix, fib_data, corr_points_fm, fm_z_values,
                       corr_points_fib, channel):
        rot_matrix = np.identity(3)
        if self.transp:
            rot_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        if self.rot:
            rot_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ rot_matrix
        if self.fliph:
            rot_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.array(
                [[0, 1, 0], [1, 0, 0], [0, 0, 1]]) @ rot_matrix
        if self.flipv:
            rot_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) @ np.array(
                [[0, 1, 0], [1, 0, 0], [0, 0, 1]]) @ rot_matrix

        p0 = np.array([0, 0, 0, 1])
        p1 = np.array([0, 0, self.voxel_size[2] / self.voxel_size[0], 1])
        z_shift = (fib_matrix @ p1)[:2] - (fib_matrix @ p0)[:2]

        fib_2d = np.zeros((3, 3))
        fib_2d[:2, :2] = fib_matrix[:2, :2]
        fib_2d[2, 2] = 1

        tf_matrix = np.copy(self.tf_matrix)
        tf_matrix[:2, 2] += self.tf_corners.min(1)[:2]

        if refine_matrix is None:
            refine_matrix = np.identity(3)

        total_matrix = refine_matrix @ fib_2d @ corr_matrix @ rot_matrix @ tf_matrix

        nx, ny = self.channel[:, :, 0].shape
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        tf_corners = total_matrix @ corners
        tf_shape = tuple([int(i) for i in (tf_corners.max(1) - tf_corners.min(1))[:2]])

        if channel == 0:
            flips = [self.transp, self.rot, self.fliph, self.flipv]
            orig_points = []
            tf_aligned_orig_shift = self.tf_matrix @ self._color_matrices[0]
            for i in range(len(corr_points_fm)):
                orig_point = self.calc_original_coordinates(corr_points_fm[i], tf_aligned_orig_shift, flips,
                                                            self.data.shape[:2])
                orig_points.append(orig_point)

            tf_points = []
            tf_img = []
            for i in range(len(orig_points)):
                img_tmp = np.zeros_like(self.channel[:, :, 0])
                img_tmp[int(orig_points[i][0]), int(orig_points[i][1])] = 1
                z = fm_z_values[i] / (self.voxel_size[2] / self.voxel_size[0])
                fib_new = np.copy(fib_2d)
                fib_new[:2, 2] += z * z_shift
                shift_matrix = refine_matrix @ fib_new @ corr_matrix @ rot_matrix @ tf_matrix @ self._color_matrices[0]
                shift_matrix[:2, 2] -= tf_corners.min(1)[:2]
                refined = ndi.affine_transform(img_tmp, np.linalg.inv(shift_matrix), order=1,
                                               output_shape=tf_shape)
                tf_img.append(refined)
                tf_point = np.where(refined == refined.max())
                tf_points.append(np.array([tf_point[0][0], tf_point[1][0]]))
                self.merge_shift = np.mean(tf_points, axis=0) - np.mean(corr_points_fib, axis=0)
            print('IMG shift: ', self.merge_shift)

        z_data = []
        for z in range(self.num_slices):
            fib_new = np.copy(fib_2d)
            z_reverse = self.num_slices - 1 - z
            fib_new[:2, 2] += z_reverse * z_shift
            total_matrix = refine_matrix @ fib_new @ corr_matrix @ rot_matrix @ tf_matrix @ self._color_matrices[channel]
            total_matrix[:2, 2] -= tf_corners.min(1)[:2]
            total_matrix[:2, 2] -= self.merge_shift.T

            refined = np.copy(ndi.affine_transform(self.channel[:, :, z], np.linalg.inv(total_matrix), order=1,
                                           output_shape=fib_data.shape))
            z_data.append(refined)

        if self.merged_3d is None:
            self.merged_3d = np.array(np.max(z_data, axis=0))
            # self.merged_3d = np.array(np.sum(z_data, axis=0))
        else:
            if self.merged_3d.ndim == 2:
                self.merged_3d = np.concatenate(
                    (np.expand_dims(self.merged_3d, axis=2), np.expand_dims(np.max(z_data, axis=0), axis=2)), axis=2)
                # self.merged_3d = np.concatenate((np.expand_dims(self.merged_3d,axis=2), np.expand_dims(np.sum(z_data, axis=0),axis=2)), axis=2)
            else:
                if channel == 2:
                    self.merged_3d = np.concatenate((self.merged_3d, np.expand_dims(np.sum(z_data, axis=0), axis=2)),
                                                    axis=2)
                else:
                    self.merged_3d = np.concatenate((self.merged_3d, np.expand_dims(np.max(z_data, axis=0), axis=2)),
                                                    axis=2)
        if channel == self.num_channels - 1:
            fib_img = np.zeros_like(refined)
            fib_img[:fib_data.shape[0], :fib_data.shape[1]] = fib_data / fib_data.max() * self.merged_3d.max()
            self.merged_3d = np.concatenate((self.merged_3d, np.expand_dims(fib_img, axis=2)), axis=2)
            # self.merged_3d[:,:,2] = self.merged_3d[:,:,2] / self.merged_3d[:,:,2].max() * self.merged_3d[:,:,3].max()

    def update_tr_matrix(self, tr_matrix, flips):
        points = np.copy(self._tf_points)
        transp, rot, fliph, flipv = self.transp, self.rot, self.fliph, self.flipv
        if 0 in flips:
            transp = not self.transp
        if 1 in flips:
            rot = not self.rot
        if 2 in flips:
            fliph = not self.fliph
        if 3 in flips:
            flipv = not self.flipv

        if transp:
            points = np.array([np.flip(point) for point in points])
        if rot:
            temp = self.data.shape[0] - 1 - points[:, 1]
            points[:, 1] = points[:, 0]
            points[:, 0] = temp
        if fliph:
            points[:, 0] = self.data.shape[0] - points[:, 0]
        if flipv:
            points[:, 1] = self.data.shape[1] - points[:, 1]

        rot_matrix = np.linalg.inv(tf.estimate_transform('affine', points, self.points).params)
        return tr_matrix @ rot_matrix

    def get_transform(self, source, dest):
        if len(source) != len(dest):
            print('Point length do not match')
            return
        self.corr_matrix = tf.estimate_transform('affine', source, dest).params
        return self.corr_matrix
