import sys
import os.path as op

import numpy as np
from scipy import ndimage as ndi
from scipy import interpolate
from skimage import transform as tf
from skimage import measure, morphology, io, feature
import read_lif
import tifffile
import xmltodict

from .ransac import Ransac
from .peak_finding import Peak_finding



class FM_ops(Peak_finding):
    def __init__(self, printer, logger):
        super(FM_ops, self).__init__()
        self._show_max_proj = False
        self._orig_points = None
        self._transformed = False
        self._tf_points = None
        self._show_mapping = False
        self._show_no_tilt = False
        self._color_matrices = []
        self._aligned_channels = []
        self._channel_idx = None

        self.print = printer
        self.log = logger

        self.base_reader = None
        self.reader = None
        self.dtype = None
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
        self.fixed_orientation = False
        self.flip_z = False
        self.sem_transform = np.identity(3)
        self.flipv_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.fliph_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.transp_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        self.rot_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        self.flip_matrix = np.identity(3)
        self.flip_z = False
        self.max_shift = 10
        self.matches = []
        self.diff_list = []
        self.old_fname = None
        self.points = None
        self.side_length = None
        self.shift = []
        self.transform_shift = 0
        self.tf_matrix = np.identity(3)
        self.tf_matrix_orig = np.identity(3)
        self.tf_max_proj_data = None
        self.cmap = None
        self.hsv_map = None
        self.tf_hsv_map = None
        self.hsv_map_no_tilt = None
        self.tf_hsv_map_no_tilt = None
        self.max_proj_status = False  # status of max_projection before doing the mapping
        self.counter_clockwise = False
        self.norm_factor = 100

    def parse(self, fname, z, series=None, reopen=True):
        ''' Parses file

        Saves parsed file in self.orig_data
        self.data is the array to be displayed
        '''
        if op.splitext(fname)[1] == '.xml' and fname == self.old_fname:
            self.orig_data = self.tif_data[:,:,z]
            self.old_fname = fname
            self.data = np.copy(self.orig_data)
            self.selected_slice = z
        elif op.splitext(fname)[1] == '.xml':
            with open(fname, 'r') as f:
                meta = xmltodict.parse(f.read())

            if list(meta.keys())[0] == 'TfsData':
                self.print('Detected IFLM XML file')
            else:
                self.print('Unknown XML format: ' + list(meta.keys())[0])
                return

            images = meta['TfsData']['ImageMatrix'][0]['Images']['Image']
            channels = meta['TfsData']['ImageMatrix'][0]['Channels']['Channel']

            self.num_channels = 1 if isinstance(channels, dict) else len(channels)
            assert len(images) % self.num_channels == 0
            self.num_slices = len(images) // self.num_channels

            imchannels = [int(imdict['Index']['Channel']) for imdict in images]
            implanes = np.array([int(imdict['Index']['Plane']) for imdict in images])
            impaths = [op.join(op.dirname(fname), imdict['RelativePath']).replace('\\', '/') for imdict in images]
            imshape = io.imread(impaths[0]).shape

            self.tif_data = np.empty(imshape + (self.num_slices, self.num_channels), dtype='f4')
            for i in range(len(images)):
                self.tif_data[:,:,implanes[i],imchannels[i]] = io.imread(impaths[i])

            # Voxel size in um
            imatrix = meta['TfsData']['ImageMatrix'][0]
            vx = float(imatrix['TileWidth']) / float(imatrix['TilePixelWidth'])
            vy = float(imatrix['TileHeight']) / float(imatrix['TilePixelHeight'])
            vz = float(images[np.where(implanes==1)[0][0]]['Position']['sem:Position']['Focus'])
            self.voxel_size = np.array([vx, vy, vz]) * 1e-6
            print('Voxel size:', self.voxel_size)

            self.orig_data = self.tif_data[:,:,z]
            self.old_fname = fname
            self.data = np.copy(self.orig_data)
            self.selected_slice = z
            [self._aligned_channels.append(False) for i in range(self.num_channels)]
            [self._color_matrices.append(np.identity(3)) for i in range(self.num_channels)]
        elif '.tif' in fname or '.tiff' in fname:
            self.tif_data = np.array(io.imread(fname)).astype('f4')
            if self.tif_data.ndim == 2:
                self.print('Single FM image. Are you sure you do not want a stack?')
                self.tif_data = self.tif_data.T[:,:,None,None]
            #self.tif_data = np.array(io.imread(fname)).astype('f4').transpose(2,1,0)
            self.num_slices = self.tif_data.shape[2]

            self.orig_data = self.tif_data[:,:,z]
            if self.orig_data.ndim == 2:
                self.orig_data = np.expand_dims(self.orig_data, axis=-1)
                self.tif_data = np.expand_dims(self.tif_data, axis=-1)

            self.num_channels = self.orig_data.shape[-1]
            for i in range(self.num_channels):
                self.orig_data[:,:,i] = (self.orig_data[:,:,i] - self.orig_data[:,:,i].min()) / \
                                          (self.orig_data[:,:,i].max() - self.orig_data[:,:,i].min())
                self.orig_data[:,:,i] *= self.norm_factor
                self.log(self.orig_data.shape)
            self.data = np.copy(self.orig_data)
            self.old_fname = fname
            self.selected_slice = z
            md_raw = tifffile.TiffFile(fname).imagej_metadata
            chapter = None
            if md_raw is not None and 'Info' in md_raw:
                chapter = 'Info'
            elif md_raw is not None and 'Labels' in md_raw:
                chapter = 'Labels'
            else:
                self.print('Unable to find metadata! Contact developers!')
                print('Unable to find metadata! Contact developers!')
                return
            labels = str(md_raw[chapter]).split(' ')
            md = {}
            for l in labels:
                i = l.split('=')
                if len(i) > 1:
                    md[i[0]] = i[1].strip('\"')
            self.voxel_size = np.array([float(md['PhysicalSizeX']), float(md['PhysicalSizeY']), float(md['PhysicalSizeZ'])]) * 1e-6
        else:
            if reopen:
                self.base_reader = read_lif.Reader(fname)
                if len(self.base_reader.getSeries()) == 1:
                    self.reader = self.base_reader.getSeries()[0]
                elif series is not None:
                    self.reader = self.base_reader.getSeries()[series]
                else:
                    return [s.getName() for s in self.base_reader.getSeries()]

                self.dtype = self.reader.getBytesInc(1)

                self.num_slices = self.reader.getFrameShape()[0]
                self.num_channels = len(self.reader.getChannels())

                md = self.reader.getMetadata()
                print('Voxel size unit: ', md['voxel_size_unit'])
                if md['voxel_size_unit'] == 'um':
                    self.voxel_size = np.array([md['voxel_size_x'], md['voxel_size_y'], md['voxel_size_z']]) * 1e-6
                elif md['voxel_size_unit'] == 'nm':
                    self.voxel_size = np.array([md['voxel_size_x'], md['voxel_size_y'], md['voxel_size_z']]) * 1e-9
                else:
                    print('Wrong voxel size unit. Contact developers')
                    return
            self.print('Voxel size: ', self.voxel_size)
            self.old_fname = fname

            # TODO: Look into modifying read_lif to get
            # a single Z-slice with all channels rather than all slices for a single channel
            #self.orig_data = np.array([self.reader.getFrame(channel=i)[z, :, :].astype('f4')
            self.orig_data = np.array([self.reader.getFrame(channel=i, dtype='u{}'.format(self.dtype))[z, :, :].astype('f4')
                                       for i in range(self.num_channels)])
            if self.num_channels < 3:
                # somehow tricks pyqtgraph when clement is started to be able to generate rgb when num_channels < 3,
                for i in range(3-self.num_channels):
                    self.orig_data = np.concatenate((self.orig_data, np.random.random(self.orig_data[0].shape)[np.newaxis,...]/1e6))

            self.orig_data = self.orig_data.transpose(1, 2, 0)
            #normalize to 100
            for i in range(self.orig_data.shape[-1]):
                self.orig_data[:, :, i] = (self.orig_data[:, :, i] - self.orig_data[:, :, i].min()) / \
                                             (self.orig_data[:, :, i].max() - self.orig_data[:, :, i].min())
                self.orig_data[:,:,i] *= self.norm_factor

            self.data = np.copy(self.orig_data)
            self.selected_slice = z
            [self._aligned_channels.append(False) for i in range(self.num_channels)]
            [self._color_matrices.append(np.identity(3)) for i in range(self.num_channels)]

        self._update_data()

    def _update_data(self):
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

        if self._transformed:
            self._update_tf_matrix()
            self.apply_transform()
        if True in self._aligned_channels and not self._show_mapping:
            self.apply_alignment()

    def _update_tf_matrix(self):
        rot_matrix = np.identity(3)
        if self.transp:
            rot_matrix = self.transp_matrix @ rot_matrix
        if self.rot:
            rot_matrix = self.rot_matrix @ rot_matrix
        if self.fliph:
            rot_matrix = self.fliph_matrix @ rot_matrix
        if self.flipv:
            rot_matrix = self.flipv_matrix @ rot_matrix

        if self.fixed_orientation:
            self.tf_matrix = np.linalg.inv(self.sem_transform) @ rot_matrix @ self.tf_matrix_orig
        else:
            self.tf_matrix = rot_matrix @ self.tf_matrix_orig

        nx, ny = self.data.shape[:-1]
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.corners = np.copy(corners)
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self._tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]

    def toggle_flip_z(self, state):
        self.data = np.flip(self.data, axis=-1)
        self.flip_z = state
        if self.peaks_z is not None:
            self.peaks_z = [self.num_slices - 1 - z for z in self.peaks_z]


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

    def toggle_original(self):
        self._update_data()

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
            self.max_proj_data = self.tif_data.max(2)
        else:
            self.max_proj_data = np.array([self.reader.getFrame(channel=i, dtype='u{}'.format(self.dtype)).max(0)
                                           for i in range(self.num_channels)]).transpose(1, 2, 0).astype('f4')
        for i in range(self.num_channels):
            self.max_proj_data[:,:,i] = (self.max_proj_data[:,:,i] - self.max_proj_data[:,:,i].min()) / \
                                        (self.max_proj_data[:,:,i].max() - self.max_proj_data[:,:,i].min())
            self.max_proj_data[:,:,i] *= self.norm_factor

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
            argmax_map = np.argmax(self.reader.getFrame(channel=self._channel_idx, dtype='u{}'.format(self.dtype)), axis=0).astype('f4').transpose((1, 0))
            self.cmap = self.create_cmaps(rot=1. / 2)
            self.hsv_map = self.colorize2d(self.max_proj_data[:, :, self._channel_idx], argmax_map, self.cmap)

        if self._transformed:
            if self.tf_hsv_map is None:
                self.apply_transform()

        self._update_data()

    def remove_tilt(self, remove_tilt):
        self._show_no_tilt = remove_tilt
        if self.hsv_map_no_tilt is None:
            if self.peaks is None:
                self.peak_finding(self.max_proj_data[:, :, self._channel_idx], transformed=False)
            ref = np.array(self.reader.getFrame(channel=self._channel_idx, dtype='u{}'.format(self.dtype)).astype('f4')).transpose((2, 1, 0))
            if self.peaks_z is None:
                self.fit_z(ref, transformed=False)
            # fit plane to peaks with least squares to remove tilt
            peaks_2d = self.peaks
            A = np.array([peaks_2d[:, 0], peaks_2d[:, 1], np.ones_like(peaks_2d[:, 0])]).T  # matrix
            beta = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), np.expand_dims(self.peaks_z, axis=1))  # params
            point = np.array([0.0, 0.0, beta[2][0]])  # point on plane
            normal = np.array(np.cross([1, 0, beta[0][0]], [0, 1, beta[1][0]]))  # normal vector of plane
            d = -point.dot(normal)  # distance to origin
            x, y = np.indices(ref[:,:,-1].shape)
            z_plane = -(normal[0] * x + normal[1] * y + d) / normal[2]  # z values of plane for whole image
            z_max_all = np.argmax(ref, axis=2)

            argmax_map_no_tilt = z_max_all - z_plane
            self.hsv_map_no_tilt = self.colorize2d(self.max_proj_data[:, :, self._channel_idx], argmax_map_no_tilt, self.cmap)

        if self._transformed:
            if self.tf_hsv_map_no_tilt is None:
                self.apply_transform()

        self._update_data()

    def calc_z(self, ind, pos, transformed=True, channel=None):
        z = None
        if channel is None:
            channel = self._channel_idx

        if ind is not None:
            z = self.peaks_z[ind]
        else:
            point = np.linalg.inv(self._color_matrices[channel]) @ np.array([pos[0], pos[1], 1])
            z = self.calc_local_z(self.channel, point, transformed=transformed, tf_matrix=self.tf_matrix)
        if z is None:
            self.print('Oops, something went wrong. Try somewhere else!')
            return None
        return z

    def load_channel(self, ind):
        if self.reader is None:
            self.channel = np.copy(self.tif_data[:,:,:,0])
        else:
            self.channel = np.array(self.reader.getFrame(channel=ind, dtype='u{}'.format(self.dtype)).astype('f4')).transpose((1, 2, 0))
            if self.flip_z:
                self.channel = np.flip(self.channel, axis=2) #flip z axis, z=0 is the most upper slice
            self.channel = (self.channel - self.channel.min()) / (self.channel.max() - self.channel.min())
            self.channel *= self.norm_factor
            self._channel_idx = ind
            self.print('Load channel {}'.format(ind+1))

    def clear_channel(self):
        self._channel_idx = None
        self.channel = None

    def estimate_alignment(self, peaks_2d, idx):
        roi_size = 20
        tmp = []
        ref = []
        err = []

        #self.green_coor = []
        #self.red_coor = []
        #self.red_coor_z = []
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
                #self.green_coor.append(tmp_i)
                diff = np.array(tmp_i - peaks_2d[i])
                if np.linalg.norm(diff) < roi_size:
                    tmp.append(tmp_i)
                    ref.append(peaks_2d[i])
                    #self.red_coor.append(peaks_2d[i])
                    #self.red_coor_z.append(self.peaks_z[i])
        #print(len(self.green_coor))
        if len(ref) != 0 and len(tmp) != 0:
            color_matrix = tf.estimate_transform('affine', np.array(tmp), np.array(ref)).params
            for i in range(len(tmp)):
                transf = color_matrix @ np.array([tmp[i][0], tmp[i][1], 1])
                diff = np.array(transf[:2] - ref[i])
                err.append(np.sqrt(diff[0] ** 2 + diff[1] ** 2))
                #print('TMP: ', tmp[i])
                #print('TRANSF: ', transf[:2])
                #print('REF: ', ref[i])
                #print('DIFF: ', diff)
            #print('Color matrix: \n', color_matrix)
            self.print('Alignment error RMS [nm]: ', np.sqrt(1 / len(err) * np.sum(err)) * self.voxel_size[0] * 1e9)
            if np.array_equal(color_matrix[2,:], np.array([0, 0, 1])):
                self._color_matrices[idx] = np.copy(color_matrix)
                self._aligned_channels[idx] = True
            else:
                self.print('Unable to align color channels. Try to adjust peak finding parameters!')
        else:
            self.print('Unable to align channels. Be sure you select a fluorescence channel!')

    def apply_alignment(self):
        for i in range(self.num_channels):
            if self._aligned_channels[i]:
                if self._transformed:
                    color_matrix = self.tf_matrix @ self._color_matrices[i] @ np.linalg.inv(self.tf_matrix)
                else:
                    color_matrix = self._color_matrices[i]
                self.log(color_matrix)
                self.data[:,:,i] = np.copy(ndi.affine_transform(self.data[:, :, i], np.linalg.inv(color_matrix), order=1))

    def calc_affine_transform(self, my_points):
        my_points = self.calc_orientation(my_points)
        self._orig_points = np.copy(my_points)
        self.log('Input points:\n', my_points)

        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))

        self.side_length = np.mean(side_list)
        self.log('ROI side length:', self.side_length, '\xb1', side_list.std())

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
        self.tf_matrix_orig = np.copy(self.tf_matrix)
        self.print('Transform matrix:\n', self.tf_matrix)
        self.print('Tf shape: ', self._tf_shape)
        self.apply_transform()

    def calc_rot_transform(self, my_points):
        self._orig_points = np.copy(my_points)
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
        self.tf_matrix_orig = np.copy(self.tf_matrix)
        self.print('Transform matrix: ', self.tf_matrix)
        self.apply_transform()

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
            self.log('counter-clockwise')
            self.counter_clockwise = True
            return points
        else:
            self.log('clockwise --> transpose points')
            order = [0, 3, 2, 1]
            return points[order]

    def apply_transform(self):
        data = np.empty(self._tf_shape + (self.data.shape[-1],))
        for i in range(self.data.shape[-1]):
            data[:,:,i] = ndi.affine_transform(self.data[:,:,i], np.linalg.inv(self.tf_matrix), order=1,
                                                    output_shape=self._tf_shape)
        self.data = np.copy(data)
        for i in range(len(self.points)):
            self.points[i] = (self.tf_matrix @ np.array([self._orig_points[i,0], self._orig_points[i,1], 1]))[:2]
        self._tf_points = np.copy(self.points)

        self._transformed = True

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
                            self.log('\rLinAlgError: ', counter)
                        if counter == 100:
                            break
            if successfull:
                if ransac.best_fit is not None:  # This should not happen, but it happens sometimes...
                    cx, cy = ransac.best_fit[0], ransac.best_fit[1]
                    coor = np.array([cx, cy]) + np.array([x, y]).T - np.array([roi_size, roi_size])
                    points_model.append(coor)
                # else:
                #    self.print('Unable to fit bead #{}! Used original coordinate instead!'.format(i))
                #    points_model.append(np.array([x, y]))
            else:
                self.print('Unable to fit bead #{}! Used original coordinate instead!'.format(i))
                points_model.append(np.array([x, y]))
        return np.array(points_model)

    @classmethod
    def get_transform(self, source, dest):
        if len(source) != len(dest):
            self.print('Point length do not match')
            return
        return tf.estimate_transform('affine', source, dest).params

