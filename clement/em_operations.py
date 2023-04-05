import sys
import numpy as np
import copy
import mrcfile as mrc
from scipy import ndimage as ndi
from skimage import transform as tf
from skimage import io, measure, feature
from sklearn import cluster, mixture
import tifffile
import xmltodict
from .ransac import Ransac
import random


class EM_ops():
    def __init__(self, printer, logger):
        self._orig_points = None
        self._grid_points_tmp = None
        self._transformed = False
        self._tf_points = None
        self._orig_points_region = None
        self._tf_points_region = None
        self._refine_matrix = None
        self._total_shift = None
        self._refine_history = [np.identity(3)]
        self._filter = 3

        self.print = printer
        self.log = logger
        self.h = None
        self.eh = None
        self.orig_data = None
        self.tf_data = None
        self.pixel_size = None  # should be in nanometer
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
        self.tf_matrix_orig = np.identity(3)
        self.tf_matrix_orig_region = np.identity(3)
        self.tf_matrix_no_shift = np.identity(3)
        self.tf_shape = None
        self.tf_shape_orig = None
        self.tf_shape_orig_region = None
        self.clockwise = False
        self.rot_angle = None
        self.first_rotation = False
        self.tf_prev = np.identity(3)
        self.gis_corrected = None
        self.gis_transf = np.identity(3)

        self.stage_origin = None

        self.points = None
        self.assembled = True
        self.cum_matrix = None
        self.dimension = None

        self.fib_matrix = None
        self.fib_shift = None
        self.fib_angle = None  # angle relative to xy-plane
        self.box_shift = None
        self.transposed = False

        self.merged = [None, None, None, None]
        self.merge_shift = None
        self.merge_matrix = None
        self.z_shift = None

    def parse_2d(self, fname):
        if '.tif' in fname or '.tiff' in fname:
            # Transposing tif images by default
            self.data = np.array(io.imread(fname).T)
            self.dimensions = self.data.shape
            self.old_fname = fname
            try:
                md = tifffile.TiffFile(fname).fei_metadata
                self.pixel_size = np.array([md['Scan']['PixelWidth'], md['Scan']['PixelHeight']]) * 1e9
            except (KeyError, TypeError):
                self.print('No pixel size found! This might cause the program to crash at some point...')
            if self.pixel_size is None:
                md = None
                with tifffile.TiffFile(fname) as f:
                    for tag in f.pages[0].tags.values():
                        if tag.name == 'FEI_TITAN':
                            md = xmltodict.parse(tag.value)

                if md is not None:
                    self.pixel_size = np.array([float(md['Metadata']['BinaryResult']['PixelSize']['X']['#text']),
                                                float(md['Metadata']['BinaryResult']['PixelSize']['Y']['#text'])]) * 1e9

        else:
            f = mrc.open(fname, 'r', permissive=True)
            if f.data is None:
                self.print('Data is empty! Check file!')
                return
            if fname != self.old_fname:
                self.h = f.header
                self.eh = np.frombuffer(f.extended_header, dtype='i2')
                self.old_fname = fname
                self.pixel_size = np.array([f.voxel_size.x, f.voxel_size.y, f.voxel_size.y]) / 10
            self.dimensions = np.array(f.data.shape)  # (dim_z, dim_y, dim_x)
            if len(self.dimensions) == 2:
                self.stacked_data = False
                self.data = np.copy(f.data)

        self.orig_data = np.copy(self.data)
        self.print('Pixel size: ', self.pixel_size)
        self._update_data()

    def parse_3d(self, step, fname):
        f = mrc.open(fname, 'r', permissive=True)
        self.dimensions = np.array(f.data.shape)  # (dim_z, dim_y, dim_x)
        if len(self.dimensions) == 3 and self.dimensions[0] > 1:
            self.stacked_data = True
            self.dimensions[1] = int(np.ceil(self.dimensions[1] / step))
            self.dimensions[2] = int(np.ceil(self.dimensions[2] / step))
            self.pos_x = self.eh[1:7 * self.dimensions[0]:7] // step
            self.pos_y = self.eh[2:7 * self.dimensions[0]:7] // step
            self.pos_x -= self.pos_x.min()
            self.pos_y -= self.pos_y.min()
            self.pos_z = self.eh[3:7 * self.dimensions[0]:7]
            self.grid_points = []
            for i in range(len(self.pos_x)):
                point = np.array((self.pos_x[i], self.pos_y[i], 1))
                box_points = [point + (0, 0, 0),
                              point + (self.dimensions[2], 0, 0),
                              point + (self.dimensions[2], self.dimensions[1], 0),
                              point + (0, self.dimensions[1], 0)]
                self.grid_points.append(box_points)

            cy, cx = np.indices(self.dimensions[1:3])

            self.data = np.zeros((self.pos_x.max() + self.dimensions[2], self.pos_y.max() + self.dimensions[1]),
                                 dtype='f4')
            self.mcounts = np.zeros_like(self.data)
            self.count_map = np.zeros_like(self.data)
            sys.stdout.write('Assembling images into %s-shaped array...' % (self.data.shape,))
            for i in range(self.dimensions[0]):
                np.add.at(self.mcounts, (cx + self.pos_x[i], cy + self.pos_y[i]), 1)
                np.add.at(self.data, (cx + self.pos_x[i], cy + self.pos_y[i]), f.data[i, ::step, ::step])
                np.add.at(self.count_map, (cx + self.pos_x[i], cy + self.pos_y[i]), i)
            sys.stdout.write('done\n')
            self.data[self.mcounts > 0] /= self.mcounts[self.mcounts > 0]
            self.count_map[self.mcounts > 1] = 0

        f.close()
        self.print(self.data.shape)
        self.orig_data = np.copy(self.data)

        self._update_data()
    def _update_data(self, filter=None, state=None):
        if filter is None:
            filter = self._filter
        else:
            self._filter = filter
        self.data = ndi.uniform_filter(self.orig_data, filter)

    def save_merge(self, fname):
        with mrc.new(fname, overwrite=True) as f:
            f.set_data(self.data)
            f.update_header_stats()

    def transpose(self):
        self.transposed = True
        self.data = self.data.T
        self.print('TRANSPOSE')
        if self.points is not None:
            self.log(self.points)
            self.points = np.array([np.flip(point) for point in self.points])
        self.log(self.points)

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
            self.tf_matrix = self.tf_matrix_orig
            self.tf_shape = self.tf_shape_orig
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
            self.tf_matrix = self.tf_matrix_orig_region
            self.tf_shape = self.tf_shape_orig_region

        if self._transformed and self._refine_matrix is not None:
            for i in range(self.points.shape[0]):
                point = np.array([self.points[i, 0], self.points[i, 1], 1])
                self.points[i] = (self._refine_matrix @ point)[:2]

    def toggle_region(self):
        self.toggle_original()

    def calc_affine_transform(self, my_points):
        my_points = self.calc_orientation(my_points)
        self.log('Input points:\n', my_points)
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))
        self.side_length = np.mean(side_list)
        self.log('ROI side length:', self.side_length, '\xb1', side_list.std())

        cen = my_points.mean(0) - np.ones(2) * self.side_length / 2.
        points_tmp = np.zeros_like(my_points)
        points_tmp[0] = cen + (0, 0)
        points_tmp[1] = cen + (self.side_length, 0)
        points_tmp[2] = cen + (self.side_length, self.side_length)
        points_tmp[3] = cen + (0, self.side_length)

        self.tf_matrix = tf.estimate_transform('affine', my_points, points_tmp).params
        self.tf_matrix_no_shift = np.copy(self.tf_matrix)
        nx, ny = self.data.shape
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self.tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] -= self.tf_corners.min(1)[:2]
        self.print('Transform matrix:\n', self.tf_matrix)
        self.print('Shift: ', -self.tf_corners.min(1)[:2])
        self.log('Assembled? ', self.assembled)
        if not self._transformed:
            if self.assembled:
                self._orig_points = np.copy(my_points)
                self.tf_matrix_orig = np.copy(self.tf_matrix)
            else:
                self._orig_points_region = np.copy(my_points)
                self.tf_matrix_orig_region = np.copy(self.tf_matrix)
        self.apply_transform(points_tmp)

    def calc_rot_transform(self, my_points):
        side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
        side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))
        self.side_length = np.mean(side_list)

        self.tf_matrix = self.calc_rot_matrix(my_points)

        center = np.mean(my_points, axis=0)
        tf_center = (self.tf_matrix @ np.array([center[0], center[1], 1]))[:2]

        points_tmp = np.zeros_like(my_points)
        points_tmp[0] = tf_center + (-self.side_length / 2, -self.side_length / 2)
        points_tmp[1] = tf_center + (self.side_length / 2, -self.side_length / 2)
        points_tmp[2] = tf_center + (self.side_length / 2, self.side_length / 2)
        points_tmp[3] = tf_center + (-self.side_length / 2, self.side_length / 2)

        nx, ny = self.data.shape
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        self.tf_corners = np.dot(self.tf_matrix, corners)
        self.tf_shape = tuple([int(i) for i in (self.tf_corners.max(1) - self.tf_corners.min(1))[:2]])
        self.tf_matrix[:2, 2] += -self.tf_corners.min(1)[:2]
        self.print('Transform matrix: ', self.tf_matrix)

        if not self._transformed:
            if self.assembled:
                self._orig_points = np.copy(my_points)
                self.tf_matrix_orig = np.copy(self.tf_matrix)
            else:
                self._orig_points_region = np.copy(my_points)
                self.tf_matrix_orig_region = np.copy(self.tf_matrix)
        self.apply_transform(points_tmp)
        self.log('New points: \n', self._tf_points)

    def calc_orientation(self, points):
        my_list = []
        for i in range(1, len(points)):
            my_list.append((points[i][0] - points[i - 1][0]) * (points[i][1] + points[i - 1][1]))
        my_list.append((points[0][0] - points[-1][0]) * (points[0][1] + points[-1][1]))
        my_sum = np.sum(my_list)
        if my_sum > 0:
            self.log('counter-clockwise')
            return points
        else:
            self.log('clockwise --> transpose points')
            order = [0, 3, 2, 1]
            return points[order]

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

    def apply_transform(self, pts):
        if self.tf_matrix is None:
            self.print('Calculate transform matrix first')
            return
        self._transformed = True

        if len(self.dimensions) == 3 and self.dimensions[0] > 1:
            self.tf_mcounts = ndi.affine_transform(self.mcounts, np.linalg.inv(self.tf_matrix), order=1,
                                                   output_shape=self.tf_shape)
            self.tf_count_map = ndi.affine_transform(self.count_map, np.linalg.inv(self.tf_matrix), order=1,
                                                     output_shape=self.tf_shape)

        if self.assembled:
            self.tf_data = ndi.affine_transform(self.data, np.linalg.inv(self.tf_matrix), order=1,
                                                output_shape=self.tf_shape)
        else:
            self.tf_region = ndi.affine_transform(self.data, np.linalg.inv(self.tf_matrix), order=1,
                                                  output_shape=self.tf_shape)

        self.transform_shift = -self.tf_corners.min(1)[:2]

        pts = np.array([point + self.transform_shift for point in pts])
        if self.assembled:
            self.tf_shape_orig = np.copy(self.tf_shape)
            self._tf_points = np.copy(pts)
            self.tf_grid_points = []
            for i in range(len(self.grid_points)):
                tf_box_points = []
                for point in self.grid_points[i]:
                    x_i, y_i, z_i = self.tf_matrix @ point
                    tf_box_points.append(np.array([x_i, y_i, z_i]))
                self.tf_grid_points.append(tf_box_points)
        else:
            self.tf_shape_orig_region = np.copy(self.tf_shape)
            self._tf_points_region = np.copy(pts)
        self.toggle_original()

    def calc_fib_transform(self, fib_angle, sem_shape, fm_voxel, sem_pixel_size, shift=np.zeros(2), sem_transpose=False):
        self.fib_angle = 90 - fib_angle
        if self.box_shift is not None:
            shift = self.box_shift - shift

        if sem_transpose:
            self.fib_matrix = np.array([[0., 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            self.fib_matrix = np.identity(4)

        # Scale according to pixel sizes
        scale = sem_pixel_size[:2] / self.pixel_size[:2]
        self.print('SEM shape: ', sem_shape)
        self.print('FIB shape: ', self.data.shape)
        self.log('Scale: ', scale)
        self.fib_matrix = np.array([[scale[0], 0, 0, 0],
                                    [0, scale[1], 0, 0],
                                    [0, 0, -fm_voxel[2]*1e9/self.pixel_size[1], 0], #z=0 is at the top, z slices shifted downwards in FIB image
                                    [0, 0, 0, 1]]) @ self.fib_matrix

        # Rotate SEM image according to sigma angles
        total_angle = self.fib_angle * np.pi / 180 #right-hand rule, clockwise rotation (looking from origin)

        #Rx
        self.fib_matrix = np.array([[1, 0, 0, 0],
                                    [0, np.cos(total_angle), -np.sin(total_angle), 0],
                                    [0, np.sin(total_angle), np.cos(total_angle), 0],
                                    [0, 0, 0, 1]]) @ self.fib_matrix


        nx, ny = sem_shape
        corners = np.array([[0, 0, 0, 1], [nx, 0, 0, 1], [nx, ny, 0, 1], [0, ny, 0, 1]]).T
        fib_corners = np.dot(self.fib_matrix, corners)
        self.fib_shift = -fib_corners.min(1)[:3]
        self.fib_matrix[:3, 3] += self.fib_shift
        self.fib_matrix[:2, 3] -= shift
        self.log('Shifted fib matrix: ', self.fib_matrix)


    def apply_fib_transform(self, points, num_slices, scaling=1):
        self.log('Num slices: ', num_slices)
        if num_slices is None:
            num_slices = 0
        else:
            num_slices *= scaling
        src = np.zeros((points.shape[0], 4))
        dst = np.zeros_like(src)
        for i in range(points.shape[0]):
            src[i, :] = [points[i, 0], points[i, 1], 17, 1] #z == 20 is an estimate for the z position of the sem grid square
            dst[i, :] = self.fib_matrix @ src[i, :]
            if self._refine_matrix is not None:
                dst[i, :3] = self._refine_matrix @ np.array([dst[i, 0], dst[i, 1], 1])
        self.points = np.array(dst[:, :2])

        if self.box_shift is None:
            com = self.points.mean(0)
            img_center = np.array([self.data.shape[0]//2, self.data.shape[1]//2])
            self.box_shift = com - img_center
        self._tf_points = np.copy(self.points)
        # if self._orig_points is None:
        self._orig_points = np.copy(self.points)

    def get_selected_region(self, coordinate):
        coordinate = coordinate.astype(int)
        try:
            if not self._transformed:
                if (0 <= coordinate[0] < self.mcounts.shape[0]) and (0 <= coordinate[1] < self.mcounts.shape[1]):
                    if self.mcounts[coordinate[0], coordinate[1]] > 1:
                        self.print('Selected region ambiguous. Try again!')
                        return
                    else:
                        counter = 0
                        my_bool = False
                        while not my_bool:
                            x_range = np.arange(self.pos_x[counter], self.pos_x[counter] + self.dimensions[2])
                            y_range = np.arange(self.pos_y[counter], self.pos_y[counter] + self.dimensions[1])
                            if coordinate[0] in x_range and coordinate[1] in y_range:
                                my_bool = True
                            counter += 1
                        self.print('Selected region: ', counter - 1)
                        return counter - 1
            else:
                if (0 <= coordinate[0] < self.tf_count_map.shape[0]) and (
                        0 <= coordinate[1] < self.tf_count_map.shape[1]):
                    if self.tf_count_map[coordinate[0], coordinate[1]] == 0:
                        self.print('Selected region ambiguous. Try again!')
                        return
                    else:
                        counter = int(self.tf_count_map[coordinate[0], coordinate[1]])
                        self.print('Selected region: ', counter)
                        return counter
        except(IndexError):
            self.print('Watch out, index error. Try again!')

    def select_region(self, coordinate, transformed):
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
            stage_x = self.eh[4:7 * self.dimensions[0]:7]
            stage_y = self.eh[5:7 * self.dimensions[0]:7]
            if self.assembled or self.selected_region is None:
                curr_region = 0
            else:
                curr_region = self.selected_region
            self.stage_origin = np.array([stage_x[curr_region], stage_y[curr_region]])
        else:
            self.stage_origin = 0
            self.pixel_size = np.array([1, 1])
        inverse_matrix = np.linalg.inv(self.tf_matrix)
        stage_positions = []
        for i in range(len(clicked_points)):
            point = np.array([clicked_points[i][0], clicked_points[i][1], 1])
            coordinate_angstrom = (inverse_matrix @ point)[:2] * self.pixel_size[:2] * downsampling
            coordinate_microns = coordinate_angstrom * 10 ** -4
            stage_positions.append(coordinate_microns + self.stage_origin)  # stage position in microns
        self.log(stage_positions)
        return stage_positions

    def calc_grid_shift(self, shift_x, shift_y):
        shift = np.array([shift_x, shift_y], dtype='f8')
        if self._total_shift is None:
            self._total_shift = shift
        else:
            self._total_shift += shift
        self.points += shift
        self._tf_points = np.copy(self.points)
        self._orig_points = np.copy(self.points)
        self.fib_matrix[:2, 3] = shift

    def calc_refine_matrix(self, src, dst, ind=None):
        refine_matrix = tf.estimate_transform('affine', src, dst).params
        if self._refine_matrix is None or ind == 0 or ind == 3:
            self._refine_matrix = refine_matrix
        else:
            self._refine_matrix = refine_matrix @ self._refine_matrix

        self._refine_history.append(refine_matrix)
        self.print('Refine matrix: \n', self._refine_matrix)

    def apply_refinement(self, points=None):
        points = np.copy(self.points)
        self.log(points)
        for i in range(points.shape[0]):
            point = np.array([points[i, 0], points[i, 1], 1])
            points[i] = (self._refine_matrix @ point)[:2]

        self.points = np.copy(points)

    def undo_refinement(self):
        if len(self._refine_history) > 1:
            self._refine_matrix = np.linalg.inv(self._refine_history[-1]) @ self._refine_matrix
            for i in range(self.points.shape[0]):
                point = np.array([self.points[i, 0], self.points[i, 1], 1])
                self.points[i] = (np.linalg.inv(self._refine_history[-1]) @ point)[:2]
            del self._refine_history[-1]
        else:
            self.print('Data not refined!')

    def _find_fiducial(self, img, num_circ=9):
        labels, num_obj = ndi.label(img)
        sizes = np.zeros(num_obj - 1)
        for i in range(1, num_obj):
            sizes[i - 1] = np.bincount(labels[labels == i]).max()

        lab_ind = sizes.argsort()[-num_circ:] + 1
        coor = np.array(ndi.center_of_mass(img, labels, lab_ind))
        coor_sorted = np.array(sorted(coor, key=lambda k: [np.cos(20 * np.pi / 180) * k[0] + k[1]]))
        return coor_sorted

    def _update_gis_points(self, point):
        point = self.gis_transf @ np.array([point[0], point[1], 1])
        return point[:2]

    def estimate_gis_transf(self, roi_pos, roi_pre, roi_post):
        roi_pre_t = roi_pre.max() - roi_pre
        roi_post_t = roi_post.max() - roi_post
        roi_pre_t[roi_pre_t < 0.95 * roi_pre_t.max()] = 0
        roi_post_t[roi_post_t < 0.95 * roi_post_t.max()] = 0
        coor_pre = self._find_fiducial(roi_pre_t) + roi_pos
        coor_post = self._find_fiducial(roi_post_t) + roi_pos

        self.gis_transf = tf.estimate_transform('affine', coor_pre, coor_post).params
        # self.gis_transf = tf.estimate_transform('euclidean', coor_pre, coor_post).params

    def align_fiducial(self, img_pre, align=False):
        if align:
            self.gis_corrected = ndi.affine_transform(img_pre, np.linalg.inv(self.gis_transf), order=1,
                                       output_shape=self.data.shape)
        else:
            self.gis_corrected = None

    def fit_circles(self, points, bead_size):
        points_model = []
        successfull = False
        roi_size = int(
            np.round(bead_size * 1000 / self.pixel_size[0] + bead_size * 1000 / (2 * self.pixel_size[0])) / 2)
        for i in range(len(points)):
            x = int(np.round(points[i, 0]))
            y = int(np.round(points[i, 1]))
            x_min = (x - roi_size) if (x - roi_size) > 0 else 0
            x_max = (x + roi_size) if (x + roi_size) < self.data.shape[0] else self.data.shape[0]
            y_min = (y - roi_size) if (y - roi_size) > 0 else 0
            y_max = (y + roi_size) if (y + roi_size) < self.data.shape[1] else self.data.shape[1]

            roi = self.data[x_min:x_max, y_min:y_max]
            edges = feature.canny(roi, 3).astype(np.float32)
            coor_x, coor_y = np.where(edges != 0)
            if len(coor_x) != 0:
                rad = bead_size * 1e3 / self.pixel_size[0] / 2  # bead size is supposed to be in microns
                ransac = Ransac(coor_x, coor_y, 100, rad)
                counter = 0
                while True:
                    try:
                        ransac.run()
                        successfull = True
                        break
                    except np.linalg.LinAlgError:
                        counter += 1
                        if counter == 100:
                            break
            if successfull:
                if ransac.best_fit is not None:  # This should not happen, but it happens sometimes...
                    cx, cy = ransac.best_fit[0], ransac.best_fit[1]
                    coor = np.array([cx, cy]) + np.array([x, y]).T - np.array([roi_size, roi_size])
                    points_model.append(coor)
            else:
                self.print('Unable to fit bead #{}! Used original coordinate instead!'.format(i))
                points_model.append(np.array([x, y]))
        return np.array(points_model)

    def calc_error(self, diff):
        clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
        clf.fit(diff)
        cov = clf.covariances_[0]
        max = np.max(np.abs(diff))
        x = np.linspace(-max, max)
        y = np.linspace(-max, max)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = -clf.score_samples(XX)
        Z = Z.reshape(X.shape)
        f = np.exp(-Z)
        f /= f.max()
        self.log('Covariance matrix: ', cov)
        return cov, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), f

    def calc_convergence(self, corr_points, em_points, min_points, refine_matrix):
        em_points = np.array(em_points)
        corr_points = np.array(corr_points)

        corr_points_refined = []
        if refine_matrix is None:
            corr_points_refined = corr_points
        else:
            for i in range(len(corr_points)):
                p = np.array([corr_points[i, 0], corr_points[i, 1], 1])
                p_new = refine_matrix @ p
                corr_points_refined.append(p_new[:2])
            # corr_points_refined = np.array(corr_points)
            corr_points_refined = np.array(corr_points_refined)

        num_sims = len(em_points) - min_points + 1
        num_iterations = 100
        precision_refined = []
        precision_free = []
        precision_all = []
        num_points = min_points
        for i in range(num_sims):
            precision_i_refined = []
            precision_i_free = []
            precision_i_all = []
            for k in range(num_iterations):
                indices = random.sample(range(len(em_points)), num_points)
                p_em = em_points[indices]
                p_corr = corr_points_refined[indices]

                refine_matrix = tf.estimate_transform('affine', p_corr, p_em).params
                calc_points = []
                for l in range(len(corr_points)):
                    p = np.array([corr_points_refined[l, 0], corr_points_refined[l, 1], 1])
                    p_refined = refine_matrix @ p
                    calc_points.append(p_refined[:2])

                diff_all = em_points - np.array(calc_points)
                diff_refined = diff_all[indices]
                diff_free = np.array([diff_all[i] for i in range(len(diff_all)) if i not in indices])

                rms_all = np.sqrt(1 / len(diff_all) * (diff_all ** 2).sum())
                rms_refined = np.sqrt(1 / len(diff_refined) * (diff_refined ** 2).sum())
                if len(diff_free) > 0:
                    rms_free = np.sqrt(1 / len(diff_free) * (diff_free ** 2).sum())
                else:
                    rms_free = 0
                precision_i_refined.append(rms_refined)
                precision_i_free.append(rms_free)
                precision_i_all.append(rms_all)
            precision_refined.append(np.mean(precision_i_refined))
            precision_free.append(np.mean(precision_i_free))
            precision_all.append(np.mean(precision_i_all))
            num_points += 1

        precision_refined = np.array(precision_refined) * self.pixel_size[0]
        precision_free = np.array(precision_free) * self.pixel_size[0]
        precision_all = np.array(precision_all) * self.pixel_size[0]
        self.print('RMS error: ', precision_all[-1])
        return [precision_refined, precision_free, precision_all]

    def apply_merge_2d(self, fm_data, channel, tr_matrix, num_channels, idx):
        if channel == 0:
            self.merge_matrix = tr_matrix
            self.merged[idx] = np.zeros(self.data.shape + (num_channels + 1,))
            self.merged[idx][:, :, -1] = self.data / self.data.max() * 100

        self.merged[idx][:, :, channel] = ndi.affine_transform(fm_data, np.linalg.inv(self.merge_matrix), order=1, output_shape=self.data.shape)
        self.print('Merged.shape: ', self.merged[idx].shape)

    def apply_merge_3d(self, fm_data_orig, tf_matrix_aligned, tr_matrices, orig_points, fm_z_values, corr_points_fib, channel,
                       num_slices, num_channels, norm_factor, idx, fibcontrols):

        #copy values from fib correlation if gis is selected
        if idx == 2:
            self.fib_matrix = fibcontrols.ops.fib_matrix
            self._refine_matrix = fibcontrols.ops._refine_matrix

        p0 = np.array([0, 0, 0, 1])
        p1 = np.array([0, 0, 1, 1])
        self.z_shift = (self.fib_matrix @ p1)[:2] - (self.fib_matrix @ p0)[:2]
        self.log('z_shift: ', self.z_shift)

        fib_2d = np.zeros((3, 3))
        fib_2d[:2, :2] = self.fib_matrix[:2, :2]
        fib_2d[2, 2] = 1

        total_matrix = self.gis_transf @ self._refine_matrix @ fib_2d @ tr_matrices @ tf_matrix_aligned

        nx, ny = fm_data_orig.shape[:2]
        corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
        tf_corners = total_matrix @ corners
        tf_shape = tuple([int(i) for i in (tf_corners.max(1) - tf_corners.min(1))[:2]])

        if channel == 0:
            tf_points = []
            corr_points_fib_red = []
            for i in range(len(orig_points)):
                img_tmp = np.zeros([nx, ny])
                img_tmp[int(np.round(orig_points[i][0])), int(np.round(orig_points[i][1]))] = 1
                z = fm_z_values[i]
                fib_new = np.copy(fib_2d)
                fib_new[:2, 2] += z * self.z_shift
                shift_matrix = self.gis_transf @ self._refine_matrix @ fib_new @ tr_matrices @ tf_matrix_aligned
                shift_matrix[:2, 2] -= tf_corners.min(1)[:2]
                refined = ndi.affine_transform(img_tmp, np.linalg.inv(shift_matrix), order=1,
                                               output_shape=tf_shape)
                if refined.max() != 0:
                    tf_point = np.where(refined == refined.max())
                    tf_points.append([tf_point[0][0], tf_point[1][0]])
                    corr_points_fib_red.append(corr_points_fib[i])

            self.merge_shift = np.mean(tf_points, axis=0) - np.mean(corr_points_fib_red, axis=0)
            self.log('IMG shift: ', self.merge_shift)

        z_data = []

        for z in range(num_slices):
            fib_new = np.copy(fib_2d)
            fib_new[:2, 2] += z * self.z_shift
            total_matrix = self.gis_transf @ self._refine_matrix @ fib_new @ tr_matrices @ tf_matrix_aligned
            total_matrix[:2, 2] -= tf_corners.min(1)[:2]
            total_matrix[:2, 2] -= self.merge_shift.T

            if z == 0:
                print('total matrix again: \n', total_matrix)
            refined = np.copy(ndi.affine_transform(fm_data_orig[:, :, z], np.linalg.inv(total_matrix), order=1,
                                           output_shape=self.data.shape))
            z_data.append(refined)

        if self.merged[idx] is None:
            self.merged[idx] = np.array(np.max(z_data, axis=0))
            #self.merged[idx] = np.array(np.sum(z_data, axis=0))
        else:
            if self.merged[idx].ndim == 2:
                self.merged[idx] = np.concatenate(
                    (np.expand_dims(self.merged[idx], axis=2), np.expand_dims(np.max(z_data, axis=0), axis=2)), axis=2)
                    #(np.expand_dims(self.merged[idx], axis=2), np.expand_dims(np.sum(z_data, axis=0), axis=2)), axis=2)
            else:
                if channel == 2:
                    self.merged[idx] = np.concatenate((self.merged[idx], np.expand_dims(np.sum(z_data, axis=0), axis=2)),
                                                    axis=2)
                else:
                    self.merged[idx] = np.concatenate((self.merged[idx], np.expand_dims(np.sum(z_data, axis=0), axis=2)),
                                                    axis=2)
        if channel == num_channels - 1:
            fib_img = np.zeros_like(refined)
            fib_img[:self.data.shape[0], :self.data.shape[1]] = self.data
            fib_img /= fib_img.max()
            fib_img *= norm_factor
            self.merged[idx] = np.concatenate((self.merged[idx], np.expand_dims(fib_img, axis=2)), axis=2)

