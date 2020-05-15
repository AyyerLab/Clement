import numpy as np
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import scipy.ndimage as ndi
import copy
from skimage import io, measure, feature, color, draw

class BaseControls(QtWidgets.QWidget):
    def __init__(self):
        super(BaseControls, self).__init__()
        # ops is EM_operations or FM_operations
        self.tag = 'base'
        self.ops = None
        self.other = None # The other controls object
        
        self._box_coordinate = None
        self._points_corr = []
        self._points_corr_z = []
        self._orig_points_corr = []
        self._points_corr_indices = []
        self._refined = False
        self._err = [None, None]
        self._std = [[None, None], [None, None]]
        self._conv = [None, None]
        self._merge_points = []
        self._merge_points_z = []

        self.tr_matrices = None
        self.show_grid_box = False
        self.show_tr_grid_box = False
        self.clicked_points = []
        self.grid_box = None
        self.tr_grid_box = None
        self.boxes = []
        self.tr_boxes = []
        self.redo_tr = False
        self.setContentsMargins(0, 0, 0, 0)
        self.counter = 0
        self.anno_list = []
        self.size = 10
        self.orig_size = 10

        self.peaks = []
        self.num_slices = None
        self.corr_points = []
        self.min_conv_points = 10

    def _init_ui(self):
        print('This message should not be seen. Please override _init_ui')

    def _imview_clicked(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
            return

        if self.ops is None:
            return

        pos = self.imview.getImageItem().mapFromScene(event.pos())
        item = self.imview.getImageItem()

        pos.setX(pos.x() - self.size/2)
        pos.setY(pos.y() - self.size/2)
        if hasattr(self, 'define_btn') and self.define_btn.isChecked():
            roi = pg.CircleROI(pos, self.size, parent=item, movable=False)
            roi.setPen(255,0,0)
            roi.removeHandle(0)
            self.imview.addItem(roi)
            self.clicked_points.append(roi)
        elif hasattr(self, 'select_btn') and self.select_btn.isChecked():
            self._draw_correlated_points(pos, item)
        elif hasattr(self, 'select_region_btn') and self.select_region_btn.isChecked():
            '''EM only: Select individual image from montage'''
            points_obj = (pos.x(), pos.y())
            self._box_coordinate = np.copy(np.array(points_obj))

            # If clicked point is inside image
            if points_obj[0] < self.ops.data.shape[0] and points_obj[1] < self.ops.data.shape[1]:
                # Get index of selected region
                ind = self.ops.get_selected_region(np.array(points_obj), not self.show_btn.isChecked())
                # If index is non-ambiguous
                if ind is not None:
                    if self.show_btn.isChecked():
                        boxes = self.boxes
                    else:
                        boxes = self.tr_boxes
                    white_pen = pg.mkPen('w')
                    [box.setPen(white_pen) for box in boxes]
                    if ind < len(boxes):
                        boxes[ind].setPen(pg.mkPen('#ed7370'))
                    else:
                        print('Oops, something went wrong. Try again!')
            else:
                print('Oops, something went wrong. Try again!')

    def _couple_views(self):
        pass

    def _draw_correlated_points(self, pos, item):
        if self.other.ops is None:
            print('Select both data first')
        
        else:
            condition = False
            if hasattr(self, 'fib') and not self.fib:
                if self.ops._tf_points is not None or self.ops._tf_points_region is not None:
                    if self.other.ops._tf_points is not None:
                        condition = True
            else:
                if self.ops._tf_points is not None:
                    if self.other.fib:
                        if self.other.sem_ops is not None:
                            if self.other.sem_ops._tf_points is not None or self.other.sem_ops._tf_points_region is not None:
                                condition = True
                    else:
                        if self.other.ops._tf_points is not None or self.other.ops._tf_points_region is not None:
                            condition = True
            if condition:
                ind = None
                point = np.array([pos.x() + self.size / 2, pos.y() + self.size / 2])
                if self.tr_matrices is not None:
                    peaks = None
                    if self.other.fib and self.ops.tf_peaks_3d is not None:
                        ind = self.ops.check_peak_index(point, self.size, True)
                        if ind is None and not self.other._refined:
                            print('You have to select a bead for the first refinement!')
                            return
                        elif ind is not None:
                            peaks = self.ops.tf_peaks_3d
                        z = self.ops.calc_z(ind, point)
                        if z is None:
                            return
                        self._points_corr_z.append(z)

                    elif not self.other.fib:
                        if self.ops.tf_peak_slices is not None and self.ops.tf_peak_slices[-1] is not None:
                            ind = self.ops.check_peak_index(point, self.size, False)
                        if ind is not None:
                            peaks = self.ops.tf_peak_slices[-1]

                    if ind is not None:
                        pos.setX(peaks[ind,0] - self.size/2)
                        pos.setY(peaks[ind,1] - self.size/2)
                        init = np.array([peaks[ind, 0], peaks[ind, 1], 1])
                    else:
                        init = np.array([point[0], point[1], 1])

                    point_obj = pg.CircleROI(pos, self.size, parent=item, movable=False, removable=True)
                    point_obj.setPen(0,255,0)
                    point_obj.removeHandle(0)
                    self.imview.addItem(point_obj)
                    self._points_corr.append(point_obj)
                    self.counter += 1
                    annotation_obj = pg.TextItem(str(self.counter), color=(0,255,0), anchor=(0,0))
                    annotation_obj.setPos(pos.x()+5, pos.y()+5)
                    self.imview.addItem(annotation_obj)
                    self.anno_list.append(annotation_obj)

                    self._points_corr_indices.append(self.counter-1)

                    print('2d init: ', init)
                    if hasattr(self.other, 'fib') and self.other.fib:
                        print('Clicked point: ', np.array([init[0], init[1], z]))
                        transf = np.dot(self.tr_matrices, init)
                        transf = self.other.ops.fib_matrix @ np.array([transf[0], transf[1], z, 1])
                        self.other._points_corr_z.append(transf[2])
                        if self.other._refined:
                            transf[:2] = (self.other.ops._refine_matrix @ np.array([transf[0], transf[1], 1]))[:2]
                    else:
                        transf = np.dot(self.tr_matrices, init)

                    print('Transformed point: ', transf)
                    pos = QtCore.QPointF(transf[0]-self.other.size/2, transf[1]-self.other.size/2)
                    point_other = pg.CircleROI(pos, self.other.size, parent=self.other.imview.getImageItem(), movable=True, removable=True)
                    point_other.setPen(0,255,255)
                    point_other.removeHandle(0)
                    self.other.imview.addItem(point_other)
                    self.other._points_corr.append(point_other)
                    self.other._orig_points_corr.append([pos.x(), pos.y()])

                    self.other.counter = self.counter
                    annotation_other = pg.TextItem(str(self.counter), color=(0,255,255), anchor=(0,0))
                    annotation_other.setPos(pos.x()+5, pos.y()+5)
                    self.other.imview.addItem(annotation_other)
                    self.other.anno_list.append(annotation_other)

                    point_obj.sigRemoveRequested.connect(lambda: self._remove_correlated_points(point_obj, other=False))
                    point_other.sigRemoveRequested.connect(lambda: self._remove_correlated_points(point_other, other=True))


            else:
                print('Transform both images before point selection')

    def _remove_correlated_points(self, point, other):
        idx = None
        for i in range(len(self._points_corr)):
            if self._points_corr[i] == point or self.other._points_corr[i] == point:
                idx = i
                break

        self.imview.removeItem(self._points_corr[idx])
        self.imview.removeItem(self.anno_list[idx])
        self.other.imview.removeItem(self.other._points_corr[idx])
        self.other.imview.removeItem(self.other.anno_list[idx])

        self._points_corr.remove(self._points_corr[idx])
        self.other._points_corr.remove(self.other._points_corr[idx])

        self.anno_list.remove(self.anno_list[idx])
        self.other.anno_list.remove(self.other.anno_list[idx])

        self._points_corr_indices.remove(self._points_corr_indices[idx])

        if (hasattr(self, 'fib') and self.fib) or (hasattr(self.other, 'fib') and self.other.fib):
            self._points_corr_z.remove(self._points_corr_z[idx])
            self.other._points_corr_z.remove(self.other._points_corr_z[idx])

        if hasattr(self, 'fib'):
            self._orig_points_corr.remove(self._orig_points_corr[idx])
        else:
            self.other._orig_points_corr.remove(self.other._orig_points_corr[idx])


    def _define_grid_toggled(self, checked):
        if self.ops is None:
            print('Select data first!')
            return

        if checked:
            print('Defining grid on %s image: Click on corners'%self.tag)
            self.show_grid_btn.setChecked(False)
        else:
            print('Done defining grid on %s image: Manually adjust fine positions'%self.tag)
            if len(self.clicked_points) == 4:
                # Initialize grid_box on original or transformed image
                if self.show_btn.isChecked():
                    self.grid_box = None
                else:
                    self.tr_grid_box = None

                # Calculate circle centers
                positions = [c.pos() for c in self.clicked_points]
                sizes = [c.size()[0] for c in self.clicked_points]
                for pos, s in zip(positions, sizes):
                    pos.setX(pos.x() + s/2)
                    pos.setY(pos.y() + s/2)
                points = np.array([(point.x(), point.y()) for point in positions])

                # If original image
                if self.show_btn.isChecked():
                    self.grid_box = pg.PolyLineROI(positions, closed=True, movable=False)
                    self.grid_box.sigRegionChangeFinished.connect(self.store_grid_box_points)
                    # If assembly is an option (EM image)
                    if hasattr(self, 'show_assembled_btn'):
                        if self.show_assembled_btn.isChecked():
                            self.ops._orig_points = points
                        else:
                            self.ops._orig_points_region = points
                    else:
                        self.ops._orig_points = points
                    self.ops.points = points
                else:
                    self.tr_grid_box = pg.PolyLineROI(positions, closed=True, movable=False)
                [self.imview.removeItem(roi) for roi in self.clicked_points]
                self.clicked_points = []

                self.transform_btn.setEnabled(True)
                self.rot_transform_btn.setEnabled(True)
                self.show_grid_btn.setEnabled(True)
                self.show_grid_btn.setChecked(True)

                if self.show_btn.isChecked():
                    self.show_grid_box = True
                else:
                    self.show_tr_grid_box = True
            else:
                print('You have to select exactly 4 points. Try again!')
                [self.imview.removeItem(roi) for roi in self.clicked_points]
                self._show_grid(None)
                self.clicked_points = []

    def _recalc_grid(self, toggle_orig=False):
        if self.ops.points is not None:
            if self.show_btn.isChecked():
                pos =  list(self.ops.points)
                poly_line = pg.PolyLineROI(pos, closed=True, movable=False)
                if self.show_grid_btn.isChecked():
                    if not toggle_orig:
                        self.imview.removeItem(self.tr_grid_box)
                        self.imview.removeItem(self.grid_box)
                        self.show_grid_box = False
                print('Recalculating original grid...')
                self.grid_box = poly_line
            else:
                pos =  list(self.ops.points)
                poly_line = pg.PolyLineROI(pos, closed=True, movable=False)
                if self.show_grid_btn.isChecked():
                    if not toggle_orig:
                        self.imview.removeItem(self.grid_box)
                        self.imview.removeItem(self.tr_grid_box)
                        self.show_tr_grid_box = False
                    if self.redo_tr:
                        self.imview.removeItem(self.tr_grid_box)
                        self.redo_tr = False
                print('Recalculating transformed grid...')
                self.tr_grid_box = poly_line
            self._show_grid(None)

    def store_grid_box_points(self):
        points_obj = self.grid_box.getState()['points']
        points = np.array([list((point[0], point[1])) for point in points_obj])
        if self.show_btn.isChecked():
            if hasattr(self, 'show_assembled_btn'):
                if self.show_assembled_btn.isChecked():
                    self.ops._orig_points = points
                else:
                    self.ops._orig_points_region = points
            else:
                self.ops._orig_points = points
            self.ops.points = points

    def _show_grid(self, state):
        if self.show_grid_btn.isChecked():
            if self.show_btn.isChecked():
                self.show_grid_box = True
                self.imview.addItem(self.grid_box)
                if self.show_tr_grid_box:
                    self.imview.removeItem(self.tr_grid_box)
                    self.show_tr_grid_box = False
            else:
                self.show_tr_grid_box = True
                self.imview.addItem(self.tr_grid_box)
                if self.show_grid_box:
                    self.imview.removeItem(self.grid_box)
                    self.show_grid_box = False
        else:
            if self.show_btn.isChecked():
                if self.show_grid_box:
                    self.imview.removeItem(self.grid_box)
                    self.show_grid_box = False
            else:
                if self.show_tr_grid_box:
                    self.imview.removeItem(self.tr_grid_box)
                    self.show_tr_grid_box = False

    def _define_corr_toggled(self, checked):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        if self.ops is None or self.other.ops is None:
            print('Select both data first')
            QtWidgets.QApplication.restoreOverrideCursor()
            return

        condition = False
        if hasattr(self, 'fib') and not self.fib:
            if self.ops._tf_points is not None or self.ops._tf_points_region is not None:
                if self.other.ops._tf_points is not None:
                    condition = True
        else:
            if self.ops._tf_points is not None:
                if self.other.fib:
                    if self.other.sem_ops is not None:
                        if self.other.sem_ops._tf_points is not None or self.other.sem_ops._tf_points_region is not None:
                            condition = True
                else:
                    if self.other.ops._tf_points is not None or self.other.ops._tf_points_region is not None:
                        condition = True

        if hasattr(self.other, 'fib') and self.other.fib:
            if self.other.ops.fib_matrix is None:
                print('You have to calculate the grid box for the FIB view first!')
                QtWidgets.QApplication.restoreOverrideCursor()
                return

        if condition:
            if checked:
                print('Select points of interest on %s image'%self.tag)
                if len(self._points_corr) != 0:
                    [self.imview.removeItem(point) for point in self._points_corr]
                    [self.other.imview.removeItem(point) for point in self.other._points_corr]
                    [self.imview.removeItem(anno) for anno in self.anno_list]
                    [self.other.imview.removeItem(anno) for anno in self.other.anno_list]
                    self.anno_list = []
                    self.other.anno_list = []
                    self.counter = 0
                    self.other.counter = 0
                    self._points_corr = []
                    self.other._points_corr = []
                    self._points_corr_z = []
                    self.other._points_corr_z = []
                    self._orig_points_corr = []
                    self.other._orig_points_corr = []
                    self._points_corr_indices = []
                    self.other._points_corr_indices = []
                    self.other.size = self.orig_size

                if hasattr(self.other, 'fib') and self.other.fib:
                    src_sorted = np.array(
                        sorted(self.ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                    dst_sorted = np.array(
                        sorted(self.other.sem_ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                    self.tr_matrices = self.other.ops.get_fib_transform(src_sorted, dst_sorted, self.other.sem_ops.tf_matrix)

                    if self.ops.tf_peaks_3d is None:
                        print('I will do you a favor and calculate the 3d peak positions. This might take a few seconds ...')
                        if self.ops.tf_peak_slices is None or self.ops.tf_peak_slices[-1] is None:
                            if self.ops.max_proj_data is None:
                                self.ops.calc_max_proj_data()
                            if self.ops.tf_peak_slices is None or self.ops.tf_peak_slices[-1] is None:
                                self.ops.peak_finding(self.ops.data[:,:,-1], transformed=True)
                        self.ops.load_channel(ind=3)
                        flip_list = [self.ops.transp, self.ops.rot, self.ops.fliph, self.ops.flipv]
                        self.ops.fit_z(self.ops.channel, transformed=True, tf_matrix=self.ops.tf_matrix,
                                                 flips=flip_list, shape=self.ops.data.shape[:-1])
                        self.ops.clear_channel()
                        print('Done.')
                    self.ops.load_channel(ind=3)
                else:
                    src_sorted = np.array(
                        sorted(self.ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                    dst_sorted = np.array(
                        sorted(self.other.ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                    self.tr_matrices = self.ops.get_transform(src_sorted, dst_sorted)
            else:
                if hasattr(self.other, 'fib') and self.ops.channel is not None:
                    self.ops.clear_channel()
                print('Done selecting points of interest on %s image'%self.tag)
        else:
            if checked:
                print('Select and transform both data first')
        QtWidgets.QApplication.restoreOverrideCursor()


    def _affine_transform(self, toggle_orig=True):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.show_btn.isChecked():
            grid_box = self.grid_box
        else:
            self.redo_tr = True
            grid_box = self.tr_grid_box

        if grid_box is not None:
            if hasattr(self, 'fliph'):
                if self.align_btn.isChecked():
                    aligned = True
                    self.align_btn.setChecked(False)
                else:
                    aligned = False
                self.fliph.setChecked(False)
                self.fliph.setEnabled(True)
                self.flipv.setChecked(False)
                self.flipv.setEnabled(True)
                self.transpose.setChecked(False)
                self.transpose.setEnabled(True)
                self.rotate.setChecked(False)
                self.rotate.setEnabled(True)
                self.size_box.setEnabled(True)
                self.auto_opt_btn.setEnabled(True)

            points_obj = grid_box.getState()['points']
            points = np.array([list((point[0], point[1])) for point in points_obj])
            if self.rot_transform_btn.isChecked():
                print('Performing rotation on %s image'%self.tag)
                self.ops.calc_rot_transform(points)
            else:
                print('Performing affine transformation on %s image'%self.tag)
                self.ops.calc_affine_transform(points)

            self.show_btn.blockSignals(True)
            self.show_btn.setEnabled(True)
            self.show_btn.setChecked(False)
            self.show_btn.blockSignals(False)
            self._recalc_grid(toggle_orig=toggle_orig)
            self._update_imview()
            self.transform_btn.setEnabled(False)
            self.rot_transform_btn.setEnabled(False)

            if hasattr(self, 'select_btn'):
                self.select_btn.setEnabled(True)
                self.merge_btn.setEnabled(True)
                self.refine_btn.setEnabled(True)
                if aligned:
                    self.align_btn.setChecked(True)

            if self.ops is not None and self.other.ops is not None:
                if hasattr(self, 'select_btn') and not self.other.fib:
                    if self.ops._transformed and self.other.ops._transformed:
                        self.other.show_peaks_btn.setEnabled(True)
                if hasattr(self, 'fib') and not self.fib:
                    if self.ops._transformed and self.other.ops._transformed:
                        self.show_peaks_btn.setEnabled(True)
        else:
            print('Define grid box on %s image first!'%self.tag)
        QtWidgets.QApplication.restoreOverrideCursor()

    def _show_original(self):
            if self.ops is not None:
                self.ops._transformed = not self.ops._transformed
                print('Transformed: ',self.ops._transformed)
                align = False
                if hasattr(self.ops, 'flipv') and not self.ops._transformed:
                    if self.align_btn.isChecked() and self.ops.color_matrix is None:
                        self.align_btn.setChecked(False)
                        align = True
                    self.flipv.setEnabled(False)
                    self.fliph.setEnabled(False)
                    self.transpose.setEnabled(False)
                    self.rotate.setEnabled(False)
                    self.select_btn.setEnabled(True)
                    self.merge_btn.setEnabled(True)
                    self.refine_btn.setEnabled(True)
                elif hasattr(self.ops, 'flipv') and self.ops._transformed:
                    if not self.ops.refined:
                        self.flipv.setEnabled(True)
                        self.fliph.setEnabled(True)
                        self.transpose.setEnabled(True)
                        self.rotate.setEnabled(True)
                        self.select_btn.setEnabled(True)
                        self.merge_btn.setEnabled(True)
                        self.refine_btn.setEnabled(True)

                print('Transformed?', self.ops._transformed)
                if hasattr(self.ops,'refined'):
                    if self.ops.refined:
                        self.ops.toggle_original(update=False)
                    else:
                        self.ops.toggle_original()
                else:
                    self.ops.toggle_original()
                self._recalc_grid(toggle_orig=True)

                show_peaks = False
                if hasattr(self, 'peak_btn'):
                    if self.peak_btn.isChecked():
                        show_peaks = True
                        self.peak_btn.setChecked(False)
                    if not self.other.fib:
                        if self.other.ops is not None:
                            if self.ops._transformed and self.other.ops._transformed:
                                self.other.show_peaks_btn.setEnabled(True)

                if hasattr(self, 'fib') and not self.fib:
                    if self.other.ops is not None:
                        if self.ops._transformed and self.other.ops._transformed:
                            self.show_peaks_btn.setEnabled(True)
                        else:
                            self.show_peaks_btn.setChecked(False)
                            self.show_peaks_btn.setEnabled(False)

                self._update_imview()
                if self.ops._transformed:
                    self.transform_btn.setEnabled(False)
                    self.rot_transform_btn.setEnabled(False)
                else:
                    self.transform_btn.setEnabled(True)
                    self.rot_transform_btn.setEnabled(True)

                if show_peaks:
                    self.peak_btn.setChecked(True)
                if align:
                    self.align_btn.setChecked(True)

    def _refine(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if len(self._points_corr) > 3:
            if not self.select_btn.isChecked():
                print('Refining...')
                dst = np.array([[point.x() + self.other.size / 2, point.y() + self.other.size / 2] for point in
                                self.other._points_corr])
                src = np.array([[point[0] + self.orig_size / 2, point[1] + self.orig_size / 2] for point in
                                self.other._orig_points_corr])
                fm_points = np.array([[point.x() + self.size / 2, point.y() + self.size / 2]
                                      for point in self._points_corr])
                em_points = np.copy(dst)
                np.save('fm_points.npy', fm_points)
                np.save('fm_points_z.npy', self._points_corr_z)
                np.save('em_points.npy', em_points)

                self.other._merge_points = np.copy(em_points)
                self._merge_points = np.copy(fm_points)

                if self.other.fib:
                    idx = 1
                    refine_matrix_old = copy.copy(self.other.ops._refine_matrix)
                    self._merge_points_z = np.copy(self._points_corr_z)
                    self.other.ops.calc_refine_matrix(src,dst)
                    self.other.ops.apply_refinement()
                    self.other._refined = True
                    self.other._calc_grid()
                    self._estimate_precision(idx, refine_matrix_old)
                    self.ops.merged_3d = None
                else:
                    idx = 0
                    refine_matrix_old = copy.copy(self.ops._refine_matrix)
                    self.ops.calc_refine_matrix(src, dst)
                    self.ops.apply_refinement()
                    self._refined = True
                    self.other._refined = True
                    self.ops.refine_grid(fm_points, em_points, self.other.ops.points)
                    self._recalc_grid()
                    self._estimate_precision(idx, refine_matrix_old)
                    self.ops.merged_2d = None

                self.fliph.setEnabled(False)
                self.flipv.setEnabled(False)
                self.auto_opt_btn.setChecked(False)
                self.transpose.setEnabled(False)
                self.rotate.setEnabled(False)
                self.other.err_plt_btn.setEnabled(True)
                self.other.convergence_btn.setEnabled(True)
                self.undo_refine_btn.setEnabled(True)

                [self.imview.removeItem(point) for point in self._points_corr]
                [self.other.imview.removeItem(point) for point in self.other._points_corr]
                [self.imview.removeItem(anno) for anno in self.anno_list]
                [self.other.imview.removeItem(anno) for anno in self.other.anno_list]

                self._update_imview()
                self.auto_opt_btn.setChecked(False)
                print('merge_points: ', self._merge_points)
            else:
                print('Confirm point selection! (Uncheck Select points of interest)')
        else:
            print('Select at least 4 points for refinement!')
        QtWidgets.QApplication.restoreOverrideCursor()

    def _undo_refinement(self):
        if self.other.fib:
            idx = 1
            self.other.ops.undo_refinement()
            self.other._calc_grid()
            if len(self.other.ops._refine_history) == 1:
                self.other._refined = False
                self.undo_refine_btn.setEnabled(False)
                self.other.err_btn.setText('0')
            else:
                self._estimate_precision(idx, self.other.ops._refine_matrix)

        else:
            idx = 0
            em_points = np.array([[point.x() + self.other.size / 2, point.y() + self.other.size / 2] for point in
                            self.other._points_corr])
            fm_points = np.array([[point.x() + self.size / 2, point.y() + self.size / 2]
                                  for point in self._points_corr])
            self.ops.undo_refinement(fm_points, em_points, self.other.ops.points)
            self._recalc_grid()
            if len(self.ops._refine_history) == 1:
                self._refined = False
                self.other._refined = False
                self.undo_refine_btn.setEnabled(False)
                self.other.err_btn.setText('0')
            else:
                self._estimate_precision(idx, self.ops._refine_matrix)
            self._update_imview()

    def _estimate_precision(self, idx, refine_matrix_old):
        sel_points = [[point.x() + self.other.size/2, point.y()+self.other.size/2] for point in self.other._points_corr]
        orig_fm_points = np.copy(self._points_corr)

        calc_points = []
        self.other.corr_points = []
        if idx == 0:
            src_sorted = np.array(
                sorted(self.ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
            dst_sorted = np.array(
                sorted(self.other.ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
            tr_matrix = self.ops.get_transform(src_sorted, dst_sorted)
            for i in range(len(orig_fm_points)):
                orig_point = np.array([orig_fm_points[i].x(), orig_fm_points[i].y()])
                init = np.array([orig_point[0] + self.size / 2, orig_point[1] + self.size / 2, 1])
                self.other.corr_points.append(np.copy((tr_matrix @ init)[:2]))
                transf = tr_matrix @ self.ops._refine_matrix @ init
                calc_points.append(transf[:2])
        else:
            orig_fm_points_z = np.copy(self._points_corr_z)
            for i in range(len(orig_fm_points)):
                orig_point = np.array([orig_fm_points[i].x(), orig_fm_points[i].y()])
                z = orig_fm_points_z[i]
                init = np.array([orig_point[0] + self.size / 2, orig_point[1] + self.size / 2, 1])
                transf = np.dot(self.tr_matrices, init)
                transf = self.other.ops.fib_matrix @ np.array([transf[0], transf[1], z, 1])
                self.other.corr_points.append(np.copy(transf[:2]))
                transf[:2] = (self.other.ops._refine_matrix @ np.array([transf[0], transf[1], 1]))[:2]
                calc_points.append(transf[:2])

        diff = np.array(sel_points) - np.array(calc_points)
        self.other._std[idx][0], self.other._std[idx][1] = self.other.ops.calc_error(diff)
        self.other._err[idx] = diff
        self.other.err_btn.setText('x: \u00B1{:.2f}, y: \u00B1{:.2f}'.format(
            self.other._std[idx][0] * self.other.ops.pixel_size[0],
            self.other._std[idx][1]* self.other.ops.pixel_size[0]))

        if len(self.other.corr_points) >= self.min_conv_points:
            min_points = self.min_conv_points-4
            convergence = self.other.ops.calc_convergence(self.other.corr_points, sel_points, min_points, refine_matrix_old)
            self.other._conv[idx] = convergence
        else:
            self.other._conv[idx] = []
        np.save('corr_points.npy', self.other.corr_points)
        np.save('corr_points_z.npy', self._points_corr_z)
        np.save('sel.npy', sel_points)
        np.save('calc.npy', calc_points)

    def _scatter_plot(self, idx):
        if self._err[idx] is not None:
            pg.plot(self._err[idx][:, 0] * self.ops.pixel_size[0], self._err[idx][:, 1] * self.ops.pixel_size[1], pen=None, symbol='o')
        else:
            print('Data not refined yet!')

    def _convergence_plot(self, idx):
        if self._conv[idx] is not None:
            if len(self._conv[idx]) == 0:
                print('To use this feature, you have to use at least 10 points for the refinement!')
            else:
                pg.plot(np.arange(self.min_conv_points-4, self.min_conv_points-4+len(self._conv[idx])), self._conv[idx])
        else:
            print('Data not refined yet!')

    def fit_circles(self):
        if self.auto_opt_btn.isChecked():
            bead_size = float(self.size_box.text())

            points_fm = np.array([[p.x() + self.size/2, p.y() + self.size/2] for p in self._points_corr])
            points_em = np.array([[p.x() + self.other.size/2, p.y() + self.other.size/2] for p in self.other._points_corr])

            #points_fm_fitted = self.ops.fit_circles(points_fm, bead_size)
            points_em_fitted = self.other.ops.fit_circles(points_em, bead_size)
            #[self.imview.removeItem(point) for point in self._points_corr]
            [self.other.imview.removeItem(point) for point in self.other._points_corr]
            #self._points_corr = []
            self.other._points_corr = []
            #circle_size_fm = bead_size * 1e-6 / self.ops.voxel_size[0]
            circle_size_em = bead_size * 1e3 / self.other.ops.pixel_size[0]
            for i in range(len(points_em_fitted)):
                #pos = QtCore.QPointF(points_fm_fitted[i,0] - circle_size_fm/2, points_fm_fitted[i,1] - circle_size_fm/2)
                #point = pg.CircleROI(pos, circle_size_fm, parent=self.imview.getImageItem(), movable=True, removable=True)
                #point.setPen(0, 255, 255)
                #point.removeHandle(0)
                #self._points_corr.append(point)
                #self.imview.addItem(point)
                #self.size = circle_size_fm

                pos = QtCore.QPointF(points_em_fitted[i,0] - circle_size_em/2, points_em_fitted[i,1] - circle_size_em/2)
                point = pg.CircleROI(pos, circle_size_em, parent=self.other.imview.getImageItem(), movable=True, removable=True)
                point.setPen(0, 255, 255)
                point.removeHandle(0)
                self.other._points_corr.append(point)
                self.other.imview.addItem(point)
                self.other.size = circle_size_em

    def _show_FM_peaks(self):
        if self.show_peaks_btn.isChecked():
            if self.other.ops is None:
                print('Select FM data first')
                return
            else:
                if self.fib and self.other.ops.tf_peaks_3d is None:
                        print('Calculate 3d FM peaks first!')
                        return
                elif (self.other.ops.tf_peak_slices is None or self.other.ops.tf_peak_slices[-1] is None):
                        print('Calculate FM peak positions for maximum projection first')
                        return

            if len(self.peaks) != 0:
                self.peaks = []
            if self.fib:
                src_sorted = np.array(
                    sorted(self.other.ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                dst_sorted = np.array(
                    sorted(self.sem_ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                tr_matrix = self.ops.get_fib_transform(src_sorted, dst_sorted, self.sem_ops.tf_matrix)
                for i in range(self.other.ops.tf_peaks_3d.shape[0]):
                    z = self.other.ops.calc_z(i, self.other.ops.tf_peaks_3d[i,:2])
                    init = np.array([self.other.ops.tf_peaks_3d[i,0], self.other.ops.tf_peaks_3d[i,1], 1])
                    transf = np.dot(tr_matrix, init)
                    transf = self.ops.fib_matrix @ np.array([transf[0], transf[1], z, 1])
                    if self._refined:
                        transf = self.ops._refine_matrix @ np.array([transf[0], transf[1], 1])
                    pos = QtCore.QPointF(transf[0] - self.other.size / 2, transf[1] - self.other.size / 2)
                    point = pg.CircleROI(pos, self.other.size, parent=self.imview.getImageItem(), movable=False,
                                               removable=False)
                    point.setPen(255, 0, 0)
                    point.removeHandle(0)
                    self.peaks.append(point)
                    self.imview.addItem(point)
            else:
                src_sorted = np.array(
                    sorted(self.other.ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                dst_sorted = np.array(
                    sorted(self.ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                tr_matrix = self.ops.get_transform(src_sorted, dst_sorted)
                for i in range(self.other.ops.tf_peak_slices[-1].shape[0]):
                    init = np.array([self.other.ops.tf_peak_slices[-1][i, 0], self.other.ops.tf_peak_slices[-1][i, 1], 1])
                    transf = tr_matrix @ self.other.ops._refine_matrix @ init
                    pos = QtCore.QPointF(transf[0] - self.other.size / 2, transf[1] - self.other.size / 2)
                    point = pg.CircleROI(pos, self.other.size, parent=self.imview.getImageItem(), movable=False,
                                         removable=False)
                    point.setPen(255, 0, 0)
                    point.removeHandle(0)
                    self.peaks.append(point)
                    self.imview.addItem(point)
        else:
            [self.imview.removeItem(point) for point in self.peaks]

    def correct_grid_z(self):
        self.ops.fib_matrix = None
        # set FIB matrix to None to recalculate with medium z slice
        self._calc_grid(scaling=self.other.ops.voxel_size[2] / self.other.ops.voxel_size[0])
        self.shift_x_btn.setText(str(self.ops._total_shift[0]))
        self.shift_y_btn.setText(str(self.ops._total_shift[1]))
        self.ops._total_shift = None
        self._refine_grid()
        self._show_grid()
        print('WARNING! Recalculate FIB grid square for z = ', self.num_slices // 2)

    def merge(self):
        if not self.other.fib:
            if self.ops.merged_2d is None:
                for i in range(self.ops.num_channels):
                    self.ops.apply_merge_2d(self.other.ops.data, self.other.ops.points, i)
                    self.progress.setValue((i+1)/self.ops.num_channels * 100)
            else:
                self.progress.setValue(100)
            if self.ops.merged_2d is not None:
                print('Merged shape: ', self.ops.merged_2d.shape)
        else:
            if self.ops.merged_3d is None:
                if self.other._refined:
                    for i in range(self.ops.num_channels):
                        self.ops.load_channel(i)
                        self.ops.apply_merge_3d(self.tr_matrices, self.other.ops.fib_matrix, self.other.ops._refine_matrix,
                                      self.other.ops.data, self._merge_points, self._merge_points_z, self.other._merge_points, i)
                        self.progress.setValue((i+1)/self.ops.num_channels * 100)
                else:
                    print('You have to perform at least one round of refinement before you can merge the images!')
            else:
                self.progress.setValue(100)
            if self.ops.merged_3d is not None:
                print('Merged shape: ', self.ops.merged_3d.shape)

    def reset_base(self):
        [self.imview.removeItem(point) for point in self._points_corr]
        [self.other.imview.removeItem(point) for point in self.other._points_corr]
        [self.imview.removeItem(anno) for anno in self.anno_list]
        [self.other.imview.removeItem(anno) for anno in self.other.anno_list]

        self.anno_list = []
        self.other.anno_list = []
        self.counter = 0
        self.other.counter = 0
        self._points_corr = []
        self.other._points_corr = []
        self._points_corr_z = []
        self.other._points_corr_z = []
        self._orig_points_corr = []
        self.other._orig_points_corr = []
        self._points_corr_indices = []
        self.other._points_corr_indices = []
        self.peaks = []