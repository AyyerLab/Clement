import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import copy
import matplotlib
import math

from . import utils

class PeakROI(pg.CircleROI):
    def __init__(self, pos, size, parent, movable=False, removable=False, resizable=False, color=None):
        super(PeakROI, self).__init__(pos, size, parent=parent,
                                      movable=movable, removable=removable, resizable=resizable)
        self.original_color = (255, 0, 0)
        self.moved_color = (0, 255, 255)
        self.original_pos = copy.copy(np.array([pos.x(), pos.y()]))
        self.has_moved = False

        if color is None:
            self.setPen(self.original_color)
        else:
            self.setPen(color)
        self.removeHandle(0)

    def contextMenuEnabled(self):
        return True

    def getMenu(self):
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle('ROI')
            resetAct = QtWidgets.QAction('Reset Peak', self.menu)
            resetAct.triggered.connect(self.resetPos)
            self.menu.addAction(resetAct)
            self.menu.resetAct = resetAct
        self.menu.setEnabled(True)
        return self.menu

    def resetPos(self):
        if self.has_moved:
            self.blockSignals(True)
            self.setPos(self.original_pos)
            self.setPen(self.original_color)
            self.has_moved = False
            self.blockSignals(False)

    def peakMoved(self, item):
        self.setPen(self.moved_color)
        self.has_moved = True

    def peakRefined(self, refined_pos):
        self.original_pos = refined_pos

class BaseControls(QtWidgets.QWidget):
    def __init__(self):
        super(BaseControls, self).__init__()
        # ops is EM_operations or FM_operations
        self.tag = 'base'
        self.ops = None
        self.other = None  # The other controls object
        self.print = None

        self._box_coordinate = None

        self._pois_raw = []
        self._pois_channel_indices = []

        self._points_corr = []
        self._orig_points_corr = []
        self._points_corr_indices = []

        self._refined = False
        self.diff = None
        self._err = [None, None, None]
        self._std = [[None, None], [None, None], [None, None]]
        self._conv = [None, None, None]
        self._dist = None

        self._points_corr_history = []
        self._points_corr_z_history = []
        self._orig_points_corr_history = []
        self._fib_vs_sem_history = []
        self._size_history = []
        self._fib_flips = []

        self.pois = []
        self.pois_sizes = []
        self.pois_base = []
        self.pois_z = []
        self.pois_err = []
        self.pois_cov = []

        self.points_raw = []
        self.points_base = []
        self.points_corr_z = []

        self.flips = [False, False, False, False]
        self.tr_matrices = None
        self.fm_sem_corr = None
        self.orig_fm_sem_corr = None
        self.fixed_orientation = False
        self.show_grid_box = False
        self.show_tr_grid_box = False
        self.clicked_points = []
        self.grid_box = None
        self.tr_grid_box = None
        self.boxes = []
        self.tr_boxes = []
        self.redo_tr = False
        self.setContentsMargins(0, 0, 0, 0)
        self.poi_counter = 0
        self.counter = 0
        self.anno_list = []
        self.poi_anno_list = []
        self.size = 10
        self.orig_size = 10
        self.peaks = []
        self.num_slices = None
        self.min_conv_points = 10
        self.show_merge = False
        self.progress = 0
        self.cov_matrix = None

    def _init_ui(self):
        self.log('This message should not be seen. Please override _init_ui')

    def select_tab(self, idx, semcontrols, fibcontrols, temcontrols):
        semcontrols.tab_index = idx
        fibcontrols.tab_index = idx
        temcontrols.tab_index = idx
        if idx == 0:
            fibcontrols.show_grid_btn.setChecked(False)
            semcontrols._update_imview()
            self.other = semcontrols
            if semcontrols._refined:
                self.undo_refine_btn.setEnabled(True)
            else:
                self.undo_refine_btn.setEnabled(False)
        elif idx == 1:
            #if self.orig_fm_sem_corr is None and fibcontrols.tab_index == 0:
            if self.orig_fm_sem_corr is None:
                if semcontrols.ops is not None and self.ops is not None:
                    if self.ops.points is not None and semcontrols.ops.points is not None:
                        self._calc_tr_matrices(em_points=semcontrols.ops.points)
            self.other = fibcontrols
            if fibcontrols.ops is not None:
                show_grid = semcontrols.show_grid_btn.isChecked()
                fibcontrols.show_grid_btn.setChecked(show_grid)
            if fibcontrols._refined:
                self.undo_refine_btn.setEnabled(True)
            else:
                self.undo_refine_btn.setEnabled(False)
            fibcontrols._update_imview()
            fibcontrols.sem_ops = semcontrols.ops
            if semcontrols.ops is not None:
                if semcontrols.ops._orig_points is not None:
                    fibcontrols.enable_buttons(enable=True)
                else:
                    fibcontrols.enable_buttons(enable=False)
                if fibcontrols.ops is not None and semcontrols.ops._tf_points is not None:
                    fibcontrols.ops._transformed = True
        else:
            fibcontrols.show_grid_btn.setChecked(False)
            temcontrols._update_imview()
            self.other = temcontrols
            if temcontrols._refined:
                self.undo_refine_btn.setEnabled(True)
            else:
                self.undo_refine_btn.setEnabled(False)


        if self.other.show_merge:
            self.progress_bar.setValue(100)
        else:
            self.progress_bar.setValue(0)
        if self.other._refined:
            self.err_btn.setText('x: \u00B1{:.2f}, y: \u00B1{:.2f}'.format(self.other._std[idx][0],
                                                                           self.other._std[idx][1]))
        else:
            self.err_btn.setText('0')

        if self.other.ops is None or not self.other.ops._transformed:
            return

        if self.ops is not None:
            if self.ops._transformed:
                self.other.size_box.setEnabled(True)
                self.other.auto_opt_btn.setEnabled(True)

        points_corr = np.copy(self._points_corr)
        if len(self._points_corr) != len(self.other._points_corr):
            if self.counter != 0:
                self.counter -= len(self._points_corr)
            for i in range(len(self._points_corr)):
                [self.imview.removeItem(point) for point in self._points_corr]
                self._points_corr = []
                self._orig_points_corr = []
                [self.imview.removeItem(anno) for anno in self.anno_list]
                self.anno_list = []
                self._points_corr_indices = []
                self.points_corr_z = []

            self._update_tr_matrices()

            item = self.imview.getImageItem()
            for i in range(len(points_corr)):
                pos = points_corr[i].pos()
                self._draw_correlated_points(pos, item)

    def _imview_clicked(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
            return

        if self.ops is None:
            return

        pos = self.imview.getImageItem().mapFromScene(event.pos())
        item = self.imview.getImageItem()

        if hasattr(self, 'define_btn') and self.define_btn.isChecked():
            pos.setX(pos.x() - self.size // 2)
            pos.setY(pos.y() - self.size // 2)
            roi = pg.CircleROI(pos, self.size, parent=item, movable=False)
            roi.setPen(255, 0, 0)
            roi.removeHandle(0)
            self.imview.addItem(roi)
            self.clicked_points.append(roi)
        elif hasattr(self, 'poi_btn') and self.poi_btn.isChecked():
            self._draw_pois(pos, item)
        elif hasattr(self, 'select_btn') and self.select_btn.isChecked():
            pos.setX(pos.x() - self.size // 2)
            pos.setY(pos.y() - self.size // 2)
            self._draw_correlated_points(pos, item)
        elif hasattr(self, 'select_region_btn') and self.select_region_btn.isChecked():
            '''EM only: Select individual image from montage'''
            points_obj = (pos.x(), pos.y())
            self._box_coordinate = np.copy(np.array(points_obj))

            # If clicked point is inside image
            if points_obj[0] < self.ops.data.shape[0] and points_obj[1] < self.ops.data.shape[1]:
                # Get index of selected region
                ind = self.ops.get_selected_region(np.array(points_obj))
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
                        self.print('Oops, something went wrong. Try again!')
            else:
                self.print('Oops, something went wrong. Try again!')

    @utils.wait_cursor('print')
    def _define_poi_toggled(self, checked):
        if checked:
            if self.select_btn.isChecked():
                self.select_btn.setChecked(False)
            self.poi_counter = len(self.pois)
            self.fliph.setEnabled(False)
            self.flipv.setEnabled(False)
            self.transpose.setEnabled(False)
            self.rotate.setEnabled(False)
            self.point_ref_btn.setEnabled(False)

            self.ops.load_channel(ind=self.point_ref_btn.currentIndex())
            self.print('Select points of interest on %s image' % self.tag)
        else:
            if self.ops.channel is not None:
                self.ops.clear_channel()
            self.print('Done selecting points of interest on %s image' % self.tag)
            if not self.other._refined:
                self.fliph.setEnabled(True)
                self.flipv.setEnabled(True)
                self.transpose.setEnabled(True)
                self.rotate.setEnabled(True)
            self.point_ref_btn.setEnabled(True)

    @utils.wait_cursor('print')
    def _define_corr_toggled(self, checked):
        if self.ops.adjusted_params == False:
            self.print('You have to adjust and save the peak finding parameters first!')
            self.select_btn.setChecked(False)
            return

        if self.other.translate_peaks_btn.isChecked() or self.other.refine_peaks_btn.isChecked():
            self.print('You have to uncheck translation buttons on SEM/FIB side first!')
            self.select_btn.setChecked(False)
            return

        if self.other.ops is not None and self.other.tab_index == 1 and self.other.ops.fib_matrix is None:
            self.print('You have to calculate the grid box for the FIB view first!')
            self.select_btn.setChecked(False)
            return

        if checked:
            if self.poi_btn.isChecked():
                self.poi_btn.setChecked(False)
            self.counter = len(self._points_corr)
            self.fliph.setEnabled(False)
            self.flipv.setEnabled(False)
            self.transpose.setEnabled(False)
            self.rotate.setEnabled(False)
            self.point_ref_btn.setEnabled(False)

            if (self.ops._transformed and self.ops.tf_peaks_z is None) or (not self.ops._transformed and self.ops.peaks_z is None):
                uncheck = False
                if self.peak_btn.isChecked == False:
                    uncheck = True
                self.peak_btn.setChecked(True)
                if uncheck:
                    self.peak_btn.setChecked(False)
                if self.peak_controls.peak_channel_btn.currentIndex() != self.ops._channel_idx:
                    self.ops.load_channel(self.peak_controls.peak_channel_btn.currentIndex())
                color_matrix = self.ops.tf_matrix @ self.ops._color_matrices[
                        self.peak_controls.peak_channel_btn.currentIndex()]
                self.ops.fit_z(self.ops.channel, transformed=self.ops._transformed, tf_matrix=color_matrix,
                               flips=self.flips, shape=self.ops.data.shape[:-1])
            if self.other.ops is not None and self.other.tab_index == 1:
                #self.fm_sem_corr = self.ops.update_tr_matrix(self.orig_fm_sem_corr, self._fib_flips)
                self.fm_sem_corr = self.ops.update_fm_sem_matrix(self.orig_fm_sem_corr, self._fib_flips)
            self.ops.load_channel(ind=self.point_ref_btn.currentIndex())
            if self.other.ops is not None:
                if self.ops.points is not None and self.other.ops.points is not None:
                    self._update_tr_matrices()
            self.print('Select reference points on %s image' % self.tag)
        else:
            if self.ops.channel is not None:
                self.ops.clear_channel()
            self.print('Done selecting reference points on %s image' % self.tag)
            if not self.other._refined:
                self.fliph.setEnabled(True)
                self.flipv.setEnabled(True)
                self.transpose.setEnabled(True)
                self.rotate.setEnabled(True)
            self.point_ref_btn.setEnabled(True)

    def _calc_tr_matrices(self, fm_points=None, em_points=None):
        if fm_points is None:
            src_sorted = np.copy(np.array(
                sorted(self.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]])))
        else:
            src_sorted = np.array(sorted(fm_points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
        if em_points is None:
            dst_sorted = np.array(sorted(self.other.ops._tf_points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
        else:
            dst_sorted = np.array(sorted(em_points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
        self.orig_fm_sem_corr = self.other.ops.get_transform(src_sorted, dst_sorted)

    def _update_tr_matrices(self):
        if self.other.tab_index == 1:
            if self.other.sem_ops._refine_matrix is not None:
                self.other.tr_matrices = self.other.ops.get_fib_transform(self.other.sem_ops._refine_matrix @ self.other.sem_ops.tf_matrix) @ self.fm_sem_corr
            else:
                self.other.tr_matrices = self.other.ops.get_fib_transform(self.other.sem_ops.tf_matrix) @ self.fm_sem_corr
        else:
            src_sorted = np.array(
                sorted(self.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
            dst_sorted = np.array(
                sorted(self.other.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
            self.other.tr_matrices = self.ops.get_transform(src_sorted, dst_sorted)

    def _draw_pois(self, pos, item):
        point = np.copy(np.array([pos.x(), pos.y()]))
        init = self._fit_poi(point)
        if init is None:
            return
        init_base = copy.copy(init)
        if self.ops._transformed:
            self._calc_base_points(init[:2], poi=True)
            init_base[:2] = [self.pois_base[-1].x(), self.pois_base[-1].y()]
        self._calc_raw_points(init_base[:2], poi=True)
        if not self.ops._transformed and not np.array_equal(np.identity(3), self.ops.tf_matrix):
            self._transform_pois(init[:2], poi=True)

        self._draw_fm_pois(init[:2], item)

        if self.other.show_merge:
            self.other.popup._update_poi(pos)

    def _calc_ellipses(self, counter):
        cov_matrix = np.copy(self.pois_cov[counter])
        if self.ops._transformed:
            cov_matrix = self._update_cov_matrix(cov_matrix)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix[:2,:2])

        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[order]
        vx, vy = eigvecs[0,0], eigvecs[0,1]
        theta = -np.arctan2(vy, vx) * 180 / np.pi
        #scale eigenvalues to 75% confidence interval and pixel size
        lambda_1, lambda_2 = 2 * np.sqrt(2.77*eigvals)
        self.log(lambda_1, lambda_2, theta)
        return lambda_1, lambda_2, theta

    def _update_cov_matrix(self, cov_matrix):
        cov = np.copy(cov_matrix)
        cov[-1,-1] = 0
        tf_cov = self.ops.tf_matrix @ cov @ self.ops.tf_matrix.T
        transp, rot, fliph, flipv = self.flips
        if flipv:
            tf_cov[0,1] *= -1
            tf_cov[1,0] *= -1
        if fliph:
            tf_cov[1,0] *= -1
        if rot:
            cov_2d = np.copy(tf_cov[:2, :2])
            rot_matrix = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]])
            cov_2d = rot_matrix @ cov_2d @ rot_matrix.T
            tf_cov[:2, :2] = cov_2d
        if transp:
            #covariance matrices are symmetric...
            pass
        return tf_cov

    def _draw_fm_pois(self, init, item):
        cmap = matplotlib.cm.get_cmap('cool')
        lin = np.linspace(0, 1, 100)
        colors = cmap(lin)
        idx = np.argmin(np.abs(lin-self.pois_err[-1][-1]))
        color = colors[idx]
        color = matplotlib.colors.to_hex(color)

        img_center = np.array(self.ops.data.shape) / 2
        lambda_1, lambda_2, theta = self._calc_ellipses(self.poi_counter)
        if math.isnan(lambda_1) or math.isnan(lambda_2):
            del self.pois_err[-1]
            del self.pois_cov[-1]
            del self._pois_raw[-1]
            del self.pois_base[-1]
            del self.pois_z[-1]
            QtWidgets.QApplication.restoreOverrideCursor()
            return

        size = (lambda_1, lambda_2)
        pos = QtCore.QPointF(init[0]-size[0]/2, init[1]-size[1]/2)
        point_obj = pg.EllipseROI(img_center, size=[size[0], size[1]], angle=0, parent=item,
                              movable=False, removable=True, resizable=False, rotatable=False)

        point_obj.setTransformOriginPoint(QtCore.QPointF(lambda_1/2, lambda_2/2))
        point_obj.setRotation(theta)
        point_obj.setPos([pos.x(), pos.y()])
        point_obj.setPen(color)
        point_obj.removeHandle(0)
        point_obj.removeHandle(0)
        self.imview.addItem(point_obj)

        self.pois.append(point_obj)
        self._pois_channel_indices.append(self.point_ref_btn.currentIndex())
        self.pois_sizes.append(size)
        self.poi_counter += 1
        annotation_obj = pg.TextItem(str(self.poi_counter), color=color, anchor=(0, 0))
        annotation_obj.setPos(pos.x() + 1, pos.y() + 1)
        self.imview.addItem(annotation_obj)
        self.poi_anno_list.append(annotation_obj)

        point_obj.sigRemoveRequested.connect(lambda: self._remove_pois(point_obj))

    def _draw_correlated_points(self, pos, item):
        point = np.array([pos.x() + self.size / 2, pos.y() + self.size / 2])
        init, pos, z = self._calc_optimized_position(point, pos)
        if init is None:
            return
        init_base = copy.copy(init)
        if self.ops._transformed:
            self._calc_base_points(init[:2])
            init_base[:2] = [self.points_base[-1].x() + self.size/2, self.points_base[-1].y()+self.size/2]
        self._calc_raw_points(init_base[:2])
        if not self.ops._transformed and not np.array_equal(np.identity(3), self.ops.tf_matrix):
            self._transform_pois(init[:2])
        self._draw_fm_points(pos, item)
        self._draw_em_points(init, z)

    def _fit_poi(self, point):
        #Dont use decorator here because of return value!
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        init, err, cov = self.ops.gauss_3d(point, self.ops._transformed, self.point_ref_btn.currentIndex())
        if init is None:
            QtWidgets.QApplication.restoreOverrideCursor()
            return None
        self.pois_z.append(init[-1])
        if self.ops._transformed:
            init = self.ops.update_points(np.expand_dims(init[:2], axis=0))[0]

        self.pois_err.append(err.tolist())
        self.pois_cov.append(cov.tolist())
        QtWidgets.QApplication.restoreOverrideCursor()
        return init

    def _calc_optimized_position(self, point, pos=None):
        peaks = None
        ind = self.ops.check_peak_index(point, self.size, transformed=self.ops._transformed)
        if ind is None and self.other.tab_index == 1 and not self.other._refined:
            self.print('If the FIB tab is selected, you have to select a bead for the first refinement!')
            return None, None, None
        if ind is not None:
            if self.ops._transformed:
                peaks = self.ops.tf_peaks
            else:
                peaks = self.ops.peaks
        z = self.ops.calc_z(ind, point, self.ops._transformed, self.point_ref_btn.currentIndex())
        if z is None:
            self.print('z is None, something went wrong here... Try another bead!')
            return None, None, None
        self.points_corr_z.append(z)

        if ind is not None:
            if pos is not None:
                pos.setX(peaks[ind, 0] - self.size / 2)
                pos.setY(peaks[ind, 1] - self.size / 2)
            init = np.array([peaks[ind, 0], peaks[ind, 1], 1])

        else:
            init = np.array([point[0], point[1], 1])
        return init, pos, z

    def _draw_fm_points(self, pos, item):
        point_obj = pg.CircleROI(pos, self.size, parent=item, movable=True, removable=True)
        point_obj.setPen(0, 255, 0)
        point_obj.removeHandle(0)
        self.imview.addItem(point_obj)

        self._points_corr.append(point_obj)
        self._orig_points_corr.append([pos.x() + self.size // 2, pos.y() + self.size // 2])
        self.counter += 1
        annotation_obj = pg.TextItem(str(self.counter), color=(0, 255, 0), anchor=(0, 0))
        annotation_obj.setPos(pos.x() + 5, pos.y() + 5)
        self.imview.addItem(annotation_obj)
        self.anno_list.append(annotation_obj)

        self._points_corr_indices.append(self.counter - 1)
        point_obj.sigRemoveRequested.connect(lambda: self._remove_correlated_points(point_obj, remove_raw=True))

    def _draw_em_points(self, init, z):
        if self.other.ops is None:
            return
        condition = False
        if self.ops._tf_points is not None:
            if self.other.tab_index == 1:
                if self.other.sem_ops is not None:
                    if self.other.sem_ops._tf_points is not None or self.other.sem_ops._tf_points_region is not None:
                        condition = True
            else:
                if self.other.ops._tf_points is not None or self.other.ops._tf_points_region is not None:
                    condition = True
        if not condition:
            self.print('Transform both images before point selection')
            return
        if self.other.tr_matrices is None:
            return

        self.log('2d init: ', init)
        self.log('Other class:', self.other, self.other.ops)
        self.log('tr_matrices:\n', self.other.tr_matrices)
        if self.other.tab_index == 1:
            transf = np.dot(self.other.tr_matrices, init)
            transf = self.other.ops.fib_matrix @ np.array([transf[0], transf[1], z, 1])
            self.other.points_corr_z.append(transf[2])
            if self.other._refined:
                transf[:2] = (self.other.ops._refine_matrix @ np.array([transf[0], transf[1], 1]))[:2]
        else:
            transf = np.dot(self.other.tr_matrices, init)

        self.print('Transformed point: ', transf)
        pos = QtCore.QPointF(transf[0] - self.other.size / 2, transf[1] - self.other.size / 2)
        point_other = pg.CircleROI(pos, self.other.size, parent=self.other.imview.getImageItem(),
                                   movable=True, removable=True)
        point_other.setPen(0, 255, 255)
        point_other.removeHandle(0)
        self.other.imview.addItem(point_other)
        self.other._points_corr.append(point_other)
        self.other._orig_points_corr.append([transf[0], transf[1]])

        self.other.counter += 1
        annotation_other = pg.TextItem(str(self.other.counter), color=(0, 255, 255), anchor=(0, 0))
        annotation_other.setPos(pos.x() + 5, pos.y() + 5)
        self.other.imview.addItem(annotation_other)
        self.other.anno_list.append(annotation_other)

        point_other.sigRemoveRequested.connect(lambda: self._remove_correlated_points(point_other, remove_raw=True))
        point_other.sigRegionChangeFinished.connect(lambda: self._update_annotations(point_other))

        self.other._points_corr_indices.append(self.counter - 1)

        for i in range(len(self.other.peaks)):
            if self.other.peaks[i].pos() == self.other._points_corr[-1].pos():
                self.other.imview.removeItem(self.other.peaks[i])

    @utils.wait_cursor('print')
    def _show_FM_peaks(self, state=None):
        if not self.show_peaks_btn.isChecked():
            # Remove already shown peaks
            [self.imview.removeItem(point) for point in self.peaks]
            self.translate_peaks_btn.setChecked(False)
            self.translate_peaks_btn.setEnabled(False)
            self.refine_peaks_btn.setChecked(False)
            self.refine_peaks_btn.setEnabled(False)
            self.peaks = []
            return

        if self.other.ops is None:
            # Check for the existence of FM image
            self.print('Select FM data first')
            self.show_peaks_btn.setChecked(False)
            return

        if self.other.ops.tf_peaks is None:
            # Check that peaks have been found for FM image
            self.print('Calculate FM peak positions for maximum projection first')
            self.show_peaks_btn.setChecked(False)
            return

        if self.tab_index == 1 and self.tr_matrices is None:
            self.other.fm_sem_corr = self.other.ops.update_fm_sem_matrix(self.other.orig_fm_sem_corr, self.other._fib_flips)
            if not self.ops._transformed:
                self.print('You have to transform the SEM image first!')
                self.show_peaks_btn.setChecked(False)
                return
        if len(self.peaks) != 0:
            self.peaks = []

        if self.ops._transformed:
            self.other._update_tr_matrices()
        if self.other.diff is not None:
            cmap = matplotlib.cm.get_cmap('cool')
            diff_normed = self.other.diff / self.other.diff.max()
            diff_abs = np.sqrt(diff_normed[:,0]**2 + diff_normed[:,1]**2)
            colors = cmap(diff_abs)
        if self.tab_index == 1:
            for i in range(len(self.other.ops.tf_peaks)):
                z = self.other.ops.tf_peaks_z[i]
                init = np.array(
                    [self.other.ops.tf_peaks[i, 0], self.other.ops.tf_peaks[i, 1], 1])
                transf = np.dot(self.tr_matrices, init)
                transf = self.ops.fib_matrix @ np.array([transf[0], transf[1], z, 1])
                if self._refined:
                    transf = self.ops._refine_matrix @ np.array([transf[0], transf[1], 1])
                pos = QtCore.QPointF(transf[0] - self.orig_size / 2, transf[1] - self.orig_size / 2)
                color = None
                if self.other.diff is not None:
                    idx = np.where(np.isclose(np.array(self.other.refined_points), np.array([transf[0], transf[1]])))
                    if not np.array_equal(idx[0], np.array([])):
                        if len(idx) == 2:
                            idx = idx[0][0]
                        color = colors[idx]
                        color = matplotlib.colors.to_hex(color)
                point = PeakROI(pos, self.size, self.imview.getImageItem(), color=color)
                self.peaks.append(point)
                for i in range(len(self._orig_points_corr)):
                    if np.allclose(transf[:2], self._orig_points_corr[i]):
                        self.imview.removeItem(point)
        else:
            for peak in self.other.ops.tf_peaks:
                init = np.array([peak[0], peak[1], 1])
                if self.show_btn.isChecked():
                    transf = np.linalg.inv(self.ops.tf_matrix) @ self.tr_matrices @ init
                else:
                    transf = self.tr_matrices @ init
                pos = QtCore.QPointF(transf[0] - self.orig_size / 2, transf[1] - self.orig_size / 2)
                color = None
                if self.other.diff is not None:
                    idx = np.where(np.isclose(np.array(self.other.refined_points), np.array([transf[0], transf[1]])))
                    if not np.array_equal(idx[0], np.array([])):
                        if len(idx) == 2:
                            idx = idx[0][0]
                        color = colors[idx]
                        color = matplotlib.colors.to_hex(color)
                point = PeakROI(pos, self.orig_size, self.imview.getImageItem(), color=color)
                self.peaks.append(point)
                # Remove FM beads information
                for i in range(len(self._orig_points_corr)):
                    if np.allclose(transf[:2], self._orig_points_corr[i]):
                        self.imview.removeItem(point)

        self.peaks = np.array(self.peaks)
        self.translate_peaks_btn.setEnabled(True)
        self.refine_peaks_btn.setEnabled(True)

    def _translate_peaks(self, active):
        if active:
            self.show_grid_btn.setChecked(False)
            self.refine_peaks_btn.setChecked(False)
            for p in self.peaks:
                p.translatable = True
                p.sigRegionChangeFinished.connect(self._translate_peaks_slot)
            self._old_peak_pos = [[p.pos().x(), p.pos().y()] for p in self.peaks]
        else:
            for p in self.peaks:
                p.sigRegionChangeFinished.disconnect()
                p.translatable = False

    def _translate_peaks_slot(self, item):
        ind = np.where(item == self.peaks)[0][0]
        shift = item.pos() - self._old_peak_pos[ind]
        self.grid_box.setPos(self.grid_box.pos().x()+shift[0], self.grid_box.pos().y()+shift[1])
        self.log(ind, shift)
        for i in range(len(self.peaks)):
            self._old_peak_pos[i][0] += shift.x()
            self._old_peak_pos[i][1] += shift.y()
            if i != ind and not self.peaks[i].has_moved:
                self.peaks[i].setPos(self._old_peak_pos[i], finish=False)

    def _refine_peaks(self, active):
        if self.other.ops is not None:
            if self.ops.points is not None and self.other.ops.points is not None:
                self.other._update_tr_matrices()
        if active:
            if self.tab_index != 1 and not (self.ops._transformed and self.other.ops._transformed):
                self.print('You have to show the transformed data on both sides!')
                self.refine_peaks_btn.setChecked(False)
                return
            elif self.tab_index == 1 and not self.other.ops._transformed:
                self.print('You have to show the transformed FM data!')
                self.refine_peaks_btn.setChecked(False)
                return
            self.show_grid_btn.setChecked(False)
            self.translate_peaks_btn.setChecked(False)
            for p in self.peaks:
                p.translatable = True
                p.sigRegionChangeFinished.connect(lambda pt=p: self._peak_to_point(pt))
        else:
            for p in self.peaks:
                p.sigRegionChangeFinished.disconnect()
                p.translatable = False

    def _peak_to_point(self, peak):
        idx = self._check_point_idx(peak)
        if idx is None:
            self.imview.removeItem(peak)
            peak.peakMoved(None)
            ref_ind = [i for i in range(len(self.peaks)) if self.peaks[i] == peak]
            pos = self.other.ops.tf_peaks[ref_ind[0]]
            point = QtCore.QPointF(pos[0] - self.other.size / 2, pos[1] - self.other.size / 2)
            self.other._draw_correlated_points(point, self.imview.getImageItem())
            self._points_corr[-1].setPos(peak.pos())

    def _check_point_idx(self, point):
        idx = None
        num_beads = len(self._points_corr)
        for i in range(len(self._points_corr)):
            if self._points_corr[i] == point:
                idx = i
                break
        if idx is None and len(self.other._points_corr) == num_beads:
            for i in range(len(self._points_corr)):
                if self.other._points_corr[i] == point:
                    idx = i
                    break
        return idx

    def _transform_pois(self, point=None, poi=False):
        if poi:
            if point is None:
                points_tf = []
                for i in range(len(self._pois_raw)):
                    pos = self._pois_raw[i]
                    point = np.array([pos.x(), pos.y()])
                    point_tf = self.ops.tf_matrix @ np.array([point[0], point[1], 1])
                    points_tf.append(point_tf[:2])
                    pos = QtCore.QPointF(points_tf[i][0], points_tf[i][1])
                    self._draw_fm_pois(np.array([points_tf[i][0], points_tf[i][1]]), self.imview.getImageItem())
                    self.pois_base.append(pos)
            else:
                point_tf = self.ops.tf_matrix @ np.array([point[0], point[1], 1])
                pos = QtCore.QPointF(point_tf[0], point_tf[1])
                self.pois_base.append(pos)
        else:
            if point is None:
                points_tf = []
                for i in range(len(self.points_raw)):
                    pos = self.points_raw[i]
                    point = np.array([pos.x() + self.size/2, pos.y() + self.size/2])
                    point_tf = self.ops.tf_matrix @ np.array([point[0], point[1], 1])
                    points_tf.append(point_tf[:2])
                    pos = QtCore.QPointF(points_tf[i][0] - self.size / 2, points_tf[i][1] - self.size / 2)
                    self._draw_fm_points(pos, self.imview.getImageItem())
                    self.points_base.append(pos)
            else:
                point_tf = self.ops.tf_matrix @ np.array([point[0], point[1], 1])
                pos = QtCore.QPointF(point_tf[0] - self.size / 2, point_tf[1] - self.size / 2)
                self.points_base.append(pos)

    def _calc_base_points(self, point, poi=False):
        point = np.copy(point)
        tf_shape = self.ops.data.shape[:-1]
        transp, rot, fliph, flipv = self.flips
        if flipv:
            point[1] = tf_shape[1] - point[1]
        if fliph:
            point[0] = tf_shape[0] - point[0]
        if rot:
            temp = tf_shape[0] - 1 - point[0]
            point[0] = point[1]
            point[1] = temp
        if transp:
            point = np.array([point[1], point[0], 1])

        if poi:
            pos = QtCore.QPointF(point[0], point[1])
            self.pois_base.append(pos)
        else:
            pos = QtCore.QPointF(point[0] - self.size / 2, point[1] - self.size / 2)
            self.points_base.append(pos)

    def _calc_raw_points(self, point, poi=False):
        point = np.copy(point)
        if poi:
           if self.ops._transformed:
               pos_raw = np.linalg.inv(self.ops.tf_matrix) @ np.array([point[0], point[1], 1])
               pos = QtCore.QPointF(pos_raw[0], pos_raw[1])
           else:
               pos = QtCore.QPointF(point[0], point[1])
           self._pois_raw.append(pos)
        else:
            if self.ops._transformed:
                pos_raw = np.linalg.inv(self.ops.tf_matrix) @ np.array([point[0], point[1], 1])
                pos = QtCore.QPointF(pos_raw[0] - self.size / 2, pos_raw[1] - self.size / 2)
            else:
                pos = QtCore.QPointF(point[0] - self.size / 2, point[1] - self.size / 2)
            self.points_raw.append(pos)

    def _update_pois_and_points(self):
        points_base = []
        for i in range(len(self.points_base)):
            pos = self.points_base[i]
            points_base.append(np.array([pos.x() + self.size/2, pos.y() + self.size/2]))
        if len(points_base) > 0:
            points_updated = self.ops.update_points(np.array(points_base))
            for i in range(len(points_updated)):
                pos = QtCore.QPointF(points_updated[i][0] - self.size / 2, points_updated[i][1] - self.size / 2)
                self._draw_fm_points(pos, self.imview.getImageItem())
        pois_base = []
        for i in range(len(self.pois_base)):
            pos = self.pois_base[i]
            pois_base.append(np.array([pos.x(), pos.y()]))
        if len(pois_base) > 0:
            pois_updated = self.ops.update_points(np.array(pois_base))
            for i in range(len(pois_updated)):
                init = np.array([pois_updated[i][0], pois_updated[i][1]])
                self._draw_fm_pois(init, self.imview.getImageItem())

        self.fixed_orientation = False

    def _update_annotations(self, point):
        idx = None
        for i in range(len(self.other._points_corr)):
            self.log(self.other._points_corr[i])
            if self.other._points_corr[i] == point:
                idx = i
                break

        anno = self.other.anno_list[idx]
        self.other.imview.removeItem(anno)
        anno.setPos(point.x() + 5, point.y() + 5)
        self.other.imview.addItem(anno)

    @utils.wait_cursor('print')
    def fit_circles(self, state=None):
        if self.ops is None:
            return

        bead_size = float(self.size_box.text())
        moved_peaks = False
        original_positions = []
        if len(self.other._points_corr) == 0:
            if self.peaks is None or len(self.peaks) == 0:
                return
            ref_ind = [i for i in range(len(self.peaks)) if self.peaks[i].has_moved]
            points_em = []
            for ind in ref_ind:
                self.imview.removeItem(self.peaks[ind])
                pos = self.peaks[ind].pos()
                original_positions.append(self.peaks[ind].original_pos)
                points_em.append(pos + np.array([self.size /2, self.size /2]))
            points_em = np.array(points_em)
            moved_peaks = True
        else:
            points_em = np.array([[p.x() + self.size / 2, p.y() + self.size / 2] for p in self._points_corr])
            [self.imview.removeItem(point) for point in self._points_corr]

        points_em_fitted = self.ops.fit_circles(points_em, bead_size)
        self._points_corr = []
        circle_size_em = bead_size * 1e3 / self.ops.pixel_size[0]
        self.size = circle_size_em
        for i in range(len(points_em_fitted)):
            pos = QtCore.QPointF(points_em_fitted[i, 0] - circle_size_em / 2,
                                 points_em_fitted[i, 1] - circle_size_em / 2)

            point = PeakROI(pos, circle_size_em, parent=self.imview.getImageItem(), movable=True)
            point.peakMoved(item=None)
            if moved_peaks:
                point.original_pos = copy.copy(original_positions[i])
                self.peaks[ref_ind[i]] = point
            else:
                self._points_corr.append(point)
            self.imview.addItem(point)
        self.size = circle_size_em

    def _remove_pois(self, point, remove_base=True):
        idx = None
        for i in range(len(self.pois)):
            if self.pois[i] == point:
                idx = i
                break
        self.imview.removeItem(self.pois[idx])
        self.imview.removeItem(self.poi_anno_list[idx])
        self.pois.remove(self.pois[idx])
        self.poi_anno_list.remove(self.poi_anno_list[idx])
        if remove_base:
            self.pois_z.remove(self.pois_z[idx])
            self.pois_sizes.remove(self.pois_sizes[idx])
            self.pois_err.remove(self.pois_err[idx])
            self.pois_cov.remove(self.pois_cov[idx])
            self._pois_raw.remove(self._pois_raw[idx])
            if len(self.pois_base) > 0:
                self.pois_base.remove(self.pois_base[idx])

        for i in range(idx, len(self.pois)):
            self.poi_anno_list[i].setText(str(i + 1))

        self.poi_counter -= 1

    def _remove_correlated_points(self, point, remove_base=True, remove_raw=False):
        idx = self._check_point_idx(point)
        num_beads = len(self._points_corr)
        if remove_raw:
            self.points_raw.remove(self.points_raw[idx])

        #Remove FM beads information
        for i in range(len(self.peaks)):
            pos = [self.peaks[i].original_pos[0] + self.orig_size/2, self.peaks[i].original_pos[1] + self.orig_size/2]
            if np.allclose(pos, self._orig_points_corr[idx]):
                self.peaks[i].resetPos()
                self.imview.addItem(self.peaks[i])
        for i in range(len(self.other.peaks)):
            pos = [self.other.peaks[i].original_pos[0] + self.other.orig_size / 2, self.other.peaks[i].original_pos[1] + self.other.orig_size / 2]
            if np.allclose(pos, self.other._orig_points_corr[idx]):
                self.other.peaks[i].resetPos()
                self.other.imview.addItem(self.other.peaks[i])

        self.imview.removeItem(self._points_corr[idx])
        self._points_corr.remove(self._points_corr[idx])
        if remove_base and len(self.points_base) > 0:
            self.points_base.remove(self.points_base[idx])
        self._orig_points_corr.remove(self._orig_points_corr[idx])
        if len(self.anno_list) > 0:
            self.imview.removeItem(self.anno_list[idx])
            self.anno_list.remove(self.anno_list[idx])
        if len(self._points_corr_indices) > 0:
            self._points_corr_indices.remove(self._points_corr_indices[idx])
        if self.other.tab_index == 1 and len(self.points_corr_z) > 0:
            self.points_corr_z.remove(self.points_corr_z[idx])
        for i in range(idx, len(self._points_corr)):
            self.anno_list[i].setText(str(i+1))
            self._points_corr_indices[i] -= 1

        # Remove EM beads information
        if len(self.other._points_corr) == num_beads:
            self.other.imview.removeItem(self.other._points_corr[idx])
            self.other._points_corr.remove(self.other._points_corr[idx])
            self.other._orig_points_corr.remove(self.other._orig_points_corr[idx])
            if len(self.other.anno_list) > 0:
                self.other.imview.removeItem(self.other.anno_list[idx])
                self.other.anno_list.remove(self.other.anno_list[idx])
            if len(self.other._points_corr_indices) > 0:
                self.other._points_corr_indices.remove(self.other._points_corr_indices[idx])
            if self.other.tab_index == 1:
                if len(self.other.points_corr_z) > 0:
                    self.other.points_corr_z.remove(self.other.points_corr_z[idx])
            for i in range(idx, len(self._points_corr)):
                self.other.anno_list[i].setText(str(i+1))
                self.other._points_corr_indices[i] -= 1
            self.log(self.other._points_corr_indices)

        self.counter -= 1
        self.other.counter -= 1

    def _clear_pois(self):
        self.counter = 0
        self.other.counter = 0
        # Remove circle from imviews
        [self.imview.removeItem(point) for point in self._points_corr]
        [self.fibcontrols.imview.removeItem(point) for point in self.fibcontrols._points_corr]
        [self.semcontrols.imview.removeItem(point) for point in self.semcontrols._points_corr]
        [self.temcontrols.imview.removeItem(point) for point in self.temcontrols._points_corr]

        # Remove ROI
        self._points_corr = []
        self.fibcontrols._points_corr = []
        self.semcontrols._points_corr = []
        self.temcontrols._points_corr = []
        # Remove original position
        self._orig_points_corr = []
        self.fibcontrols._orig_points_corr = []
        self.semcontrols._orig_points_corr = []
        self.temcontrols._orig_points_corr = []

        #Remove base point
        self.points_base = []
        self.fibcontrols.points_base = []
        self.semcontrols.points_base = []
        self.temcontrols.points_base = []

        #Remove raw_points
        self.points_raw = []

        # Remove annotation
        if len(self.anno_list) > 0:
            [self.imview.removeItem(anno) for anno in self.anno_list]
            [self.fibcontrols.imview.removeItem(anno) for anno in self.fibcontrols.anno_list]
            [self.semcontrols.imview.removeItem(anno) for anno in self.semcontrols.anno_list]
            [self.temcontrols.imview.removeItem(anno) for anno in self.temcontrols.anno_list]
            self.anno_list = []
            self.fibcontrols.anno_list = []
            self.semcontrols.anno_list = []
            self.temcontrols.anno_list = []

        # Remove correlation index
        self._points_corr_indices = []
        self.fibcontrols._points_corr_indices = []
        self.semcontrols._points_corr_indices = []
        self.temcontrols._points_corr_indices = []

        # Remove FIB z-position
        self.points_corr_z = []
        self.fibcontrols.points_corr_z = []
        self.semcontrols.points_corr_z = []
        self.temcontrols.points_corr_z = []

    def _remove_points_flip(self):
        for i in range(len(self._points_corr)):
            self._remove_correlated_points(self._points_corr[0], remove_base=False)
        for i in range(len(self.pois)):
            self._remove_pois(self.pois[0], remove_base=False)
        self.fixed_orientation = False

    def _store_fib_flips(self, idx):
        if idx in self._fib_flips:
            del self._fib_flips[self._fib_flips == idx]
        else:
            self._fib_flips.append(idx)

    def _define_grid_toggled(self, checked):
        if self.ops is None:
            self.print('Select data first!')
            return

        if checked:
            self.print('Defining grid on %s image: Click on corners' % self.tag)
            self.show_grid_btn.setChecked(False)
            return

        self.print('Done defining grid on %s image: Manually adjust fine positions' % self.tag)
        if len(self.clicked_points) != 4:
            self.print('You have to select exactly 4 points. Try again!')
            [self.imview.removeItem(roi) for roi in self.clicked_points]
            self._show_grid()
            self.clicked_points = []
            return

        # Initialize grid_box on original or transformed image
        if self.show_btn.isChecked():
            self.grid_box = None
        else:
            self.tr_grid_box = None

        # Calculate circle centers
        positions = [c.pos() for c in self.clicked_points]
        sizes = [c.size()[0] for c in self.clicked_points]
        for pos, s in zip(positions, sizes):
            pos.setX(pos.x() + s / 2)
            pos.setY(pos.y() + s / 2)
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
        #self.rot_transform_btn.setEnabled(True)
        self.show_grid_btn.setEnabled(True)
        self.show_grid_btn.setChecked(True)

        if self.show_btn.isChecked():
            self.show_grid_box = True
        else:
            self.show_tr_grid_box = True

    def _recalc_grid(self, toggle_orig=False):
        if self.ops.points is None:
            return

        if self.show_btn.isChecked():
            pos = list(self.ops.points)
            poly_line = pg.PolyLineROI(pos, closed=True, movable=False)
            if self.show_grid_btn.isChecked():
                if not toggle_orig:
                    if self.tr_grid_box is not None:
                        self.imview.removeItem(self.tr_grid_box)
                    if self.grid_box is not None:
                        self.imview.removeItem(self.grid_box)
                    self.show_grid_box = False
            self.print('Recalculating original grid...')
            self.grid_box = poly_line
        else:
            pos = list(self.ops.points)
            poly_line = pg.PolyLineROI(pos, closed=True, movable=False)
            if self.show_grid_btn.isChecked():
                if not toggle_orig:
                    if self.grid_box is not None:
                        self.imview.removeItem(self.grid_box)
                    if self.tr_grid_box is not None:
                        self.imview.removeItem(self.tr_grid_box)
                    self.show_tr_grid_box = False
                if self.redo_tr:
                    if self.tr_grid_box is not None:
                        self.imview.removeItem(self.tr_grid_box)
                    self.redo_tr = False
            self.print('Recalculating transformed grid...')
            self.tr_grid_box = poly_line
        self._show_grid()

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

    def _show_grid(self):
        if self.show_grid_btn.isChecked():
            if self.show_btn.isChecked():
                self.show_grid_box = True
                self.imview.addItem(self.grid_box)
                if self.show_tr_grid_box:
                    if self.tr_grid_box is not None:
                        self.imview.removeItem(self.tr_grid_box)
                    self.show_tr_grid_box = False
            else:
                self.show_tr_grid_box = True
                self.imview.addItem(self.tr_grid_box)
                if self.show_grid_box:
                    if self.grid_box is not None:
                        self.imview.removeItem(self.grid_box)
                    self.show_grid_box = False
        else:
            if self.show_btn.isChecked():
                if self.show_grid_box:
                    if self.grid_box is not None:
                        self.imview.removeItem(self.grid_box)
                    self.show_grid_box = False
            else:
                if self.show_tr_grid_box:
                    if self.tr_grid_box is not None:
                        self.imview.removeItem(self.tr_grid_box)
                    self.show_tr_grid_box = False

    @utils.wait_cursor('print')
    def _affine_transform(self, toggle_orig=True):
        if not np.array_equal(np.identity(3), self.ops.tf_matrix):
            self.ops.tf_matrix = np.identity(3)
            self.ops.tf_peaks = None
            self.ops.orig_tf_peaks = None

        if self.show_btn.isChecked():
            grid_box = self.grid_box
        else:
            self.redo_tr = True
            grid_box = self.tr_grid_box

        if grid_box is None:
            self.print('Define grid box on %s image first!' % self.tag)
            return

        if hasattr(self, 'fliph'):
            self.fliph.setChecked(False)
            self.fliph.setEnabled(True)
            self.flipv.setChecked(False)
            self.flipv.setEnabled(True)
            self.transpose.setChecked(False)
            self.transpose.setEnabled(True)
            self.rotate.setChecked(False)
            self.rotate.setEnabled(True)
            self.other.size_box.setEnabled(True)
            self.other.auto_opt_btn.setEnabled(True)

        points_obj = grid_box.getState()['points']
        points = np.array([list((point[0], point[1])) for point in points_obj])
        #if self.rot_transform_btn.isChecked():
        #    self.print('Performing rotation on %s image' % self.tag)
        #    self.ops.calc_rot_transform(points)
        #else:
        self.print('Performing affine transformation on %s image' % self.tag)
        self.ops.calc_affine_transform(points)

        self.show_btn.blockSignals(True)
        self.show_btn.setEnabled(True)
        self.show_btn.setChecked(False)
        self.show_btn.blockSignals(False)
        self._recalc_grid(toggle_orig=toggle_orig)
        self._update_imview()
        self.transform_btn.setEnabled(False)
        #self.rot_transform_btn.setEnabled(False)
        self.define_btn.setEnabled(False)

        for i in range(len(self._points_corr)):
            self._remove_correlated_points(self._points_corr[0], remove_base=False)
        for i in range(len(self.pois)):
            self._remove_pois(self.pois[0], remove_base=False)
        if hasattr(self, 'select_btn'):
            self.point_ref_btn.setEnabled(True)
            self.merge_btn.setEnabled(True)
            self.refine_btn.setEnabled(True)
            self._transform_pois(poi=True)
            self._transform_pois(poi=False)

        if self.ops is not None and self.other.ops is not None:
            if self.ops._transformed and self.other.ops._transformed:
                if hasattr(self, 'tab_index'):
                    if self.tab_index != 1:
                        self.show_peaks_btn.setEnabled(True)

                elif not hasattr(self, 'tab_index') and self.other.tab_index != 1:
                    self.other.show_peaks_btn.setEnabled(True)

    def _show_original(self):
        if self.ops is None:
            return
        [self.imview.removeItem(poi) for poi in self.pois]
        [self.imview.removeItem(anno) for anno in self.poi_anno_list]
        self.pois = []
        self.poi_anno_list = []
        self.poi_counter = 0

        [self.imview.removeItem(point) for point in self._points_corr]
        [self.imview.removeItem(anno) for anno in self.anno_list]
        self._points_corr = []
        self.anno_list = []
        self.counter = 0


        self.ops._transformed = not self.ops._transformed
        align = False
        if hasattr(self.ops, 'flipv') and not self.ops._transformed:
            if self.peak_controls is not None and self.peak_controls.align_btn.isChecked() and self.ops.color_matrix is None:
                self.peak_controls.align_btn.setChecked(False)
                align = True
            self.flipv.setEnabled(False)
            self.fliph.setEnabled(False)
            self.transpose.setEnabled(False)
            self.rotate.setEnabled(False)
            self.point_ref_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.merge_btn.setEnabled(False)
            self.refine_btn.setEnabled(False)
        elif hasattr(self.ops, 'flipv') and self.ops._transformed:
            self.point_ref_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.merge_btn.setEnabled(True)
            self.refine_btn.setEnabled(True)
            if not self.fixed_orientation:
                self.flipv.setEnabled(True)
                self.fliph.setEnabled(True)
                self.transpose.setEnabled(True)
                self.rotate.setEnabled(True)

        self.log('Transformed?', self.ops._transformed)
        self.ops.toggle_original()
        self._recalc_grid(toggle_orig=True)

        show_peaks = False
        if hasattr(self, 'peak_btn'):
            if self.peak_btn.isChecked():
                show_peaks = True
                self.peak_btn.setChecked(False)
            if self.other.tab_index != 1:
                if self.other.ops is not None:
                    if self.ops._transformed and self.other.ops._transformed:
                        self.other.show_peaks_btn.setEnabled(True)

        if hasattr(self, 'tab_index') and not self.tab_index != 1:
            if self.other.ops is not None:
                if self.show_peaks_btn.isChecked():
                    self.show_peaks_btn.setChecked(False)
                    self.show_peaks_btn.setChecked(True)

        self._update_imview()
        if self.ops._transformed:
            self.transform_btn.setEnabled(False)
            #self.rot_transform_btn.setEnabled(False)
            self.define_btn.setEnabled(False)
        else:
            self.transform_btn.setEnabled(True)
            #self.rot_transform_btn.setEnabled(True)
            self.define_btn.setEnabled(True)

        if show_peaks:
            self.peak_btn.setChecked(True)
        if align:
            self.peak_controls.align_btn.setChecked(True)

        if hasattr(self, 'select_btn'):
            if self.ops._transformed:
                self._update_pois_and_points()
            else:
                for i in range(len(self.points_raw)):
                    self._draw_fm_points(self.points_raw[i], self.imview.getImageItem())
                for i in range(len(self._pois_raw)):
                    init = np.array([self._pois_raw[i].x(), self._pois_raw[i].y()])
                    self._draw_fm_pois(init, self.imview.getImageItem())

    @utils.wait_cursor('print')
    def _refine(self, state=None):
        if self.select_btn.isChecked():
            self.print('Confirm point selection! (Uncheck Select points of interest)')
            return

        if self.other.translate_peaks_btn.isChecked() or self.other.refine_peaks_btn.isChecked():
            self.print('Confirm peak translation (uncheck collective/individual translation)')
            return

        if len(self._points_corr) < 4:
            self.print('Select at least 4 points for refinement!')
            return

        self.print('Processing shown FM peaks: %d peaks refined' % len(self._points_corr))
        self.print('Refining...')
        dst = np.array([[point.x() + self.other.size / 2, point.y() + self.other.size / 2] for point in
                        self.other._points_corr])
        src = np.array([[point[0], point[1]] for point in self.other._orig_points_corr])

        self._points_corr_history.append(copy.copy(self._points_corr))
        self._points_corr_z_history.append(copy.copy(self.points_corr_z))
        self._orig_points_corr_history.append(copy.copy(self._orig_points_corr))
        self.other._points_corr_history.append(copy.copy(self.other._points_corr))
        self.other._points_corr_z_history.append(copy.copy(self.other.points_corr_z))
        self.other._orig_points_corr_history.append(copy.copy(self.other._orig_points_corr))
        self._fib_vs_sem_history.append(self.other.tab_index)
        self.other._size_history.append(self.other.size)

        self.other.ops.merged[self.other.tab_index] = None

        refine_matrix_old = copy.copy(self.other.ops._refine_matrix)
        self.other.ops.calc_refine_matrix(src, dst)
        self.other.ops.apply_refinement()
        self.other._refined = True
        if self.other.show_grid_btn.isChecked():
            self.other._show_grid()
        self._estimate_precision(self.other.tab_index, refine_matrix_old)
        self.other.size = self.other.orig_size

        self.fixed_orientation = True
        self.fliph.setEnabled(False)
        self.flipv.setEnabled(False)
        self.transpose.setEnabled(False)
        self.rotate.setEnabled(False)
        self.err_plt_btn.setEnabled(True)
        self.convergence_btn.setEnabled(True)
        self.undo_refine_btn.setEnabled(True)

        for i in range(len(self._points_corr)):
            self._remove_correlated_points(self._points_corr[0])

        self.other.size = copy.copy(self.size)
        self._update_imview()
        if self.other.show_peaks_btn.isChecked():
            self.other.show_peaks_btn.setChecked(False)
            self.other.show_peaks_btn.setChecked(True)

        self._points_corr = []
        self.points_corr_z = []
        self._orig_points_corr = []
        self.other._points_corr = []
        self.other.points_corr_z = []
        self.other._orig_points_corr = []

    def _undo_refinement(self):
        self.other.ops.undo_refinement()
        for i in range(len(self._points_corr)):
            self._remove_correlated_points(self._points_corr[0])

        if len(self.other.ops._refine_history) == 1:
            self.fliph.setEnabled(True)
            self.flipv.setEnabled(True)
            self.rotate.setEnabled(True)
            self.transpose.setEnabled(True)
            self.other._refined = False
            self.other._recalc_grid()
            self.undo_refine_btn.setEnabled(False)
            self.err_btn.setText('0')
            self.fixed_orientation = False
            self.err_plt_btn.setEnabled(False)
            self.convergence_btn.setEnabled(False)
            self.other.grid_box.movable = True
        else:
            self.other._recalc_grid()
            self.other._points_corr_indices = np.arange(len(self.other._points_corr)).tolist()

        self._update_tr_matrices()
        self.other.size = copy.copy(self.other._size_history[-1])
        for i in range(len(self._points_corr_history[-1])):
            point = self._points_corr_history[-1][i]
            self._draw_correlated_points(point.pos(), self.imview.getImageItem())
            point_other = self.other._points_corr_history[-1][i]
            self.other._points_corr[i].setPos(point_other.pos())
            self.other.anno_list[i].setPos(point_other.pos().x()+5, point_other.pos().y()+5)

        if self.other.show_peaks_btn.isChecked():
            self.other.show_peaks_btn.setChecked(False)
            self.other.show_peaks_btn.setChecked(True)

        self.other.ops.merged[self.other.tab_index] = None

        id = len(self._fib_vs_sem_history) - self._fib_vs_sem_history[::-1].index(self.other.tab_index) - 1

        del self._fib_vs_sem_history[id]
        del self._points_corr_history[-1]
        del self._points_corr_z_history[-1]
        del self._orig_points_corr_history[-1]

        del self.other._points_corr_history[-1]
        del self.other._points_corr_z_history[-1]
        del self.other._orig_points_corr_history[-1]
        del self.other._size_history[-1]

        if len(self.other.ops._refine_history) > 1:
            idx = self.other.tab_index
            self._estimate_precision(idx, self.other.ops._refine_matrix)
            self.other.size = copy.copy(self.size)



    @utils.wait_cursor('print')
    def _estimate_precision(self, idx, refine_matrix_old):
        sel_points = [[point.x() + self.other.size / 2, point.y() + self.other.size / 2] for point in
                      self.other._points_corr_history[-1]]
        orig_fm_points = np.copy(self._points_corr_history[-1])

        self.refined_points = []
        corr_points = []
        self._update_tr_matrices()
        if idx != 1:
            for i in range(len(orig_fm_points)):
                orig_point = np.array([orig_fm_points[i].x(), orig_fm_points[i].y()])
                init = np.array([orig_point[0] + self.size // 2, orig_point[1] + self.size // 2, 1])
                corr_points.append(np.copy((self.other.tr_matrices @ init)[:2]))
                transf = self.other.tr_matrices @ init
                self.refined_points.append(transf[:2])
        else:
            #orig_fm_points_z = np.copy(self.points_corr_z)
            orig_fm_points_z = np.copy(self._points_corr_z_history[-1])
            for i in range(len(orig_fm_points)):
                orig_point = np.array([orig_fm_points[i].x(), orig_fm_points[i].y()])
                z = orig_fm_points_z[i]
                init = np.array([orig_point[0] + self.size // 2, orig_point[1] + self.size // 2, 1])
                transf = np.dot(self.other.tr_matrices, init)
                transf = self.other.ops.fib_matrix @ np.array([transf[0], transf[1], z, 1])
                corr_points.append(np.copy(transf[:2]))
                transf[:2] = (self.other.ops._refine_matrix @ np.array([transf[0], transf[1], 1]))[:2]
                self.refined_points.append(transf[:2])

        self.diff = np.array(sel_points) - np.array(self.refined_points)
        self.log(self.diff.shape)
        self.diff *= self.other.ops.pixel_size[0]
        self.other.cov_matrix, self.other._std[idx][0], self.other._std[idx][1], self.other._dist = self.other.ops.calc_error(self.diff)

        self.other._err[idx] = self.diff
        self.err_btn.setText(
            'x: \u00B1{:.2f}, y: \u00B1{:.2f}'.format(self.other._std[idx][0], self.other._std[idx][1]))

        if len(corr_points) >= self.min_conv_points:
            min_points = self.min_conv_points - 4
            convergence = self.other.ops.calc_convergence(corr_points, sel_points, min_points, refine_matrix_old)
            self.other._conv[idx] = convergence
        else:
            self.other._conv[idx] = []

    def correct_grid_z(self):
        self.ops.fib_matrix = None
        # set FIB matrix to None to recalculate with medium z slice
        self._recalc_grid(scaling=self.other.ops.voxel_size[2] / self.other.ops.voxel_size[0])
        self._show_grid()
        self.print('WARNING! Recalculate FIB grid square for z = ', self.num_slices // 2)

    def merge(self):
        if not self._refined:
            self.print('You have to do at least one round of refinement before merging is allowed!')
            return False
        if hasattr(self, 'show_btn'):
            if (self.show_btn.isChecked() or self.other.show_btn.isChecked()):
                self.print('Merge only allowed with transformed data. Uncheck show original data buttons on both sides!')
                return
        else:
            if self.other.show_btn.isChecked():
                self.print('Merge only allowed with transformed data. Uncheck show original data buttons on both sides!')
                return

        size_em = copy.copy(self._size_history[-1])
        dst = np.array([[point.x() + size_em / 2, point.y() + size_em / 2] for point in
                        self._points_corr_history[-1]])
        src = np.array([[point.x() + self.other.size / 2, point.y() + self.other.size / 2]
                        for point in self.other._points_corr_history[-1]])

        if self.tab_index != 1:
            if self.ops.merged[self.tab_index] is None:
                for i in range(self.other.ops.num_channels):
                    if self.show_assembled_btn.isChecked():
                        show_region = False
                        #self.ops.apply_merge_2d(self.other.ops.orig_data, self.other.ops.tf_matrix_orig,
                        #                        self.other.ops.data.shape, self.other.ops.points, i)
                    else:
                        show_region = True
                    self.ops.apply_merge_2d(self.other.ops.data[:,:,i], self.other.ops.points, i,
                                            show_region, self.other.ops.num_channels, self.tab_index)
                    self.other.progress_bar.setValue((i + 1) / self.other.ops.num_channels * 100)
                self.progress = 100
            else:
                self.other.progress_bar.setValue(100)
                self.progress = 100
            if self.ops.merged[self.tab_index] is not None:
                self.print('Merged shape: ', self.ops.merged[self.tab_index].shape)
            self.show_merge = True
            return True
        else:
            if self.ops.merged[self.tab_index] is None:
                if self._refined:
                    src_z = copy.copy(self.other._points_corr_z_history[-1])
                    flip_list = [self.other.ops.transp, self.other.ops.rot, self.other.ops.fliph, self.other.ops.flipv]
                    for i in range(self.other.ops.num_channels):
                        self.other.ops.load_channel(i)
                        #self.ops.apply_merge_3d(self.other.tr_matrices, self.other.ops.fib_matrix,
                        #                        self.other.ops._refine_matrix,
                        #                        self.other.ops.data, src, src_z, dst, i)
                        orig_coor = []
                        tf_aligned_orig_shift = self.other.ops.tf_matrix @ self.other.ops._color_matrices[0]
                        for k in range(len(src)):
                            orig_pt = self.other.ops.calc_original_coordinates(src[k], tf_aligned_orig_shift,
                                                                               flip_list, self.other.ops.data.shape[:2])
                            orig_coor.append(orig_pt)
                        self.ops.apply_merge_3d(self.other.ops.channel, self.tr_matrices, self.other.ops.tf_matrix,
                                                self.other.ops.tf_corners, self.other.ops._color_matrices, flip_list,
                                                src, orig_coor, src_z, dst, i, self.other.ops.voxel_size,
                                                self.other.num_slices, self.other.ops.num_channels,
                                                self.other.ops.norm_factor, self.tab_index)
                        self.other.progress_bar.setValue((i + 1) / self.other.ops.num_channels * 100)
                    self.progress = 100
                else:
                    self.print('You have to perform at least one round of refinement before you can merge the images!')
            else:
                self.other.progress_bar.setValue(100)
                self.progress = 100
            if self.ops.merged[self.tab_index] is not None:
                self.print('Merged shape: ', self.ops.merged[self.tab_index].shape)
            self.show_merge = True
            return True

    def _couple_views(self):
        if (self.ops is None or self.other.ops is None or
                not self.ops._transformed or not self.other.ops._transformed):
            return

        vrange = self.imview.getImageItem().getViewBox().targetRect()
        p1 = np.array([vrange.bottomLeft().x(), vrange.bottomLeft().y()])
        p2 = np.array([vrange.bottomRight().x(), vrange.bottomRight().y()])
        p3 = np.array([vrange.topRight().x(), vrange.topRight().y()])
        p4 = np.array([vrange.topLeft().x(), vrange.topLeft().y()])
        points = [p1, p2, p3, p4]
        if hasattr(self, 'select_btn') and self.other.tab_index != 1:
            src_sorted = np.array(
                sorted(self.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
            dst_sorted = np.array(
                sorted(self.other.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
            tr_matrices = self.ops.get_transform(src_sorted, dst_sorted)
            tf_points = np.array([(tr_matrices @ np.array([point[0], point[1], 1]))[:2] for point in points])
        elif hasattr(self.other, 'select_btn') and self.tab_index != 1:
            src_sorted = np.array(
                sorted(self.other.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
            dst_sorted = np.array(
                sorted(self.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
            tr_matrices = self.other.ops.get_transform(src_sorted, dst_sorted)
            tf_points = np.array(
                [(np.linalg.inv(tr_matrices) @ np.array([point[0], point[1], 1]))[:2] for point in points])
        else:
            return
        xmin = tf_points.min(0)[0]
        xmax = tf_points.max(0)[0]
        ymin = tf_points.min(0)[1]
        ymax = tf_points.max(0)[1]
        self.other.imview.getImageItem().getViewBox().blockSignals(True)
        self.other.imview.getImageItem().getViewBox().setRange(xRange=(xmin, xmax), yRange=(ymin, ymax))
        self.other.imview.getImageItem().getViewBox().blockSignals(False)

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
        self.points_corr_z = []
        self.other.points_corr_z = []
        self._orig_points_corr = []
        self.other._orig_points_corr = []
        self._points_corr_indices = []
        self.other._points_corr_indices = []
        self.peaks = []
