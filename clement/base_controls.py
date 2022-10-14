import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import copy
import matplotlib

from . import utils

class PeakROI(pg.CircleROI):
    def __init__(self, pos, size, parent, movable=False, removable=False, resizable=False, color=None):
        super(PeakROI, self).__init__(pos, size, parent=parent,
                                      movable=movable, removable=removable, resizable=resizable)
        self.original_color = (255, 0, 0)
        self.moved_color = (0, 255, 255)
        self.original_pos = copy.copy(np.array([pos.x(), pos.y()]))
        self._old_peak_pos = copy.copy(self.original_pos)
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

        self._points_corr = []
        self._orig_points_corr = []
        self._points_corr_indices = []

        self._refined = False
        self.diff = None
        self._err = None
        self._std = [None, None]
        self._conv = None
        self._dist = None

        self._points_corr_history = []
        self._points_corr_z_history = []
        self._orig_points_corr_history = []
        self._fib_vs_sem_history = []
        self._size_history = []


        self.points_corr_z = []

        self.flips = [False, False, False, False]
        self.tr_matrices = None
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
        self.peaks = []
        self.peak_colors = []
        self.num_slices = None
        self.min_conv_points = 10
        self.show_merge = False
        self.progress = 0
        self.cov_matrix = None

        self.tf_points_indices = []

    def _init_ui(self):
        self.log('This message should not be seen. Please override _init_ui')

    def select_tab(self, idx, show_grid=False):
        self.tab_index = idx
        if idx == 0:
            self.semcontrols._update_imview()
            self.other = self.semcontrols
            if self.semcontrols._refined:
                self.undo_refine_btn.setEnabled(True)
            else:
                self.undo_refine_btn.setEnabled(False)
        elif idx == 1:
            if self.tr_matrices is None:
                if self.semcontrols.ops is not None and self.ops is not None:
                    if self.ops.points is not None and self.semcontrols.ops.points is not None:
                        self._calc_tr_matrices()
            self.other = self.fibcontrols
            if self.fibcontrols._refined:
                self.undo_refine_btn.setEnabled(True)
            else:
                self.undo_refine_btn.setEnabled(False)
            self.fibcontrols._update_imview()
            self.fibcontrols.sem_ops = self.semcontrols.ops
            if self.semcontrols.ops is not None:
                if self.semcontrols.ops._orig_points is not None:
                    self.fibcontrols.enable_buttons(enable=True)
                else:
                    self.fibcontrols.enable_buttons(enable=False)
                if self.fibcontrols.ops is not None and self.semcontrols.ops._tf_points is not None:
                    self.fibcontrols.ops._transformed = True
        elif idx == 2:
            self.giscontrols.sem_ops = self.semcontrols.ops
            self.giscontrols.fib_ops = self.fibcontrols.ops
            self.giscontrols._init_fib_params(self.fibcontrols)

            self.other = self.giscontrols
            self.giscontrols._update_imview()

        else:
            self.temcontrols._update_imview()
            self.other = self.temcontrols
            if self.temcontrols._refined:
                self.undo_refine_btn.setEnabled(True)
            else:
                self.undo_refine_btn.setEnabled(False)

        if show_grid:
            self.show_grid_btn.setChecked(True)

        if self.other.show_merge:
            self.progress_bar.setValue(100)
        else:
            self.progress_bar.setValue(0)
        if self.other._refined:
            self.err_btn.setText('x: \u00B1{:.2f}, y: \u00B1{:.2f}'.format(self.other._std[0],
                                                                           self.other._std[1]))
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

            self._calc_tr_matrices()

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

        if hasattr(self, 'define_btn') and self.define_btn.isChecked(): #Any
            pos.setX(pos.x() - self.size // 2)
            pos.setY(pos.y() - self.size // 2)
            roi = pg.CircleROI(pos, self.size, parent=item, movable=False)
            roi.setPen(255, 0, 0)
            roi.removeHandle(0)
            self.imview.addItem(roi)
            self.clicked_points.append(roi)
        elif hasattr(self, 'poi_btn') and self.poi_btn.isChecked(): #FM
            self._calc_pois(pos)
        elif hasattr(self, 'select_btn') and self.select_btn.isChecked(): #FM
            pos.setX(pos.x() - self.size // 2)
            pos.setY(pos.y() - self.size // 2)
            #print('pos: ', pos)
            self._draw_correlated_points(pos, item)
        elif hasattr(self, 'select_region_btn') and self.select_region_btn.isChecked(): #SEM
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
    def _show_FM_peaks(self, state=None):
        if not self.show_peaks_btn.isChecked():
            # Remove already shown peaks
            [self.imview.removeItem(point) for point in self.peaks]
            self.peaks = []
            if hasattr(self, 'translate_peaks_btn'):
                self.translate_peaks_btn.setChecked(False)
                self.translate_peaks_btn.setEnabled(False)
                self.refine_peaks_btn.setChecked(False)
                self.refine_peaks_btn.setEnabled(False)
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

        if self.other.tab_index == 1 and self.tr_matrices is None:
            if not self.ops._transformed:
                self.print('You have to transform the SEM image first!')
                self.show_peaks_btn.setChecked(False)
                return

        if not self.other.fixed_orientation and not (self.other.semcontrols.ops._transformed and self.other.ops._transformed):
            self.print('You have to transform both FM and SEM images or confirm correct orientation first!')
            self.show_peaks_btn.setChecked(False)
            return

        #self.other._calc_tr_matrices(self.other.ops.points, self.other.semcontrols.ops.points)
        #self.other._calc_tr_matrices()

        if len(self.peaks) != 0:
            self.peaks = []
            self.peak_colors = []

        if self.other.diff is not None:
            cmap = matplotlib.cm.get_cmap('cool')
            diff_normed = self.other.diff / self.other.diff.max()
            diff_abs = np.sqrt(diff_normed[:,0]**2 + diff_normed[:,1]**2)
            colors = cmap(diff_abs)

        transf = np.zeros_like(self.other.ops.peaks)
        for i in range(len(transf)):
            init = np.array([self.other.ops.tf_peaks[i, 0], self.other.ops.tf_peaks[i, 1], 1])
            transf[i] = (self.other.tr_matrices @ init)[:2]

        if self.other.tab_index == 1:
            for i in range(len(transf)):
                z = self.other.ops.peaks_z[i]
                #if self.other.semcontrols.ops._transformed:
                if not self.other.fixed_orientation:
                    transf[i] = (np.linalg.inv(self.other.semcontrols.ops.tf_matrix) @ np.array([transf[i,0], transf[i,1], 1]))[:2]
                transf_i = self.ops.fib_matrix @ np.array([transf[i,0], transf[i,1], z, 1])
                if self._refined:
                    transf_i = self.ops._refine_matrix @ np.array([transf_i[0], transf_i[1], 1])
                pos = QtCore.QPointF(transf_i[0] - self.orig_size / 2, transf_i[1] - self.orig_size / 2)
                color = None
                if self.other.diff is not None:
                    idx = np.where(np.isclose(np.array(self.other.refined_points), np.array([transf_i[0], transf_i[1]])))
                    if not np.array_equal(idx[0], np.array([])):
                        if len(idx) == 2:
                            idx = idx[0][0]
                        color = colors[idx]
                        color = matplotlib.colors.to_hex(color)
                point = PeakROI(pos, self.size, self.imview.getImageItem(), color=color)
                self.peaks.append(point)
                self.peak_colors.append(color)
                for i in range(len(self._orig_points_corr)):
                    if np.allclose(transf[:2], self._orig_points_corr[i]):
                        self.imview.removeItem(point)
        elif self.other.tab_index == 2:
            if self.other.fibcontrols.peaks is not None:
                peaks = self.other.fibcontrols.peaks
                if self.ops.gis_corrected is not None:
                    for i in range(len(peaks)):
                        size = peaks[i].size()[0]
                        point = np.array([peaks[i].pos().x() + size/2, peaks[i].pos().y()+size/2])
                        point = self.ops._update_gis_points(point)
                        pos = QtCore.QPointF(point[0]-size/2, point[1]-size/2)
                        self.peaks.append(PeakROI(pos, size, self.imview.getImageItem(), color=self.other.fibcontrols.peak_colors[i]))
                [self.imview.addItem(peak) for peak in self.peaks]
            else:
                self.print('You have to correlate FM and FIB first!')
        else: #SEM or TEM
            for i in range(len(transf)):
                pos = QtCore.QPointF(transf[i,0] - self.orig_size / 2, transf[i,1] - self.orig_size / 2)
                color = None
                if self.other.diff is not None:
                    idx = np.where(np.isclose(np.array(self.other.refined_points), np.array([transf[i,0], transf[i,1]])))
                    if not np.array_equal(idx[0], np.array([])):
                        if len(idx) == 2:
                            idx = idx[0][0]
                        color = colors[idx]
                        color = matplotlib.colors.to_hex(color)
                point = PeakROI(pos, self.orig_size, self.imview.getImageItem(), color=color)
                self.peaks.append(point)
                # Remove FM beads information
                for k in range(len(self._orig_points_corr)):
                    if np.allclose(transf[i, :2], self._orig_points_corr[k]):
                        self.imview.removeItem(point)

        self.peaks = np.array(self.peaks)
        if hasattr(self, 'translate_peaks_btn') and self.other.fixed_orientation:
            self.translate_peaks_btn.setEnabled(True)
            self.refine_peaks_btn.setEnabled(True)

    def _translate_peaks(self, active):
        if not self.other.fixed_orientation:
            self.print('You have to confirm the FM orientation first!')
            return

        if active:
            self.show_grid_btn.setChecked(False)
            self.refine_peaks_btn.setChecked(False)
            for p in self.peaks:
                p.translatable = True
                p.sigRegionChangeFinished.connect(self._translate_peaks_slot)
            #self._old_peak_pos = [[p.pos().x(), p.pos().y()] for p in self.peaks]
        else:
            for p in self.peaks:
                p.sigRegionChangeFinished.disconnect()
                p.translatable = False

    def _translate_peaks_slot(self, item):
        ind = np.where(item == self.peaks)[0][0]
        shift = item.pos() - self.peaks[ind]._old_peak_pos
        #self.grid_box.setPos(self.grid_box.pos().x()+shift[0], self.grid_box.pos().y()+shift[1])
        self.log(ind, shift)
        for i in range(len(self.peaks)):
            self.peaks[i]._old_peak_pos[0] += shift.x()
            self.peaks[i]._old_peak_pos[1] += shift.y()
            if i != ind and not self.peaks[i].has_moved:
                pass
            else:
                self.peaks[i].has_moved = False
            self.peaks[i].setPos(self.peaks[i]._old_peak_pos, finish=False)
                #self.peaks[i].setPos(self._old_peak_pos[i], finish=False)

    def _refine_peaks(self, active):
        if active:
            if not self.other.fixed_orientation: #and not( self.other.ops._transformed and self.other.semcontrols.ops._transformed):
                self.print('You have confirm correct orientation first!')
                self.refine_peaks_btn.setChecked(False)
                return
            self.show_grid_btn.setChecked(False)
            self.translate_peaks_btn.setChecked(False)
            for p in self.peaks:
                p.translatable = True
                p.sigRegionChangeFinished.connect(lambda pt=p: self._peak_to_point(pt))
        else:
            if not self.other.fixed_orientation: #and not( self.other.ops._transformed and self.other.semcontrols.ops._transformed):
                return
            for p in self.peaks:
                p.sigRegionChangeFinished.disconnect()
                p.translatable = False

    def _peak_to_point(self, peak):
        idx = self._check_point_idx(peak)
        if idx is None:
            self.imview.removeItem(peak)
            peak.peakMoved(None)
            ref_ind = [i for i in range(len(self.peaks)) if self.peaks[i] == peak]
            pos = copy.copy(self.other.ops.peaks[ref_ind[0]])
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
        if self.ops is not None and self.other.ops is not None:
            if self.ops.points is not None and self.other.ops.points is not None:
                if hasattr(self, 'tab_index') and self.tab_index != 1:
                    self._calc_tr_matrices()
                elif hasattr(self.other, 'tab_index') and self.other.tab_index != 1:
                    self.other._calc_tr_matrices()

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
            self.confirm_btn.setEnabled(True)
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
        self.transform_btn.setEnabled(False)
        #self.rot_transform_btn.setEnabled(False)
        self.define_btn.setEnabled(False)

        if hasattr(self, 'select_btn'):
            self._transform_pois()

        if self.ops is not None and self.other.ops is not None:
            if self.ops._transformed and self.other.ops._transformed:
                if hasattr(self, 'tab_index'):
                    if self.tab_index != 1:
                        self.other.show_peaks_btn.setEnabled(True)
                else:
                    self.show_peaks_btn.setEnabled(True)

        self._recalc_grid(toggle_orig=toggle_orig)
        self._update_imview()


    @utils.wait_cursor('print')
    def _show_original(self, state=None):
        if self.ops is None:
            return
        if hasattr(self, 'poi_btn'):
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
            self.confirm_btn.setChecked(False)
            self.confirm_btn.setEnabled(False)
            self.flipv.setEnabled(False)
            self.fliph.setEnabled(False)
            self.transpose.setEnabled(False)
            self.rotate.setEnabled(False)
            #self.poi_ref_btn.setEnabled(False)
            #self.clear_btn.setEnabled(False)
            self.merge_btn.setEnabled(False)
            self.refine_btn.setEnabled(False)
        elif hasattr(self.ops, 'flipv') and self.ops._transformed:
            #self.poi_ref_btn.setEnabled(True)
            self.confirm_btn.setEnabled(True)
            #self.clear_btn.setEnabled(True)
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
        if hasattr(self, 'tab_index'):
            if self.peak_btn.isChecked():
                show_peaks = True
                self.peak_btn.setChecked(False)
            if self.tab_index != 1:
                if self.other.ops is not None:
                    if self.ops._transformed and self.other.ops._transformed:
                        self.other.show_peaks_btn.setEnabled(True)

        #if hasattr(self, 'tab_index') and not self.tab_index != 1:
        if not hasattr(self, 'tab_index'):
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
                self._update_pois()
            else:
                self._draw_pois()


    def correct_grid_z(self):
        self.ops.fib_matrix = None
        # set FIB matrix to None to recalculate with medium z slice
        self._recalc_grid(scaling=self.other.ops.voxel_size[2] / self.other.ops.voxel_size[0])
        self._show_grid()
        self.print('WARNING! Recalculate FIB grid square for z = ', self.num_slices // 2)


    def _couple_views(self):
        if self.ops is None or self.other.ops is None or self.ops.points is None or self.other.ops.points is None:
            return
        if not self.fixed_orientation:
            if not self.ops._transformed or not self.other.ops._transformed:
                return
        else:
            if hasattr(self, 'select_btn'):
                if not self.ops._transformed or self.other.ops._transformed:
                    return
            else:
                if not self.other.ops._transformed or self.ops._transformed:
                    return

        vrange = self.imview.getImageItem().getViewBox().targetRect()
        p1 = np.array([vrange.bottomLeft().x(), vrange.bottomLeft().y()])
        p2 = np.array([vrange.bottomRight().x(), vrange.bottomRight().y()])
        p3 = np.array([vrange.topRight().x(), vrange.topRight().y()])
        p4 = np.array([vrange.topLeft().x(), vrange.topLeft().y()])
        points = [p1, p2, p3, p4]
        if hasattr(self, 'select_btn') and self.tab_index != 1:
            if self.tr_matrices is None:
                self._calc_tr_matrices()
            #src_sorted = np.array(
            #    sorted(self.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
            #dst_sorted = np.array(
            #    sorted(self.other.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
            #tr_matrices = self.ops.get_transform(src_sorted, dst_sorted)
            tf_points = np.array([(self.tr_matrices @ np.array([point[0], point[1], 1]))[:2] for point in points])
        elif hasattr(self.other, 'select_btn') and self.other.tab_index != 1:
            if self.other.tr_matrices is None:
                self.other._calc_tr_matrices()
            #    src_sorted = np.array(
        #        sorted(self.other.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
        #    dst_sorted = np.array(
        #        sorted(self.ops.points, key=lambda k: [np.cos(60 * np.pi / 180) * k[0] + k[1]]))
        #    tr_matrices = self.other.ops.get_transform(src_sorted, dst_sorted)
            tf_points = np.array([(np.linalg.inv(self.other.tr_matrices) @ np.array([point[0], point[1], 1]))[:2] for point in points])
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
