import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
from .em_operations import EM_ops
from . import utils

class FIBControls(BaseControls):
    def __init__(self, imview, vbox, sem_ops, printer, logger):
        super(FIBControls, self).__init__()
        self.tag = 'FIB'
        self.imview = imview
        self.ops = None
        self.sem_ops = sem_ops
        self.num_slices = None
        self.popup = None
        self.show_merge = False

        self.show_grid_box = False
        self.grid_box = None
        self.old_pos0 = None
        self.box_shift = None
        self.mrc_fname = None
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)

        self._curr_folder = None
        self._file_name = None
        self._fib_angle = None
        self._refined = False

        self.orig_size = None
        self.size = self.orig_size
        self.print = printer
        self.log = logger

        self._init_ui(vbox)

    def _init_ui(self, vbox):
        utils.add_montage_line(self, vbox, 'FIB', downsampling=False)

        # ---- Specify FIB orientation
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Angles:', self)
        line.addWidget(label)

        label = QtWidgets.QLabel('Milling angle:', self)
        line.addWidget(label)
        self.angle_btn = QtWidgets.QLineEdit(self)
        self.angle_btn.setMaximumWidth(30)
        self.angle_btn.setText('0')
        self._fib_angle = float(self.angle_btn.text())
        self.angle_btn.textChanged.connect(self._recalc_sigma)
        self.angle_btn.setEnabled(False)
        line.addWidget(self.angle_btn)

        line.addStretch(1)

        # ---- Calculate grid square
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Grid:')
        line.addWidget(label)
        self.show_grid_btn = QtWidgets.QCheckBox('Show grid square', self)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_grid_btn.stateChanged.connect(self._show_grid)
        line.addWidget(self.show_grid_btn)
        line.addStretch(1)

        utils.add_fmpeaks_line(self, vbox)

        self.size = self.orig_size
        self.show()

    #@utils.wait_cursor('print')
    def _load_mrc(self, jump=False):
        if not jump:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            self._file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                       'Select FIB data',
                                                                       self._curr_folder,
                                                                       'All (*.tif *.tiff *.mrc);;*.mrc;;*.tif;;*tiff')
            self._curr_folder = os.path.dirname(self._file_name)

        if self._file_name != '':
            if self.ops is not None:
                self.reset_init()
                self.other.tab_index = 1
            self.mrc_fname.setText(os.path.basename(self._file_name))

            self.ops = EM_ops(self.print, self.log)
            self.ops.parse_2d(self._file_name)

            self.orig_size = float(self.size_box.text()) * 1e3 / self.ops.pixel_size[0]
            self.size = self.orig_size
            self.imview.setImage(self.ops.data)
            self.grid_box = None
            self.transp_btn.setEnabled(True)
            self.angle_btn.setEnabled(True)
            if self.sem_ops is not None and self.sem_ops._orig_points is not None:
                self.show_grid_btn.setEnabled(True)
            if self.sem_ops is not None and self.sem_ops._tf_points is not None:
                self.ops._transformed = True
            self.show_grid_btn.setChecked(False)
        else:
            self.print('You have to choose a file first!')

    @utils.wait_cursor('print')
    def _update_imview(self, state=None):
        if self.ops is not None and self.ops.data is not None:
            old_shape = self.imview.image.shape
            new_shape = self.ops.data.shape
            if old_shape == new_shape:
                vr = self.imview.getImageItem().getViewBox().targetRect()
            levels = self.imview.getHistogramWidget().item.getLevels()
            self.imview.setImage(self.ops.data, levels=levels)
            if old_shape == new_shape:
                self.imview.getImageItem().getViewBox().setRange(vr, padding=0)

    @utils.wait_cursor('print')
    def _transpose(self, state=None):
        self.ops.transpose()
        self._recalc_grid(recalc_matrix=False)
        self._update_imview()

    def enable_buttons(self, enable=False):
        if self.ops is not None and self.ops.data is not None:
            self.show_grid_btn.setEnabled(enable)


    @utils.wait_cursor('print')
    def _show_grid(self, state=2):
        if state > 0:
            self.show_grid_box = True
            self._recalc_grid()
        else:
            self.show_grid_box = False
            if self.grid_box is not None:
                self.imview.removeItem(self.grid_box)

    @utils.wait_cursor('print')
    def _recalc_sigma(self, state=None):
        if self.angle_btn.text() is not '':
            self._fib_angle = float(self.angle_btn.text())
            self.box_shift = None
            self.ops.box_shift = None
            self._recalc_grid()


    @utils.wait_cursor('print')
    def _recalc_grid(self, state=None, recalc_matrix=True, scaling=1, shift=np.array([0,0])):
        if self.sem_ops is not None and self.sem_ops.points is not None and recalc_matrix:
            if self.box_shift is None:
                shift = np.zeros(2)
                redo = True
            else:
                shift = self.box_shift
                redo = False

            fib_angle = float(self.angle_btn.text())
            #is_transposed = not self.transp_btn.isChecked()

            self.ops.calc_fib_transform(fib_angle, self.sem_ops.data.shape,
                                        self.other.ops.voxel_size, self.sem_ops.pixel_size, shift=shift, sem_transpose=False)


            tf_points = []
            for i in range(len(self.other.ops.points)):
                p = self.other.ops.points[i]
                tf_points.append((self.other.tr_matrices @ np.array([p[0], p[1], 1]))[:2])

            #self.ops.apply_fib_transform(self.sem_ops._orig_points, self.num_slices, scaling)
            self.ops.apply_fib_transform(np.array(tf_points), self.num_slices, scaling)

        if self.ops.points is not None:
            pos = list(self.ops.points)
            if self.show_grid_box and self.grid_box is not None:
                self.imview.removeItem(self.grid_box)
            self.grid_box = pg.PolyLineROI(pos, closed=True, movable=not self._refined, resizable=False, rotatable=False)
            if self.old_pos0 is None:
                self.old_pos0 = [0, 0]
            self.grid_box.sigRegionChangeFinished.connect(self._update_shifts)
            if redo:
                self.ops.calc_fib_transform(fib_angle, self.sem_ops.data.shape, self.other.ops.voxel_size,
                                            self.sem_ops.pixel_size, shift=np.zeros(2), sem_transpose=False)
                #self.ops.apply_fib_transform(self.sem_ops._orig_points, self.num_slices, scaling)
                self.ops.apply_fib_transform(self.sem_ops.points, self.num_slices, scaling)
                pos = list(self.ops.points)
                self.grid_box = pg.PolyLineROI(pos, closed=True, movable=not self._refined, resizable=False,
                                               rotatable=False)
                self.old_pos0 = self.grid_box.pos()
                self.box_shift = np.zeros(2)
                self.grid_box.sigRegionChangeFinished.connect(self._update_shifts)
            self.print('Box origin at:', self.old_pos0)

        if self.grid_box is not None:
            if self.show_grid_box:
                self.imview.addItem(self.grid_box)
            self.show_peaks_btn.setEnabled(True)
            self.auto_opt_btn.setEnabled(True)
            self.size_box.setEnabled(True)
            if self.show_peaks_btn.isChecked():
                self.show_peaks_btn.setChecked(False)
                self.show_peaks_btn.setChecked(True)

    @utils.wait_cursor('print')
    def _update_shifts(self, state):
        new_pos = self.grid_box.pos() + self.old_pos0
        self.box_shift = np.array(new_pos)
        self.ops.points = np.copy([point + self.box_shift for point in self.ops.points])
        self.old_pos0 = new_pos
        self._recalc_grid()

    @utils.wait_cursor('print')
    def _save_mrc_montage(self, state=None):
        if self.ops is None:
            self.print('No montage to save!')
        else:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Binned Montage', self._curr_folder,
                                                                 '*.mrc')
            self._curr_folder = os.path.dirname(file_name)
            if file_name != '':
                self.ops.save_merge(file_name)

    @utils.wait_cursor('print')
    def reset_init(self, state=None):
        if self.show_grid_btn.isChecked() and self.grid_box is not None:
            self.imview.removeItem(self.grid_box)
        self.show_peaks_btn.setChecked(False)
        self.show_peaks_btn.setEnabled(False)

        self._box_coordinate = None
        self._points_corr = []
        self._points_corr_z = []
        self._orig_points_corr = []
        self._points_corr_indices = []
        self._refined = False
        self._err = None
        self._std = [None, None]
        self._conv = None
        self._dist = None

        self.popup = None

        self._points_corr_history = []
        self._points_corr_z_history = []
        self._orig_points_corr_history = []
        self._fib_vs_sem_history = []
        self._size_history = []

        self.flips = [False, False, False, False]
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
        self.orig_size = None
        self.size = self.orig_size
        self.fixed_orientation = False
        self.peaks = []
        self.num_slices = None
        self.min_conv_points = 10

        self.show_grid_box = False
        self.grid_box = None
        self.transp_btn.setEnabled(False)
        self.transp_btn.setChecked(False)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)

        self.translate_peaks_btn.setChecked(False)
        self.translate_peaks_btn.setEnabled(False)
        self.refine_peaks_btn.setChecked(False)
        self.refine_peaks_btn.setEnabled(False)
        self.size_box.setEnabled(False)
        self.auto_opt_btn.setEnabled(False)

        self.ops.__init__(self.print, self.log)
