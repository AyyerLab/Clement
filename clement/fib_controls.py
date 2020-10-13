import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
from .em_operations import EM_ops
from . import utils

class FIBControls(BaseControls):
    def __init__(self, imview, vbox, sem_ops):
        super(FIBControls, self).__init__()
        self.tag = 'EM'
        self.imview = imview
        self.ops = None
        self.sem_ops = sem_ops
        self.fib = False
        self.num_slices = None

        self.show_grid_box = False
        self.grid_box = None
        self.mrc_fname = None
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)

        self._curr_folder = None
        self._file_name = None
        self._sigma_angle = None
        self._refined = False
        self._init_ui(vbox)

    def _init_ui(self, vbox):
        utils.add_montage_line(self, vbox, 'FIB', downsampling=False)

        # ---- Specify FIB orientation
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Angles:', self)
        line.addWidget(label)

        label = QtWidgets.QLabel('\u03c3_SEM - \u03c3_FIB:', self)
        line.addWidget(label)
        self.sigma_btn = QtWidgets.QLineEdit(self)
        self.sigma_btn.setText('0')
        self._sigma_angle = int(self.sigma_btn.text())
        self.sigma_btn.setEnabled(False)
        line.addWidget(self.sigma_btn)

        #label = QtWidgets.QLabel('\u03c6:', self)
        #line.addWidget(label)
        #self.phi_box = QtWidgets.QComboBox(self)
        #listview = QtWidgets.QListView(self)
        #self.phi_box.setView(listview)
        #self.phi_box.addItems([str(i) for i in range(0,360,180)])
        #self.phi_box.setCurrentIndex(0)
        #self.phi_box.currentIndexChanged.connect(self._phi_changed)
        #line.addWidget(self.phi_box)

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
        shift_x_label = QtWidgets.QLabel('Shift x:')
        line.addWidget(shift_x_label)
        self.shift_x_btn = QtWidgets.QLineEdit(self)
        self.shift_x_btn.setText('0')
        self.shift_x_btn.setEnabled(False)
        line.addWidget(self.shift_x_btn)
        shift_y_label = QtWidgets.QLabel('Shift y:')
        line.addWidget(shift_y_label)
        self.shift_y_btn = QtWidgets.QLineEdit(self)
        self.shift_y_btn.setText('0')
        self.shift_y_btn.setEnabled(False)
        line.addWidget(self.shift_y_btn)
        button = QtWidgets.QPushButton('Recalculate', self)
        button.clicked.connect(self._recalc_grid)
        line.addWidget(button)
        line.addStretch(1)

        utils.add_fmpeaks_line(self, vbox)

        utils.add_precision_line(self, vbox)

        self.show()

    #@utils.wait_cursor
    def _load_mrc(self, jump=False):
        if not jump:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            self._file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                       'Select .mrc file',
                                                                       self._curr_folder,
                                                                       '*.tif;;*tiff;;*.mrc')
            self._curr_folder = os.path.dirname(self._file_name)

        if self._file_name != '':
            self.reset_init()
            self.mrc_fname.setText(self._file_name)

            self.ops = EM_ops()
            self.ops.parse_2d(self._file_name)
            self.imview.setImage(self.ops.data)
            self.grid_box = None
            self.transp_btn.setEnabled(True)
            self.sigma_btn.setEnabled(True)
            if self.sem_ops is not None and self.sem_ops._orig_points is not None:
                self.show_grid_btn.setEnabled(True)
                self.shift_x_btn.setEnabled(True)
                self.shift_y_btn.setEnabled(True)
            self.show_grid_btn.setChecked(False)
        else:
            print('You have to choose a file first!')

    def _update_imview(self):
        if self.ops is not None and self.ops.data is not None:
            old_shape = self.imview.image.shape
            new_shape = self.ops.data.shape
            if old_shape == new_shape:
                vr = self.imview.getImageItem().getViewBox().targetRect()
            levels = self.imview.getHistogramWidget().item.getLevels()
            self.imview.setImage(self.ops.data, levels=levels)
            if old_shape == new_shape:
                self.imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _transpose(self):
        self.ops.transpose()
        self._recalc_grid(recalc_matrix=False)
        self._update_imview()

    def enable_buttons(self, enable=False):
        if self.ops is not None and self.ops.data is not None:
            self.show_grid_btn.setEnabled(enable)

    @utils.wait_cursor
    def _show_grid(self, state=2):
        if state > 0:
            self.show_grid_box = True
            if self.grid_box is None:
                self._recalc_grid()
            self.imview.addItem(self.grid_box)
        else:
            if self.grid_box is not None:
                self.imview.removeItem(self.grid_box)
            self.show_grid_box = False

    @utils.wait_cursor
    def _recalc_grid(self, state=None, recalc_matrix=True, scaling=1):
        if self.sem_ops is not None and recalc_matrix:
            sigma_angle = float(self.sigma_btn.text())
            #phi_angle = float(self.phi_box.currentText()) * np.pi / 180.
            is_transposed = self.transp_btn.isChecked()

            self.ops.calc_fib_transform(sigma_angle, self.sem_ops.data.shape,
                                        self.sem_ops.pixel_size,sem_transpose=is_transposed)

            if self.ops.points is not None:
            #if self.ops.points is not None and scaling != 1:
                xshift = float(self.shift_x_btn.text())
                yshift = float(self.shift_y_btn.text())
                self.ops.calc_grid_shift(xshift, yshift)
            print(self.ops.fib_matrix)

            self.ops.apply_fib_transform(self.sem_ops._orig_points, self.num_slices, scaling)

        if self.ops.points is not None:
            self._show_grid(0) # Hide grid

            pos = list(self.ops.points)
            self.grid_box = pg.PolyLineROI(pos, closed=True, movable=not self._refined, resizable=False, rotatable=False)
            self.old_pos0 = [float(self.shift_x_btn.text()), float(self.shift_y_btn.text())]
            print('Box origin at:', self.old_pos0)
            self.grid_box.sigRegionChanged.connect(self._update_shifts)
            self.grid_box.sigRegionChangeFinished.connect(self._recalc_grid)
            self._show_grid(2) # Show grid

        if self.grid_box is not None:
            self.show_peaks_btn.setEnabled(True)
            if self.show_peaks_btn.isChecked():
                self.show_peaks_btn.setChecked(False)
                self.show_peaks_btn.setChecked(True)

            self.shift_x_btn.setEnabled(True)
            self.shift_y_btn.setEnabled(True)

    def _update_shifts(self, state):
        diff_pos = state.pos() + self.old_pos0
        self.shift_x_btn.setText('%.2f'%diff_pos.x())
        self.shift_y_btn.setText('%.2f'%diff_pos.y())

    def _phi_changed(self, index):
        self.shift_x_btn.setText('0')
        self.shift_y_btn.setText('0')

    @utils.wait_cursor
    def _save_mrc_montage(self):
        if self.ops is None:
            print('No montage to save!')
        else:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Binned Montage', self._curr_folder,
                                                                 '*.mrc')
            self._curr_folder = os.path.dirname(file_name)
            if file_name != '':
                self.ops.save_merge(file_name)

    def reset_init(self):
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
        self._err = [None, None]
        self._std = [[None, None], [None, None]]
        self._conv = [None, None]
        self._dist = None

        self._points_corr_history = []
        self._points_corr_z_history = []
        self._orig_points_corr_history = []
        self._fib_vs_sem_history = []
        self._size_history = []
        self._fib_flips = []

        self.flips = [False, False, False, False]
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

        self.err_btn.setText('0')
        self.err_plt_btn.setEnabled(False)
        self.convergence_btn.setEnabled(False)

        self.ops.__init__()
