import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
from .em_operations import EM_ops


class FIBControls(BaseControls):
    def __init__(self, imview, vbox, sem_ops):
        super(FIBControls, self).__init__()
        self.tag = 'EM'
        self.imview = imview
        self.ops = None
        self.sem_ops = sem_ops
        self.fib = False

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
        # ---- Assemble montage
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        button = QtWidgets.QPushButton('Load FIB Image:', self)
        button.clicked.connect(self._load_mrc)
        line.addWidget(button)
        self.mrc_fname = QtWidgets.QLabel(self)
        line.addWidget(self.mrc_fname, stretch=1)

        self.transp_btn = QtWidgets.QCheckBox('Transpose', self)
        self.transp_btn.clicked.connect(self._transpose)
        self.transp_btn.setEnabled(False)
        line.addWidget(self.transp_btn)

        # ---- Specify FIB orientation
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Angles:', self)
        line.addWidget(label)

        label = QtWidgets.QLabel('Sigma:', self)
        line.addWidget(label)
        self.sigma_btn = QtWidgets.QLineEdit(self)
        self.sigma_btn.setText('20')
        self._sigma_angle = int(self.sigma_btn.text())
        self.sigma_btn.setEnabled(False)
        line.addWidget(self.sigma_btn)
        line.addStretch(1)

        # ---- Calculate grid square
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Grid box:')
        line.addWidget(label)
        self.show_grid_btn = QtWidgets.QCheckBox('Recalculate grid square',self)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_grid_btn.stateChanged.connect(self._show_grid)
        shift_x_label = QtWidgets.QLabel('Shift x:')
        shift_y_label = QtWidgets.QLabel('Shift y:')
        self.shift_x_btn = QtWidgets.QLineEdit(self)
        self.shift_y_btn = QtWidgets.QLineEdit(self)
        self.shift_x_btn.setText('-1000')
        self.shift_y_btn.setText('250')
        self.shift_x = int(self.shift_x_btn.text())
        self.shift_y = int(self.shift_y_btn.text())
        self.shift_x_btn.setEnabled(False)
        self.shift_y_btn.setEnabled(False)
        self.shift_btn = QtWidgets.QPushButton('Shift box')
        self.shift_btn.clicked.connect(self._refine_grid)
        self.shift_btn.setEnabled(False)
        line.addWidget(self.show_grid_btn)
        line.addWidget(shift_x_label)
        line.addWidget(self.shift_x_btn)
        line.addWidget(shift_y_label)
        line.addWidget(self.shift_y_btn)
        line.addWidget(self.shift_btn)
        line.addStretch(1)

        # ---- Show FM peaks
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Peaks:')
        line.addWidget(label)
        self.show_peaks_btn = QtWidgets.QCheckBox('Show FM peaks',self)
        self.show_peaks_btn.setChecked(False)
        self.show_peaks_btn.stateChanged.connect(self._show_FM_peaks)
        self.show_peaks_btn.setEnabled(False)
        line.addWidget(self.show_peaks_btn)
        line.addStretch(1)

        # ---- Refinement
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Refinement precision [nm]:', self)
        line.addWidget(label)
        self.refine_btn = QtWidgets.QPushButton('Refine', self)
        self.refine_btn.setEnabled(False)
        self.err_btn = QtWidgets.QLabel('0')
        self.err_plt_btn = QtWidgets.QPushButton('Show error distribution')
        self.err_plt_btn.clicked.connect(lambda : self._scatter_plot(idx=1))
        self.err_plt_btn.setEnabled(False)
        line.addWidget(self.err_btn)
        line.addWidget(self.err_plt_btn)
        line.addStretch(1)

        # ---- Quit button
        vbox.addStretch(1)
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        line.addStretch(1)
        self.quit_button = QtWidgets.QPushButton('Quit', self)
        line.addWidget(self.quit_button)

        self.show()

    def _load_mrc(self, jump=False):
        if not jump:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            self._file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                 'Select .mrc file',
                                                                 self._curr_folder,
                                                                 '*.mrc;;*.tif;;*tiff')
            self._curr_folder = os.path.dirname(self._file_name)

        if self._file_name is not '':
            self.reset_init()
            self.mrc_fname.setText(self._file_name)

            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.ops = EM_ops()
            self.ops.parse_2d(self._file_name)
            self.imview.setImage(self.ops.data)
            self.grid_box = None
            self.transp_btn.setEnabled(True)
            self.sigma_btn.setEnabled(True)
            if self.sem_ops is not None and self.sem_ops._orig_points is not None:
                self.show_grid_btn.setEnabled(True)
                self.shift_btn.setEnabled(True)
                self.shift_x_btn.setEnabled(True)
                self.shift_y_btn.setEnabled(True)
            self.show_grid_btn.setChecked(False)
            QtWidgets.QApplication.restoreOverrideCursor()
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
        self._calc_grid(transpose=True)
        self._update_imview()

    def enable_buttons(self, enable=False):
        if self.ops is not None and self.ops.data is not None:
            self.show_grid_btn.setEnabled(enable)
            self.shift_btn.setEnabled(enable)

    def _show_grid(self):
        if self.show_grid_btn.isChecked():
            self._calc_grid()
            self.show_grid_box = True
            self.imview.addItem(self.grid_box)
        else:
            self.imview.removeItem(self.grid_box)
            self.show_grid_box = False

    def _calc_grid(self, transpose=False, scaling=1):
        if not transpose:
            if self.sem_ops is not None:
                if self.ops.fib_matrix is None:
                    self.ops.calc_fib_transform(int(self.sigma_btn.text()), self.sem_ops.data.shape, self.sem_ops.pixel_size)
                    self.ops.apply_fib_transform(self.sem_ops._orig_points, self.num_slices, scaling)

        if self.ops.points is not None:
            if self.show_grid_btn.isChecked():
                self.imview.removeItem(self.grid_box)
            pos = list(self.ops.points)
            self.grid_box = pg.PolyLineROI(pos, closed=True, movable=False)
            self.refine_btn.setEnabled(True)
            if self.show_grid_btn.isChecked():
                self.imview.addItem(self.grid_box)

        if self.grid_box is not None:
            self.show_peaks_btn.setEnabled(True)

    def _refine_grid(self):
        if self.ops.points is not None:
            print('Orig Points: \n', self.ops.points)
            xshift = int(self.shift_x_btn.text())
            yshift = int(self.shift_y_btn.text())
            self.ops.calc_grid_shift(xshift, yshift)
            print('New Points: \n', self.ops.points)
            self._calc_grid()
            self.shift_x_btn.setText('0')
            self.shift_y_btn.setText('0')
        else:
            print('You have to calculate the grid box first!')

    def _refine_fib(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if len(self._points_corr) < 4:
            print('You have to select at least 4 points!')
        else:
            self._refine()
            #self.ops.apply_refinement(self.ops.points)
            self._calc_grid()
            self.err_btn.setText('{:.2f}'.format(self._rms*self.ops.pixel_size[0]*1e9))
        QtWidgets.QApplication.restoreOverrideCursor()

    def _save_mrc_montage(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is None:
            print('No montage to save!')
        else:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Binned Montage', self._curr_folder, '*.mrc')
            self._curr_folder = os.path.dirname(file_name)
            if file_name is not '':
                self.ops.save_merge(file_name)
        QtWidgets.QApplication.restoreOverrideCursor()

    def reset_init(self):
        if self.show_grid_btn.isChecked():
           self.imview.removeItem(self.grid_box)

        self._points_corr = []
        self._points_corr_indices= []
        self.show_grid_box = False
        self.grid_box = None
        self.transp_btn.setEnabled(False)
        self.transp_btn.setChecked(False)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_peaks_btn.setEnabled(False)
        self.show_peaks_btn.setChecked(False)

        self.ops.__init__()



