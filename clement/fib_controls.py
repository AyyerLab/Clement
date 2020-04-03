import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
#from .fib_operations import FIB_ops
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

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Angles:', self)
        line.addWidget(label)

        label = QtWidgets.QLabel('Sigma:', self)
        line.addWidget(label)
        self.sigma_btn = QtWidgets.QLineEdit(self)
        self.sigma_btn.setText('20')
        self._sigma_angle = int(self.sigma_btn.text())
        line.addWidget(self.sigma_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Grid box:')
        line.addWidget(label)
        self.show_grid_btn = QtWidgets.QCheckBox('Recalculate grid box',self)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_grid_btn.stateChanged.connect(self._show_grid)
        line.addWidget(self.show_grid_btn)
        line.addStretch(1)

        # ---- Points of interest
        #line = QtWidgets.QHBoxLayout()
        #vbox.addLayout(line)
        #label = QtWidgets.QLabel('Point transform:', self)
        #line.addWidget(label)
        self.select_btn = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn.setCheckable(True)
        self.select_btn.setEnabled(False)
        self.select_btn.toggled.connect(self._define_corr_toggled)
        #line.addWidget(self.select_btn)
        #line.addStretch(1)

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
            #self.ops = FIB_ops()
            self.ops = EM_ops()
            self.ops.parse(self.mrc_fname.text(), step=1)
            self.imview.setImage(self.ops.data)
            self.grid_box = None
            self.transp_btn.setEnabled(True)
            if self.sem_ops is not None and self.sem_ops._orig_points is not None:
                self.show_grid_btn.setEnabled(True)
            self.show_grid_btn.setChecked(False)
            #if self.sem_ops is not None and (self.sem_ops._tf_points is not None or self.sem_ops._tf_points_region is not None):
            #    self.select_btn.setEnabled(True)
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
        self._recalc_grid()
        self._update_imview()

    def enable_buttons(self, enable=False):
        if self.ops is not None and self.ops.data is not None:
            self.show_grid_btn.setEnabled(enable)
            #self.select_btn.setEnabled(enable)

    def _show_grid(self, state):
        if self.show_grid_btn.isChecked():
            self._calc_grid()
            self.show_grid_box = True
            self.imview.addItem(self.grid_box)
        else:
            self.imview.removeItem(self.grid_box)
            self.show_grid_box = False

    def _calc_grid(self):
        if self.ops.fib_matrix is None:
            self.ops.calc_fib_transform(int(self.sigma_btn.text()), self.sem_ops.data.shape)

        self.ops.apply_fib_transform(self.sem_ops._orig_points, self.sem_ops.data.shape)

        pos = list(self.ops.points)
        self.grid_box = pg.PolyLineROI(pos, closed=True, movable=False)
        self.imview.addItem(self.grid_box)

    def _recalc_grid(self, toggle_orig=False):
        if self.ops.points is not None:
            if self.show_grid_btn.isChecked():
                self.imview.removeItem(self.grid_box)

            pos = list(self.ops.points)
            self.grid_box = pg.PolyLineROI(pos, closed=True, movable=False)
            if self.show_grid_btn.isChecked():
                self.imview.addItem(self.grid_box)


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
        #self.select_btn.setEnabled(False)

        self.ops.__init__()
