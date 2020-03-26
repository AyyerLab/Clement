import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
#from .fib_operations import FIB_ops
from .em_operations import EM_ops

class FIBControls(BaseControls):
    def __init__(self, imview, vbox):
        super(FIBControls, self).__init__()
        self.tag = 'EM'
        self.imview = imview
        self.ops = None
        self.fib = None

        self.show_grid_box = False
        self.grid_box = None
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)

        self._curr_folder = None
        self._file_name = None
        self._downsampling = None
        self._select_region_original = True

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
        line.addWidget(self.transp_btn)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Grid transform:', self)
        line.addWidget(label)
        self.show_grid_btn = QtWidgets.QCheckBox('Show grid box',self)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_grid_btn.stateChanged.connect(self._show_grid)
        line.addWidget(self.show_grid_btn)
        line.addStretch(1)

        # ---- Points of interest
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Point transform:', self)
        line.addWidget(label)
        self.select_btn = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn.setCheckable(True)
        self.select_btn.setEnabled(False)
        self.select_btn.toggled.connect(self._define_corr_toggled)
        line.addWidget(self.select_btn)
        line.addStretch(1)

        # ---- Quit button
        vbox.addStretch(1)
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        line.addStretch(1)
        self.quit_button = QtWidgets.QPushButton('Quit', self)
        line.addWidget(self.quit_button)

        self.show()

    def _load_mrc(self):
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
            self.show_grid_btn.setEnabled(False)
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
        self._update_imview()

    def enable_buttons(self, enable=False):
        self.show_grid_btn.setEnabled(enable)
        self.select_btn.setEnabled(enable)

    def _show_grid(self, state):
        if self.grid_box is not None:
            if self.show_grid_btn.isChecked():
                self.show_grid_box = True
                self.imview.addItem(self.grid_box)
            else:
                self.imview.removeItem(self.grid_box)
                self.show_grid_box = False

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
        #self.ops = None
        #self.other = None # The other controls object
        if self.show_grid_btn.isChecked():
            if self.ops._transformed:
                self.imview.removeItem(self.tr_grid_box)
            else:
                self.imview.removeItem(self.grid_box)

        self._box_coordinate = None
        self._points_corr = []
        self._points_corr_indices= []
        self._size_ops = 3
        self._size_other = 3
        self._refined = False
        self._refine_history = []
        self._refine_counter = 0

        self.tr_matrices = None
        self.show_grid_box = False
        self.show_tr_grid_box = False
        self.clicked_points = []
        self.grid_box = None
        self.tr_grid_box = None
        self.boxes = []
        self.tr_boxes = []
        self.original_help = True
        self.redo_tr = False
        self.setContentsMargins(0, 0, 0, 0)
        self.counter = 0
        self.anno_list = []

        self.show_boxes = False
        self._downsampling = None

        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.select_btn.setEnabled(False)

        self.ops.__init__()
