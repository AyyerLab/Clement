import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
from .em_operations import EM_ops

class EMControls(BaseControls):
    def __init__(self, imview, vbox):
        super(EMControls, self).__init__()
        self.tag = 'EM'
        self.imview = imview
        self.ops = None
        self.fib = None

        self.show_boxes = False
        self.mrc_fname = None
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
        button = QtWidgets.QPushButton('EM Image:', self)
        button.clicked.connect(self._load_mrc)
        line.addWidget(button)
        self.mrc_fname = QtWidgets.QLabel(self)
        line.addWidget(self.mrc_fname, stretch=1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        step_label = QtWidgets.QLabel(self)
        step_label.setText('Downsampling factor:')
        self.step_box = QtWidgets.QLineEdit(self)
        self.step_box.setMaximumWidth(30)
        self.step_box.setText('10')
        self.step_box.setEnabled(False)
        self._downsampling = self.step_box.text()
        line.addWidget(step_label)
        line.addWidget(self.step_box)
        line.addStretch(1)
        self.assemble_btn = QtWidgets.QPushButton('Assemble', self)
        self.assemble_btn.clicked.connect(self._assemble_mrc)
        self.assemble_btn.setEnabled(False)
        line.addWidget(self.assemble_btn)
        self.transp_btn = QtWidgets.QCheckBox('Transpose', self)
        self.transp_btn.clicked.connect(self._transpose)
        self.transp_btn.setEnabled(False)
        line.addWidget(self.transp_btn)

        # ---- Assembly grid options
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Assembly grid:', self)
        line.addWidget(label)
        self.select_region_btn = QtWidgets.QPushButton('Select subregion',self)
        self.select_region_btn.setCheckable(True)
        self.select_region_btn.toggled.connect(self._select_box)
        self.select_region_btn.setEnabled(False)
        self.show_assembled_btn = QtWidgets.QCheckBox('Show assembled image',self)
        self.show_assembled_btn.stateChanged.connect(self._show_assembled)
        self.show_assembled_btn.setChecked(True)
        self.show_assembled_btn.setEnabled(False)
        line.addWidget(self.select_region_btn)
        line.addWidget(self.show_assembled_btn)
        line.addStretch(1)

        # ---- Define grid
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Grid:', self)
        line.addWidget(label)
        self.define_btn = QtWidgets.QPushButton('Define grid square', self)
        self.define_btn.setCheckable(True)
        self.define_btn.toggled.connect(self._define_grid_toggled)
        self.define_btn.setEnabled(False)
        line.addWidget(self.define_btn)
        self.show_grid_btn = QtWidgets.QCheckBox('Show grid square',self)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_grid_btn.stateChanged.connect(self._show_grid)
        line.addWidget(self.show_grid_btn)
        line.addStretch(1)

        # ---- Transformations
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Transformations:', self)
        line.addWidget(label)
        self.transform_btn = QtWidgets.QPushButton('Transform image', self)
        self.transform_btn.clicked.connect(self._affine_transform)
        self.transform_btn.setEnabled(False)
        line.addWidget(self.transform_btn)
        self.rot_transform_btn = QtWidgets.QCheckBox('Disable Shearing', self)
        self.rot_transform_btn.setEnabled(False)
        line.addWidget(self.rot_transform_btn)
        self.show_btn = QtWidgets.QCheckBox('Show original data', self)
        self.show_btn.setEnabled(False)
        self.show_btn.setChecked(True)
        self.show_btn.stateChanged.connect(self._show_original)
        line.addWidget(self.show_btn)
        line.addStretch(1)

        # ---- Show FM peaks
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Peaks:')
        line.addWidget(label)
        self.show_peaks_btn = QtWidgets.QCheckBox('Show FM peaks',self)
        self.show_peaks_btn.setEnabled(True)
        self.show_peaks_btn.setChecked(False)
        self.show_peaks_btn.stateChanged.connect(self._show_FM_peaks)
        self.show_peaks_btn.setEnabled(False)
        line.addWidget(self.show_peaks_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Refinement precision [nm]:', self)
        line.addWidget(label)
        self.err_btn = QtWidgets.QLabel('0')
        self.err_plt_btn = QtWidgets.QPushButton('Show error distribution')
        self.err_plt_btn.setEnabled(False)

        self.convergence_btn = QtWidgets.QPushButton('Show RMS convergence')
        self.convergence_btn.setEnabled(False)
        line.addWidget(self.err_btn)
        line.addWidget(self.err_plt_btn)
        line.addWidget(self.convergence_btn)
        line.addStretch(1)

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

        if self._file_name != '':
            self.reset_init()
            self.mrc_fname.setText(self._file_name)
            self.assemble_btn.setEnabled(True)
            self.step_box.setEnabled(True)

            self.ops = EM_ops()
            self.ops.parse_2d(self._file_name)
            if len(self.ops.dimensions) == 2:
                self.step_box.setText('1')
                self._assemble_mrc()
                self.assemble_btn.setEnabled(False)

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

            if self.show_assembled_btn.isChecked():
                if self.show_boxes:
                    self._show_boxes()
            else:
                self.show_boxes = False

    def _assemble_mrc(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.step_box.text() == '':
            self._downsampling = 10
        else:
            self._downsampling = self.step_box.text()

        if self._file_name != '':
            if len(self.ops.dimensions) == 3:
                self.ops.parse_3d(int(self._downsampling), self._file_name)
            self.imview.setImage(self.ops.data)
            self.define_btn.setEnabled(True)
            self.show_btn.setChecked(True)
            self.transform_btn.setEnabled(False)
            self.rot_transform_btn.setEnabled(False)
            self.transp_btn.setEnabled(True)
            if self.tr_grid_box is not None:
                self.imview.removeItem(self.tr_grid_box)
            if self.grid_box is not None:
                self.imview.removeItem(self.grid_box)
            self.grid_box = None
            self.ops._transformed = False
            self.show_grid_btn.setEnabled(False)

            if self.ops.stacked_data:
                self.select_region_btn.setEnabled(True)
            else:
                self.select_region_btn.setEnabled(False)
                self.show_assembled_btn.setEnabled(False)
            self.boxes = []
            self.show_grid_btn.setChecked(False)
        else:
            print('You have to choose a file first!')
        QtWidgets.QApplication.restoreOverrideCursor()

    def _transpose(self):
        self.ops.transpose()
        self._recalc_grid()
        self._update_imview()

    def _show_boxes(self):
        if self.ops is None:
            return
        handle_pen = pg.mkPen('#00000000')
        if self.show_btn.isChecked():
            if self.show_boxes:
                [self.imview.removeItem(box) for box in self.tr_boxes]
            if len(self.boxes) == 0:
                for i in range(len(self.ops.pos_x)):
                    roi = pg.PolyLineROI([], closed=True, movable=False)
                    roi.handlePen = handle_pen
                    roi.setPoints(self.ops.grid_points[i])
                    self.boxes.append(roi)
                    self.imview.addItem(roi)
            else:
                [self.imview.addItem(box) for box in self.boxes]
        else:
            if self.show_boxes:
                [self.imview.removeItem(box) for box in self.boxes]
            if len(self.tr_boxes) == 0:
                for i in range(len(self.ops.tf_grid_points)):
                    roi = pg.PolyLineROI([], closed=True, movable=False)
                    roi.handlePen = handle_pen
                    roi.setPoints(self.ops.tf_grid_points[i])
                    self.tr_boxes.append(roi)
                    self.imview.addItem(roi)
            else:
                [self.imview.addItem(box) for box in self.tr_boxes]
        self.show_boxes = True

    def _hide_boxes(self):
        if self.show_btn.isChecked():
            if self.show_boxes:
                [self.imview.removeItem(box) for box in self.boxes]
        else:
            if self.show_boxes:
                [self.imview.removeItem(box) for box in self.tr_boxes]
        self.show_boxes = False

    def _select_box(self, state=None):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.select_region_btn.isChecked():
            self._show_boxes()
            self.ops.orig_region = None
            self.show_assembled_btn.setEnabled(False)
            print('Select box!')
            if self.show_btn.isChecked():
                self._select_region_original = True
            else:
                self._select_region_original = False
        else:
            if self._box_coordinate is not None:
                if self.show_btn.isChecked():
                    transformed = False
                else:
                    transformed = True
                points_obj = self._box_coordinate
                self.ops.select_region(np.array(points_obj),transformed)
                self._hide_boxes()
                if self.ops.orig_region is None:
                    print('Ooops, something went wrong. Try again!')
                    return
                self.show_assembled_btn.setEnabled(True)
                self.show_assembled_btn.setChecked(False)
                self.show_grid_btn.setChecked(False)
                if self.ops._orig_points_region is not None:
                    self.show_grid_btn.setEnabled(True)
                else:
                    self.show_grid_btn.setEnabled(False)
                self.show_btn.setChecked(True)
                self.show_btn.setEnabled(False)
                self.transform_btn.setEnabled(True)
            else:
                self._hide_boxes()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _show_assembled(self):
        if self.ops is None:
            return
        if self.grid_box is not None:
            self.imview.removeItem(self.grid_box)
        if self.tr_grid_box is not None:
            self.imview.removeItem(self.tr_grid_box)
        if self.show_assembled_btn.isChecked():
            self.ops.assembled = True
            if self.ops._orig_points is None:
                self.show_grid_btn.setEnabled(False)
                self.show_grid_btn.setChecked(False)
            else:
                self.show_grid_btn.setEnabled(True)

            self.select_region_btn.setEnabled(True)

            if self.ops.tf_data is None:
                self.show_btn.setChecked(True)
                self.show_btn.setEnabled(False)
            else:
                self.show_btn.setEnabled(True)
        else:
            self.ops.assembled = False
            self.select_region_btn.setEnabled(False)
            self.show_grid_btn.setEnabled(True)
            if self.ops.tf_region is not None:
                self.show_btn.setEnabled(True)
            else:
                self.show_btn.setEnabled(False)
            if self.ops._orig_points_region is None:
                self.show_grid_btn.setEnabled(False)
                self.show_grid_btn.setChecked(False)

        self.ops.toggle_region()
        self._recalc_grid(self.imview)
        self._update_imview()

    def _save_mrc_montage(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is None:
            print('No montage to save!')
        else:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Binned Montage', self._curr_folder, '*.mrc')
            self._curr_folder = os.path.dirname(file_name)
            if file_name != '':
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
        if self.select_region_btn.isChecked():
            if self.ops._transformed:
                [self.imview.removeItem(box) for box in self.tr_boxes]
            else:
                [self.imview.removeItem(box) for box in self.boxes]
            self.select_region_btn.setChecked(False)
            self.select_region_btn.setEnabled(False)

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

        self.show_boxes = False
        self._downsampling = None


        self.step_box.setEnabled(False)
        self.assemble_btn.setEnabled(False)
        self.define_btn.setEnabled(False)
        self.show_btn.setChecked(True)
        self.show_btn.setEnabled(False)
        self.rot_transform_btn.setEnabled(False)
        self.transform_btn.setEnabled(False)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_assembled_btn.setChecked(True)
        self.show_assembled_btn.setEnabled(False)

        self.show_peaks_btn.setChecked(False)
        self.show_peaks_btn.setEnabled(False)

        self.err_btn.setText('0')
        self.err_plt_btn.setEnabled(False)
        self.convergence_btn.setEnabled(False)
        self.ops.__init__()
