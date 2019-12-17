import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
from .em_operations import EM_ops

class EMControls(BaseControls):
    def __init__(self, imview):
        super(EMControls, self).__init__()
        self.tag = 'EM'
        self.imview = imview
        self.ops = None
        
        self.show_boxes = False
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)

        self._curr_folder = None
        self._file_name = None
        self._downsampling = None

        self._init_ui()

    def _init_ui(self):
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

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

        # ---- Define and align to grid
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Grid transform:', self)
        line.addWidget(label)
        self.define_btn = QtWidgets.QPushButton('Define Grid', self)
        self.define_btn.setCheckable(True)
        self.define_btn.toggled.connect(self._define_grid_toggled)
        self.define_btn.setEnabled(False)
        line.addWidget(self.define_btn)
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
        self.show_grid_btn = QtWidgets.QCheckBox('Show grid box',self)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_grid_btn.stateChanged.connect(self._show_grid)
        line.addWidget(self.show_grid_btn)
        line.addStretch(1)

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

        # ---- Points of interest
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Point transform:', self)
        line.addWidget(label)
        self.select_btn = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn.setCheckable(True)
        self.select_btn.setEnabled(False)
        self.select_btn.toggled.connect(self._define_corr_toggled)
        #self.refine_btn = QtWidgets.QPushButton('Refinement')
        #self.refine_btn.clicked.connect(self._refine)
        line.addWidget(self.select_btn)
        #line.addWidget(self.refine_btn)
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
                                                             '*.mrc')
        self._curr_folder = os.path.dirname(self._file_name)

        if self._file_name is not '':
            self.mrc_fname.setText(self._file_name)
            self.assemble_btn.setEnabled(True)
            self.step_box.setEnabled(True)

    def _update_imview(self):
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
        if self.step_box.text() is '':
            self._downsampling = 10
        else:
            self._downsampling = self.step_box.text()

        if self.mrc_fname.text() is not '':
            self.ops = EM_ops()
            self.ops.parse(self.mrc_fname.text(), int(self._downsampling))
            self.imview.setImage(self.ops.data)
            self.define_btn.setEnabled(True)
            self.rot_transform_btn.setEnabled(True)
            self.show_btn.setChecked(True)
            self.show_btn.setEnabled(True)
            self.transform_btn.setEnabled(False)
            if self.tr_grid_box is not None:
                self.imview.removeItem(self.tr_grid_box)
            if self.grid_box is not None:
                self.imview.removeItem(self.grid_box)
            self.grid_box = None
            self.ops.transformed = False
            self.show_grid_btn.setEnabled(False)

            if self.ops.stacked_data:
                self.select_region_btn.setEnabled(True)
                self.select_btn.setEnabled(True)
            else:
                self.select_region_btn.setEnabled(False)
                self.select_btn.setEnabled(False)
                self.show_assembled_btn.setEnabled(False)
            self.boxes = []
            self.show_grid_btn.setChecked(False)
        else:
            print('You have to choose an .mrc file first!')
        QtWidgets.QApplication.restoreOverrideCursor()

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
                self.show_grid_btn.setEnabled(False)
                self.original_help = False
                self.show_btn.setChecked(True)
                self.original_help = True
                self.show_btn.setEnabled(False)
                self.transform_btn.setEnabled(True)
            else:
                self._hide_boxes()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _show_assembled(self):
        if self.ops is None:
            return
        self.imview.removeItem(self.grid_box)
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
            if file_name is not '':
                self.ops.save_merge(file_name)
        QtWidgets.QApplication.restoreOverrideCursor()

