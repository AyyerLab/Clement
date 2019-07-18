#!/usr/bin/env python

import sys
import os
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg

import assemble
import align_fm
import affine_transform
import fm_operations

warnings.simplefilter('ignore', category=FutureWarning)

class GUI(QtGui.QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        self.fm = None
        self.assembler = None
        self.ind = 0
        self.clicked_points = []
        self.clicked_points_em = []
        self.points_corr = []
        self.points_corr_em = []
        self.grid_box = None
        self.grid_box_em = None
        self.grid_box_transformed = None
        self.grid_box_transformed_list = []
        self.curr_mrc_folder = None
        self.curr_fm_folder = None
        self._init_ui()

    def _init_ui(self):
        self.resize(1000,800)
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        # Menu bar
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        # -- File menu
        filemenu = menubar.addMenu('&File')
        action = QtWidgets.QAction('Load &FM image(s)', self)
        action.triggered.connect(self._load_fm_images)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Load &EM montage', self)
        action.triggered.connect(self._load_mrc)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Save binned montage', self)
        action.triggered.connect(self._save_mrc_montage)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Quit', self)
        action.triggered.connect(self.close)
        filemenu.addAction(action)

        # Image views
        splitter_images = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter_images, stretch=1)

        # -- FM Image view
        self.fm_imview = pg.ImageView()
        self.fm_imview.ui.roiBtn.hide()
        self.fm_imview.ui.menuBtn.hide()
        self.fm_imview.scene.sigMouseClicked.connect(self._imview_clicked)
        splitter_images.addWidget(self.fm_imview)

        # -- EM Image view
        self.em_imview = pg.ImageView()
        self.em_imview.ui.roiBtn.hide()
        self.em_imview.ui.menuBtn.hide()
        self.em_imview.scene.sigMouseClicked.connect(self._imview_clicked_em)
        splitter_images.addWidget(self.em_imview)

        # Options
        options = QtWidgets.QHBoxLayout()
        layout.addLayout(options)

        self._init_fm_options(options)
        self._init_em_options(options)

        self.show()

    def _init_fm_options(self, parent_layout):
        vbox = QtWidgets.QVBoxLayout()
        parent_layout.addLayout(vbox)

        # ---- Select file
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        button = QtWidgets.QPushButton('FM image:', self)
        button.clicked.connect(self._load_fm_images)
        line.addWidget(button)
        self.fm_fname = QtWidgets.QLabel(self)
        line.addWidget(self.fm_fname, stretch=1)
        button = QtWidgets.QPushButton('\u2190', self)
        button.setFixedWidth(16)
        button.clicked.connect(self._prev_file)
        line.addWidget(button)
        button = QtWidgets.QPushButton('\u2192', self)
        button.setFixedWidth(16)
        button.clicked.connect(self._next_file)
        line.addWidget(button)

        # ---- Define and align to grid
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.define_btn = QtWidgets.QPushButton('Define Grid', self)
        self.define_btn.setCheckable(True)
        self.define_btn.toggled.connect(self._define_toggled)
        line.addWidget(self.define_btn)
        self.transform_btn = QtWidgets.QPushButton('Transform image', self)
        self.transform_btn.clicked.connect(self._affine_transform)
        line.addWidget(self.transform_btn)
        self.show_btn = QtWidgets.QCheckBox('Show original data',self)
        self.show_btn.stateChanged.connect(self._show_original)
        self.show_btn.setEnabled(False)
        self.show_btn.setChecked(True)
        self.show_grid_btn = QtWidgets.QCheckBox('Show grid box',self)
        self.show_grid_btn.stateChanged.connect(self._show_grid)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        line.addWidget(self.show_btn)
        line.addWidget(self.show_grid_btn)

        # ---- Align colors
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.peak_btn = QtWidgets.QPushButton('Peak finding',self)
        self.peak_btn.clicked.connect(self._find_peaks)
        self.align_btn = QtWidgets.QPushButton('Align color channels', self)
        self.align_btn.clicked.connect(self._calc_shift)
        line.addWidget(self.peak_btn)
        line.addWidget(self.align_btn)

        # ---- Flips and rotates
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.fliph = QtWidgets.QCheckBox('Flip horizontally', self)
        self.fliph.stateChanged.connect(self._fliph)
        line.addWidget(self.fliph)
        self.flipv = QtWidgets.QCheckBox('Flip vertically', self)
        self.flipv.stateChanged.connect(self._flipv)
        line.addWidget(self.flipv)
        self.transpose = QtWidgets.QCheckBox('Transpose',self)
        self.transpose.stateChanged.connect(self._trans)
        line.addWidget(self.transpose)
        self.rotate = QtWidgets.QCheckBox('Rotate 90 deg',self)
        self.rotate.stateChanged.connect(self._rot)
        line.addWidget(self.rotate)
        vbox.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.select_points = QtWidgets.QPushButton('Select points of interest',self)
        self.select_points.setCheckable(True)
        self.select_points.toggled.connect(self._define_toggeled_corr)
        line.addWidget(self.select_points)

    def _init_em_options(self, parent_layout):
        vbox = QtWidgets.QVBoxLayout()
        parent_layout.addLayout(vbox)

        # ---- Assemble montage
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        button = QtWidgets.QPushButton('EM Montage:', self)
        button.clicked.connect(self._load_mrc)
        line.addWidget(button)
        self.mrc_fname = QtWidgets.QLabel(self)
        line.addWidget(self.mrc_fname, stretch=1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        step_label = QtWidgets.QLabel(self)
        step_label.setText('Downsampling factor:')
        self.step_box = QtWidgets.QLineEdit(self)
        self.step_box.setText('100')
        line.addWidget(step_label)
        line.addWidget(self.step_box)
        line.addStretch(1)
        button = QtWidgets.QPushButton('Assemble', self)
        button.clicked.connect(self._assemble_mrc)
        line.addWidget(button)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.define_btn_em = QtWidgets.QPushButton('Define EM Grid', self)
        self.define_btn_em.setCheckable(True)
        self.define_btn_em.toggled.connect(self._define_toggled_em)
        line.addWidget(self.define_btn_em)
        self.transform_btn_em = QtWidgets.QPushButton('Transform EM image', self)
        self.transform_btn_em.clicked.connect(self._affine_transform_em)
        line.addWidget(self.transform_btn_em)
        self.show_btn_em = QtWidgets.QCheckBox('Show original EM data',self)
        self.show_btn_em.stateChanged.connect(self._show_original_em)
        self.show_btn_em.setEnabled(False)
        self.show_btn_em.setChecked(True)
        line.addWidget(self.show_btn_em)


        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.select_points_em = QtWidgets.QPushButton('Select points of interest',self)
        self.select_points_em.setCheckable(True)
        self.select_points_em.toggled.connect(self._define_toggeled_corr_em)
        line.addWidget(self.select_points_em)

        # ---- Quit button
        vbox.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        line.addStretch(1)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        line.addWidget(button)

    def _imview_clicked(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
        if self.define_btn.isChecked():
            roi = pg.CircleROI(self.fm_imview.getImageItem().mapFromScene(event.pos()),
                               5,
                               parent=self.fm_imview.getImageItem(),
                               movable=False)
            roi.removeHandle(0)
            self.fm_imview.addItem(roi)
            self.clicked_points.append(roi)
        elif self.select_points.isChecked():
            point = pg.CircleROI(self.fm_imview.getImageItem().mapFromScene(event.pos()),
                               100,
                               parent=self.fm_imview.getImageItem(),
                               movable=True)
            point.removeHandle(0)
            self.fm_imview.addItem(point)
            self.points_corr.append(point)
        else:
            pass

    def _imview_clicked_em(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
        if self.define_btn_em.isChecked():
            roi = pg.CircleROI(self.em_imview.getImageItem().mapFromScene(event.pos()),
                               5,
                               parent=self.em_imview.getImageItem(),
                               movable=False)
            roi.removeHandle(0)
            self.em_imview.addItem(roi)
            self.clicked_points_em.append(roi)
        elif self.select_points_em.isChecked():
            point = pg.CircleROI(self.em_imview.getImageItem().mapFromScene(event.pos()),
                               100,
                               parent=self.em_imview.getImageItem(),
                               movable=True)
            point.removeHandle(0)
            self.em_imview.addItem(point)
            self.points_corr_em.append(point)
        else:
            pass

    def keyPressEvent(self, event):
        key = event.key()
        mod = int(event.modifiers())

        if QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+P'):
            self._prev_file()
        elif QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+N'):
            self._next_file()
        elif QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+W'):
            self.close()
        else:
            event.ignore()

    def closeEvent(self, event):
        fm_operations.javabridge.kill_vm()
        event.accept()

    # ---- FM functions

    def _load_fm_images(self):
        if self.curr_fm_folder is None:
            #self.curr_fm_folder = os.getcwd()
            self.curr_fm_folder = '/beegfs/cssb/user/kaufmanr/cryoCLEM-software/clem_dataset/'

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select FM file', self.curr_fm_folder , '*.lif')
        self.curr_fm_folder = os.path.dirname(file_name)

        if file_name is not '':
            self.fm_fname.setText(file_name)

        self.fm = fm_operations.FM_ops()
        self.fm.parse(self.fm_fname.text(), z=0)
        self.num_channels = self.fm.num_channels

        self.fm_imview.setImage(self.fm.data, levels=(self.fm.data.min(), self.fm.data.mean()*2))

    def _update_fm_imview(self):
        vr = self.fm_imview.getImageItem().getViewBox().targetRect()
        levels = self.fm_imview.getHistogramWidget().item.getLevels()

        self.fm_imview.setImage(self.fm.data, levels=levels)
        self.fm_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _define_toggled(self, checked):
        if checked:
            print('Defining grid: Click on corners')
            if self.grid_box is not None:
                self.fm_imview.removeItem(self.grid_box)
                self.grid_box = None
                if self.fm is not None:
                    self.fm_imview.removeItem(self.grid_box_transformed)
        else:
            print('Done defining grid: Manually adjust fine positions')
            self.grid_box = pg.PolyLineROI([c.pos() for c in self.clicked_points], closed=True, movable=False)
            self.fm_imview.addItem(self.grid_box)
            print(self.grid_box)
            [self.fm_imview.removeItem(roi) for roi in self.clicked_points]
            self.clicked_points = []
            self.show_grid_btn.setEnabled(True)
            self.show_grid_btn.setChecked(True)

    def _define_toggeled_corr(self, checked):
        if checked:
            print('Select points of interest')
            if len(self.points_corr) != 0:
                [self.fm_imview.removeItem(point) for point in self.points_corr]
                [self.em_imview.removeItem(point) for point in self.points_corr]
                self.points_corr = []
            if len(self.points_corr_em) != 0:
                [self.em_imview.removeItem(point) for point in self.points_corr_em]
                [self.em_imview.removeItem(point) for point in self.points_corr_em]
                self.points_corr_em = []
        else:
            print('Done selecting points of interest')
            if self.assembler is not None:
                [self.em_imview.addItem(point) for point in self.points_corr]
                [self.fm_imview.addItem(point) for point in self.points_corr]

    def _show_original(self, state):
        if self.fm is not None:
            self.fm.toggle_original(state==0)
            self._update_fm_imview() 
            if self.show_btn.isChecked():
                self.fm_imview.removeItem(self.grid_box_transformed)
                if self.show_grid_btn.isChecked():
                    self.fm_imview.addItem(self.grid_box)
            else:
                self.fm_imview.removeItem(self.grid_box)
                if self.grid_box_transformed is not None:
                    self.fm_imview.addItem(self.grid_box_transformed)
            
    def _show_grid(self,state):
        if self.show_btn.isChecked():
            if self.show_grid_btn.isChecked():
                self.fm_imview.addItem(self.grid_box)
            else:
                self.fm_imview.removeItem(self.grid_box)
        else:
            if self.fm is not None:
                if self.show_grid_btn.isChecked():
                    self.fm_imview.addItem(self.grid_box_transformed)
                else:
                    self.fm_imview.removeItem(self.grid_box_transformed)

    def _fliph(self,state):
        self.fm.flip_horizontal(state == QtCore.Qt.Checked)
        self._update_fm_imview()

    def _flipv(self,state):
        self.fm.flip_vertical(state == QtCore.Qt.Checked)
        self._update_fm_imview()

    def _trans(self,state):
        self.fm.transpose(state == QtCore.Qt.Checked)
        self._update_fm_imview()

    def _rot(self,state):
        self.fm.rotate_clockwise(state == QtCore.Qt.Checked)
        self._update_fm_imview()

    def _next_file(self):
        if self.fm is None:
            print('Pick FM image first')
            return
        self.ind = (self.ind + 1 + self.num_channels) % self.num_channels
        self.fm.parse(fname=self.fm.old_fname, z=self.ind)
        self._update_fm_imview()

    def _prev_file(self):
        if self.fm is None:
            print('Pick FM image first')
            return
        self.ind = (self.ind - 1 + self.num_channels) % self.num_channels
        self.fm.parse(fname=self.fm.old_fname, z=self.ind)
        self._update_fm_imview()

    def _find_peaks(self):
        if self.fm is not None:
            self.fm.peak_finding()
        #print(self.fm.diff_list)
        else:
            print('You have to select the data first!')

    def _calc_shift(self):
        print('Align color channels')
        return
        # TODO fix this
        new_list = align_fm.calc_shift(self.flist, self.fm.data)

        self.fselector.addItems(new_list)
        self.fselector.currentIndexChanged.connect(self._file_changed)

        data_shifted = [np.array(Image.open(fname)) for fname in new_list]
        for i in range(len(data_shifted)):
            self.fm.data.append(data_shifted[i])

        self.align_btn.setEnabled(False)

    def _affine_transform(self):
        if self.grid_box is not None:
            print('Perform affine transformation')
            if self.grid_box_transformed is not None:
                points_obj = self.grid_box_transformed.getState()['points']
            else:
                points_obj = self.grid_box.getState()['points']
            points = np.array([list((point[0],point[1])) for point in points_obj])

            self.fm.calc_transform(points)
            self.fm.toggle_original()
            self._update_fm_imview()

            self.fm_imview.removeItem(self.grid_box)
            if self.grid_box_transformed is not None:
                self.fm_imview.removeItem(self.grid_box_transformed)
            self.show_btn.setEnabled(True)
            self.show_btn.setChecked(False)
            for i in range(self.fm.new_points.shape[0]):
                roi = pg.CircleROI(self.fm.new_points[i],
                                    5,
                                    parent=self.fm_imview.getImageItem(),
                                    movable=False)
                roi.removeHandle(0)
                self.fm_imview.addItem(roi)
                self.grid_box_transformed_list.append(roi)
            
            self.grid_box_transformed = pg.PolyLineROI([c.pos() for c in self.grid_box_transformed_list], closed=True, movable=False)
            self.fm_imview.addItem(self.grid_box_transformed)
            [self.fm_imview.removeItem(roi) for roi in self.grid_box_transformed_list]
            self.grid_box_transformed_list = []

        else:
            print('Define grid box first!')

    # ---- EM functions

    def _load_mrc(self):
        if self.curr_mrc_folder is None:
            self.curr_mrc_folder = os.getcwd()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select .mrc file', self.curr_mrc_folder , '*.mrc')
        self.curr_mrc_folder = os.path.dirname(file_name)

        if file_name is not '':
            self.mrc_fname.setText(file_name)
            #self._assemble_mrc()

    def _update_em_imview(self):
        vr = self.em_imview.getImageItem().getViewBox().targetRect()
        levels = self.em_imview.getHistogramWidget().item.getLevels()

        self.em_imview.setImage(self.assembler.data, levels=levels)
        self.em_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _define_toggled_em(self, checked):
        if checked:
            print('Defining grid: Click on corners')
            if self.grid_box_em is not None:
                self.em_imview.removeItem(self.grid_box_em)
                self.grid_box_em = None
        else:
            print('Done defining grid: Manually adjust fine positions')
            self.grid_box_em = pg.PolyLineROI([c.pos() for c in self.clicked_points_em], closed=True, movable=False)
            self.em_imview.addItem(self.grid_box_em)
            print(self.grid_box_em)
            [self.em_imview.removeItem(roi) for roi in self.clicked_points_em]
            self.clicked_points_em = []

    def _define_toggeled_corr_em(self, checked):
        if checked:
            print('Select points of interest')
            if len(self.points_corr_em) != 0:
                [self.em_imview.removeItem(point) for point in self.points_corr_em]
                [self.fm_imview.removeItem(point) for point in self.points_corr_em]
                self.points_corr_em = []
            if len(self.points_corr) != 0:
                [self.fm_imview.removeItem(point) for point in self.points_corr]
                [self.em_imview.removeItem(point) for point in self.points_corr]
                self.points_corr = []
        else:
            print('Done selecting points of interest')
            if self.assembler is not None:
                [self.fm_imview.addItem(i) for i in self.points_corr_em]

    def _show_original_em(self, state):
        if self.assembler is not None:
            self.assembler.toggle_original(state==0)
            self._update_em_imview()

    def _assemble_mrc(self):
        if self.step_box.text() is '':
            step = 100
        else:
            step = self.step_box.text()

        if self.mrc_fname.text() is not '':
            self.assembler = assemble.Assembler(step=int(step))
            self.assembler.parse(self.mrc_fname.text())
            self.assembler.assemble()
            print('Done')
            self.em_imview.setImage(self.assembler.data)
        else:
            print('You have to choose .mrc file first!')

    def _save_mrc_montage(self):
        if self.assembler is None:
            print('No montage to save')
        else:
            if self.curr_mrc_folder is None:
                self.curr_mrc_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Binned Montage', self.curr_mrc_folder , '*.mrc')
            self.curr_mrc_folder = os.path.dirname(file_name)
            if file_name is not '':
                self.assembler.save_merge(file_name)

    def _affine_transform_em(self):
        if self.grid_box_em is not None:
            print('Perform affine transformation')
            points_obj = self.grid_box_em.getState()['points']
            points = np.array([list((point[0],point[1])) for point in points_obj])

            self.assembler.affine_transform(points)
            self.assembler.toggle_original()
            self._update_em_imview()

            self.em_imview.removeItem(self.grid_box_em)
            self.grid_box_em = None
            self.show_btn_em.setEnabled(True)
            self.show_btn_em.setChecked(False)
        else:
            print('Define grid box first!')

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    gui = GUI()
    sys.exit(app.exec_())
