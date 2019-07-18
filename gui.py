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

        self.boxes = []
        self.box_coordinate = None
        # In each of the following lists, first is for FM image and second for EM image
        self.clicked_points = [[], []]
        self.points_corr = [[], []]
        self.grid_box = [None, None]
        self.tr_grid_box = [None, None]
        self.tr_grid_box_list = [[], []]
        self.tr_matrices = [None, None]

        self.curr_mrc_folder = None
        self.curr_fm_folder = None
        self._init_ui()

    # ---- UI functions

    def _init_ui(self):
        self.resize(1600, 800)
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
        self.fm_imview.scene.sigMouseClicked.connect(lambda evt, par=self.fm_imview: self._imview_clicked(evt, par))
        splitter_images.addWidget(self.fm_imview)

        # -- EM Image view
        self.em_imview = pg.ImageView()
        self.em_imview.ui.roiBtn.hide()
        self.em_imview.ui.menuBtn.hide()
        self.em_imview.scene.sigMouseClicked.connect(lambda evt, par=self.em_imview: self._imview_clicked(evt, par))
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
        self.define_btn.toggled.connect(lambda state, par=self.fm_imview: self._define_grid_toggled(state, par))
        line.addWidget(self.define_btn)
        self.transform_btn = QtWidgets.QPushButton('Transform image', self)
        self.transform_btn.clicked.connect(lambda: self._affine_transform(self.fm_imview))
        line.addWidget(self.transform_btn)
        self.show_btn = QtWidgets.QCheckBox('Show original data', self)
        self.show_btn.setEnabled(False)
        self.show_btn.setChecked(True)
        self.show_btn.stateChanged.connect(lambda state, par=self.fm_imview: self._show_original(state, par))
        line.addWidget(self.show_btn)
        self.show_grid_btn = QtWidgets.QCheckBox('Show grid box',self)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_grid_btn.stateChanged.connect(lambda state, par=self.fm_imview: self._show_grid(state, par))
        line.addWidget(self.show_grid_btn)

        # ---- Align colors
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.peak_btn = QtWidgets.QPushButton('Peak finding', self)
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
        self.transpose = QtWidgets.QCheckBox('Transpose', self)
        self.transpose.stateChanged.connect(self._trans)
        line.addWidget(self.transpose)
        self.rotate = QtWidgets.QCheckBox('Rotate 90 deg', self)
        self.rotate.stateChanged.connect(self._rot)
        line.addWidget(self.rotate)
        vbox.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.select_btn = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(lambda state, par=self.fm_imview: self._define_corr_toggled(state, par))
        line.addWidget(self.select_btn)

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
        self.define_btn_em.toggled.connect(lambda state, par=self.em_imview: self._define_grid_toggled(state, par))
        line.addWidget(self.define_btn_em)
        self.transform_btn_em = QtWidgets.QPushButton('Transform EM image', self)
        self.transform_btn_em.clicked.connect(lambda: self._affine_transform(self.em_imview))
        line.addWidget(self.transform_btn_em)
        self.show_btn_em = QtWidgets.QCheckBox('Show original EM data', self)
        self.show_btn_em.setEnabled(False)
        self.show_btn_em.setChecked(True)
        self.show_btn_em.stateChanged.connect(lambda state, par=self.em_imview: self._show_original(state, par))
        line.addWidget(self.show_btn_em)
        self.show_grid_btn_em = QtWidgets.QCheckBox('Show grid box',self)
        self.show_grid_btn_em.setEnabled(False)
        self.show_grid_btn_em.setChecked(False)
        self.show_grid_btn_em.stateChanged.connect(lambda state, par=self.em_imview: self._show_grid(state, par))
        line.addWidget(self.show_grid_btn_em)
        
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.show_boxes_btn = QtWidgets.QCheckBox('Show boxes',self)
        self.show_boxes_btn.stateChanged.connect(self._show_boxes)
        self.select_region_btn = QtWidgets.QPushButton('Select subregion',self)
        self.select_region_btn.setCheckable(True)
        self.select_region_btn.toggled.connect(lambda state, par=self.em_imview: self._select_box(state,par))
        self.show_assembled_btn = QtWidgets.QCheckBox('Show assembled image',self)
        self.show_assembled_btn.stateChanged.connect(self._show_assembled)
        self.show_assembled_btn.setEnabled(False)
        line.addWidget(self.show_boxes_btn)
        line.addWidget(self.select_region_btn)
        line.addWidget(self.show_assembled_btn) 


        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.select_btn_em = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn_em.setCheckable(True)
        self.select_btn_em.toggled.connect(lambda state, par=self.em_imview: self._define_corr_toggled(state, par))
        line.addWidget(self.select_btn_em)

        # ---- Quit button
        vbox.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        line.addStretch(1)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        line.addWidget(button)

    def _imview_clicked(self, event, parent):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()

        if parent == self.fm_imview:
            if self.fm is None:
                return
            else:
                index = 0
                obj = self.fm
                dbtn = self.define_btn
                selbtn = self.select_btn
                if obj.side_length is None:
                    size = 0.01 * obj.data.shape[0]
                else:
                    size = obj.side_length / 25
                other = self.em_imview
                other_obj = self.assembler

        if parent == self.em_imview:
            if self.assembler is None:
                return
            else:
                index = 1
                obj = self.assembler
                dbtn = self.define_btn_em
                selbtn = self.select_btn_em
                if obj.side_length is None:
                    size = 0.004 * obj.data.shape[0]
                else:
                    size = obj.side_length / 25
                clicked_points = self.clicked_points[index]
                other = self.fm_imview
                other_obj = self.fm
        
        pos = parent.getImageItem().mapFromScene(event.pos())
        pos.setX(pos.x() - size/2)
        pos.setY(pos.y() - size/2)
        item = parent.getImageItem()

        if dbtn.isChecked():
            roi = pg.CircleROI(pos, size, parent=item, movable=False)
            roi.setPen(255,0,0)
            roi.removeHandle(0)
            parent.addItem(roi)
            self.clicked_points[index].append(roi)
        elif selbtn.isChecked():
            point = pg.CircleROI(pos, size, parent=item, movable=True)
            point.setPen(0,255,0)
            point.removeHandle(0)
            parent.addItem(point)
            self.points_corr[index].append(point)

            # Coordinates in clicked image
            shift = obj.transform_shift + [obj.side_length/2]*2
            init = np.array([pos.x(), pos.y(), 1])
            transf = np.dot(self.tr_matrices[index], init)

            cen = other_obj.side_length / 100
            pos = QtCore.QPointF(transf[0]-cen, transf[1]-cen)
            point = pg.CircleROI(pos, 2*cen, parent=other.getImageItem(), movable=False)
            point.setPen(0,255,255)
            point.removeHandle(0)
            other.addItem(point)
            self.points_corr[1-index].append(point)
        elif self.select_region_btn.isChecked():
            self.box_coordinate = pos

    def _define_grid_toggled(self, checked, parent):
        if parent == self.fm_imview:
            tag = 'FM'
            index = 0
            show_grid_btn = self.show_grid_btn
        else:
            tag = 'EM'
            index = 1
            show_grid_btn = self.show_grid_btn_em

        if checked:
            print('Defining grid on %s image: Click on corners'%tag)
            if self.grid_box[index] is not None:
                parent.removeItem(self.grid_box[index])
                self.grid_box[index] = None
        else:
            print('Done defining grid on %s image: Manually adjust fine positions'%tag)
            positions = [c.pos() for c in self.clicked_points[index]]
            sizes = [c.size()[0] for c in self.clicked_points[index]]
            for pos, s in zip(positions, sizes):
                pos.setX(pos.x() + s/2)
                pos.setY(pos.y() + s/2)
            self.grid_box[index] = pg.PolyLineROI(positions, closed=True, movable=False)
            parent.addItem(self.grid_box[index])
            [parent.removeItem(roi) for roi in self.clicked_points[index]]
            self.clicked_points[index] = []
            show_grid_btn.setEnabled(True)
            show_grid_btn.setChecked(True)

    def _define_corr_toggled(self, checked, parent):
        if parent == self.fm_imview:
            tag = 'FM'
            index = 0
            obj = self.fm
            other_obj = self.assembler
        else:
            tag = 'EM'
            index = 1
            obj = self.assembler
            other_obj = self.fm
        if obj is None:
            return

        if checked:
            print('Select points of interest on %s image'%tag)
            if len(self.points_corr[index]) != 0:
                [parent.removeItem(point) for point in self.points_corr[index]]
                self.points_corr[index] = []
            self.tr_matrices[index] = obj.get_transform(obj.points, other_obj.points)
        else:
            print('Done selecting points of interest on %s image'%tag)

    def _affine_transform(self, parent):
        if parent == self.fm_imview:
            tag = 'FM'
            index = 0
            obj = self.fm
            show_btn = self.show_btn
        else:
            tag = 'EM'
            index = 1
            obj = self.assembler
            show_btn = self.show_btn_em

        if self.grid_box[index] is not None:
            print('Performing affine transformation on %s image'%tag)
            if self.tr_grid_box[index] is not None:
                points_obj = self.tr_grid_box[index].getState()['points']
            else:
                points_obj = self.grid_box[index].getState()['points']
            points = np.array([list((point[0], point[1])) for point in points_obj])

            obj.calc_transform(points)
            if obj.data is None:
                obj.toggle_original()
            else:
                obj.toggle_original(True)
            self._update_fm_imview() if index == 0 else self._update_em_imview()

            parent.removeItem(self.grid_box[index])
            if self.tr_grid_box[index] is not None:
                parent.removeItem(self.tr_grid_box[index])
            show_btn.setEnabled(True)
            show_btn.setChecked(False)

            for i in range(obj.new_points.shape[0]):
                roi = pg.CircleROI(obj.new_points[i], 5, parent=parent.getImageItem(), movable=False)
                roi.removeHandle(0)
                parent.addItem(roi)
                self.tr_grid_box_list[index].append(roi)
            
            positions = [c.pos() for c in self.tr_grid_box_list[index]]
            self.tr_grid_box[index] = pg.PolyLineROI(positions, closed=True, movable=False)
            parent.addItem(self.tr_grid_box[index])
            [parent.removeItem(roi) for roi in self.tr_grid_box_list[index]]
            self.tr_grid_box_list[index] = []
        else:
            print('Define grid box on %s image first!'%tag)

    def _show_original(self, state, parent):
        if parent == self.fm_imview:
            index = 0
            obj = self.fm
            updater = self._update_fm_imview
            orig_btn = self.show_btn
            grid_btn = self.show_grid_btn
        else:
            index = 1
            obj = self.assembler
            updater = self._update_em_imview
            orig_btn = self.show_btn_em
            grid_btn = self.show_grid_btn_em

        if obj is not None:
            obj.toggle_original(state==0)
            if grid_btn.isChecked():
                if orig_btn.isChecked():
                    parent.removeItem(self.tr_grid_box[index])
                else:
                    parent.removeItem(self.grid_box[index])
            updater()
            if orig_btn.isChecked():
                if grid_btn.isChecked():
                    parent.addItem(self.grid_box[index])
            else:
                if grid_btn.isChecked() and self.tr_grid_box[index] is not None:
                    parent.addItem(self.tr_grid_box[index])

    def _show_grid(self, state, parent):
        if parent == self.fm_imview:
            index = 0
            obj = self.fm
            orig_btn = self.show_btn
            grid_btn = self.show_grid_btn
        else:
            index = 1
            obj = self.assembler
            orig_btn = self.show_btn_em
            grid_btn = self.show_grid_btn_em

        if orig_btn.isChecked():
            if grid_btn.isChecked():
                self._recalc_grid(orig_btn.isChecked())
                parent.addItem(self.grid_box[index])
            else:
                parent.removeItem(self.grid_box[index])
        else:
            if obj is not None:
                if grid_btn.isChecked():
                    self._recalc_grid(orig_btn.isChecked())
                    parent.addItem(self.tr_grid_box[index])
                else:
                    parent.removeItem(self.tr_grid_box[index])

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

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             'Select FM file',
                                                             self.curr_fm_folder,
                                                             '*.lif')
        if file_name is not '':
            self.curr_fm_folder = os.path.dirname(file_name)
            self._parse_fm_images(file_name)

    def _parse_fm_images(self, file_name):
        self.fm = fm_operations.FM_ops()
        self.fm.parse(file_name, z=0)
        self.num_channels = self.fm.num_channels

        if file_name is not '':
            self.fm_fname.setText(file_name + ' [0/%d]'%self.fm.num_channels)

        self.fm_imview.setImage(self.fm.data, levels=(self.fm.data.min(), self.fm.data.mean()*2))

    def _update_fm_imview(self):
        vr = self.fm_imview.getImageItem().getViewBox().targetRect()
        levels = self.fm_imview.getHistogramWidget().item.getLevels()

        self.fm_imview.setImage(self.fm.data, levels=levels)
        self.fm_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _fliph(self, state):
        self.fm.flip_horizontal(state == QtCore.Qt.Checked)
        self._update_fm_imview()

    def _flipv(self, state):
        self.fm.flip_vertical(state == QtCore.Qt.Checked)
        self._update_fm_imview()

    def _trans(self, state):
        self.fm.transpose(state == QtCore.Qt.Checked)
        self._update_fm_imview()

    def _rot(self, state):
        self.fm.rotate_clockwise(state == QtCore.Qt.Checked)
        self._update_fm_imview()

    def _recalc_grid(self, orig=True):
        if self.fm.orig_points is None:
            return
        if orig:
            print('Recalc orig')
            self.fm._update_data()
            pos = [QtCore.QPointF(point[0], point[1]) for point in self.fm.points]
            self.grid_box[0] = pg.PolyLineROI(pos, closed=True, movable=False)
            print(pos[0].x(), pos[0].y())
        else:
            print('Recalc transf')
            self.fm._update_data()
            pos = [QtCore.QPointF(point[0], point[1]) for point in self.fm.points]
            self.tr_grid_box[0] = pg.PolyLineROI(pos, closed=True, movable=False)
            print(pos[0].x(), pos[0].y())

    def _next_file(self):
        if self.fm is None:
            print('Pick FM image first')
            return
        self.ind = (self.ind + 1 + self.num_channels) % self.num_channels
        self.fm.parse(fname=self.fm.old_fname, z=self.ind)
        self._update_fm_imview()
        fname, indstr = self.fm_fname.text().split()
        self.fm_fname.setText(fname + ' [%d/%d]'%(self.ind, self.num_channels))

    def _prev_file(self):
        if self.fm is None:
            print('Pick FM image first')
            return
        self.ind = (self.ind - 1 + self.num_channels) % self.num_channels
        self.fm.parse(fname=self.fm.old_fname, z=self.ind)
        self._update_fm_imview()
        fname, indstr = self.fm_fname.text().split()
        self.fm_fname.setText(fname + ' [%d/%d]'%(self.ind, self.num_channels))

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
        '''
        new_list = align_fm.calc_shift(self.flist, self.fm.data)

        self.fselector.addItems(new_list)
        self.fselector.currentIndexChanged.connect(self._file_changed)

        data_shifted = [np.array(Image.open(fname)) for fname in new_list]
        for i in range(len(data_shifted)):
            self.fm.data.append(data_shifted[i])

        self.align_btn.setEnabled(False)
        '''

    # ---- EM functions

    def _load_mrc(self):
        if self.curr_mrc_folder is None:
            self.curr_mrc_folder = os.getcwd()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             'Select .mrc file',
                                                             self.curr_mrc_folder,
                                                             '*.mrc')
        self.curr_mrc_folder = os.path.dirname(file_name)

        if file_name is not '':
            self.mrc_fname.setText(file_name)
            #self._assemble_mrc()

    def _update_em_imview(self):
        vr = self.em_imview.getImageItem().getViewBox().targetRect()
        levels = self.em_imview.getHistogramWidget().item.getLevels()

        self.em_imview.setImage(self.assembler.data, levels=levels)
        self.em_imview.getImageItem().getViewBox().setRange(vr, padding=0)

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
    
    def _show_boxes(self):
        if self.show_boxes_btn.isChecked():
            if self.assembler is not None:
                if len(self.boxes) == 0:
                    for i in range(len(self.assembler.pos_x)):
                        roi = pg.RectROI([self.assembler.pos_x[i],self.assembler.pos_y[i]],
                                        [self.assembler.stacked_data.shape[1],self.assembler.stacked_data.shape[2]],
                                        movable=False)
                        #roi.removeHandle(0)
                        self.boxes.append(roi)
                        self.em_imview.addItem(roi)
                else:
                    [self.em_imview.addItem(box) for box in self.boxes]
        else:
            [self.em_imview.removeItem(box) for box in self.boxes]

    def _select_box(self,state,parent):
        if self.select_region_btn.isChecked():
            print('Select box')
            parent.setImage(self.assembler.data)
        else:
            if self.box_coordinate is not None:
                points_obj = (self.box_coordinate.x(),self.box_coordinate.y())
                print(points_obj)
                self.assembler.select_region(np.array(points_obj))
                parent.setImage(self.assembler.data)
                #[parent.removeItem(box) for box in self.boxes]
                self.show_boxes_btn.setChecked(False)
                self.box_coordinate = None
                self.show_assembled_btn.setEnabled(True)
                self.show_assembled_btn.setChecked(False)

    def _show_assembled(self):
        if self.show_assembled_btn.isChecked():
            assembled = True
        else:
            assembled = False
        if self.assembler is not None:
            self.assembler.toggle_region(assembled)
            self.em_imview.setImage(self.assembler.data)
    
    def _save_mrc_montage(self):
        if self.assembler is None:
            print('No montage to save')
        else:
            if self.curr_mrc_folder is None:
                self.curr_mrc_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Binned Montage', self.curr_mrc_folder, '*.mrc')
            self.curr_mrc_folder = os.path.dirname(file_name)
            if file_name is not '':
                self.assembler.save_merge(file_name)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    gui = GUI()
    sys.exit(app.exec_())
