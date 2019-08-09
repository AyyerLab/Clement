#!/usr/bin/env python

import sys
import os
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
import matplotlib.colors as cm
import matplotlib.pyplot as plt

import em_operations
import align_fm
import affine_transform
import fm_operations

warnings.simplefilter('ignore', category=FutureWarning)

class GUI(QtGui.QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        self.fm = None
        self.em = None
        self.ind = 0
         
        self.channels = [True, True, True, True] 
        self.colors = ['#ff0000', '#00ff00', '#0000ff', '#808080']
        self.color_data = None
        self.overlay = True
        
        self.box_points = []
        self.boxes = []
        self.tr_box_points = []
        self.tr_boxes = []
        self.box_coordinate = None
        
        # In each of the following lists, first is for FM image and second for EM image
        self.clicked_points = [[], []]
        self.points_corr = [[], []]
        self.grid_box = [None, None]
        self.tr_grid_box = [None, None]
        self.tr_matrices = [None, None]

        self.curr_mrc_folder = None
        self.curr_fm_folder = None

        self.settings = QtCore.QSettings('MPSD-CNI', 'CLEMGui', self)
        self.colors = self.settings.value('channel_colors', defaultValue=['#ff0000', '#00ff00', '#0000ff', '#808080'])
        self._init_ui()

    # ---- UI functions

    def _init_ui(self):
        geom = self.settings.value('geometry')
        if geom is None:
            self.resize(1600, 800)
        else:
            self.setGeometry(geom)
        theme = self.settings.value('theme')
        if theme is None:
            theme = 'none'
        self._set_theme(theme)

        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
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

        # -- Theme menu
        thememenu = menubar.addMenu('&Theme')
        agroup = QtWidgets.QActionGroup(self)
        agroup.setExclusive(True)
        action = QtWidgets.QAction('None', self)
        action.triggered.connect(lambda: self._set_theme('none'))
        thememenu.addAction(action)
        agroup.addAction(action)
        action = QtWidgets.QAction('Dark', self)
        action.triggered.connect(lambda: self._set_theme('dark'))
        thememenu.addAction(action)
        agroup.addAction(action)

        # Image views
        splitter_images = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter_images, stretch=1)

        # -- FM Image view
        #pg.setConfigOption('background', '#1d262a')
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
        options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(options)

        self._init_fm_options(options)
        self._init_em_options(options)

        self.show()

    def _init_fm_options(self, parent_layout):
        vbox = QtWidgets.QVBoxLayout()
        parent_layout.addLayout(vbox)

        # ---- Select file or show max projection
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        button = QtWidgets.QPushButton('FM image:', self)
        button.clicked.connect(self._load_fm_images)
        line.addWidget(button)
        self.fm_fname = QtWidgets.QLabel(self)
        line.addWidget(self.fm_fname, stretch=1)
        self.max_proj_btn = QtWidgets.QCheckBox('Max projection')
        self.max_proj_btn.stateChanged.connect(self._show_max_projection)
        line.addWidget(self.max_proj_btn) 
        self.prev_btn = QtWidgets.QPushButton('\u2190', self)
        self.prev_btn.setFixedWidth(32)
        self.prev_btn.clicked.connect(self._prev_file)
        line.addWidget(self.prev_btn)
        self.next_btn = QtWidgets.QPushButton('\u2192', self)
        self.next_btn.setFixedWidth(32)
        self.next_btn.clicked.connect(self._next_file)
        line.addWidget(self.next_btn)
        
        # ---- Select channels
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Colors:', self)
        line.addWidget(label)
        self.channel1_btn = QtWidgets.QCheckBox(' ',self)
        self.channel2_btn = QtWidgets.QCheckBox(' ',self)
        self.channel3_btn = QtWidgets.QCheckBox(' ',self)
        self.channel4_btn = QtWidgets.QCheckBox(' ',self)
        self.overlay_btn = QtWidgets.QCheckBox('Overlay',self)
        self.channel1_btn.stateChanged.connect(lambda state, channel=0: self._show_channels(state,channel))
        self.channel2_btn.stateChanged.connect(lambda state, channel=1: self._show_channels(state,channel))
        self.channel3_btn.stateChanged.connect(lambda state, channel=2: self._show_channels(state,channel))
        self.channel4_btn.stateChanged.connect(lambda state, channel=3: self._show_channels(state,channel))
        self.overlay_btn.stateChanged.connect(self._show_overlay)
        self.channel1_btn.setChecked(True)
        self.channel2_btn.setChecked(True)
        self.channel3_btn.setChecked(True)
        self.channel4_btn.setChecked(True)
        self.overlay_btn.setChecked(True)

        self.c1_btn = QtWidgets.QPushButton(' ', self)
        self.c1_btn.clicked.connect(lambda: self._sel_color(0, self.c1_btn))
        width = self.c1_btn.fontMetrics().boundingRect(' ').width() + 24
        self.c2_btn.setFixedWidth(width)
        self.c1_btn.setMaximumHeight(width)
        self.c1_btn.setStyleSheet('background-color: {}'.format(self.colors[0]))
        self.c2_btn = QtWidgets.QPushButton(' ', self)
        self.c2_btn.clicked.connect(lambda: self._sel_color(1, self.c2_btn))
        self.c2_btn.setMaximumHeight(width)
        self.c2_btn.setFixedWidth(width)
        self.c2_btn.setStyleSheet('background-color: {}'.format(self.colors[1]))
        self.c3_btn = QtWidgets.QPushButton(' ', self)
        self.c3_btn.setMaximumHeight(width)
        self.c3_btn.setFixedWidth(width)
        self.c3_btn.clicked.connect(lambda: self._sel_color(2, self.c3_btn))
        self.c3_btn.setStyleSheet('background-color: {}'.format(self.colors[2]))
        self.c4_btn = QtWidgets.QPushButton(' ', self)
        self.c4_btn.setMaximumHeight(width)
        self.c4_btn.setFixedWidth(width)
        self.c4_btn.clicked.connect(lambda: self._sel_color(3, self.c4_btn))
        self.c4_btn.setStyleSheet('background-color: {}'.format(self.colors[3]))

        line.addWidget(self.c1_btn)
        line.addWidget(self.channel1_btn)
        line.addWidget(self.c2_btn)
        line.addWidget(self.channel2_btn) 
        line.addWidget(self.c3_btn)
        line.addWidget(self.channel3_btn)
        line.addWidget(self.c4_btn)
        line.addWidget(self.channel4_btn)
        line.addWidget(self.overlay_btn)
        line.addStretch(1)

        # ---- Flips and rotates
        label = QtWidgets.QLabel('Flips:', self)
        line.addWidget(label)

        self.fliph = QtWidgets.QPushButton('\u2345', self)
        width = self.fliph.fontMetrics().boundingRect(' ').width() + 25
        font = self.fliph.font()
        font.setPointSize(24)
        self.fliph.setFixedWidth(width)
        self.fliph.setFixedHeight(width)
        self.fliph.setCheckable(True)
        self.fliph.setFont(font)
        self.fliph.toggled.connect(self._fliph)
        line.addWidget(self.fliph)

        self.flipv = QtWidgets.QPushButton('\u2356', self)
        self.flipv.setCheckable(True)
        self.flipv.setFixedWidth(width)
        self.flipv.setFixedHeight(width)
        self.flipv.setFont(font)
        self.flipv.toggled.connect(self._flipv)
        line.addWidget(self.flipv)

        self.transpose = QtWidgets.QPushButton('\u292f', self)
        self.transpose.setCheckable(True)
        self.transpose.setFixedWidth(width)
        self.transpose.setFixedHeight(width)
        font.setPointSize(20)
        self.transpose.setFont(font)
        self.transpose.toggled.connect(self._trans)
        line.addWidget(self.transpose)

        self.rotate = QtWidgets.QPushButton('\u293e', self)
        self.rotate.setCheckable(True)
        self.rotate.setFixedWidth(width)
        self.rotate.setFixedHeight(width)
        self.rotate.setFont(font)
        self.rotate.toggled.connect(self._rot)
        line.addWidget(self.rotate)
        line.addStretch(1)

        # ---- Define and align to grid
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Grid transform:', self)
        line.addWidget(label)
        self.define_btn = QtWidgets.QPushButton('Define Grid', self)
        self.define_btn.setCheckable(True)
        self.define_btn.toggled.connect(lambda state, par=self.fm_imview: self._define_grid_toggled(state, par))
        line.addWidget(self.define_btn)
        self.transform_btn = QtWidgets.QPushButton('Transform image', self)
        self.transform_btn.clicked.connect(lambda: self._affine_transform(self.fm_imview))
        line.addWidget(self.transform_btn)
        self.rot_transform_btn = QtWidgets.QCheckBox('Disable Shearing', self)
        self.rot_transform_btn.stateChanged.connect(lambda state, par=self.fm_imview: self._allow_rotation_only(state, par))
        line.addWidget(self.rot_transform_btn)              
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
        line.addStretch(1)

        # ---- Align colors
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Peak finding:', self)
        line.addWidget(label)
        self.peak_btn = QtWidgets.QPushButton('Find peaks', self)
        self.peak_btn.clicked.connect(self._find_peaks)
        self.align_btn = QtWidgets.QPushButton('Align color channels', self)
        self.align_btn.clicked.connect(self._calc_shift)
        line.addWidget(self.peak_btn)
        line.addWidget(self.align_btn)
        line.addStretch(1)

        # Select points
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Point transform:', self)
        line.addWidget(label)
        self.select_btn = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(lambda state, par=self.fm_imview: self._define_corr_toggled(state, par))
        self.refine_btn = QtWidgets.QPushButton('Refinement')
        self.refine_btn.clicked.connect(self._refine)
        line.addWidget(self.select_btn)
        line.addWidget(self.refine_btn)
        line.addStretch(1)
        vbox.addStretch(1)

    def _init_em_options(self, parent_layout):
        vbox = QtWidgets.QVBoxLayout()
        parent_layout.addLayout(vbox)

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
        line.addWidget(step_label)
        line.addWidget(self.step_box)
        line.addStretch(1)
        button = QtWidgets.QPushButton('Assemble', self)
        button.clicked.connect(self._assemble_mrc)
        line.addWidget(button)

        # ---- Define and align to grid
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Grid transform:', self)
        line.addWidget(label)
        self.define_btn_em = QtWidgets.QPushButton('Define Grid', self)
        self.define_btn_em.setCheckable(True)
        self.define_btn_em.toggled.connect(lambda state, par=self.em_imview: self._define_grid_toggled(state, par))
        line.addWidget(self.define_btn_em)
        self.transform_btn_em = QtWidgets.QPushButton('Transform image', self)
        self.transform_btn_em.clicked.connect(lambda: self._affine_transform(self.em_imview))
        line.addWidget(self.transform_btn_em)
        self.rot_transform_btn_em = QtWidgets.QCheckBox('Disable Shearing', self)
        self.rot_transform_btn_em.stateChanged.connect(lambda state, par=self.em_imview: self._allow_rotation_only(state,par))
        line.addWidget(self.rot_transform_btn_em)
        self.show_btn_em = QtWidgets.QCheckBox('Show original data', self)
        self.show_btn_em.setEnabled(False)
        self.show_btn_em.setChecked(True)
        self.show_btn_em.stateChanged.connect(lambda state, par=self.em_imview: self._show_original(state, par))
        line.addWidget(self.show_btn_em)
        self.show_grid_btn_em = QtWidgets.QCheckBox('Show grid box',self)
        self.show_grid_btn_em.setEnabled(False)
        self.show_grid_btn_em.setChecked(False)
        self.show_grid_btn_em.stateChanged.connect(lambda state, par=self.em_imview: self._show_grid(state, par))
        line.addWidget(self.show_grid_btn_em)
        line.addStretch(1)
        
        # ---- Assembly grid options
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Assembly grid:', self)
        line.addWidget(label)
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
        line.addStretch(1)

        # ---- Points of interest
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Point transform:', self)
        line.addWidget(label)
        self.select_btn_em = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn_em.setCheckable(True)
        self.select_btn_em.toggled.connect(lambda state, par=self.em_imview: self._define_corr_toggled(state, par))
        self.refine_btn_em = QtWidgets.QPushButton('Refinement')
        self.refine_btn_em.clicked.connect(self._refine)
        line.addWidget(self.select_btn_em)
        line.addWidget(self.refine_btn_em)
        line.addStretch(1)

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
                clicked_points = self.clicked_points[index]
                points_corr = self.points_corr
                other = self.em_imview
                other_obj = self.em

        if parent == self.em_imview:
            if self.em is None:
                return
            else:
                index = 1
                obj = self.em
                dbtn = self.define_btn_em
                selbtn = self.select_btn_em
                if obj.side_length is None:
                    size = 0.004 * obj.data.shape[0]
                else:
                    size = obj.side_length / 25
                clicked_points = self.clicked_points[index]
                points_corr = self.points_corr
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
            clicked_points.append(roi)
        elif selbtn.isChecked():
            if other_obj is not None:
                if obj.transformed and other_obj.transformed:
                    point = pg.CircleROI(pos, size, parent=item, movable=False)
                    point.setPen(0,255,0)
                    point.removeHandle(0)
                    parent.addItem(point)
                    point_updated = self.fm.update_points(np.array((point.x(),point.y())).reshape(1,2))
                    points_corr[index].append(point)
                    
                    # Coordinates in clicked image

                    shift = obj.transform_shift + [obj.side_length/2]*2
                    shift = self.fm.update_points(shift.reshape(1,2))[0]
                    init = np.array([point_updated[0,0], point_updated[0,1], 1])
                    transf = np.dot(self.tr_matrices[index], init)
                    cen = other_obj.side_length / 100
                    pos = QtCore.QPointF(transf[0]-cen, transf[1]-cen)  
                    point = pg.CircleROI(pos, 2*cen, parent=other.getImageItem(), movable=True)
                    point.setPen(0,255,255)
                    point.removeHandle(0)
                    other.addItem(point)
                    points_corr[1-index].append(point)
                else:
                    print('Transform images before point selection')
        elif self.select_region_btn.isChecked():
            self.box_coordinate = pos
            points_obj = (self.box_coordinate.x(),self.box_coordinate.y())
            ind = self.em.get_selected_region(np.array(points_obj), not self.show_btn_em.isChecked())
            if ind is not None:
                if self.show_btn_em.isChecked():
                    boxes = self.boxes
                else:
                    boxes = self.tr_boxes
                white_pen = pg.mkPen('w')
                [box.setPen(white_pen) for box in boxes]
                boxes[ind].setPen(pg.mkPen('#ed7370'))

    def _define_grid_toggled(self, checked, parent):
        if parent == self.fm_imview:
            tag = 'FM'
            index = 0
            show_grid_btn = self.show_grid_btn
            obj = self.fm
            show_btn = self.show_btn
        else:
            tag = 'EM'
            index = 1
            show_grid_btn = self.show_grid_btn_em
            obj = self.em
            show_btn = self.show_btn_em

        if checked:
            print('Defining grid on %s image: Click on corners'%tag)
            show_grid_btn.setChecked(True)
            if show_btn.isChecked():
                if self.grid_box[index] is not None:
                    parent.removeItem(self.grid_box[index])
                    self.grid_box[index] = None
            else:
                if self.tr_grid_box[index] is not None:
                    parent.removeItem(self.tr_grid_box[index])
                    self.tr_grid_box[index] = None
        else:
            if show_btn.isChecked():
                print('Done defining grid on %s image: Manually adjust fine positions'%tag)
                positions = [c.pos() for c in self.clicked_points[index]]
                sizes = [c.size()[0] for c in self.clicked_points[index]]
                for pos, s in zip(positions, sizes):
                    pos.setX(pos.x() + s/2)
                    pos.setY(pos.y() + s/2)
                self.grid_box[index] = pg.PolyLineROI(positions, closed=True, movable=False)
                parent.addItem(self.grid_box[index])
                [parent.removeItem(roi) for roi in self.clicked_points[index]]
                points_obj = self.grid_box[index].getState()['points']
                points = np.array([list((point[0], point[1])) for point in points_obj])
                obj.orig_points = points
                self.clicked_points[index] = []
                if obj is not None:
                    show_grid_btn.setEnabled(True)
                    show_grid_btn.setChecked(True)
            else:
                print('Done defining grid on %s image: Manually adjust fine positions'%tag)
                positions = [c.pos() for c in self.clicked_points[index]]
                sizes = [c.size()[0] for c in self.clicked_points[index]]
                for pos, s in zip(positions, sizes):
                    pos.setX(pos.x() + s/2)
                    pos.setY(pos.y() + s/2)
                self.tr_grid_box[index] = pg.PolyLineROI(positions, closed=True, movable=False)
                parent.addItem(self.tr_grid_box[index])
                [parent.removeItem(roi) for roi in self.clicked_points[index]]
                points_obj = self.tr_grid_box[index].getState()['points']
                points = np.array([list((point[0], point[1])) for point in points_obj])
                obj.orig_points = points
                self.clicked_points[index] = []
                if obj is not None:
                    show_grid_btn.setEnabled(True)
                    show_grid_btn.setChecked(True)

    def _define_corr_toggled(self, checked, parent):
        if parent == self.fm_imview:
            tag = 'FM'
            index = 0
            obj = self.fm
            other_obj = self.em
        else:
            tag = 'EM'
            index = 1
            obj = self.em
            other_obj = self.fm
        if obj is None:
            return
        
        if other_obj is not None:
            if checked:
                print('Select points of interest on %s image'%tag)
                if len(self.points_corr[index]) != 0:
                    [parent.removeItem(point) for point in self.points_corr[index]]
                    self.points_corr[index] = []
                self.tr_matrices[index] = obj.get_transform(obj.points, other_obj.points)
            else:
                print('Done selecting points of interest on %s image'%tag)
        else:
            if checked:
                if obj == self.fm:
                    print('Open corresponding EM image first')
                else:
                    print('Open corresponding FM image first')

    def _affine_transform(self, parent):
        if parent == self.fm_imview:
            tag = 'FM'
            index = 0
            obj = self.fm
            show_btn = self.show_btn
        else:
            tag = 'EM'
            index = 1
            obj = self.em
            show_btn = self.show_btn_em
        self.tr_boxes = []
        if self.grid_box[index] is not None:
            if self.rot_transform_btn.isChecked():
                print('Performing rotation on %s image'%tag)
            else:
                print('Performing affine transformation on %s image'%tag)

            if show_btn.isChecked():
                points_obj = self.grid_box[index].getState()['points']
            else:
                points_obj = self.tr_grid_box[index].getState()['points']     

            points = np.array([list((point[0], point[1])) for point in points_obj])
            
            if obj == self.fm:
                if self.rot_transform_btn.isChecked():
                    obj.calc_rot_transform(points)
                else:
                    obj.calc_affine_transform(points)
            else:
                if self.rot_transform_btn.isChecked():
                    obj.calc_rot_transform(points)
                else:
                    obj.calc_affine_transform(points)
            
            if obj.data is None:
                obj.toggle_original()
            else:
                obj.toggle_original(True)
                        
            self._update_fm_imview() if index == 0 else self._update_em_imview()

            show_btn.setEnabled(True)
            show_btn.setChecked(False)
            
            if not show_btn.isChecked() and self.tr_grid_box[index] is not None:
                parent.removeItem(self.tr_grid_box[index])

            positions = [point for point in obj.new_points]
            self.tr_grid_box[index] = pg.PolyLineROI(positions, closed=True, movable=False)
            parent.addItem(self.tr_grid_box[index]) 
        else:
            print('Define grid box on %s image first!'%tag)

    def _allow_rotation_only(self,checked,parent):
        if parent == self.fm_imview:
            obj = self.fm
        else:
            obj = self.em
        if obj is not None:
            if checked:
                obj.no_shear = True
            else:
                obj.no_shear = False
            
    def _show_original(self, state, parent):
        if parent == self.fm_imview:
            index = 0
            obj = self.fm
            updater = self._update_fm_imview
            orig_btn = self.show_btn
            grid_btn = self.show_grid_btn
        else:
            index = 1
            obj = self.em
            updater = self._update_em_imview
            orig_btn = self.show_btn_em
            grid_btn = self.show_grid_btn_em

        if obj is not None:
            obj.toggle_original(state==0)
            if grid_btn.isChecked():
                if state:
                    if self.tr_grid_box[index] is not None:
                        parent.removeItem(self.tr_grid_box[index])
                    parent.addItem(self.grid_box[index])
                else:
                    parent.removeItem(self.grid_box[index])
                    if self.tr_grid_box[index] is not None:
                        parent.addItem(self.tr_grid_box[index])
        if obj == self.em:
            if self.show_boxes_btn.isChecked():
                if state:
                    [parent.removeItem(box) for box in self.tr_boxes]
                    if len(self.boxes) == 0:
                        self._show_boxes()
                    else:
                        [parent.addItem(box) for box in self.boxes]
                else:
                    [parent.removeItem(box) for box in self.boxes]
                    [parent.addItem(box) for box in self.tr_boxes]

        updater()

    def _show_grid(self, state, parent):
        if parent == self.fm_imview:
            index = 0
            obj = self.fm
            orig_btn = self.show_btn
            grid_btn = self.show_grid_btn
        else:
            index = 1
            obj = self.em
            orig_btn = self.show_btn_em
            grid_btn = self.show_grid_btn_em
        
        if self.grid_box[index] is not None:
            if orig_btn.isChecked():
                if state:
                    parent.addItem(self.grid_box[index])
                else:
                    parent.removeItem(self.grid_box[index])
            else:
                if state:
                    parent.addItem(self.tr_grid_box[index])
                else:
                    parent.removeItem(self.tr_grid_box[index])

    def _set_theme(self, name):
        if name == 'none':
            self.setStyleSheet('')
        else:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'styles/%s.css'%name), 'r') as f:
                self.setStyleSheet(f.read())
        self.settings.setValue('theme', name)

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
        self.settings.setValue('channel_colors', self.colors)
        self.settings.setValue('geometry', self.geometry())
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
        self._update_fm_imview()  
        
    def _show_max_projection(self):
        if self.max_proj_btn.isChecked():
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
        self.fm.calc_max_projection()
        self._update_fm_imview()

    def _calc_colors(self,my_channels):
        my_channels = [np.repeat(channel[:,:,np.newaxis],3,axis=2) for channel in my_channels]
        for i in range(len(my_channels)):
            my_channels[i] = my_channels[i] * cm.hex2color(self.colors[i])
        return my_channels

    def _update_fm_imview(self):
        if self.fm is not None:
            channels = []
            for i in range(len(self.channels)):
                if self.channels[i]:
                    channels.append(self.fm.data[:,:,i])
                else:
                    if self.overlay_btn.isChecked():
                        channels.append(np.zeros_like(self.fm.data[:,:,i]))
                                
            if len(channels) == 0:
                channels.append(np.zeros_like(self.fm.data[:,:,0]))
            
            color_channels = self._calc_colors(channels)
            
            self.color_data = np.array(color_channels)  
            if self.overlay_btn.isChecked():
                self.color_data = np.sum(self.color_data,axis=0)
            
            self._recalc_grid(not self.fm.transformed)
            vr = self.fm_imview.getImageItem().getViewBox().targetRect()
            levels = self.fm_imview.getHistogramWidget().item.getLevels()
            self.fm_imview.setImage(self.color_data, levels=levels)
            self.fm_imview.getImageItem().getViewBox().setRange(vr, padding=0)
        
    def _show_overlay(self,checked):
        if self.fm is not None:
            self.overlay = not self.overlay
            self._update_fm_imview()

    def _show_channels(self,checked,my_channel):       
        if self.fm is not None:
           self.channels[my_channel] = not self.channels[my_channel]          
           self._update_fm_imview()            

    def _sel_color(self, index, button):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            cname = color.name()
            self.colors[index] = cname
            button.setStyleSheet('background-color: {}'.format(cname))
            self._update_fm_imview()
        else:
            print('Invalid color')

    def _fliph(self, state):
        if self.fm is not None:
            self.fm.flip_horizontal(state)
            self._update_fm_imview()

    def _flipv(self, state):
        if self.fm is not None:
            self.fm.flip_vertical(state)
            self._update_fm_imview()

    def _trans(self, state):
        if self.fm is not None:
            self.fm.transpose(state)
            self._update_fm_imview()

    def _rot(self, state):
        if self.fm is not None:
            self.fm.rotate_clockwise(state)
            self._update_fm_imview()

    def _recalc_grid(self, orig=True):
        if self.fm.orig_points is None:
            return
        if orig:
            print('Recalc orig')
            self.fm_imview.removeItem(self.grid_box[0])
            self.fm._update_data()
            pos = [QtCore.QPointF(point[0], point[1]) for point in self.fm.points]
            self.grid_box[0] = pg.PolyLineROI(pos, closed=True, movable=False)
            self.fm_imview.addItem(self.grid_box[0])
            print(pos[0].x(), pos[0].y())
        else:
            print('Recalc transf')
            self.fm_imview.removeItem(self.tr_grid_box[0])
            self.fm._update_data()
            pos = [QtCore.QPointF(point[0], point[1]) for point in self.fm.points]
            self.tr_grid_box[0] = pg.PolyLineROI(pos, closed=True, movable=False)
            self.fm_imview.addItem(self.tr_grid_box[0])
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
    
    def _refine(self):
        pass

    # ---- EM functions

    def _load_mrc(self):
        if self.curr_mrc_folder is None:
            #self.curr_mrc_folder = os.getcwd()
            self.curr_mrc_folder = '/beegfs/cssb/user/kaufmanr/cryoCLEM-software/clem_dataset/'
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

        self.em_imview.setImage(self.em.data, levels=levels)
        self.em_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _assemble_mrc(self):
        if self.step_box.text() is '':
            step = 100
        else:
            step = self.step_box.text()

        if self.mrc_fname.text() is not '':
            self.em = em_operations.EM_ops(step=int(step))
            self.em.parse(self.mrc_fname.text())
            self.em.assemble()
            print('Done')
            self.em_imview.setImage(self.em.data)
        else:
            print('You have to choose .mrc file first!')
    
    def _show_boxes(self):
        if self.show_boxes_btn.isChecked():
            if self.em is not None:
                handle_pen = pg.mkPen('#00000000')
                if self.show_btn_em.isChecked():
                    if len(self.boxes) == 0:
                        for i in range(len(self.em.pos_x)):
                            roi = pg.PolyLineROI([], closed=True, movable=False)
                            roi.handlePen = handle_pen
                            roi.setPoints(self.em.grid_points[i])
                            self.boxes.append(roi)
                            self.em_imview.addItem(roi)
                    else:
                        [self.em_imview.addItem(box) for box in self.boxes]
                else:
                    if len(self.tr_boxes) == 0:
                        for i in range(len(self.em.tr_grid_points)):
                            roi = pg.PolyLineROI([], closed=True, movable=False)
                            roi.handlePen = handle_pen
                            roi.setPoints(self.em.tr_grid_points[i])
                            self.tr_boxes.append(roi)
                            self.em_imview.addItem(roi)
                    else:
                        [self.em_imview.addItem(box) for box in self.tr_boxes]
        else:
            if self.show_btn_em.isChecked():
                [self.em_imview.removeItem(box) for box in self.boxes]
            else:
                [self.em_imview.removeItem(box) for box in self.tr_boxes]

    def _select_box(self,state,parent):
        if self.select_region_btn.isChecked():
            print('Select box!')
            #parent.setImage(self.em.data)
        else:
            if self.box_coordinate is not None:
                if self.show_btn_em.isChecked():
                    transformed = False
                else:
                    transformed = True
                points_obj = (self.box_coordinate.x(),self.box_coordinate.y())
                print(points_obj)
                self.em.select_region(np.array(points_obj),transformed)
                self._update_em_imview() 
                #parent.setImage(self.em.data)
                #[parent.removeItem(box) for box in self.boxes]
                self.show_boxes_btn.setChecked(False)
                self.box_coordinate = None
                self.show_assembled_btn.setEnabled(True)
                self.show_assembled_btn.setChecked(False)
                self.show_grid_btn_em.setChecked(False)

    def _show_assembled(self):
        if self.show_assembled_btn.isChecked():
            assembled = True
        else:
            assembled = False
            self.show_boxes_btn.setChecked(False)
        if self.show_btn_em.isChecked():
            transformed = False
        else:
            transformed = True
        if self.em is not None:
            self.em.toggle_region(transformed,assembled)
            self._update_em_imview()
    
    def _save_mrc_montage(self):
        if self.em is None:
            print('No montage to save!')
        else:
            if self.curr_mrc_folder is None:
                self.curr_mrc_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Binned Montage', self.curr_mrc_folder, '*.mrc')
            self.curr_mrc_folder = os.path.dirname(file_name)
            if file_name is not '':
                self.em.save_merge(file_name)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    gui = GUI()
    sys.exit(app.exec_())
