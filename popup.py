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
import popup

warnings.simplefilter('ignore', category=FutureWarning)

class Merge(QtGui.QMainWindow,):
    def __init__(self, parent):
        super(Merge, self).__init__()
        self.parent = parent 
        self.theme = self.parent.theme
        self.channels = [False, False, False, False]
        self.colors = list(np.copy(self.parent.colors))
        if self.parent.fm.merged is not None:
            self.data = np.copy(self.parent.fm.merged)
            self.channels.append(False)
            self.colors.append('#808080')
        else:
            self.data = np.copy(self.parent.fm.data)
        print(self.data.shape)
        self.num_channels = self.parent.num_channels
        self.ind = self.parent.ind
        print(self.channels)
        print(self.colors)
        self.color_data = None
        self.overlay = True
        self.settings = QtCore.QSettings('MPSD-CNI', 'CLEMGui', self)
        self._init_ui()
    
    def _init_ui(self):
        self.resize(800, 800)
        self.parent._set_theme(self.theme)
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
        action = QtWidgets.QAction('&Save merged image', self)
        action.triggered.connect(self._save_merged_image)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Quit', self)
        action.triggered.connect(self.close)
        filemenu.addAction(action) 
        self._set_theme(self.theme)
        
        self.imview = pg.ImageView()
        self.imview.ui.roiBtn.hide()
        self.imview.ui.menuBtn.hide()
        self.imview.setImage(np.sum(self.data,axis=2), levels=(self.data.min(), self.data.mean()*2))   
        layout.addWidget(self.imview)
      

        options = QtWidgets.QHBoxLayout()
        options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(options) 
        self._init_options(options)       
    
    def _init_options(self,parent_layout):
        vbox = QtWidgets.QVBoxLayout()
        parent_layout.addLayout(vbox)
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)

        button = QtWidgets.QLabel('FM image:', self)
        line.addWidget(button)
        self.fm_fname  = self.parent.fm_fname
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

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        button = QtWidgets.QLabel('EM Image:', self)
        line.addWidget(button)
        self.em_fname = self.parent.mrc_fname
        line.addWidget(self.em_fname, stretch=1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Colors:', self)
        line.addWidget(label)
        self.channel1_btn = QtWidgets.QCheckBox(' ',self)
        self.channel2_btn = QtWidgets.QCheckBox(' ',self)
        self.channel3_btn = QtWidgets.QCheckBox(' ',self)
        self.channel4_btn = QtWidgets.QCheckBox(' ',self)
        self.channel5_btn = QtWidgets.QCheckBox(' ',self)
        self.overlay_btn = QtWidgets.QCheckBox('Overlay',self)
        self.channel1_btn.stateChanged.connect(lambda state, channel=0: self._show_channels(state,channel))
        self.channel2_btn.stateChanged.connect(lambda state, channel=1: self._show_channels(state,channel))
        self.channel3_btn.stateChanged.connect(lambda state, channel=2: self._show_channels(state,channel))
        self.channel4_btn.stateChanged.connect(lambda state, channel=3: self._show_channels(state,channel))
        if len(self.channels) == 5:
            self.channel5_btn.stateChanged.connect(lambda state, channel=4: self._show_channels(state,channel))
        self.overlay_btn.stateChanged.connect(self._show_overlay)
        self.channel1_btn.setChecked(True)
        self.channel2_btn.setChecked(True)
        self.channel3_btn.setChecked(True)
        self.channel4_btn.setChecked(True)
        self.channel5_btn.setChecked(True)
        self.overlay_btn.setChecked(True)

        self.c1_btn = QtWidgets.QPushButton(' ', self)
        self.c1_btn.clicked.connect(lambda: self._sel_color(0, self.c1_btn))
        width = self.c1_btn.fontMetrics().boundingRect(' ').width() + 24
        self.c1_btn.setFixedWidth(width)
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
        self.c5_btn = QtWidgets.QPushButton(' ', self)
        self.c5_btn.setMaximumHeight(width)
        self.c5_btn.setFixedWidth(width)
        if len(self.channels) == 5:
            self.c5_btn.clicked.connect(lambda: self._sel_color(4, self.c5_btn))
            self.c5_btn.setStyleSheet('background-color: {}'.format(self.colors[4]))

        
        line.addWidget(self.c1_btn)
        line.addWidget(self.channel1_btn)
        line.addWidget(self.c2_btn)
        line.addWidget(self.channel2_btn)
        line.addWidget(self.c3_btn)
        line.addWidget(self.channel3_btn)
        line.addWidget(self.c4_btn)
        line.addWidget(self.channel4_btn)
        line.addWidget(self.c5_btn)
        line.addWidget(self.channel5_btn)
        line.addWidget(self.overlay_btn)
        line.addStretch(1)

    def _show_overlay(self,checked):
        self.overlay = not self.overlay
        self._update_imview()

    def _show_channels(self,checked,my_channel):
        self.channels[my_channel] = not self.channels[my_channel]
        self._update_imview()
 
    def _sel_color(self, index, button):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            cname = color.name()
            self.colors[index] = cname
            button.setStyleSheet('background-color: {}'.format(cname))
            self._update_imview()
        else:
            print('Invalid color')
    
    def _calc_colors(self,my_channels):
        my_channels = [np.repeat(channel[:,:,np.newaxis],3,axis=2) for channel in my_channels]
        my_channels_red = []
        for i in range(len(my_channels)):
            if self.channels[i]:
                my_channels_red.append(my_channels[i] * cm.hex2color(self.colors[i]))
        return my_channels_red

    def _update_imview(self):
        channels = []
        for i in range(len(self.channels)):
            if self.channels[i]:
                channels.append(self.data[:,:,i])
            else:
                channels.append(np.zeros_like(self.data[:,:,i]))

        color_channels = self._calc_colors(channels)
        if len(color_channels) == 0:
            color_channels.append(np.zeros_like(self.data[:,:,0]))
                                                  
        self.color_data = np.array(color_channels)
        if self.overlay_btn.isChecked():
            self.color_data = np.sum(self.color_data,axis=0)

        vr = self.imview.getImageItem().getViewBox().targetRect()
        levels = self.imview.getHistogramWidget().item.getLevels()
        self.imview.setImage(self.color_data, levels=levels)
        self.imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _save_merged_image(self):
        if self.data is None:
            print('No image to save!')
        else:
            if self.curr_mrc_folder is None:
                self.curr_mrc_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Merged Image', parent.curr_mrc_folder, '*.tif')
            #self.curr_mrc_folder = os.path.dirname(file_name)
            #if file_name is not '':
            #    self.em.save_merge(file_name)
    
    def _show_max_projection(self):
        if self.max_proj_btn.isChecked():
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
        self.parent.fm.calc_max_projection()
        self._update_imview()

    def _next_file(self):
        self.ind = (self.ind + 1 + self.num_channels) % self.num_channels
        self.parent.fm.parse(fname=self.parent.fm.old_fname, z=self.ind)
        self._update_imview()
        fname, indstr = self.fm_fname.text().split()
        self.fm_fname.setText(fname + ' [%d/%d]'%(self.ind, self.num_channels))

    def _prev_file(self):
        self.ind = (self.ind - 1 + self.num_channels) % self.num_channels
        self.parent.fm.parse(fname=self.parent.fm.old_fname, z=self.ind)
        self._update_imview()
        fname, indstr = self.fm_fname.text().split()
        self.fm_fname.setText(fname + ' [%d/%d]'%(self.ind, self.num_channels))

    def _set_theme(self, name):
        if name == 'none':
            self.setStyleSheet('')
        else:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'styles/%s.css'%name), 'r') as f:
                self.setStyleSheet(f.read())
        self.settings.setValue('theme', name)

    def closeEvent(self, event):
        event.accept()

