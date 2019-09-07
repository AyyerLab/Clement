import sys
import os
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
import csv 
import mrcfile as mrc

warnings.simplefilter('ignore', category=FutureWarning)

class Merge(QtGui.QMainWindow,):
    def __init__(self, parent):
        super(Merge, self).__init__(parent)
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
        self.curr_mrc_folder = self.parent.emcontrols.curr_mrc_folder
        self.num_slices = self.parent.fmcontrols.num_slices
        self._current_slice = self.parent.fmcontrols._current_slice
        self.ind = self.parent.fmcontrols.ind
        self.color_data = None
        self.overlay = True
        self.clicked_points = []
        self.annotations = []
        self.counter = 0
        self.stage_positions = None
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
        action.triggered.connect(self._save_data)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Quit', self)
        action.triggered.connect(self.close)
        filemenu.addAction(action) 
        self._set_theme(self.theme)
        
        self.imview = pg.ImageView()
        self.imview.ui.roiBtn.hide()
        self.imview.ui.menuBtn.hide()
        self.imview.scene.sigMouseClicked.connect(lambda evt: self._imview_clicked(evt))
        self.imview.setImage(np.sum(self.data,axis=2), levels=(self.data.min(), self.data.max()//3))   
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

        label = QtWidgets.QLabel('FM image:', self)
        line.addWidget(label)
        self.fm_fname  = self.parent.fmcontrols.fm_fname.text()
        label = QtWidgets.QLabel(self.fm_fname, self)
        line.addWidget(label, stretch=1)
        self.max_proj_btn = QtWidgets.QCheckBox('Max projection')
        self.max_proj_btn.stateChanged.connect(self._show_max_projection)
        line.addWidget(self.max_proj_btn)
        self.slice_select_btn = QtWidgets.QSpinBox(self)
        self.slice_select_btn.editingFinished.connect(self._slice_changed)
        self.slice_select_btn.setRange(0, self.parent.fm.num_slices)
        self.slice_select_btn.setValue(self.parent.fmcontrols.slice_select_btn.value())
        line.addWidget(self.slice_select_btn)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('EM Image:', self)
        line.addWidget(label)
        self.em_fname = self.parent.emcontrols.mrc_fname.text()
        label = QtWidgets.QLabel(self.em_fname, self)
        line.addWidget(label, stretch=1)

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

        # Select and save coordinates
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Select and save coordinates', self)
        line.addWidget(label)
        self.select_btn = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self._calc_stage_positions)
        self.save_btn = QtWidgets.QPushButton('Save data')
        self.save_btn.clicked.connect(self._save_data)
        line.addWidget(self.select_btn)
        line.addWidget(self.save_btn)
        line.addStretch(1)

    def _imview_clicked(self, event):
        if self.select_btn.isChecked():
            if event.button() == QtCore.Qt.RightButton:
                event.ignore()
                return
            self.counter += 1
            size = 0.01 * self.data.shape[0]

            pos = self.imview.getImageItem().mapFromScene(event.pos())
            pos.setX(pos.x() - size/2)
            pos.setY(pos.y() - size/2)
            item = self.imview.getImageItem()
            point = pg.CircleROI(pos, size, parent=item, movable=False, removable=True)
            point.setPen(0,255,0)
            point.removeHandle(0)
            self.imview.addItem(point)
            self.clicked_points.append(point) 
            annotation = pg.TextItem(str(self.counter), color=(0,255,0), anchor=(0,0))
            annotation.setPos(pos.x()+5, pos.y()+5)
            self.annotations.append(annotation)
            self.imview.addItem(annotation)
            point.sigRemoveRequested.connect(lambda: self._remove_correlated_points(point,annotation))

    def _remove_correlated_points(self,pt,anno):
        self.imview.removeItem(pt)
        self.clicked_points.remove(pt)
        self.imview.removeItem(anno)
        self.annotations.remove(anno)

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
                rgb = tuple([int(self.colors[i][1+2*c:3+2*c], 16)/255. for c in range(3)])
                my_channels_red.append(my_channels[i] * rgb)
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

    def _calc_stage_positions(self,checked):
        if checked:
            print('Select points of interest!')
            if len(self.clicked_points) != 0:
                [self.imview.removeItem(point) for point in self.clicked_points]
                [self.imview.removeItem(annotation) for annotation in self.annotations]
                self.clicked_points = []
                self.annotations = []
                self.counter = 0
                self.stage_positions = None
        else:
            size = 0.01 * self.data.shape[0]
            coordinates = [np.array([point.x()+size/2,point.y()+size/2]) for point in self.clicked_points]
            self.stage_positions = self.parent.emcontrols.ops.calc_stage_positions(coordinates)
            print('Done selecting points of interest!')

    def _save_data(self):
        if self.data is None:
            print('No image to save!')
            return
        if len(self.clicked_points) == 0:
            print('No coordinates selected!')
            return
        if self.stage_positions is None:
            print('Confirm selected points first!')
            return
    
        if self.curr_mrc_folder is None:
            self.curr_mrc_folder = os.getcwd()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Merged Image', self.curr_mrc_folder)
        screenshot = self.imview.grab()
        if file_name is not '':
            screenshot.save(file_name,'tif')
            self._save_merge(file_name)
            self._save_coordinates(file_name)
    
    def _save_merge(self, fname):
        try:
            with mrc.new(fname+'.mrc', overwrite=True) as f:
                f.set_data(self.data.astype(np.float32))
                f.update_header_stats()
        except PermissionError:
            pass

    def _save_coordinates(self, fname):
        enumerated = []
        for i in range(len(self.stage_positions)):
            enumerated.append((i+1,self.stage_positions[i][0],self.stage_positions[i][1]))
        try:
            with open(fname+'.txt', 'a', newline='') as f:
                csv.writer(f, delimiter=' ').writerows(enumerated)

        except PermissionError:
            print('Permission error! Choose a different directory!')
            self._save_data()
            
    def _show_max_projection(self):
        self.slice_select_btn.setEnabled(not self.max_proj_btn.isChecked())
        self.parent.fm.calc_max_projection()
        self.parent.fm.apply_merge()
        self.data = np.copy(self.parent.fm.merged)       
        self._update_imview()

    def _slice_changed(self):
        num = self.slice_select_btn.value()
        if num != self._current_slice:
            self.parent.fm.parse(fname=self.fm_fname, z=num, reopen=False)
            self.parent.fm.apply_merge()
            self.data = np.copy(self.parent.fm.merged)
            self._update_imview()
            fname, indstr = self.fm_fname.split()
            self.fm_fname = (fname + ' [%d/%d]'%(num, self.parent.fm.num_slices))
            self._current_slice = num 
            self.slice_select_btn.clearFocus()

    def _set_theme(self, name):
        if name == 'none':
            self.setStyleSheet('')
        else:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'styles/%s.qss'%name), 'r') as f:
                self.setStyleSheet(f.read())
        self.settings.setValue('theme', name)

    def closeEvent(self, event):
        event.accept()

