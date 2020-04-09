import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import scipy.ndimage.interpolation as interpol

from skimage.color import hsv2rgb

from .base_controls import BaseControls
from .fm_operations import FM_ops
#import align_fm

class SeriesPicker(QtWidgets.QDialog):
    def __init__(self, parent, names):
        super(SeriesPicker, self).__init__(parent)
        self.current_series = 0

        self.setWindowTitle('Pick image series...')
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        self.picker = QtWidgets.QComboBox(self)
        layout.addWidget(self.picker)
        self.picker.addItems(names)
        self.picker.currentIndexChanged.connect(self.series_changed)

        button = QtWidgets.QPushButton('Confirm', self)
        layout.addWidget(button)
        button.clicked.connect(self.accept)

    def series_changed(self, i):
        self.current_series = i

    def closeEvent(self, event):
        self.current_series = -1
        event.accept()

class FMControls(BaseControls):
    def __init__(self, imview, colors):
        super(FMControls, self).__init__()
        self.tag = 'FM'
        self.imview = imview
        self.ops = None
        #self.ind = 0
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)

        self._colors = colors
        self._channels = [True, True, True, True]
        self._overlay = True
        self._curr_folder = None
        self._file_name = None
        self._series = None
        self._current_slice = 0
        self._peaks = []
        self._shift = None 
        self._init_ui()
    
    def _init_ui(self):
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

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
        self.max_proj_btn.setEnabled(False)
        line.addWidget(self.max_proj_btn)
        self.slice_select_btn = QtWidgets.QSpinBox(self)
        self.slice_select_btn.setRange(0, 0)
        self.slice_select_btn.setEnabled(False)
        #self.slice_select_btn.valueChanged.connect(self._slice_changed)
        self.slice_select_btn.editingFinished.connect(self._slice_changed)
        line.addWidget(self.slice_select_btn)

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
        self.channel1_btn.setEnabled(False)
        self.channel2_btn.setEnabled(False)
        self.channel3_btn.setEnabled(False)
        self.channel4_btn.setEnabled(False)
        self.overlay_btn.setEnabled(False)

        self.c1_btn = QtWidgets.QPushButton(' ', self)
        self.c1_btn.clicked.connect(lambda: self._sel_color(0, self.c1_btn))
        width = self.c1_btn.fontMetrics().boundingRect(' ').width() + 24
        self.c1_btn.setFixedWidth(width)
        self.c1_btn.setMaximumHeight(width)
        self.c1_btn.setStyleSheet('background-color: {}'.format(self._colors[0]))
        self.c2_btn = QtWidgets.QPushButton(' ', self)
        self.c2_btn.clicked.connect(lambda: self._sel_color(1, self.c2_btn))
        self.c2_btn.setMaximumHeight(width)
        self.c2_btn.setFixedWidth(width)
        self.c2_btn.setStyleSheet('background-color: {}'.format(self._colors[1]))
        self.c3_btn = QtWidgets.QPushButton(' ', self)
        self.c3_btn.setMaximumHeight(width)
        self.c3_btn.setFixedWidth(width)
        self.c3_btn.clicked.connect(lambda: self._sel_color(2, self.c3_btn))
        self.c3_btn.setStyleSheet('background-color: {}'.format(self._colors[2]))
        self.c4_btn = QtWidgets.QPushButton(' ', self)
        self.c4_btn.setMaximumHeight(width)
        self.c4_btn.setFixedWidth(width)
        self.c4_btn.clicked.connect(lambda: self._sel_color(3, self.c4_btn))
        self.c4_btn.setStyleSheet('background-color: {}'.format(self._colors[3]))

        self.c1_btn.setEnabled(False)
        self.c2_btn.setEnabled(False)
        self.c3_btn.setEnabled(False)
        self.c4_btn.setEnabled(False)

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

        self.align_btn = QtWidgets.QCheckBox('Align color channels', self)
        self.align_btn.stateChanged.connect(self._calc_shift)
        self.align_btn.setEnabled(False)
        line.addWidget(self.align_btn)
        #line.addStretch(1)

        # ---- Define and align to grid
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Grid transform:', self)
        line.addWidget(label)
        self.define_btn = QtWidgets.QPushButton('Define grid box', self)
        self.define_btn.setCheckable(True)
        self.define_btn.toggled.connect(self._define_grid_toggled)
        self.define_btn.setEnabled(False)
        line.addWidget(self.define_btn)
        self.transform_btn = QtWidgets.QPushButton('Transform image', self)
        self.transform_btn.clicked.connect(self._affine_transform)
        self.transform_btn.setEnabled(False)
        line.addWidget(self.transform_btn)
        self.rot_transform_btn = QtWidgets.QCheckBox('Disable shearing', self)
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
        self.show_grid_btn.clicked.connect(self._update_imview)
        line.addWidget(self.show_grid_btn)
        line.addStretch(1)

        # ---- Align colors
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Peak finding:', self)
        line.addWidget(label)
        self.peak_btn = QtWidgets.QPushButton('Show peaks', self)
        self.peak_btn.setCheckable(True)
        self.peak_btn.toggled.connect(self._find_peaks)
        self.peak_btn.setEnabled(False)
        line.addWidget(self.peak_btn)
        line.addStretch(1)

        # 3D mapping
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('3D mapping: ', self)
        line.addWidget(label)
        self.map_btn = QtWidgets.QPushButton('Z mapping', self)
        self.map_btn.setCheckable(True)
        self.map_btn.toggled.connect(self._mapping)
        self.map_btn.setEnabled(False)
        self.remove_tilt_btn = QtWidgets.QCheckBox('Remove tilt effect', self)
        self.remove_tilt_btn.setEnabled(False)
        self.remove_tilt_btn.setChecked(False)
        self.remove_tilt_btn.stateChanged.connect(self._remove_tilt)
        line.addWidget(self.map_btn)
        line.addWidget(self.remove_tilt_btn)
        line.addStretch(1)

        # Select points
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Point transform:', self)
        line.addWidget(label)
        self.select_btn = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self._define_corr_toggled)
        self.select_btn.setEnabled(False)
        self.refine_btn = QtWidgets.QPushButton('Refinement')
        self.refine_btn.clicked.connect(self._refine)
        self.refine_btn.setEnabled(False)

        #self.auto_opt_btn = QtWidgets.QCheckBox('Auto-optimize', self)
        #self.auto_opt_btn.setEnabled(False)
        #self.auto_opt_btn.stateChanged.connect(self._optimize)
        line.addWidget(self.select_btn)
        #line.addWidget(self.auto_opt_btn)
        line.addWidget(self.refine_btn)
        line.addStretch(1)

        # ---- Flips and rotates
        label = QtWidgets.QLabel('Flips:', self)
        line.addWidget(label)

        #self.fliph = QtWidgets.QPushButton('\u2345', self)
        self.fliph = QtWidgets.QPushButton('', self)
        self.fliph.setObjectName('fliph')
        width = self.fliph.fontMetrics().boundingRect(' ').width() + 25
        font = self.fliph.font()
        font.setPointSize(24)
        self.fliph.setFixedWidth(width)
        self.fliph.setFixedHeight(width)
        self.fliph.setCheckable(True)
        self.fliph.setFont(font)
        self.fliph.toggled.connect(self._fliph)
        self.fliph.setEnabled(False)
        line.addWidget(self.fliph)

        #self.flipv = QtWidgets.QPushButton('\u2356', self)
        self.flipv = QtWidgets.QPushButton('', self)
        self.flipv.setObjectName('flipv')
        self.flipv.setCheckable(True)
        self.flipv.setFixedWidth(width)
        self.flipv.setFixedHeight(width)
        self.flipv.setFont(font)
        self.flipv.toggled.connect(self._flipv)
        self.flipv.setEnabled(False)
        line.addWidget(self.flipv)

        #self.transpose = QtWidgets.QPushButton('\u292f', self)
        self.transpose = QtWidgets.QPushButton('', self)
        self.transpose.setObjectName('transpose')
        self.transpose.setCheckable(True)
        self.transpose.setFixedWidth(width)
        self.transpose.setFixedHeight(width)
        font.setPointSize(20)
        self.transpose.setFont(font)
        self.transpose.toggled.connect(self._trans)
        self.transpose.setEnabled(False)
        line.addWidget(self.transpose)

        #self.rotate = QtWidgets.QPushButton('\u293e', self)
        self.rotate = QtWidgets.QPushButton('', self)
        self.rotate.setObjectName('rotate')
        self.rotate.setCheckable(True)
        self.rotate.setFixedWidth(width)
        self.rotate.setFixedHeight(width)
        self.rotate.setFont(font)
        self.rotate.toggled.connect(self._rot)
        line.addWidget(self.rotate)
        self.rotate.setEnabled(False)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Merge FM and EM:', self)
        line.addWidget(label)
        self.merge_btn = QtWidgets.QPushButton('Merge',self)
        self.merge_btn.setEnabled(False)
        line.addWidget(self.merge_btn)
        line.addStretch(1)
        vbox.addStretch(1)

        self.show()

    def _update_imview(self):
        if self.ops is not None:
            print(self.ops.data.shape)
            if self.ops._show_mapping:
                vr = self.imview.getImageItem().getViewBox().targetRect()
                self.imview.setImage(hsv2rgb(self.ops.data))
                self.imview.getImageItem().getViewBox().setRange(vr, padding=0)
            else:
                self._calc_color_channels()
                vr = self.imview.getImageItem().getViewBox().targetRect()
                levels = self.imview.getHistogramWidget().item.getLevels()
                self.imview.setImage(self.color_data, levels=levels)
                self.imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _load_fm_images(self):
        if self._curr_folder is None:
            self._curr_folder = os.getcwd()

        self._file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             'Select FM file',
                                                             self._curr_folder,
                                                             '*.lif;;*.tif;;*.tiff')
        if self._file_name is not '':
            self.reset_init()
            self._curr_folder = os.path.dirname(self._file_name)
            self._current_slice = self.slice_select_btn.value()
            self._parse_fm_images(self._file_name)
            
    def _parse_fm_images(self, file_name, series=None):
        self.ops = FM_ops()
        retval = self.ops.parse(file_name, z=0, series=series)
        if retval is not None:
            picker = SeriesPicker(self, retval)
            picker.exec_()
            self._series = picker.current_series
            if self._series < 0:
                self.ops = None
                return
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.ops.parse(file_name, z=0, series=self._series)

        self.num_slices = self.ops.num_slices

        if file_name is not '':
            self.fm_fname.setText(file_name + ' [0/%d]'%self.num_slices)
            self.slice_select_btn.setRange(0, self.num_slices)

            self.imview.setImage(self.ops.data, levels=(self.ops.data.min(), self.ops.data.mean()*2))
            self._update_imview()
            self.max_proj_btn.setEnabled(True)
            self.slice_select_btn.setEnabled(True)
            self.channel1_btn.setEnabled(True)
            self.channel2_btn.setEnabled(True)
            self.channel3_btn.setEnabled(True)
            self.channel4_btn.setEnabled(True)
            self.c1_btn.setEnabled(True)
            self.c2_btn.setEnabled(True)
            self.c3_btn.setEnabled(True)
            self.c4_btn.setEnabled(True)
            self.overlay_btn.setEnabled(True)
            self.define_btn.setEnabled(True)
            self.transform_btn.setEnabled(True)
            self.rot_transform_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.refine_btn.setEnabled(True)
            self.merge_btn.setEnabled(True)
            self.peak_btn.setEnabled(True)
            self.align_btn.setEnabled(True)
            self.map_btn.setEnabled(True)
            self.remove_tilt_btn.setEnabled(True)

        QtWidgets.QApplication.restoreOverrideCursor()

    def _show_max_projection(self):
        self.slice_select_btn.setEnabled(not self.max_proj_btn.isChecked())
        if self.ops is not None:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.ops.calc_max_projection()
            self._update_imview()
            QtWidgets.QApplication.restoreOverrideCursor()

    def _calc_color_channels(self):
        self.color_data = np.zeros((len(self._channels),) + self.ops.data[:,:,0].shape + (3,))
        for i in range(len(self._channels)):
            if self._channels[i]:
                my_channel = self.ops.data[:,:,i]
                my_channel_rgb = np.repeat(my_channel[:,:,np.newaxis],3,axis=2)
                rgb = tuple([int(self._colors[i][1+2*c:3+2*c], 16)/255. for c in range(3)])
                self.color_data[i,:,:,:] = my_channel_rgb * rgb
            else:
                self.color_data[i,:,:,:] = np.zeros((self.ops.data[:,:,i].shape + (3,)))

        if self.overlay_btn.isChecked():
            self.color_data = np.sum(self.color_data,axis=0)

    def _show_overlay(self,checked):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is not None:
            self._overlay = not self._overlay
            self._update_imview()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _show_channels(self,checked,my_channel):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is not None and self.ops.data is not None:
           self._channels[my_channel] = not self._channels[my_channel]
           self._update_imview()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _sel_color(self, index, button):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            cname = color.name()
            self._colors[index] = cname
            button.setStyleSheet('background-color: {}'.format(cname))
            self._update_imview()
        else:
            print('Invalid color')

    def _fliph(self, state):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is not None:
            self.ops.flip_horizontal(state)
            self._recalc_grid()
            self._update_imview()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _flipv(self, state):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is not None:
            self.ops.flip_vertical(state)
            self._recalc_grid()
            self._update_imview()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _trans(self, state):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is not None:
            self.ops.transpose(state)
            self._recalc_grid()
            self._update_imview()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _rot(self, state):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is not None:
            self.ops.rotate_clockwise(state)
            self._recalc_grid()
            self._update_imview()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _slice_changed(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is None:
            print('Pick FM image first')
            return

        num = self.slice_select_btn.value()
        if num != self._current_slice:
            self.ops.parse(fname=self.ops.old_fname, z=num%self.num_slices, reopen=False)
            self._update_imview()
            fname, indstr = self.fm_fname.text().split()
            self.fm_fname.setText(fname + ' [%d/%d]'%(num, self.num_slices))
            self._current_slice = num
            self.slice_select_btn.clearFocus()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _find_peaks(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.ops is not None:
            print(self.ops.data.shape)
            if self.peak_btn.isChecked():
                if self.map_btn.isChecked():
                    if self.ops._transformed:
                        if self.ops.tf_peak_slices is None or self.ops.tf_peak_slices[-1] is None:
                            self.ops.peak_finding(self.ops.tf_max_proj_data[:,:,-1], transformed=True)
                        peaks_2d = self.ops.tf_peak_slices[-1]
                    else:
                        if self.ops.peak_slices is None or self.ops.peak_slices[-1] is None:
                            self.ops.peak_finding(self.ops.max_proj_data[:, :, -1], transformed=False)
                        peaks_2d = self.ops.peak_slices[-1]
                else:
                    if self.max_proj_btn.isChecked():
                        if self.ops._transformed:
                            if self.ops.tf_peak_slices is None or self.ops.tf_peak_slices[-1] is None:
                                self.ops.peak_finding(self.ops.data[:,:,-1], transformed=True)
                            peaks_2d = self.ops.tf_peak_slices[-1]
                        else:
                            if self.ops.peak_slices is None or self.ops.peak_slices[-1] is None:
                                self.ops.peak_finding(self.ops.data[:, :, -1], transformed=False)
                            peaks_2d = self.ops.peak_slices[-1]
                    else:
                        if self.ops._transformed:
                            if self.ops.tf_peak_slices is None or self.ops.tf_peak_slices[self._current_slice] is None:
                                self.ops.peak_finding(self.ops.data[:,:,-1], transformed=True, curr_slice=self._current_slice)
                            peaks_2d = self.ops.tf_peak_slices[self._current_slice]
                        else:
                            if self.ops.peak_slices is None or self.ops.peak_slices[self._current_slice] is None:
                                self.ops.peak_finding(self.ops.data[:,:,-1], transformed=False, curr_slice=self._current_slice)
                            peaks_2d = self.ops.peak_slices[self._current_slice]

                print(peaks_2d.shape)
                for i in range(len(peaks_2d)):
                    pos = QtCore.QPointF(peaks_2d[i][0]-self.size_ops/2, peaks_2d[i][1]-self.size_ops/2)
                    point_obj= pg.CircleROI(pos, self.size_ops, parent=self.imview.getImageItem(), movable=False, removable=True)
                    point_obj.removeHandle(0)
                    self.imview.addItem(point_obj)
                    self._peaks.append(point_obj)
            else:
                [self.imview.removeItem(point) for point in self._peaks]
                self._peaks = []
        else:
            print('You have to select the data first!')
        QtWidgets.QApplication.restoreOverrideCursor()

    def _calc_shift(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        print('Align color channels')
        if self.ops is not None:
            if self.align_btn.isChecked():
                undo_max_proj = False
                if not self.max_proj_btn.isChecked():
                    self.max_proj_btn.setChecked(True)
                    undo_max_proj = True

                if self._shift is None:
                    fm_max = np.copy(self.ops.data[:, :, -1])
                    if self.ops.peaks_2d is None:
                        self.ops.peak_finding(fm_max)
                        #self.ops.wshed_peaks(fm_max)

                    roi_size = 40
                    green_coor = []
                    for i in range(len(self.ops.peaks_2d)):
                        roi_min_0 = int(self.ops.peaks_2d[i][0]-roi_size//2) if int(self.ops.peaks_2d[i][0]-roi_size//2) > 0 else 0
                        roi_min_1 = int(self.ops.peaks_2d[i][1]-roi_size//2) if int(self.ops.peaks_2d[i][1]-roi_size//2) > 0 else 0
                        roi_max_0 = int(self.ops.peaks_2d[i][0]+roi_size//2) if int(self.ops.peaks_2d[i][0]+roi_size//2) < fm_max.shape[0] else fm_max.shape[0]
                        roi_max_1 = int(self.ops.peaks_2d[i][1]+roi_size//2) if int(self.ops.peaks_2d[i][1]+roi_size//2) < fm_max.shape[1] else fm_max.shape[1]
                        green_coor_i = self.ops.peak_finding(self.ops.data[:,:,-2][roi_min_0:roi_max_0, roi_min_1:roi_max_1], roi=True)
                        #green_coor_i = self.ops.wshed_peaks(self.ops.data[:,:,-2][roi_min_0:roi_max_0, roi_min_1:roi_max_1], roi=True)
                        green_coor_0 = green_coor_i[0]+self.ops.peaks_2d[i][0]-roi_size//2 if green_coor_i[0]+self.ops.peaks_2d[i][0]-roi_size//2 > 0 else green_coor_i[0]
                        green_coor_1 = green_coor_i[1]+self.ops.peaks_2d[i][1]-roi_size//2 if green_coor_i[0]+self.ops.peaks_2d[i][0]-roi_size//2 > 0 else green_coor_i[1]
                        green_coor.append((green_coor_0,green_coor_1))

                    self._shift = np.median((np.array(green_coor)-np.array(self.ops.peaks_2d)), axis=0)
                    if undo_max_proj:
                        self.max_proj_btn.setChecked(False)
                
                print('Color shift: ', self._shift)
                self.ops.data[:,:,-2] = interpol.shift(self.ops.data[:,:,-2], -self._shift)
            else:
                self.ops.data[:,:,-2] = interpol.shift(self.ops.data[:,:,-2], self._shift)
            self._update_imview()
        else:
            print('You have to select the data first!')
        QtWidgets.QApplication.restoreOverrideCursor()

        # TODO fix this
        '''
        new_list = align_fm.calc_shift(self.flist, self.ops.data)

        self.fselector.addItems(new_list)
        self.fselector.currentIndexChanged.connect(self._file_changed)

        data_shifted = [np.array(Image.open(fname)) for fname in new_list]
        for i in range(len(data_shifted)):
            self.ops.data.append(data_shifted[i])

        self.align_btn.setEnabled(False)
        '''

    def _mapping(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.ops.calc_mapping()
        self._update_imview()
        if self.remove_tilt_btn.isChecked():
            self._remove_tilt()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _remove_tilt(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.map_btn.isChecked():
            self.ops.remove_tilt(self.remove_tilt_btn.isChecked())
            self._update_imview()
        QtWidgets.QApplication.restoreOverrideCursor()


    def _side_view(self):
        pass

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
        #self._merged = False
        
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


        self._overlay = True
        self._channels = [True, True, True, True]
        #self.ind = 0
        self._curr_folder = None
        self._series = None
        self._current_slice = 0
        self.channel1_btn.setChecked(True)
        self.channel2_btn.setChecked(True)
        self.channel3_btn.setChecked(True)
        self.channel4_btn.setChecked(True)
        self.overlay_btn.setChecked(True)
        self.channel1_btn.setEnabled(False)
        self.channel2_btn.setEnabled(False)
        self.channel3_btn.setEnabled(False)
        self.channel4_btn.setEnabled(False)
        self.overlay_btn.setEnabled(False)

        self.c1_btn.setEnabled(False)
        self.c2_btn.setEnabled(False)
        self.c3_btn.setEnabled(False)
        self.c4_btn.setEnabled(False)
        self.max_proj_btn.setChecked(False)        

        self.define_btn.setEnabled(False)
        self.transform_btn.setEnabled(False)
        self.rot_transform_btn.setEnabled(False)
        self.show_btn.setEnabled(False)
        self.show_btn.setChecked(True)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.peak_btn.setEnabled(False)
        self.align_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.refine_btn.setEnabled(False)   
        #self.auto_opt_btn.setEnabled(False)
        self.fliph.setEnabled(False)
        self.flipv.setEnabled(False)
        self.transpose.setEnabled(False)
        self.rotate.setEnabled(False)
        self.merge_btn.setEnabled(False)

        self.map_btn.setEnabled(False)
        self.map_btn.setChecked(False)
        self.remove_tilt_btn.setEnabled(False)
        self.remove_tilt_btn.setChecked(False)
        
        self.ops.__init__()
