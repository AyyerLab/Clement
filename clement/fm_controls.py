import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from skimage.color import hsv2rgb
import matplotlib
import math
import copy

from .base_controls import BaseControls
from .fm_operations import FM_ops
from . import utils

class SeriesPicker(QtWidgets.QDialog):
    def __init__(self, parent, names):
        super(SeriesPicker, self).__init__(parent)
        self.current_series = 0

        self.setWindowTitle('Pick image series...')
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        self.picker = QtWidgets.QComboBox(self)
        listview = QtWidgets.QListView(self)
        self.picker.setView(listview)
        self.picker.addItems(names)
        self.picker.currentIndexChanged.connect(self.series_changed)
        layout.addWidget(self.picker)

        button = QtWidgets.QPushButton('Confirm', self)
        layout.addWidget(button)
        button.clicked.connect(self.accept)

    def series_changed(self, i):
        self.current_series = i

    def closeEvent(self, event):
        self.current_series = -1
        event.accept()

class FMControls(BaseControls):
    def __init__(self, imview, colors, refine_layout, merge_layout, semcontrols, fibcontrols, giscontrols, temcontrols,
                 tab_index, printer, logger):
        super(FMControls, self).__init__()
        self.tag = 'FM'
        self.imview = imview
        self.ops = None
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)
        self.merge_layout = merge_layout
        self.refine_layout = refine_layout
        self.num_slices = None
        self.peak_controls = None
        self.picker = None
        self.semcontrols = semcontrols
        self.fibcontrols = fibcontrols
        self.giscontrols = giscontrols
        self.temcontrols = temcontrols
        self.tab_index = tab_index

        self._colors = colors
        self._channels = []
        self._overlay = True
        self._curr_folder = None
        self._file_name = None
        self._series = None
        self._current_slice = 0
        self._peaks = []
        self._bead_size = None

        self.orig_size = 10
        self.size = 10
        self.pois = []
        self._pois_orig = []
        self.pois_sizes = []
        self.pois_z = []
        self.pois_err = []
        self.pois_cov = []
        self._pois_channel_indices = []
        self._pois_slices = []


        self.print = printer
        self.log = logger

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
        self.flip_z_btn = QtWidgets.QCheckBox('Flip z')
        self.flip_z_btn.stateChanged.connect(self._flip_z)
        self.flip_z_btn.setEnabled(False)
        line.addWidget(self.flip_z_btn)
        self.slice_select_btn = QtWidgets.QSpinBox(self)
        self.slice_select_btn.setRange(0, 0)
        self.slice_select_btn.setEnabled(False)
        self.slice_select_btn.editingFinished.connect(self._slice_changed)
        self.slice_select_btn.valueChanged.connect(self._slice_changed)
        line.addWidget(self.slice_select_btn)

        # ---- Select channels
        self.channel_line = QtWidgets.QHBoxLayout()
        vbox.addLayout(self.channel_line)
        label = QtWidgets.QLabel('Show color channels:', self)
        self.channel_line.addWidget(label)
        self.channel_btns = []
        self.color_btns = []

        # line.addStretch(1)

        # ---- Peak finding
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Peak finding:', self)
        line.addWidget(label)

        self.set_params_btn = QtWidgets.QPushButton('Set peak finding parameters', self)
        self.set_params_btn.setEnabled(False)

        self.peak_btn = QtWidgets.QPushButton('Show peaks', self)
        self.peak_btn.setCheckable(True)
        self.peak_btn.toggled.connect(self._find_peaks)
        self.peak_btn.setEnabled(False)

        line.addWidget(self.set_params_btn)
        line.addWidget(self.peak_btn)
        line.addStretch(1)

        # 3D mapping
        #line = QtWidgets.QHBoxLayout()
        #vbox.addLayout(line)
        #label = QtWidgets.QLabel('3D mapping: ', self)
        #line.addWidget(label)
        #self.map_btn = QtWidgets.QPushButton('Z mapping', self)
        #self.map_btn.setCheckable(True)
        #self.map_btn.toggled.connect(self._mapping)
        #self.map_btn.setEnabled(False)
        #self.remove_tilt_btn = QtWidgets.QCheckBox('Remove tilt effect', self)
        #self.remove_tilt_btn.setEnabled(False)
        #self.remove_tilt_btn.setChecked(False)
        #self.remove_tilt_btn.stateChanged.connect(self._remove_tilt)
        #line.addWidget(self.map_btn)
        #line.addWidget(self.remove_tilt_btn)
        #line.addStretch(1)

        utils.add_define_grid_line(self, vbox)
        line = utils.add_transform_grid_line(self, vbox, show_original=False)

        # ---- Flips and rotates
        label = QtWidgets.QLabel('Flips:', self)
        line.addWidget(label)

        # self.fliph = QtWidgets.QPushButton('\u2345', self)
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

        # self.flipv = QtWidgets.QPushButton('\u2356', selfroi_pos)
        self.flipv = QtWidgets.QPushButton('', self)
        self.flipv.setObjectName('flipv')
        self.flipv.setCheckable(True)
        self.flipv.setFixedWidth(width)
        self.flipv.setFixedHeight(width)
        self.flipv.setFont(font)
        self.flipv.toggled.connect(self._flipv)
        self.flipv.setEnabled(False)
        line.addWidget(self.flipv)

        # self.transpose = QtWidgets.QPushButton('\u292f', self)
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

        # self.rotate = QtWidgets.QPushButton('\u293e', self)
        self.rotate = QtWidgets.QPushButton('', self)
        self.rotate.setObjectName('rotate')
        self.rotate.setCheckable(True)
        self.rotate.setFixedWidth(width)
        self.rotate.setFixedHeight(width)
        self.rotate.setFont(font)
        self.rotate.toggled.connect(self._rot)
        line.addWidget(self.rotate)
        self.rotate.setEnabled(False)

        self.confirm_btn = QtWidgets.QCheckBox('Confirm orientation', self)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setChecked(False)
        self.confirm_btn.stateChanged.connect(self._confirm_transf)
        line.addWidget(self.confirm_btn)
        line.addStretch(1)

        self.show_btn = QtWidgets.QCheckBox('Show original data', self)
        self.show_btn.setEnabled(False)
        self.show_btn.setChecked(True)
        self.show_btn.stateChanged.connect(self._show_original)
        line.addWidget(self.show_btn)

        # Select points
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Point transform reference:', self)
        line.addWidget(label)

        self.select_btn = QtWidgets.QPushButton('Select references', self)
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self._define_corr_toggled)
        self.select_btn.setEnabled(False)
        self.clear_btn = QtWidgets.QPushButton('Remove references')
        self.clear_btn.clicked.connect(self._clear_points)
        self.clear_btn.setEnabled(False)
        line.addWidget(self.select_btn)
        line.addWidget(self.clear_btn)
        line.addStretch(1)

        # Select POIs
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Points of Interest:', self)
        line.addWidget(label)
        self.poi_ref_btn = QtWidgets.QComboBox()
        listview = QtWidgets.QListView(self)
        self.poi_ref_btn.setView(listview)
        self.poi_ref_btn.setMinimumWidth(100)
        self.poi_ref_btn.setEnabled(False)
        self.poi_btn = QtWidgets.QPushButton('Select POIs', self)
        self.poi_btn.setCheckable(True)
        self.poi_btn.toggled.connect(self._define_poi_toggled)
        self.poi_btn.setEnabled(False)

        line.addWidget(self.poi_ref_btn)
        line.addWidget(self.poi_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        self.refine_layout.addLayout(line)
        line.addStretch(1)
        label = QtWidgets.QLabel('Refinement:', self)
        line.addWidget(label)

        self.refine_btn = QtWidgets.QPushButton('Refine')
        self.refine_btn.clicked.connect(self._refine)
        self.refine_btn.setEnabled(False)
        self.undo_refine_btn = QtWidgets.QPushButton('Undo last refinement')
        self.undo_refine_btn.clicked.connect(self._undo_refinement)
        self.undo_refine_btn.setEnabled(False)
        self.err_plt_btn = QtWidgets.QPushButton('Show error distribution')
        self.err_plt_btn.setEnabled(False)
        self.convergence_btn = QtWidgets.QPushButton('Show RMS convergence')
        self.convergence_btn.setEnabled(False)

        line.addWidget(self.refine_btn)
        line.addWidget(self.undo_refine_btn)
        line.addStretch(1)
        label = QtWidgets.QLabel('Refinement precision [nm]:', self)
        line.addWidget(label)
        self.err_btn = QtWidgets.QLabel('0')
        line.addWidget(self.err_btn)
        line.addWidget(self.err_plt_btn)
        line.addWidget(self.convergence_btn)
        line.addStretch(2)

        line = QtWidgets.QHBoxLayout()
        self.merge_layout.addLayout(line)
        line.addStretch(1)
        label = QtWidgets.QLabel('Merging:', self)
        line.addWidget(label)

        self.merge_btn = QtWidgets.QPushButton('Merge', self)
        self.merge_btn.setEnabled(False)
        label = QtWidgets.QLabel('Progress:')
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setMaximum(100)

        line.addWidget(self.merge_btn)
        line.addWidget(label)
        line.addWidget(self.progress_bar)
        line.addStretch(1)

        vbox.addStretch(1)

        self.show()

    @utils.wait_cursor('print')
    def _update_imview(self, state=None):
        channel_idx = None
        if self.imview.axes['t'] is not None:
            channel_idx = self.imview.timeLine.value()
        if self.ops is None:
            return
        self.print(self.ops.data.shape)
        if self.peak_btn.isChecked():
            self.peak_btn.setChecked(False)
            self.peak_btn.setChecked(True)

        if self.semcontrols.show_peaks_btn.isChecked():
            self.semcontrols.show_peaks_btn.setChecked(False)
            self.semcontrols.show_peaks_btn.setChecked(True)

        if self.ops._show_mapping:
            self.imview.setImage(hsv2rgb(self.ops.data))
            vr = self.imview.getImageItem().getViewBox().targetRect()
            self.imview.getImageItem().getViewBox().setRange(vr, padding=0)
        else:
            self._calc_color_channels()
            old_shape = self.imview.image.shape
            new_shape = self.color_data.shape
            if old_shape == new_shape:
                vr = self.imview.getImageItem().getViewBox().targetRect()
            levels = self.imview.getHistogramWidget().item.getLevels()
            if self.color_data.max() > 0:
                self.imview.setImage(self.color_data, levels=levels)
            else:
                self.imview.setImage(self.color_data)
            if old_shape == new_shape:
                self.imview.getImageItem().getViewBox().setRange(vr, padding=0)
        if channel_idx is not None:
            self.imview.timeLine.setValue(channel_idx)

    def _load_fm_images(self):
        if self._curr_folder is None:
            self._curr_folder = os.getcwd()

        self._file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                   'Select FM file',
                                                                   self._curr_folder,
                                                                   'All(*.lif *.tif *.tiff *.xml);;*.lif;;*.tif;;*.tiff;;*.xml')
        if self._file_name != '':
            self.reset_init()
            self._curr_folder = os.path.dirname(self._file_name)
            self._current_slice = self.slice_select_btn.value()
            self._parse_fm_images(self._file_name)

    @utils.wait_cursor('print')
    def _parse_fm_images(self, file_name, series=None):
        self.log(file_name)
        self.ops = FM_ops(self.print, self.log)
        retval = self.ops.parse(file_name, z=0, series=series)
        if retval is not None:
            self.picker = SeriesPicker(self, retval)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.picker.exec_()
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self._series = self.picker.current_series
            if self._series < 0:
                self.ops = None
                return
            self.ops.parse(file_name, z=0, series=self._series)
            self.print(self.ops.data.shape)

        self.num_slices = self.ops.num_slices
        if file_name != '':
            if self.picker is not None:
                self.fm_fname.setText('File: ' + os.path.basename(file_name) + '; Series: ' + retval[self.picker.current_series]  + '; Slice ' + '[0/%d]' % self.num_slices)
            elif self.picker is None and series is not None:
                series_name = self.ops.base_reader.getSeries()[series].getName()
                self.fm_fname.setText('File: ' + os.path.basename(file_name) + '; Series: ' + series_name + '; Slice ' + '[0/%d]' % self.num_slices)
            else:
                self.fm_fname.setText('File: ' + os.path.basename(file_name) + '; Slice ' + '[0/%d]' % self.num_slices)
            self.slice_select_btn.setRange(0, self.num_slices - 1)

            self._colors = self._colors[:self.ops.num_channels]
            for i in range(1, self.ops.num_channels + 1):
                self._channels.append(True)
                channel_btn = QtWidgets.QCheckBox(' ', self)
                channel_btn.setChecked(True)
                channel_btn.stateChanged.connect(lambda state, channel=(i-1): self._show_channels(state, channel))
                self.channel_btns.append(channel_btn)
                color_btn = QtWidgets.QPushButton(' ', self)
                color_btn.clicked.connect(lambda state, channel=i-1: self._sel_color(state, channel))
                width = color_btn.fontMetrics().boundingRect(' ').width() + 24
                color_btn.setFixedWidth(width)
                color_btn.setMaximumHeight(width)
                if i > (len(self._colors)-1):
                    self._colors.append('#808080')

                color_btn.setStyleSheet('background-color: {}'.format(self._colors[i-1]))
                self.color_btns.append(color_btn)
                self.channel_line.addWidget(color_btn)
                self.channel_line.addWidget(channel_btn)
                self.poi_ref_btn.addItem('Channel ' + str(i))

            self.poi_ref_btn.setCurrentIndex(self.ops.num_channels-1)
            self.overlay_btn = QtWidgets.QCheckBox('Overlay', self)
            self.overlay_btn.stateChanged.connect(self._show_overlay)
            self.channel_line.addWidget(self.overlay_btn)
            self.channel_line.addStretch(1)

            self.imview.setImage(self.ops.data, levels=(self.ops.data.min(), self.ops.data.mean() * 2))
            self._update_imview()
            self.max_proj_btn.setEnabled(True)
            self.max_proj_btn.setChecked(True)
            self.define_btn.setEnabled(True)
            self.set_params_btn.setEnabled(True)
            self.peak_btn.setEnabled(True)
            self.poi_ref_btn.setEnabled(True)
            self.poi_btn.setEnabled(True)
            self.overlay_btn.setChecked(True)
            self.overlay_btn.setEnabled(True)
            if self.num_slices > 1:
                self.flip_z_btn.setEnabled(True)
                self.slice_select_btn.setEnabled(True)

    @utils.wait_cursor('print')
    def _show_max_projection(self, state=None):
        self.slice_select_btn.setEnabled(not self.max_proj_btn.isChecked())
        if self.max_proj_btn.isChecked():
            self.ops.selected_slice = None
        else:
            self.ops.selected_slice = self.slice_select_btn.value()
        if self.ops is not None:
            self.ops.calc_max_projection()
            self._update_imview()

    @utils.wait_cursor('print')
    def _flip_z(self, state=None):
        if self.ops is not None:
            self.ops.toggle_flip_z(self.flip_z_btn.isChecked())
        if not self.max_proj_btn.isChecked():
            self._slice_changed(None)
        if self.tab_index == 1:
            if self.other.show_peaks_btn.isChecked():
                self.other.show_peaks_btn.setChecked(False)
                self.other.show_peaks_btn.setChecked(True)

    @utils.wait_cursor('print')
    def _calc_color_channels(self, state=None):
        self.print('Num channels: ', len(self._channels))
        self.color_data = np.zeros((len(self._channels),) + self.ops.data[:, :, 0].shape + (3,))
        for i in range(len(self._channels)):
            self.log(self._channels[i])
            if self._channels[i]:
                my_channel = self.ops.data[:, :, i]
                my_channel_rgb = np.repeat(my_channel[:, :, np.newaxis], 3, axis=2)
                rgb = tuple([int(self._colors[i][1 + 2 * c:3 + 2 * c], 16) / 255. for c in range(3)])
                self.color_data[i, :, :, :] = my_channel_rgb * rgb
            else:
                self.color_data[i, :, :, :] = np.zeros((self.ops.data[:, :, i].shape + (3,)))

        if self.overlay_btn.isChecked():
            self.color_data = np.sum(self.color_data, axis=0)

    @utils.wait_cursor('print')
    def _show_overlay(self, state=None):
        if self.ops is not None:
            self._overlay = not self._overlay
            self._update_imview()

    @utils.wait_cursor('print')
    def _show_channels(self, checked, my_channel):
        if self.ops is not None and self.ops.data is not None:
            self._channels[my_channel] = not self._channels[my_channel]
            self._update_imview()

    def _sel_color(self, state, index):
        button = self.color_btns[index]
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            cname = color.name()
            self._colors[index] = cname
            button.setStyleSheet('background-color: {}'.format(cname))
            self._update_imview()
        else:
            self.print('Invalid color')

    @utils.wait_cursor('print')
    def _fliph(self, state):
        if self.ops is None:
            return
        if self.fliph.isChecked():
            self.flips[2] = True
        else:
            self.flips[2] = False
        self._remove_points_flip()
        self.ops.flip_horizontal(state)
        self._recalc_grid()
        self._update_imview()
        self._update_pois()

    @utils.wait_cursor('print')
    def _flipv(self, state):
        if self.ops is None:
            return
        if self.flipv.isChecked():
            self.flips[3] = True
        else:
            self.flips[3] = False
        self._remove_points_flip()
        self.ops.flip_vertical(state)
        self._recalc_grid()
        self._update_imview()
        self._update_pois()

    @utils.wait_cursor('print')
    def _trans(self, state):
        if self.ops is None:
            return
        if self.transpose.isChecked():
            self.flips[0] = True
        else:
            self.flips[0] = False
        self._remove_points_flip()
        self.ops.transpose(state)
        self._recalc_grid()
        self._update_imview()
        self._update_pois()

    @utils.wait_cursor('print')
    def _rot(self, state):
        if self.ops is None:
            return
        if self.rotate.isChecked():
            self.flips[1] = True
        else:
            self.flips[1] = False
        self._remove_points_flip()
        self.ops.rotate_clockwise(state)
        self._recalc_grid()
        self._update_imview()
        self._update_pois()

    @utils.wait_cursor('print')
    def _confirm_transf(self, state):
        if self.other.ops is None:
            self.print('You have to load and transform an SEM ')
            self.confirm_btn.setChecked(False)
            return

        self.show_grid_btn.setChecked(False)
        if state:
            self.ops.sem_transform = self.semcontrols.ops.tf_matrix_no_shift
            self.ops.fixed_orientation = True
            self.fixed_orientation = True
            self.other.fixed_orientation = True
            self.ops.orig_tf_peaks = None
            self.flipv.setEnabled(False)
            self.fliph.setEnabled(False)
            self.transpose.setEnabled(False)
            self.rotate.setEnabled(False)
            self.show_btn.setEnabled(False)
            self.semcontrols.show_btn.setChecked(True)
            self.semcontrols.show_btn.setEnabled(False)
            self.semcontrols.define_btn.setEnabled(False)
            self.semcontrols.transform_btn.setEnabled(False)
            self.refine_btn.setEnabled(True)
            self.merge_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
        else:
            self.fixed_orientation = False
            self.ops.fixed_orientation = False
            self.other.fixed_orientation = False
            self.ops.orig_tf_peaks = None
            self.flipv.setEnabled(True)
            self.fliph.setEnabled(True)
            self.transpose.setEnabled(True)
            self.rotate.setEnabled(True)
            self.show_btn.setEnabled(True)
            self.semcontrols.show_btn.setChecked(False)
            self.semcontrols.show_btn.setEnabled(True)
            self.refine_btn.setEnabled(False)
            self.select_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.merge_btn.setEnabled(False)


        self.ops._update_data()
        self._calc_tr_matrices()
        self._update_pois()
        self._recalc_grid()
        self.show_grid_btn.setChecked(True)
        self._update_imview()


    @utils.wait_cursor('print')
    def _slice_changed(self, state=None):
        if self.ops is None:
            self.print('Pick FM image first')
            return

        num = self.slice_select_btn.value()
        if self.ops.flip_z:
            num = self.num_slices - 1 - num
        if num == self._current_slice:
            return
        self.ops.parse(fname=self.ops.old_fname, z=num, reopen=False)
        self._update_imview()
        fname = self.fm_fname.text().split('[')[0]
        self.fm_fname.setText(fname + '[%d/%d]' % (num, self.num_slices))
        self._current_slice = num
        self.slice_select_btn.clearFocus()


    @utils.wait_cursor('print')
    def _find_peaks(self, state=None):
        if self.ops is None:
            self.print('You have to select the data first!')
            return

        if self.ops.adjusted_params == False:
            self.print('You have to adjust and save the peak finding parameters first!')
            self.peak_btn.setChecked(False)
            return

        if not self.peak_btn.isChecked():
            [self.imview.removeItem(point) for point in self._peaks]
            self._peaks = []
            return

        channel = self.peak_controls.peak_channel_btn.currentIndex()
        self.print('Perform peak finding on channel: ', channel+1)

        self.ops.update_peaks(self.ops.tf_matrix, self.ops._transformed)

        if self.ops.peaks is not None:
            for i in range(len(self.ops.peaks)):
                pos = QtCore.QPointF(self.ops.peaks[i,0] - self.size / 2, self.ops.peaks[i,1] - self.size / 2)
                point_obj = pg.CircleROI(pos, self.size, parent=self.imview.getImageItem(), movable=False,
                                         removable=True)
                point_obj.removeHandle(0)
                self.imview.addItem(point_obj)
                self._peaks.append(point_obj)

            if self.num_slices > 1:
                if self.ops.peaks_z is None:
                    self.ops.load_channel(self.peak_controls.peak_channel_btn.currentIndex())
                    self.ops.fit_z(self.ops.channel)

    @utils.wait_cursor('print')
    def _align_colors(self, idx, state):
        if self.ops is None:
            self.print('You have to select the data first!')
            return

        if self.ops.adjusted_params == False:
            self.print('You have to adjust and save the peak finding parameters first!')
            self.peak_controls.action_btns[idx].setChecked(False)
            return

        if not state:
            self.ops._aligned_channels[idx] = False
            self.ops._update_data()
            self._update_imview()
            return

        self.print('Align channel ', idx+1)
        reference = self.peak_controls.peak_channel_btn.currentIndex()
        if idx != reference and np.array_equal(self.ops._color_matrices[idx], np.identity(3)):
            self.ops.aligning = True
            show_transformed = False
            if not self.show_btn.isChecked():
                self.show_btn.setChecked(True)
                show_transformed = True
            undo_max_proj = False
            if not self.max_proj_btn.isChecked():
                self.max_proj_btn.setChecked(True)
                undo_max_proj = True

            peak_channel_idx = self.peak_controls.peak_channel_btn.currentIndex()
            if reference == peak_channel_idx:
                peaks_2d = self.ops.peaks
            else:
                self.peak_controls.peak_channel_btn.setCurrentIndex(reference)
                peaks_2d = None
            if peaks_2d is None:
                self.peak_controls.peak_btn.setChecked(True)
                self.peak_controls.peak_btn.setChecked(False)
                peaks_2d = self.ops.peaks_align_ref
                self.peak_controls.peak_channel_btn.setCurrentIndex(peak_channel_idx)
            self.ops.estimate_alignment(peaks_2d, idx)
            if undo_max_proj:
                self.max_proj_btn.setChecked(False)
            if show_transformed:
                self.show_btn.setChecked(False)

            self.ops.aligning = False

        self.ops._aligned_channels[idx] = True
        self.ops._update_data()
        self._update_imview()

    @utils.wait_cursor('print')
    def _mapping(self, state=None):
        if self.ops.adjusted_params == False:
            self.print('You have to adjust and save the peak finding parameters first!')
            self.map_btn.setChecked(False)
            return
        self.ops.calc_mapping()
        self._update_imview()
        if self.remove_tilt_btn.isChecked():
            self._remove_tilt()

    @utils.wait_cursor('print')
    def _remove_tilt(self, state=None):
        if self.map_btn.isChecked():
            self.ops.remove_tilt(self.remove_tilt_btn.isChecked())
            self._update_imview()

    @utils.wait_cursor('print')
    def _define_poi_toggled(self, checked):
        if checked:
            if self.select_btn.isChecked():
                self.select_btn.setChecked(False)
            self.poi_counter = len(self.pois)
            self.fliph.setEnabled(False)
            self.flipv.setEnabled(False)
            self.transpose.setEnabled(False)
            self.rotate.setEnabled(False)
            self.poi_ref_btn.setEnabled(False)
            self.ops.load_channel(ind=self.poi_ref_btn.currentIndex())
            self.print('Select points of interest on %s image' % self.tag)
        else:
            if self.ops.channel is not None:
                self.ops.clear_channel()
            self.print('Done selecting points of interest on %s image' % self.tag)
            if not self.other._refined or not self.confirm_btn.isChecked():
                self.fliph.setEnabled(True)
                self.flipv.setEnabled(True)
                self.transpose.setEnabled(True)
                self.rotate.setEnabled(True)
            self.poi_ref_btn.setEnabled(True)

    @utils.wait_cursor('print')
    def _define_corr_toggled(self, checked):
        if self.ops.adjusted_params == False:
            self.print('You have to adjust and save the peak finding parameters first!')
            self.select_btn.setChecked(False)
            return

        if self.other.translate_peaks_btn.isChecked() or self.other.refine_peaks_btn.isChecked():
            self.print('You have to uncheck translation buttons on SEM/FIB side first!')
            self.select_btn.setChecked(False)
            return

        if self.other.ops is not None and self.tab_index == 1 and self.other.ops.fib_matrix is None:
            self.print('You have to calculate the grid box for the FIB view first!')
            self.select_btn.setChecked(False)
            return

        if checked:
            if self.poi_btn.isChecked():
                self.poi_btn.setChecked(False)
            self.counter = len(self._points_corr)
            self.fliph.setEnabled(False)
            self.flipv.setEnabled(False)
            self.transpose.setEnabled(False)
            self.rotate.setEnabled(False)

            if self.peak_controls.peak_channel_btn.currentIndex() != self.ops._channel_idx:
                self.ops.load_channel(ind=self.peak_controls.peak_channel_btn.currentIndex())
            if self.other.ops is not None:
                if self.ops.points is not None and self.other.ops.points is not None:
                    if self.tab_index != 1:
                            self._calc_tr_matrices()
            self.print('Select reference points on %s image' % self.tag)
        else:
            if self.ops.channel is not None:
                self.ops.clear_channel()
            self.print('Done selecting reference points on %s image' % self.tag)
            if not self.other._refined:
                self.fliph.setEnabled(True)
                self.flipv.setEnabled(True)
                self.transpose.setEnabled(True)
                self.rotate.setEnabled(True)
            self.poi_ref_btn.setEnabled(True)

    def _calc_tr_matrices(self):
        if not self.fixed_orientation:
            src_init = np.copy(np.array(sorted(self.ops.points, key=lambda k: [np.cos(35 * np.pi / 180) * (k[0]) + k[1]])))
            self.tf_points_indices = [i for i,v in sorted(enumerate(self.ops.points), key=lambda k: [np.cos(35 * np.pi / 180) * (k[1][0]) + k[1][1]])]
            dst_init = np.array(
                sorted(self.semcontrols.ops.points, key=lambda k: [np.cos(35 * np.pi / 180) * k[0] + k[1]]))
            self.other.tf_points_indices = [i for i,v in sorted(enumerate(self.semcontrols.ops.points), key=lambda k: [np.cos(35 * np.pi / 180) * (k[1][0]) + k[1][1]])]

            self.tr_matrices = self.ops.get_transform(src_init, dst_init)
        else:
            src_init = np.copy(self.ops.points[self.tf_points_indices])
            dst_init = np.copy(self.other.ops.points[self.other.tf_points_indices])
            self.tr_matrices = self.ops.get_transform(src_init, dst_init)

    def _calc_pois(self, pos=None):
        if pos is None:
            pass
        else:
            point = np.copy(np.array([pos.x(), pos.y()]))
            self.ops.load_channel(self.poi_ref_btn.currentIndex())
            init = self._fit_poi(point)
            if init is None:
                return

        self._pois_orig.append(init)
        if not self.ops._transformed:
            self._draw_pois(init)
        else:
            self._transform_pois(init[:2])

        # if len(self.pois_z) > 0:
        #    self._draw_correlated_points(QtCore.QPointF(init[0]-self.size/2, init[1]-self.size/2), self.imview.getImageItem(), skip=True)

        # if self.other.show_merge:
        #    self.other.popup._update_poi(pos)

        if self.max_proj_btn.isChecked():
            self._pois_slices.append(None)
        else:
            self._pois_slices.append(self._current_slice)

    def _calc_ellipses(self, counter):
        cov_matrix = np.copy(self.pois_cov[counter])
        if self.ops._transformed:
            cov_matrix = self._update_cov_matrix(cov_matrix)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix[:2, :2])

        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[order]
        vx, vy = eigvecs[0, 0], eigvecs[0, 1]
        theta = -np.arctan2(vy, vx) * 180 / np.pi
        # scale eigenvalues to 75% confidence interval and pixel size
        lambda_1, lambda_2 = 2 * np.sqrt(2.77 * eigvals)
        self.log(lambda_1, lambda_2, theta)
        return lambda_1, lambda_2, theta

    def _update_cov_matrix(self, cov_matrix):
        cov = np.copy(cov_matrix)
        cov[-1, -1] = 0
        tf_cov = self.ops.tf_matrix @ cov @ self.ops.tf_matrix.T
        transp, rot, fliph, flipv = self.flips
        if flipv:
            tf_cov[0, 1] *= -1
            tf_cov[1, 0] *= -1
        if fliph:
            tf_cov[1, 0] *= -1
        if rot:
            cov_2d = np.copy(tf_cov[:2, :2])
            rot_matrix = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)], [np.sin(np.pi / 2), np.cos(np.pi / 2)]])
            cov_2d = rot_matrix @ cov_2d @ rot_matrix.T
            tf_cov[:2, :2] = cov_2d
        if transp:
            # covariance matrices are symmetric...
            pass
        return tf_cov

    def _draw_pois(self, point=None):
        if point is None:
            points = self._pois_orig
        else:
            points = [point]

        for point in points:
            img_center = np.array(self.ops.data.shape) / 2
            lambda_1, lambda_2, theta = self._calc_ellipses(self.poi_counter)
            if math.isnan(lambda_1) or math.isnan(lambda_2):
                del self.pois_err[-1]
                del self.pois_cov[-1]
                del self.pois_z[-1]
                QtWidgets.QApplication.restoreOverrideCursor()
                return

            cmap = matplotlib.cm.get_cmap('cool')
            lin = np.linspace(0, 1, 100)
            colors = cmap(lin)
            #idx = np.argmin(np.abs(lin - self.pois_err[-1][-1]))
            idx = np.argmin(np.abs(lin - self.pois_err[self.poi_counter][-1]))
            color = colors[idx]
            color = matplotlib.colors.to_hex(color)

            size = (lambda_1, lambda_2)
            pos = QtCore.QPointF(point[0] - lambda_1 / 2 + 0.5, point[1] - lambda_2 / 2 + 0.5)
            point_obj = pg.EllipseROI(img_center, size=[size[0], size[1]], angle=0, parent=self.imview.getImageItem(),
                                      movable=False, removable=True, resizable=False, rotatable=False)

            point_obj.setTransformOriginPoint(QtCore.QPointF(lambda_1 / 2, lambda_2 / 2))
            point_obj.setRotation(theta)
            point_obj.setPos([pos.x(), pos.y()])
            point_obj.setPen(color)
            point_obj.removeHandle(0)
            point_obj.removeHandle(0)

            self._pois_channel_indices.append(self.poi_ref_btn.currentIndex())
            self.pois_sizes.append(size)
            self.poi_counter += 1
            annotation_obj = pg.TextItem(str(self.poi_counter), color=color, anchor=(0, 0))
            annotation_obj.setPos(pos.x() + 1, pos.y() + 1)
            self.poi_anno_list.append(annotation_obj)
            point_obj.sigRemoveRequested.connect(lambda: self._remove_pois(point_obj))
            self.pois.append(point_obj)

            self.imview.addItem(point_obj)
            self.imview.addItem(annotation_obj)

    def _fit_poi(self, point):
        # Dont use decorator here because of return value!
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.max_proj_btn.isChecked():
            init, err, cov = self.ops.gauss_3d(point, self.ops._transformed, self.poi_ref_btn.currentIndex(), size=self.size)
        else:
            init, err, cov = self.ops.gauss_3d(point, self.ops._transformed, self.poi_ref_btn.currentIndex(),
                                               slice=self._current_slice, size=self.size)
        if init is None:
            QtWidgets.QApplication.restoreOverrideCursor()
            return None
        init[:2] = (self.ops._color_matrices[self.poi_ref_btn.currentIndex()] @ np.array([init[0], init[1], 1]))[:2]

        self.pois_z.append(init[-1])
        self.pois_err.append(err.tolist())
        self.pois_cov.append(cov.tolist())
        QtWidgets.QApplication.restoreOverrideCursor()
        return init

    def _calc_optimized_position(self, point, pos=None):
        peaks = None
        ind = self.ops.check_peak_index(point, self.size)
        if ind is None and self.tab_index == 1 and not self.other._refined:
            self.print('If the FIB tab is selected, you have to select a bead for the first refinement!')
            return None, None, None
        if ind is not None:
            peaks = self.ops.peaks
            z = self.ops.peaks_z[ind]
        else:
            z = self.ops.calc_z(ind, point, self.ops._transformed, self.poi_ref_btn.currentIndex())
        if z is None:
            self.print('z is None, something went wrong here... Try another bead!')
            return None, None, None
        self.points_corr_z.append(z)

        if ind is not None:
            if pos is not None:
                pos.setX(peaks[ind, 0] - self.size / 2)
                pos.setY(peaks[ind, 1] - self.size / 2)
            init = np.array([peaks[ind, 0], peaks[ind, 1], 1])

        else:
            init = np.array([point[0], point[1], 1])
        return init, pos, z

    def _draw_correlated_points(self, pos, item, skip=False):
        point = np.array([pos.x() + self.size / 2, pos.y() + self.size / 2])
        if not skip:
            init, pos, z = self._calc_optimized_position(point, pos)
        else:
            init = np.array([point[0], point[1], 1])
            z = self.pois_z[-1]
        if init is None:
            return
        self._draw_fm_points(pos, item)
        self._draw_em_points(init, z)

    def _draw_fm_points(self, pos, item):
        point_obj = pg.CircleROI(pos, self.size, parent=item, movable=True, removable=True)
        point_obj.setPen(0, 255, 0)
        point_obj.removeHandle(0)
        self.imview.addItem(point_obj)

        self._points_corr.append(point_obj)
        self._orig_points_corr.append([pos.x() + self.size // 2, pos.y() + self.size // 2])
        self.counter += 1
        annotation_obj = pg.TextItem(str(self.counter), color=(0, 255, 0), anchor=(0, 0))
        annotation_obj.setPos(pos.x() + 5, pos.y() + 5)
        self.imview.addItem(annotation_obj)
        self.anno_list.append(annotation_obj)

        self._points_corr_indices.append(self.counter - 1)
        point_obj.sigRemoveRequested.connect(lambda: self._remove_correlated_points(point_obj))

    def _draw_em_points(self, init, z):
        if self.other.ops is None:
            return
        condition = False
        if self.ops._tf_points is not None:
            if self.tab_index == 1:
                if self.other.sem_ops is not None:
                    if self.other.sem_ops._tf_points is not None or self.other.sem_ops._tf_points_region is not None:
                        condition = True
            else:
                if self.other.ops._tf_points is not None or self.other.ops._tf_points_region is not None:
                    condition = True
        if not condition:
            self.print('Transform both images before point selection')
            return
        if self.tr_matrices is None:
            return

        self.log('2d init: ', init)
        self.log('Other class:', self.other, self.other.ops)
        self.log('tr_matrices:\n', self.other.tr_matrices)
        if self.tab_index == 1:
            transf = np.dot(self.tr_matrices, init)
            transf = self.other.ops.fib_matrix @ np.array([transf[0], transf[1], z, 1])
            self.other.points_corr_z.append(transf[2])
            if self.other._refined:
                transf[:2] = (self.other.ops._refine_matrix @ np.array([transf[0], transf[1], 1]))[:2]
        else:
            transf = np.dot(self.tr_matrices, init)

        self.print('Transformed point: ', transf)
        pos = QtCore.QPointF(transf[0] - self.other.size / 2, transf[1] - self.other.size / 2)
        point_other = pg.CircleROI(pos, self.other.size, parent=self.other.imview.getImageItem(),
                                   movable=True, removable=True)
        point_other.setPen(0, 255, 255)
        point_other.removeHandle(0)
        self.other.imview.addItem(point_other)
        self.other._points_corr.append(point_other)
        self.other._orig_points_corr.append([transf[0], transf[1]])

        self.other.counter += 1
        annotation_other = pg.TextItem(str(self.other.counter), color=(0, 255, 255), anchor=(0, 0))
        annotation_other.setPos(pos.x() + 5, pos.y() + 5)
        self.other.imview.addItem(annotation_other)
        self.other.anno_list.append(annotation_other)

        point_other.sigRemoveRequested.connect(lambda: self._remove_correlated_points(point_other))
        point_other.sigRegionChangeFinished.connect(lambda: self._update_annotations(point_other))

        self.other._points_corr_indices.append(self.counter - 1)

        for i in range(len(self.other.peaks)):
            if self.other.peaks[i].pos() == self.other._points_corr[-1].pos():
                self.other.imview.removeItem(self.other.peaks[i])

    def _transform_pois(self, point=None):
        '''
        Transform POIs if they were defined before aligning the FM image
        '''
        if point is None:
            #[self.imview.removeItem(p) for p in self.pois]
            #[self.imview.removeItem(a) for a in self.poi_anno_list]
            points_tf = []
            for i in range(len(self._pois_orig)):
                point = self._pois_orig[i]
                self._remove_pois(point, remove_base=False, transformed=False)
                point_tf = self.ops.tf_matrix @ np.array([point[0], point[1], 1])
                points_tf.append(point_tf[:2])
                self._draw_pois(np.array([points_tf[i][0], points_tf[i][1]]))
        else:
            point_tf = self.ops.tf_matrix @ np.array([point[0], point[1], 1])
            self._draw_pois(np.array([point_tf[0], point_tf[1]]))

    def _update_annotations(self, point):
        idx = None
        for i in range(len(self.other._points_corr)):
            self.log(self.other._points_corr[i])
            if self.other._points_corr[i] == point:
                idx = i
                break

        anno = self.other.anno_list[idx]
        self.other.imview.removeItem(anno)
        anno.setPos(point.x() + 5, point.y() + 5)
        self.other.imview.addItem(anno)

    def _update_pois(self, point=None):
        for _ in range(len(self.pois)):
            self._remove_pois(self.pois[0], remove_base=False)
        for i in range(len(self._pois_orig)):
            self._transform_pois(self._pois_orig[i])

    def _remove_pois(self, point, remove_base=True, transformed=True):
        idx = None
        if transformed:
            for i in range(len(self.pois)):
                if self.pois[i] == point:
                    idx = i
                    break
        else:
            for i in range(len(self._pois_orig)):
                if np.allclose(self._pois_orig[i], point):
                    idx = i
                    break
        if idx is None:
            return
        self.imview.removeItem(self.pois[idx])
        self.imview.removeItem(self.poi_anno_list[idx])
        self.pois.remove(self.pois[idx])
        self.poi_anno_list.remove(self.poi_anno_list[idx])
        self._pois_channel_indices.remove(self._pois_channel_indices[idx])
        if remove_base:
            self._pois_slices.remove(self._pois_slices[idx])
            self.pois_z.remove(self.pois_z[idx])
            self.pois_sizes.remove(self.pois_sizes[idx])
            self.pois_err.remove(self.pois_err[idx])
            self.pois_cov.remove(self.pois_cov[idx])
            self._pois_orig.remove(self._pois_orig[idx])

        for i in range(idx, len(self.pois)):
            self.poi_anno_list[i].setText(str(i + 1))

        self.poi_counter -= 1

    def _remove_correlated_points(self, point):
        idx = self._check_point_idx(point)
        num_beads = len(self._points_corr)

        #Remove FM beads information
        for i in range(len(self.peaks)):
            pos = [self.peaks[i].original_pos[0] + self.orig_size/2, self.peaks[i].original_pos[1] + self.orig_size/2]
            if np.allclose(pos, self._orig_points_corr[idx]):
                self.peaks[i].resetPos()
                self.imview.addItem(self.peaks[i])

        for i in range(len(self.other.peaks)):
            pos = [self.other.peaks[i].original_pos[0] + self.other.orig_size / 2, self.other.peaks[i].original_pos[1] + self.other.orig_size / 2]
            if np.allclose(pos, self.other._orig_points_corr[idx]):
                self.other.peaks[i].resetPos()
                self.other.imview.addItem(self.other.peaks[i])

        self.imview.removeItem(self._points_corr[idx])
        self._points_corr.remove(self._points_corr[idx])
        self._orig_points_corr.remove(self._orig_points_corr[idx])
        if len(self.anno_list) > 0:
            self.imview.removeItem(self.anno_list[idx])
            self.anno_list.remove(self.anno_list[idx])
        if len(self._points_corr_indices) > 0:
            self._points_corr_indices.remove(self._points_corr_indices[idx])
        if self.tab_index == 1 and len(self.points_corr_z) > 0:
            self.points_corr_z.remove(self.points_corr_z[idx])
        for i in range(idx, len(self._points_corr)):
            self.anno_list[i].setText(str(i+1))
            self._points_corr_indices[i] -= 1

        # Remove EM beads information
        if len(self.other._points_corr) == num_beads:
            self.other.imview.removeItem(self.other._points_corr[idx])
            self.other._points_corr.remove(self.other._points_corr[idx])
            self.other._orig_points_corr.remove(self.other._orig_points_corr[idx])
            if len(self.other.anno_list) > 0:
                self.other.imview.removeItem(self.other.anno_list[idx])
                self.other.anno_list.remove(self.other.anno_list[idx])
            if len(self.other._points_corr_indices) > 0:
                self.other._points_corr_indices.remove(self.other._points_corr_indices[idx])
            if self.tab_index == 1:
                if len(self.other.points_corr_z) > 0:
                    self.other.points_corr_z.remove(self.other.points_corr_z[idx])
            for i in range(idx, len(self._points_corr)):
                self.other.anno_list[i].setText(str(i+1))
                self.other._points_corr_indices[i] -= 1
            self.log(self.other._points_corr_indices)

        self.counter -= 1
        self.other.counter -= 1

    def _clear_points(self):
        self.counter = 0
        self.other.counter = 0
        # Remove circle from imviews
        [self.imview.removeItem(point) for point in self._points_corr]
        [self.fibcontrols.imview.removeItem(point) for point in self.fibcontrols._points_corr]
        [self.semcontrols.imview.removeItem(point) for point in self.semcontrols._points_corr]
        [self.temcontrols.imview.removeItem(point) for point in self.temcontrols._points_corr]

        # Remove ROI
        self._points_corr = []
        self.fibcontrols._points_corr = []
        self.semcontrols._points_corr = []
        self.temcontrols._points_corr = []
        # Remove original position
        self._orig_points_corr = []
        self.fibcontrols._orig_points_corr = []
        self.semcontrols._orig_points_corr = []
        self.temcontrols._orig_points_corr = []

        # Remove annotation
        if len(self.anno_list) > 0:
            [self.imview.removeItem(anno) for anno in self.anno_list]
            [self.fibcontrols.imview.removeItem(anno) for anno in self.fibcontrols.anno_list]
            [self.semcontrols.imview.removeItem(anno) for anno in self.semcontrols.anno_list]
            [self.temcontrols.imview.removeItem(anno) for anno in self.temcontrols.anno_list]
            self.anno_list = []
            self.fibcontrols.anno_list = []
            self.semcontrols.anno_list = []
            self.temcontrols.anno_list = []

        # Remove correlation index
        self._points_corr_indices = []
        self.fibcontrols._points_corr_indices = []
        self.semcontrols._points_corr_indices = []
        self.temcontrols._points_corr_indices = []

        # Remove FIB z-position
        self.points_corr_z = []
        self.fibcontrols.points_corr_z = []
        self.semcontrols.points_corr_z = []
        self.temcontrols.points_corr_z = []

        if self.other.show_peaks_btn.isChecked():
            self.other.show_peaks_btn.setChecked(False)
            self.other.show_peaks_btn.setChecked(True)

    def _remove_points_flip(self):
        for i in range(len(self._points_corr)):
            self._remove_correlated_points(self._points_corr[0])
        for i in range(len(self.pois)):
            self._remove_pois(self.pois[0], remove_base=False)

    @utils.wait_cursor('print')
    def _refine(self, state=None):
        ''' self is FM, other is SEM/FIB etc.'''
        if self.select_btn.isChecked():
            self.print('Confirm point selection! (Uncheck Select points of interest)')
            return

        if self.other.translate_peaks_btn.isChecked() or self.other.refine_peaks_btn.isChecked():
            self.print('Confirm peak translation (uncheck collective/individual translation)')
            return

        if len(self._points_corr) < 4:
            self.print('Select at least 4 points for refinement!')
            return

        self.print('Processing shown FM peaks: %d peaks refined' % len(self._points_corr))
        self.print('Refining...')
        dst = np.array([[point.x() + self.other.size / 2, point.y() + self.other.size / 2] for point in
                        self.other._points_corr])
        src = np.array([[point[0], point[1]] for point in self.other._orig_points_corr])

        self._points_corr_history.append(copy.copy(self._points_corr))
        self._points_corr_z_history.append(copy.copy(self.points_corr_z))
        self._orig_points_corr_history.append(copy.copy(self._orig_points_corr))
        self.other._points_corr_history.append(copy.copy(self.other._points_corr))
        self.other._points_corr_z_history.append(copy.copy(self.other.points_corr_z))
        self.other._orig_points_corr_history.append(copy.copy(self.other._orig_points_corr))
        self._fib_vs_sem_history.append(self.tab_index)
        self.other._size_history.append(self.other.size)

        self.other.ops.merged[self.tab_index] = None

        refine_matrix_old = copy.copy(self.other.ops._refine_matrix)
        self.other.ops.calc_refine_matrix(src, dst, ind=self.tab_index)

        self.other.ops.apply_refinement()
        self.other._refined = True
        self.other._recalc_grid()
        if self.other.show_grid_btn.isChecked():
            self.other._show_grid()
        self._estimate_precision(self.tab_index, refine_matrix_old)
        self.other.size = self.other.orig_size

        self.fliph.setEnabled(False)
        self.flipv.setEnabled(False)
        self.transpose.setEnabled(False)
        self.rotate.setEnabled(False)
        self.confirm_btn.setEnabled(False)
        self.err_plt_btn.setEnabled(True)
        self.convergence_btn.setEnabled(True)
        self.undo_refine_btn.setEnabled(True)

        for i in range(len(self._points_corr)):
            self._remove_correlated_points(self._points_corr[0])

        self._update_imview()
        if self.other.show_peaks_btn.isChecked():
            self.other.show_peaks_btn.setChecked(False)
            self.other.show_peaks_btn.setChecked(True)

        self._points_corr = []
        self.points_corr_z = []
        self._orig_points_corr = []
        self.other._points_corr = []
        self.other.points_corr_z = []
        self.other._orig_points_corr = []

        if self.tab_index == 0 or self.tab_index == 3:
            self._calc_tr_matrices()

    def _undo_refinement(self):
        ''' self is FM, other is SEM/FIB etc.'''
        self.other.ops.undo_refinement()
        for i in range(len(self._points_corr)):
            self._remove_correlated_points(self._points_corr[0])

        if len(self.other.ops._refine_history) == 1:
            self.fliph.setEnabled(True)
            self.flipv.setEnabled(True)
            self.rotate.setEnabled(True)
            self.transpose.setEnabled(True)
            self.confirm_btn.setEnabled(True)
            self.other._refined = False
            self.other._recalc_grid()
            self.undo_refine_btn.setEnabled(False)
            self.err_btn.setText('0')
            self.err_plt_btn.setEnabled(False)
            self.convergence_btn.setEnabled(False)
            self.other.grid_box.movable = True
        else:
            self.other._recalc_grid()
            self.other._points_corr_indices = np.arange(len(self.other._points_corr)).tolist()

        if self.tab_index == 0 or self.tab_index == 3:
            self._calc_tr_matrices()
        #print('tr matrices after undo refinement: \n', self.tr_matrices)
        #self.other.size = copy.copy(self.other._size_history[-1])
        # self.other._orig_points_corr = self.other._orig_points_corr_history[-1]
        # self.other._points_corr = []
        # self.other._orig_points_corr = []
        # for i in range(len(self._points_corr_history[-1])):
        #    point = self._points_corr_history[-1][i]
        #    self._draw_correlated_points(point.pos(), self.imview.getImageItem())
        #    point_other = self.other._points_corr_history[-1][i]
        #    self.other._points_corr[i].setPos(point_other.pos())
        #    #self._peak_to_point(self.other._points_corr[i])
        #    self.other.anno_list[i].setPos(point_other.pos().x()+5, point_other.pos().y()+5)

        # if self.other.show_peaks_btn.isChecked():
        #    self.other.size = self.size
        #    self.other.show_peaks_btn.setChecked(False)
        #    self.other.show_peaks_btn.setChecked(True)

        self.other.ops.merged[self.tab_index] = None

        id = len(self._fib_vs_sem_history) - self._fib_vs_sem_history[::-1].index(self.tab_index) - 1

        del self._fib_vs_sem_history[id]
        del self._points_corr_history[-1]
        del self._points_corr_z_history[-1]
        del self._orig_points_corr_history[-1]

        del self.other._points_corr_history[-1]
        del self.other._points_corr_z_history[-1]
        del self.other._orig_points_corr_history[-1]
        del self.other._size_history[-1]

        if len(self.other.ops._refine_history) > 1:
            idx = self.tab_index
            self._estimate_precision(idx, self.other.ops._refine_matrix)
            #self.other.size = copy.copy(self.size)

        if self.other.show_peaks_btn.isChecked():
            self.other.show_peaks_btn.setChecked(False)
            self.other.show_peaks_btn.setChecked(True)

    @utils.wait_cursor('print')
    def _estimate_precision(self, idx, refine_matrix_old):
        sel_points = [[point.x() + self.other.size / 2, point.y() + self.other.size / 2] for point in
                      self.other._points_corr_history[-1]]
        orig_fm_points = np.copy(self._points_corr_history[-1])

        self.refined_points = []
        corr_points = []
        if self.tab_index == 0 or self.tab_index == 3:
            self._calc_tr_matrices()
        if idx != 1:
            for i in range(len(orig_fm_points)):
                orig_point = np.array([orig_fm_points[i].x(), orig_fm_points[i].y()])
                init = np.array([orig_point[0] + self.size // 2, orig_point[1] + self.size // 2, 1])
                corr_points.append(np.copy((self.tr_matrices @ init)[:2]))
                transf = self.tr_matrices @ init
                self.refined_points.append(transf[:2])
        else:
            # orig_fm_points_z = np.copy(self.points_corr_z)
            orig_fm_points_z = np.copy(self._points_corr_z_history[-1])
            for i in range(len(orig_fm_points)):
                orig_point = np.array([orig_fm_points[i].x(), orig_fm_points[i].y()])
                z = orig_fm_points_z[i]
                init = np.array([orig_point[0] + self.size // 2, orig_point[1] + self.size // 2, 1])
                transf = np.dot(self.tr_matrices, init)
                transf = self.other.ops.fib_matrix @ np.array([transf[0], transf[1], z, 1])
                corr_points.append(np.copy(transf[:2]))
                transf[:2] = (self.other.ops._refine_matrix @ np.array([transf[0], transf[1], 1]))[:2]
                self.refined_points.append(transf[:2])
        #print('sel points: \n', sel_points)

        self.diff = np.array(sel_points) - np.array(self.refined_points)
        self.log(self.diff.shape)
        self.diff *= self.other.ops.pixel_size[0]
        self.other.cov_matrix, self.other._std[0], self.other._std[1], self.other._dist = self.other.ops.calc_error(
            self.diff)

        self.other._err = self.diff
        self.err_btn.setText(
            'x: \u00B1{:.2f}, y: \u00B1{:.2f}'.format(self.other._std[0], self.other._std[1]))

        if len(corr_points) >= self.min_conv_points:
            min_points = self.min_conv_points - 4
            convergence = self.other.ops.calc_convergence(corr_points, sel_points, min_points, refine_matrix_old)
            self.other._conv = convergence
        else:
            self.other._conv = None

    def perform_merge(self):
        if not self.other._refined:
            self.print('You have to do at least one round of refinement before merging is allowed!')
            return False
        if not self.fixed_orientation:
            self.print('Merge only allowed when orientation is confirmed!')
            return

        if self.tab_index == 1:
            size_em = copy.copy(self.other._size_history[-1])
            dst = np.array([[point.x() + size_em / 2, point.y() + size_em / 2] for point in
                        self.other._points_corr_history[-1]])
        else:
            size_em = copy.copy(self.other._size_history[-1])
            dst = np.array([[point.x() + size_em / 2, point.y() + size_em / 2] for point in
                        self.other._points_corr_history[-1]])

        src = np.array([[point.x() + self.size / 2, point.y() + self.size / 2]
                        for point in self._points_corr_history[-1]])

        if self.tab_index == 0 or self.tab_index == 3:
            if self.other.ops.merged[self.tab_index] is None:
                for i in range(self.ops.num_channels):
                    self.other.ops.apply_merge_2d(self.ops.data[:,:,i], i, self.tr_matrices,
                                                self.ops.num_channels, self.tab_index)
                    self.progress_bar.setValue(int(np.rint((i + 1) / self.ops.num_channels * 100)))
                self.other.progress = 100
            else:
                self.progress_bar.setValue(100)
                self.other.progress = 100
            if self.other.ops.merged[self.tab_index] is not None:
                self.print('Merged shape: ', self.other.ops.merged[self.tab_index].shape)
            self.other.show_merge = True
            return True
        else:
            if self.other.ops.merged[self.tab_index] is None:
                if self.other._refined:
                    src_z = copy.copy(self._points_corr_z_history[-1])
                    for i in range(self.ops.num_channels):
                        self.ops.load_channel(i)
                        orig_coor = []
                        for k in range(len(src)):
                            #test = self.ops.calc_original_coordinates(tf_aligned_orig_shift, src[k])
                            idx = self.ops.check_peak_index(src[k], self.size)
                            orig_pt = self.ops.peaks_orig[idx]
                            orig_coor.append(orig_pt)

                        tf_matrix_aligned = self.ops.tf_matrix @ self.ops._color_matrices[i]

                        self.other.ops.apply_merge_3d(self.ops.channel, tf_matrix_aligned, self.tr_matrices,
                                                    orig_coor, src_z, dst, i, self.num_slices, self.ops.num_channels,
                                                    self.ops.norm_factor, self.tab_index, self.fibcontrols)
                        self.progress_bar.setValue(int(np.rint((i + 1) / self.ops.num_channels * 100)))
                    self.other.progress = 100
                else:
                    self.print('You have to perform at least one round of refinement before you can merge the images!')
            else:
                self.progress_bar.setValue(100)
                self.other.progress = 100
            if self.other.ops.merged[self.tab_index] is not None:
                self.print('Merged shape: ', self.other.ops.merged[self.tab_index].shape)
            self.other.show_merge = True
            return True

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())


    @utils.wait_cursor('print')
    def reset_init(self, state=None):
        # self.ops = None
        # self.other = None # The other controls object
        if self.show_grid_btn.isChecked():
            if self.ops._transformed:
                if self.tr_grid_box is not None:
                    self.imview.removeItem(self.tr_grid_box)
            else:
                if self.grid_box is not None:
                    self.imview.removeItem(self.grid_box)

        self.peak_btn.setChecked(False)
        self.peak_btn.setEnabled(False)

        self._box_coordinate = None
        self._points_corr = []
        self._points_corr_z = []
        self._orig_points_corr = []
        self._points_corr_indices = []
        self._refined = False
        self._err = [None, None, None]
        self._std = [[None, None], [None, None], [None, None], [None, None]]
        self._conv = [None, None, None]
        self._dist = None

        self._points_corr_history = []
        self._points_corr_z_history = []
        self._orig_points_corr_history = []
        self._fib_vs_sem_history = []
        self._size_history = []

        self.tr_matrices = None
        self.show_grid_box = False
        self.show_tr_grid_box = False
        self.clicked_points = []
        self.grid_box = None
        self.tr_grid_box = None
        self.redo_tr = False
        self.setContentsMargins(0, 0, 0, 0)
        self.counter = 0
        self.anno_list = []

        self._overlay = True
        self._channels = []
        # self.ind = 0
        self._curr_folder = None
        self._series = None

        self.peak_controls = None

        self._current_slice = 0
        self.clearLayout(self.channel_line)
        label = QtWidgets.QLabel('Show color channels:', self)
        self.channel_line.addWidget(label)

        for i in range(len(self.channel_btns)):
        #    self.align_menu.removeAction(self.action_btns[i])
            self.poi_ref_btn.removeItem(0)

        self.channel_btns = []
        self.color_btns = []

        self.max_proj_btn.setChecked(False)

        self.define_btn.setEnabled(False)
        self.transform_btn.setEnabled(False)
        #self.rot_transform_btn.setEnabled(False)
        self.show_btn.setEnabled(False)
        self.show_btn.setChecked(True)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)

        self.poi_ref_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.refine_btn.setEnabled(False)

        self.fliph.setEnabled(False)
        self.flipv.setEnabled(False)
        self.transpose.setEnabled(False)
        self.rotate.setEnabled(False)
        self.fliph.setChecked(False)
        self.flipv.setChecked(False)
        self.transpose.setChecked(False)
        self.rotate.setChecked(False)

        self.set_params_btn.setEnabled(False)

        self.merge_btn.setEnabled(False)

        #self.map_btn.setEnabled(False)
        #self.map_btn.setChecked(False)
        #self.remove_tilt_btn.setEnabled(False)
        #self.remove_tilt_btn.setChecked(False)

        self.err_btn.setText('0')
        self.err_plt_btn.setEnabled(False)
        self.convergence_btn.setEnabled(False)

        self.ops.__init__(self.print, self.log)

