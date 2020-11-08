import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import scipy.ndimage.interpolation as interpol

from skimage.color import hsv2rgb

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
    def __init__(self, imview, colors, merge_layout, printer, logger):
        super(FMControls, self).__init__()
        self.tag = 'FM'
        self.imview = imview
        self.ops = None
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)
        self.merge_layout = merge_layout
        self.num_slices = None
        self.peak_controls = None

        self._colors = colors
        self._channels = []
        self._overlay = True
        self._curr_folder = None
        self._file_name = None
        self._series = None
        self._current_slice = 0
        self._peaks = []
        self._bead_size = None

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
        self.slice_select_btn = QtWidgets.QSpinBox(self)
        self.slice_select_btn.setRange(0, 0)
        self.slice_select_btn.setEnabled(False)
        self.slice_select_btn.editingFinished.connect(self._slice_changed)
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

        self.point_ref_btn = QtWidgets.QComboBox()
        listview = QtWidgets.QListView(self)
        self.point_ref_btn.setView(listview)
        #self.point_ref_btn.currentIndexChanged.connect(self._change_point_ref)
        self.point_ref_btn.setMinimumWidth(100)
        self.point_ref_btn.setEnabled(False)

        self.select_btn = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self._define_corr_toggled)
        self.select_btn.setEnabled(False)


        line.addWidget(self.point_ref_btn)
        line.addWidget(self.select_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        self.merge_layout.addLayout(line)
        line.addStretch(1)
        label = QtWidgets.QLabel('Refinement & Merging:', self)
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

        self.merge_btn = QtWidgets.QPushButton('Merge', self)
        self.merge_btn.setEnabled(False)
        label = QtWidgets.QLabel('Progress:')
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setMaximum(100)
        line.addWidget(self.refine_btn)
        line.addWidget(self.undo_refine_btn)

        line.addWidget(self.merge_btn)
        line.addWidget(label)
        line.addWidget(self.progress_bar)

        line.addStretch(0.5)
        label = QtWidgets.QLabel('Refinement precision [nm]:', self)
        line.addWidget(label)
        self.err_btn = QtWidgets.QLabel('0')
        line.addWidget(self.err_btn)
        line.addWidget(self.err_plt_btn)
        line.addWidget(self.convergence_btn)
        line.addStretch(1)
        vbox.addStretch(1)

        self.show()

    @utils.wait_cursor('print')
    def _update_imview(self, state=None):
        if self.ops is None:
            return
        self.print(self.ops.data.shape)
        if self.peak_btn.isChecked():
            self.peak_btn.setChecked(False)
            self.peak_btn.setChecked(True)
        if self.ops._show_mapping:
            self.imview.setImage(hsv2rgb(self.ops.data))
            vr = self.imview.getImageItem().getViewBox().targetRect()
            self.imview.getImageItem().getViewBox().setRange(vr, padding=0)
        else:
            self._calc_color_channels()
            self.imview.setImage(self.color_data)
            vr = self.imview.getImageItem().getViewBox().targetRect()
            self.imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _load_fm_images(self):
        if self._curr_folder is None:
            self._curr_folder = os.getcwd()

        self._file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                   'Select FM file',
                                                                   self._curr_folder,
                                                                   '*.lif;;*.tif;;*.tiff')
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
            picker = SeriesPicker(self, retval)
            QtWidgets.QApplication.restoreOverrideCursor()
            picker.exec_()
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self._series = picker.current_series
            if self._series < 0:
                self.ops = None
                return
            self.ops.parse(file_name, z=0, series=self._series)
            self.print(self.ops.data.shape)

        self.num_slices = self.ops.num_slices
        if file_name != '':
            self.fm_fname.setText(os.path.basename(file_name) + ' [0/%d]' % self.num_slices)
            self.slice_select_btn.setRange(0, self.num_slices - 1)

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
                self.point_ref_btn.addItem('Channel ' + str(i))

            self.point_ref_btn.setCurrentIndex(self.ops.num_channels-1)
            self.overlay_btn = QtWidgets.QCheckBox('Overlay', self)
            self.overlay_btn.stateChanged.connect(self._show_overlay)
            self.channel_line.addWidget(self.overlay_btn)
            self.channel_line.addStretch(1)

            self.imview.setImage(self.ops.data, levels=(self.ops.data.min(), self.ops.data.mean() * 2))
            self._update_imview()
            self.max_proj_btn.setEnabled(True)
            self.max_proj_btn.setChecked(True)
            self.slice_select_btn.setEnabled(True)
            self.define_btn.setEnabled(True)
            self.set_params_btn.setEnabled(True)
            self.peak_btn.setEnabled(True)
            self.map_btn.setEnabled(True)
            self.remove_tilt_btn.setEnabled(True)

            self.overlay_btn.setChecked(True)
            self.overlay_btn.setEnabled(True)

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
    def _calc_color_channels(self, state=None):
        self.color_data = np.zeros((len(self._channels),) + self.ops.data[:, :, 0].shape + (3,))
        self.print('Num channels: ', len(self._channels))
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
        if self.other.fib and self.other.ops is not None:
            self._store_fib_flips(idx=2)
        if self.fliph.isChecked():
            self.flips[2] = True
        else:
            self.flips[2] = False
        self._remove_points_flip()
        self.ops.flip_horizontal(state)
        self._recalc_grid()
        self._update_imview()

    @utils.wait_cursor('print')
    def _flipv(self, state):
        if self.ops is None:
            return
        if self.other.fib and self.other.ops is not None:
            self._store_fib_flips(idx=3)
        if self.flipv.isChecked():
            self.flips[3] = True
        else:
            self.flips[3] = False
        self._remove_points_flip()
        self.ops.flip_vertical(state)
        self._recalc_grid()
        self._update_imview()

    @utils.wait_cursor('print')
    def _trans(self, state):
        if self.ops is None:
            return
        if self.other.fib and self.other.ops is not None:
            self._store_fib_flips(idx=0)
        if self.transpose.isChecked():
            self.flips[0] = True
        else:
            self.flips[0] = False
        self._remove_points_flip()
        self.ops.transpose(state)
        self._recalc_grid()
        self._update_imview()

    @utils.wait_cursor('print')
    def _rot(self, state):
        if self.ops is None:
            return
        if self.other.fib and self.other.ops is not None:
            self._store_fib_flips(idx=1)
        if self.rotate.isChecked():
            self.flips[1] = True
        else:
            self.flips[1] = False
        self._remove_points_flip()
        self.ops.rotate_clockwise(state)
        self._recalc_grid()
        self._update_imview()

    @utils.wait_cursor('print')
    def _slice_changed(self, state=None):
        if self.ops is None:
            self.print('Pick FM image first')
            return

        num = self.slice_select_btn.value()
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
            # self.max_proj_btn.setChecked(show_max_proj)
            return

        flip = False
        if self.ops._transformed and self.ops.orig_tf_peak_slices is None:
            fliph, flipv, transp, rot = self.ops.fliph, self.ops.flipv, self.ops.transp, self.ops.rot
            self.ops.fliph = False
            self.ops.flipv = False
            self.ops.rot = False
            self.ops.transp = False
            self.ops._update_data()
            flip = True

        channel = self.peak_controls.peak_channel_btn.currentIndex()
        self.print('Perform peak finding on channel: ', channel+1)
        peaks_2d = None
        if self.ops._transformed:
            if self.map_btn.isChecked() or self.max_proj_btn.isChecked():
                peaks_2d = self.ops.peak_slices[-1]
                if self.ops.tf_peak_slices is None or self.ops.tf_peak_slices[-1] is None:
                    self.ops.calc_transformed_coordinates(peaks_2d, self.ops.tf_matrix, self.peak_controls.data_roi.shape,
                                                          self.peak_controls.roi_pos)
                peaks_2d = self.ops.tf_peak_slices[-1]
            else:
                peaks_2d = self.ops.peak_slices[self._current_slice]
                if self.ops.tf_peak_slices is None or self.ops.tf_peak_slices[self._current_slice] is None:
                    self.ops.calc_transformed_coordinates(peaks_2d, self.ops.tf_matrix, self.peak_controls.data_roi.shape,
                                                          self.peak_controls.roi_pos, slice=self._current_slice)
                peaks_2d = self.ops.tf_peak_slices[self._current_slice]
        else:
            if self.map_btn.isChecked():
                #if self.ops.peak_slices is None or self.ops.peak_slices[-1] is None:
                #    self.ops.peak_finding(self.peak_controls.data_roi[:, :, channel], transformed=False,
                #                          roi_pos=self.peak_controls.roi_pos)
                peaks_2d = self.ops.peak_slices[-1]
            elif self.max_proj_btn.isChecked():
                #if self.ops.peak_slices is None or self.ops.peak_slices[-1] is None:
                #    self.ops.peak_finding(self.peak_controls.data_roi[:, :, channel], transformed=False,
                #                          roi_pos=self.peak_controls.roi_pos)
                peaks_2d = self.ops.peak_slices[-1]
            else:
                #if self.ops.peak_slices is None or self.ops.peak_slices[self._current_slice] is None:
                #    self.ops.peak_finding(self.peak_controls.data_roi[:, :, channel], transformed=False,
                #                          curr_slice = self._current_slice, roi_pos=self.peak_controls.roi_pos)
                peaks_2d = self.ops.peak_slices[self._current_slice]

        if len(peaks_2d.shape) > 0:
            for i in range(len(peaks_2d)):
                pos = QtCore.QPointF(peaks_2d[i][0] - self.size / 2, peaks_2d[i][1] - self.size / 2)
                point_obj = pg.CircleROI(pos, self.size, parent=self.imview.getImageItem(), movable=False,
                                         removable=True)
                point_obj.removeHandle(0)
                self.imview.addItem(point_obj)
                self._peaks.append(point_obj)
            if flip:
                self.ops.fliph = fliph
                self.ops.flipv = flipv
                self.ops.rot = rot
                self.ops.transp = transp
                self.ops._update_data()
                self._update_imview()

        if not self.other._refined:
            if self.ops.tf_peaks_z is None:
                if self.peak_controls.peak_channel_btn.currentIndex() != self.ops._channel_idx:
                    self.ops.load_channel(self.peak_controls.peak_channel_btn.currentIndex())
                color_matrix = self.ops.tf_matrix @ self.ops._color_matrices[self.peak_controls.peak_channel_btn.currentIndex()]
                self.ops.fit_z(self.ops.channel, transformed=self.ops._transformed, tf_matrix=color_matrix,
                               flips=self.flips, shape=self.ops.data.shape[:-1])

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
        reference = self.peak_controls.ref_btn.currentIndex()
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
                if self.ops.peak_slices is None or self.ops.peak_slices[-1] is None:
                    peaks_2d = None
                else:
                    peaks_2d = self.ops.peak_slices[-1]
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
        #self.align_btn.setEnabled(not self.map_btn.isChecked())
        self.ops.calc_mapping()
        self._update_imview()
        if self.remove_tilt_btn.isChecked():
            self._remove_tilt()

    @utils.wait_cursor('print')
    def _remove_tilt(self, state=None):
        if self.map_btn.isChecked():
            self.ops.remove_tilt(self.remove_tilt_btn.isChecked())
            self._update_imview()

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
            self.point_ref_btn.removeItem(0)

        self.channel_btns = []
        self.color_btns = []

        self.max_proj_btn.setChecked(False)

        self.define_btn.setEnabled(False)
        self.transform_btn.setEnabled(False)
        self.rot_transform_btn.setEnabled(False)
        self.show_btn.setEnabled(False)
        self.show_btn.setChecked(True)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)

        self.point_ref_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
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

        self.map_btn.setEnabled(False)
        self.map_btn.setChecked(False)
        self.remove_tilt_btn.setEnabled(False)
        self.remove_tilt_btn.setChecked(False)

        self.err_btn.setText('0')
        self.err_plt_btn.setEnabled(False)
        self.convergence_btn.setEnabled(False)

        self.ops.__init__(self.print, self.log)

