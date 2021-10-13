import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
from .em_operations import EM_ops
from . import utils

class GISControls(BaseControls):
    def __init__(self, imview, vbox, sem_ops, fib_ops, printer, logger):
        super(GISControls, self).__init__()
        self.tag = 'EM'
        self.imview = imview
        self.ops = None
        self.sem_ops = sem_ops
        self.fib_ops = fib_ops
        self.tab_index = 0
        self.opacity = 1
        self.img_pre = None
        self.img_post = None
        self.roi_pos = None
        self.roi_size = None
        self.roi_pre = None
        self.roi_post = None
        self.num_slices = None
        self.popup = None
        self.show_merge = False

        self.show_grid_box = False
        self.grid_box = None
        self.old_pos0 = None
        self.box_shift = None
        self.mrc_fname = None
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)

        self._curr_folder = None
        self._file_name = None
        self._sigma_angle = None
        self._refined = False

        self.print = printer
        self.log = logger

        self._init_ui(vbox)

    def _init_ui(self, vbox):
        utils.add_montage_line(self, vbox, 'GIS', downsampling=False)

        # ---- Calculate grid square
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Overlay pre- and post-GIS:')
        line.addWidget(label)
        self.show_grid_btn = QtWidgets.QCheckBox('Show grid square', self)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_grid_btn.stateChanged.connect(self._show_grid)
        line.addWidget(self.show_grid_btn)
        #utils.add_fmpeaks_line(self, vbox)
        self.show_peaks_btn = QtWidgets.QCheckBox('Show FM peaks', self)
        self.show_peaks_btn.setEnabled(True)
        self.show_peaks_btn.setChecked(False)
        self.show_peaks_btn.toggled.connect(self._show_FM_peaks)
        self.show_peaks_btn.setEnabled(False)
        line.addWidget(self.show_peaks_btn)

        line.addStretch(0.5)
        label = QtWidgets.QLabel('Overlay pre-/post-GIS:')
        line.addWidget(label)

        self.slider = QtWidgets.QSlider(self)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setSingleStep(5)
        self.slider.valueChanged.connect(self._update_opacity)
        self.slider.sliderReleased.connect(self._overlay_images)
        self.slider.setEnabled(False)
        line.addWidget(self.slider)

        self.overlay_label = QtWidgets.QLabel('Opacity: 100 %')
        line.addWidget(self.overlay_label)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Correlate pre- and post-GIS:')
        line.addWidget(label)

        self.select_btn = QtWidgets.QPushButton('Select Fiducial ROI')
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self._draw_roi)
        self.select_btn.setEnabled(False)
        line.addWidget(self.select_btn)

        line.addStretch(1)
        vbox.addStretch(1)

        self.show()

    #@utils.wait_cursor('print')
    def _load_mrc(self, jump=False):
        if not jump:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            self._file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                       'Select GIS data',
                                                                       self._curr_folder,
                                                                       '*.tif;;*tiff;;*.mrc')
            self._curr_folder = os.path.dirname(self._file_name)

        if self._file_name != '':
            if self.ops is not None:
                self.reset_init()
                self.tab_index = 1
            self.mrc_fname.setText(os.path.basename(self._file_name))

            self.ops = EM_ops(self.print, self.log)
            self.ops.parse_2d(self._file_name)
            self._update_imview()
            #self.imview.setImage(self.ops.data)
            self.grid_box = None
            self.transp_btn.setEnabled(True)
            if self.fib_ops is not None:
                if self.fib_ops.points is not None:
                    self.show_grid_btn.setEnabled(True)
                if self.fib_ops._transformed is not None:
                    self.ops._transformed = True
                if self.fib_ops.data is not None:
                    self.slider.setEnabled(True)
                    self.select_btn.setEnabled(True)
            self.show_grid_btn.setChecked(False)
        else:
            self.print('You have to choose a file first!')

    @utils.wait_cursor('print')
    def _update_imview(self, state=None):
        if self.img_post is not None:
            self.imview.removeItem(self.img_post)
        if self.img_pre is not None:
            self.imview.removeItem(self.img_pre)
        if self.ops is not None and self.ops.data is not None:
            self.img_post = pg.ImageItem(self.ops.data)
            if self.fib_ops is not None and self.fib_ops.data is not None:
                self.img_pre = pg.ImageItem(self.fib_ops.data)
            else:
                self.img_pre = pg.ImageItem(np.zeros_like(self.ops.data))

            self.imview.addItem(self.img_pre)
            self.imview.addItem(self.img_post)
            self.img_post.setZValue(10)
            self.img_post.setOpacity(self.opacity)

    def _update_opacity(self):
        self.opacity = (100 - self.slider.value()) / 100
        self.overlay_label.setText('Opacity: {}'.format(self.opacity))

    def _overlay_images(self):
        self._update_imview()

    def _draw_roi(self, checked):
        if checked:
            pos = np.array([self.ops.data.shape[0] // 2, self.ops.data.shape[1] //2])
            size = np.array([200, 200])
            qrect = None
            self.roi = pg.ROI(pos=pos, size=size, maxBounds=qrect, resizable=True, rotatable=False, removable=False)
            self.roi.addScaleHandle(pos=[1, 1], center=[0.5, 0.5])
            self.imview.addItem(self.roi)
        else:
            self.imview.removeItem(self.roi)
            self.roi_pos = np.rint(np.array([self.roi.pos().x(), self.roi.pos().y()])).astype(int)
            self.roi_size = np.rint(np.array([self.roi.size().x(), self.roi.size().y()])).astype(int)
            self.roi_pre = self.fib_ops.data[self.roi_pos[0]:self.roi_pos[0]+self.roi_size[0], self.roi_pos[1]:self.roi_pos[1]+self.roi_size[1]]
            self.roi_post = self.ops.data[self.roi_pos[0]:self.roi_pos[0]+self.roi_size[0], self.roi_pos[1]:self.roi_pos[1]+self.roi_size[1]]

            #self.imview.removeItem(self.img_pre)
            #self.imview.removeItem(self.img_post)
            #self.imview.setImage(self.roi_post)


    @utils.wait_cursor('print')
    def _transpose(self, state=None):
        self.ops.transpose()
        self._recalc_grid(recalc_matrix=False)
        self._update_imview()

    def enable_buttons(self, overlay=False, enable=False):
        if self.ops is not None and self.ops.data is not None:
            if enable:
                self.show_grid_btn.setEnabled(True)
                self.show_peaks_btn.setEnabled(True)
            if self.fib_ops.data is not None and self.ops.data is not None:
                self.slider.setEnabled(overlay)
                self.select_btn.setEnabled(overlay)

    @utils.wait_cursor('print')
    def _show_grid(self, state=2):
        if state > 0:
            self.show_grid_box = True
            self._recalc_grid()
        else:
            self.show_grid_box = False
            if self.grid_box is not None:
                self.imview.removeItem(self.grid_box)

    @utils.wait_cursor('print')
    def _recalc_grid(self, state=None, recalc_matrix=True, scaling=1, shift=np.array([0,0])):
        if self.sem_ops is not None and self.sem_ops.points is not None and recalc_matrix:
            if self.box_shift is None:
                shift = np.zeros(2)
                redo = True
            else:
                shift = self.box_shift
                redo = False

            sigma_angle = float(self.sigma_btn.text())
            #is_transposed = not self.transp_btn.isChecked()

            self.ops.calc_fib_transform(sigma_angle, self.sem_ops.data.shape,
                                        self.other.ops.voxel_size, self.sem_ops.pixel_size, shift=shift, sem_transpose=False)
            self.ops.apply_fib_transform(self.sem_ops._orig_points, self.num_slices, scaling)

        if self.ops.points is not None:
            pos = list(self.ops.points)
            if self.show_grid_box and self.grid_box is not None:
                self.imview.removeItem(self.grid_box)
            self.grid_box = pg.PolyLineROI(pos, closed=True, movable=not self._refined, resizable=False, rotatable=False)
            if self.old_pos0 is None:
                self.old_pos0 = [0, 0]
            self.grid_box.sigRegionChangeFinished.connect(self._update_shifts)
            if redo:
                self.ops.calc_fib_transform(sigma_angle, self.sem_ops.data.shape, self.other.ops.voxel_size,
                                            self.sem_ops.pixel_size, shift=np.zeros(2), sem_transpose=False)
                self.ops.apply_fib_transform(self.sem_ops._orig_points, self.num_slices, scaling)
                pos = list(self.ops.points)
                self.grid_box = pg.PolyLineROI(pos, closed=True, movable=not self._refined, resizable=False,
                                               rotatable=False)
                self.old_pos0 = self.grid_box.pos()
                self.box_shift = np.zeros(2)
                self.grid_box.sigRegionChangeFinished.connect(self._update_shifts)
            self.print('Box origin at:', self.old_pos0)

        if self.grid_box is not None:
            if self.show_grid_box:
                self.imview.addItem(self.grid_box)
            self.show_peaks_btn.setEnabled(True)
            self.auto_opt_btn.setEnabled(True)
            self.size_box.setEnabled(True)
            if self.show_peaks_btn.isChecked():
                self.show_peaks_btn.setChecked(False)
                self.show_peaks_btn.setChecked(True)

    @utils.wait_cursor('print')
    def _update_shifts(self, state):
        new_pos = self.grid_box.pos() + self.old_pos0
        self.box_shift = np.array(new_pos)
        self.ops.points = np.copy([point + self.box_shift for point in self.ops.points])
        self.old_pos0 = new_pos
        self._recalc_grid()

    @utils.wait_cursor('print')
    def _save_mrc_montage(self, state=None):
        if self.ops is None:
            self.print('No montage to save!')
        else:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Binned Montage', self._curr_folder,
                                                                 '*.mrc')
            self._curr_folder = os.path.dirname(file_name)
            if file_name != '':
                self.ops.save_merge(file_name)

    @utils.wait_cursor('print')
    def reset_init(self, state=None):
        if self.show_grid_btn.isChecked() and self.grid_box is not None:
            self.imview.removeItem(self.grid_box)
        self.show_peaks_btn.setChecked(False)
        self.show_peaks_btn.setEnabled(False)

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

        self.popup = None

        self._points_corr_history = []
        self._points_corr_z_history = []
        self._orig_points_corr_history = []
        self._fib_vs_sem_history = []
        self._size_history = []

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
        self.tab_index = 0

        self.show_grid_box = False
        self.grid_box = None
        self.transp_btn.setEnabled(False)
        self.transp_btn.setChecked(False)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)

        self.translate_peaks_btn.setChecked(False)
        self.translate_peaks_btn.setEnabled(False)
        self.refine_peaks_btn.setChecked(False)
        self.refine_peaks_btn.setEnabled(False)
        self.size_box.setEnabled(False)
        self.auto_opt_btn.setEnabled(False)

        self.ops.__init__(self.print, self.log)
