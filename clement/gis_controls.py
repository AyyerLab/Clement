import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
from .em_operations import EM_ops
from . import utils
import time
import copy

class GISControls(BaseControls):
    def __init__(self, imview, vbox, sem_ops, fib_ops, printer, logger):
        super(GISControls, self).__init__()
        self.tag = 'EM'
        self.imview = imview
        self.ops = None
        self.sem_ops = sem_ops
        self.fib_ops = fib_ops
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
        self._refined = False

        self.print = printer
        self.log = logger

        self._init_ui(vbox)
        self._init_fib_params()

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

        line.addStretch(1)
        label = QtWidgets.QLabel('Overlay pre-/post-GIS:')
        line.addWidget(label)

        self.slider = QtWidgets.QSlider(self)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setSingleStep(5)
        self.time = time.time()

        self.slider.valueChanged.connect(self._update_opacity)
        self.slider.sliderReleased.connect(self._update_imview)
        self.slider.setEnabled(False)
        line.addWidget(self.slider)

        self.overlay_label = QtWidgets.QLabel('Opacity: 100 %')
        line.addWidget(self.overlay_label)
        line.addStretch(2)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Correlate pre- and post-GIS:')
        line.addWidget(label)

        self.select_fiducial_btn = QtWidgets.QPushButton('Select Fiducial ROI')
        self.select_fiducial_btn.setCheckable(True)
        self.select_fiducial_btn.toggled.connect(self._draw_roi)
        self.select_fiducial_btn.setEnabled(False)
        line.addWidget(self.select_fiducial_btn)

        self.correct_btn = QtWidgets.QCheckBox('Align pre- and post-GIS images')
        self.correct_btn.toggled.connect(self._align_gis)
        self.correct_btn.setEnabled(False)
        line.addWidget(self.correct_btn)

        line.addStretch(1)
        vbox.addStretch(1)

        self.show()

    def _init_fib_params(self, fibcontrols=None):
        if fibcontrols is None:
            return

        if fibcontrols.ops is not None:
            self.enable_buttons(overlay=True, enable=(fibcontrols.ops.points is not None))
            if self.ops is not None and fibcontrols.ops is not None:
                self.ops._tranformed = fibcontrols.ops._transformed
            self._refined = fibcontrols._refined
            self.tr_matrices = fibcontrols.tr_matrices
            self.cov_matrix = fibcontrols.cov_matrix

        if fibcontrols.show_grid_btn.isChecked():
            self.show_grid_btn.setChecked(True)

        if self.ops is None or np.array_equal(self.ops.gis_transf, np.identity(3)):
            self._err = copy.copy(self.other.fibcontrols._err)
            self._std = copy.copy(self.other.fibcontrols._std)
            self._conv = copy.copy(self.other.fibcontrols._conv)
            self._dist = copy.copy(self.other.fibcontrols._dist)

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
                self.other.tab_index = 2
            self.mrc_fname.setText(os.path.basename(self._file_name))

            self.ops = EM_ops(self.print, self.log)
            self.ops.parse_2d(self._file_name)
            self._update_imview()
            #self.imview.setImage(self.ops.data)
            self.grid_box = None
            self.transp_btn.setEnabled(True)
            self._init_fib_params(self.other.fibcontrols)
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
                if self.correct_btn.isChecked():
                    self.img_pre = pg.ImageItem(self.ops.gis_corrected)
                else:
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
        now = time.time()
        if (now - self.time) > 0.4:
            self._update_imview()
        self.time = now

    def _draw_roi(self, checked):
        if checked:
            pos = np.array([self.ops.data.shape[0] // 2, self.ops.data.shape[1] //2])
            size = np.array([200, 200])
            qrect = None
            self.roi = pg.ROI(pos=pos, size=size, maxBounds=qrect, resizable=True, rotatable=False, removable=False)
            self.roi.addScaleHandle(pos=[1, 1], center=[0.5, 0.5])
            self.imview.addItem(self.roi)
            self.roi.setZValue(20)
        else:
            self.imview.removeItem(self.roi)
            self.roi_pos = np.rint(np.array([self.roi.pos().x(), self.roi.pos().y()])).astype(int)
            self.roi_size = np.rint(np.array([self.roi.size().x(), self.roi.size().y()])).astype(int)
            self.roi_pre = self.fib_ops.data[self.roi_pos[0]:self.roi_pos[0]+self.roi_size[0], self.roi_pos[1]:self.roi_pos[1]+self.roi_size[1]]
            self.roi_post = self.ops.data[self.roi_pos[0]:self.roi_pos[0]+self.roi_size[0], self.roi_pos[1]:self.roi_pos[1]+self.roi_size[1]]

            #np.save('roi_pre', self.roi_pre)
            #np.save('roi_post', self.roi_post)
            #self.imview.removeItem(self.img_pre)
            #self.imview.removeItem(self.img_post)
            #self.imview.setImage(self.roi_post)

    @utils.wait_cursor('print')
    def _align_gis(self, state):
        if self.select_fiducial_btn.isChecked() or self.roi_post is None:
            self.correct_btn.setChecked(False)
            self.print('You have to specify the ROI of the fiducial first!')
        else:
            if state:
                self.ops.estimate_gis_transf(self.roi_pos, self.roi_pre, self.roi_post)
                self.ops.align_fiducial(self.fib_ops.data, state)
            else:
                self.ops.align_fiducial(None, state)
            self._update_imview()
            self._show_grid()
            if self.show_peaks_btn.isChecked():
                self.show_peaks_btn.setChecked(False)
                self.show_peaks_btn.setChecked(True)


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
                self.select_fiducial_btn.setEnabled(overlay)
                self.correct_btn.setEnabled(overlay)

    @utils.wait_cursor('print')
    def _show_grid(self, state=2):
        if self.grid_box is not None:
            self.imview.removeItem(self.grid_box)
        if state > 0:
            self.show_grid_box = True
            self._recalc_grid()
            self.imview.addItem(self.grid_box)
        else:
            self.show_grid_box = False

    @utils.wait_cursor('print')
    def _recalc_grid(self, state=None, recalc_matrix=True, scaling=1, shift=np.array([0,0])):
        print('recalc grid')
        if self.ops is None or self.ops.gis_corrected is None:
            self.grid_box = pg.PolyLineROI(self.fib_ops.points, closed=True, movable=not self._refined, resizable=False,
                                               rotatable=False)
        else:
            print('yes sir')
            points = [self.ops._update_gis_points(p) for p in self.fib_ops.points]
            self.grid_box = pg.PolyLineROI(points, closed=True, movable=not self._refined, resizable=False,
                                           rotatable=False)

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
        self._err = None
        self._std = [None, None]
        self._conv = None
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
        self.peaks = []
        self.num_slices = None

        self.show_grid_box = False
        self.grid_box = None
        self.transp_btn.setEnabled(False)
        self.transp_btn.setChecked(False)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)

        self.ops.__init__(self.print, self.log)
