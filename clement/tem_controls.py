import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from .base_controls import BaseControls
from .em_operations import EM_ops
from . import utils

class TEMControls(BaseControls):
    def __init__(self, imview, vbox, printer, logger):
        super(TEMControls, self).__init__()
        self.tag = 'TEM'
        self.imview = imview
        self.ops = None
        self.popup = None
        self.show_merge = False

        self.show_boxes = False
        self.mrc_fname = None
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)

        self._curr_folder = None
        self._file_name = None
        self._downsampling = None
        self._select_region_original = True

        self.print = printer
        self.log = logger
        self._init_ui(vbox)

    def _init_ui(self, vbox):
        utils.add_montage_line(self, vbox, 'TEM', downsampling=True)

        # ---- Assembly grid options
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Assembly grid:', self)
        line.addWidget(label)
        self.select_region_btn = QtWidgets.QPushButton('Select subregion',self)
        self.select_region_btn.setCheckable(True)
        self.select_region_btn.toggled.connect(self._select_box)
        self.select_region_btn.setEnabled(False)
        self.show_assembled_btn = QtWidgets.QCheckBox('Show assembled image',self)
        self.show_assembled_btn.stateChanged.connect(self._show_assembled)
        self.show_assembled_btn.setChecked(True)
        self.show_assembled_btn.setEnabled(False)
        line.addWidget(self.select_region_btn)
        line.addWidget(self.show_assembled_btn)
        line.addStretch(1)

        utils.add_define_grid_line(self, vbox)
        utils.add_transform_grid_line(self, vbox, show_original=True)
        utils.add_fmpeaks_line(self, vbox)

        self.show()

    def _load_mrc(self, jump=False):
        if not jump:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            self._file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                 'Select TEM data',
                                                                 self._curr_folder,
                                                                 'All (*.tif *.tiff *.mrc);;*.mrc;;*.tif;;*tiff')
            self._curr_folder = os.path.dirname(self._file_name)

        if self._file_name != '':
            if self.ops is not None:
                self.reset_init()
            self.mrc_fname.setText(os.path.basename(self._file_name))
            self.assemble_btn.setEnabled(True)
            self.step_box.setEnabled(True)

            self.ops = EM_ops(self.print, self.log)
            self.ops.parse_2d(self._file_name)
            if len(self.ops.dimensions) == 2:
                self.step_box.setText('1')
                self._assemble_mrc()
                self.assemble_btn.setEnabled(False)

    @utils.wait_cursor('print')
    def _update_imview(self, state=None):
        if self.ops is not None and self.ops.data is not None:
            old_shape = self.imview.image.shape
            new_shape = self.ops.data.shape
            if old_shape == new_shape:
                vr = self.imview.getImageItem().getViewBox().targetRect()
            levels = self.imview.getHistogramWidget().item.getLevels()
            self.imview.setImage(self.ops.data, levels=levels)
            if old_shape == new_shape:
                self.imview.getImageItem().getViewBox().setRange(vr, padding=0)

            if self.show_assembled_btn.isChecked():
                if self.show_boxes:
                    self._show_boxes()
            else:
                self.show_boxes = False

    @utils.wait_cursor('print')
    def _assemble_mrc(self, state=None):
        if self.step_box.text() == '':
            self._downsampling = 10
        else:
            self._downsampling = self.step_box.text()

        if self._file_name != '':
            if len(self.ops.dimensions) == 3:
                self.ops.parse_3d(int(self._downsampling), self._file_name)
            self.imview.setImage(self.ops.data)
            self.define_btn.setEnabled(True)
            self.show_btn.setChecked(True)
            self.transform_btn.setEnabled(False)
            #self.rot_transform_btn.setEnabled(False)
            self.transp_btn.setEnabled(True)
            if self.tr_grid_box is not None:
                self.imview.removeItem(self.tr_grid_box)
            if self.grid_box is not None:
                self.imview.removeItem(self.grid_box)
            self.grid_box = None
            self.ops._transformed = False
            self.show_grid_btn.setEnabled(False)

            if self.ops.stacked_data:
                self.select_region_btn.setEnabled(True)
            else:
                self.select_region_btn.setEnabled(False)
                self.show_assembled_btn.setEnabled(False)
            self.boxes = []
            self.show_grid_btn.setChecked(False)
        else:
            self.print('You have to choose a file first!')

    @utils.wait_cursor('print')
    def _transpose(self, state=None):
        self.ops.transpose()
        self._recalc_grid()
        self._update_imview()

    @utils.wait_cursor('print')
    def _show_boxes(self, state=None):
        if self.ops is None:
            return
        handle_pen = pg.mkPen('#00000000')
        if self.show_btn.isChecked():
            if self.show_boxes:
                [self.imview.removeItem(box) for box in self.tr_boxes]
            if len(self.boxes) == 0:
                for i in range(len(self.ops.pos_x)):
                    roi = pg.PolyLineROI([], closed=True, movable=False)
                    roi.handlePen = handle_pen
                    roi.setPoints(self.ops.grid_points[i])
                    self.boxes.append(roi)
                    self.imview.addItem(roi)
            else:
                [self.imview.addItem(box) for box in self.boxes]
        else:
            if self.show_boxes:
                [self.imview.removeItem(box) for box in self.boxes]
            if len(self.tr_boxes) == 0:
                for i in range(len(self.ops.tf_grid_points)):
                    roi = pg.PolyLineROI([], closed=True, movable=False)
                    roi.handlePen = handle_pen
                    roi.setPoints(self.ops.tf_grid_points[i])
                    self.tr_boxes.append(roi)
                    self.imview.addItem(roi)
            else:
                [self.imview.addItem(box) for box in self.tr_boxes]
        self.show_boxes = True

    @utils.wait_cursor('print')
    def _hide_boxes(self, state=None):
        if self.show_btn.isChecked():
            if self.show_boxes:
                [self.imview.removeItem(box) for box in self.boxes]
        else:
            if self.show_boxes:
                [self.imview.removeItem(box) for box in self.tr_boxes]
        self.show_boxes = False

    @utils.wait_cursor('print')
    def _select_box(self, state=None):
        if self.select_region_btn.isChecked():
            self._show_boxes()
            self.ops.orig_region = None
            self.show_assembled_btn.setEnabled(False)
            self.print('Select box!')
            if self.show_btn.isChecked():
                self._select_region_original = True
            else:
                self._select_region_original = False
        else:
            if self._box_coordinate is not None:
                if self.show_btn.isChecked():
                    transformed = False
                else:
                    transformed = True
                points_obj = self._box_coordinate
                self.ops.select_region(np.array(points_obj),transformed)
                self._hide_boxes()
                if self.ops.orig_region is None:
                    self.print('Ooops, something went wrong. Try again!')
                    return
                self.show_assembled_btn.setEnabled(True)
                self.show_assembled_btn.setChecked(False)
                self.show_grid_btn.setChecked(False)
                if self.ops._orig_points_region is not None:
                    self.show_grid_btn.setEnabled(True)
                else:
                    self.show_grid_btn.setEnabled(False)
                self.show_btn.setChecked(True)
                self.show_btn.setEnabled(False)
                self.transform_btn.setEnabled(True)
            else:
                self._hide_boxes()

    @utils.wait_cursor('print')
    def _show_assembled(self, state=None):
        if self.ops is None:
            return
        if self.grid_box is not None:
            self.imview.removeItem(self.grid_box)
        if self.tr_grid_box is not None:
            self.imview.removeItem(self.tr_grid_box)
        if self.show_assembled_btn.isChecked():
            self.ops.assembled = True
            if self.ops._orig_points is None:
                self.show_grid_btn.setEnabled(False)
                self.show_grid_btn.setChecked(False)
            else:
                self.show_grid_btn.setEnabled(True)

            self.select_region_btn.setEnabled(True)

            if self.ops.tf_data is None:
                self.show_btn.setChecked(True)
                self.show_btn.setEnabled(False)
            else:
                self.show_btn.setEnabled(True)
        else:
            self.ops.assembled = False
            self.select_region_btn.setEnabled(False)
            self.show_grid_btn.setEnabled(True)
            if self.ops.tf_region is not None:
                self.show_btn.setEnabled(True)
            else:
                self.show_btn.setEnabled(False)
            if self.ops._orig_points_region is None:
                self.show_grid_btn.setEnabled(False)
                self.show_grid_btn.setChecked(False)

        self.ops.toggle_region()
        self._recalc_grid(self.imview)
        self._update_imview()

    @utils.wait_cursor('print')
    def _save_mrc_montage(self, state=None):
        if self.ops is None:
            self.print('No montage to save!')
        else:
            if self._curr_folder is None:
                self._curr_folder = os.getcwd()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Binned Montage', self._curr_folder, '*.mrc')
            self._curr_folder = os.path.dirname(file_name)
            if file_name != '':
                self.ops.save_merge(file_name)

    @utils.wait_cursor('print')
    def reset_init(self, state=None):
        #self.ops = None
        #self.other = None # The other controls object
        if self.show_grid_btn.isChecked():
            if self.ops._transformed:
                self.imview.removeItem(self.tr_grid_box)
            else:
                self.imview.removeItem(self.grid_box)
        if self.select_region_btn.isChecked():
            if self.ops._transformed:
                [self.imview.removeItem(box) for box in self.tr_boxes]
            else:
                [self.imview.removeItem(box) for box in self.boxes]
            self.select_region_btn.setChecked(False)
            self.select_region_btn.setEnabled(False)

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
        self.size = 10
        self.orig_size = 10
        self.fixed_orientation = False
        self.peaks = []
        self.num_slices = None
        self.min_conv_points = 10

        self.show_boxes = False
        self._downsampling = None

        self.step_box.setEnabled(False)
        self.assemble_btn.setEnabled(False)
        self.define_btn.setEnabled(False)
        self.show_btn.setChecked(True)
        self.show_btn.setEnabled(False)
        #self.rot_transform_btn.setEnabled(False)
        self.transform_btn.setEnabled(False)
        self.show_grid_btn.setEnabled(False)
        self.show_grid_btn.setChecked(False)
        self.show_assembled_btn.setChecked(True)
        self.show_assembled_btn.setEnabled(False)

        self.show_peaks_btn.setChecked(False)
        self.show_peaks_btn.setEnabled(False)

        self.translate_peaks_btn.setChecked(False)
        self.translate_peaks_btn.setEnabled(False)
        self.refine_peaks_btn.setChecked(False)
        self.refine_peaks_btn.setEnabled(False)

        self.size_box.setEnabled(False)
        self.auto_opt_btn.setEnabled(False)

        self.ops.__init__(self.print, self.log)
