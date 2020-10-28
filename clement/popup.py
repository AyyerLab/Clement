import sys
import os
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
import csv
import mrcfile as mrc
import copy
from skimage.color import hsv2rgb
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter

warnings.simplefilter('ignore', category=FutureWarning)

from . import utils

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # self.fig = Figure(figsize=(5, 5), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _scatter_plot(self):
        pass

    def _convergence_plot(self):
        pass


class Scatter_Plot(MplCanvas):
    def __init__(self, base):
        MplCanvas.__init__(self)
        self.base = base
        self._scatter_plot()

    def _scatter_plot(self):
        if self.base.fib:
            idx = 1
        else:
            idx = 0
        diff = self.base._err[idx]
        max = np.max(np.abs(diff))
        x = np.linspace(-max, max)
        y = np.linspace(-max, max)
        X, Y = np.meshgrid(x, y)
        levels = np.array([0.5, 0.75, 0.95])
        cset = self.axes.contour(X, Y, 1 - self.base._dist, colors='k', levels=levels)
        self.axes.clabel(cset, fmt='%1.2f', inline=1, fontsize=10)
        if self.base.fib:
            scatter = self.axes.scatter(diff[:, 0], diff[:, 1], c=self.base.other._points_corr_z_history[-1])
            cbar = self.fig.colorbar(scatter)
            cbar.set_label('z position of beads in FM')
        else:
            self.axes.scatter(diff[:, 0], diff[:, 1])
        self.axes.xaxis.set_minor_locator(AutoMinorLocator())
        self.axes.yaxis.set_minor_locator(AutoMinorLocator())
        self.axes.tick_params(which='both', axis="y", direction="in")
        self.axes.tick_params(which='both', axis="x", direction="in")
        self.axes.set_xlabel('x error [nm]')
        self.axes.set_ylabel('y error [nm]')
        self.axes.title.set_text('Error distribution and GMM model confidence intervals')


class Convergence_Plot(MplCanvas):
    def __init__(self, base):
        MplCanvas.__init__(self)
        self.base = base
        self.min_points = self.base.min_conv_points
        self._convergence_plot()

    def _convergence_plot(self):
        if self.base.fib:
            idx = 1
        else:
            idx = 0
        refined, free, all = self.base._conv[idx]
        final = all[-1]
        x = np.arange(self.min_points - 4, self.min_points - 4 + len(refined))
        self.axes.plot(x, refined, label='Refined beads')
        self.axes.plot(x, free, label='Non-refined beads')
        self.axes.plot(x, all, label='All beads')
        self.axes.axhspan(final, final + 0.1 * final, facecolor='0.2', alpha=0.3)
        # self.axes.set_xticks(x)
        # self.axes.set_xticklabels(x)
        self.axes.xaxis.set_major_locator(MultipleLocator(5))
        self.axes.xaxis.set_minor_locator(MultipleLocator(1))
        self.axes.yaxis.set_minor_locator(AutoMinorLocator())
        self.axes.tick_params(which='both', axis="y", direction="in")
        self.axes.tick_params(which='both', axis="x", direction="in")

        self.axes.legend()
        self.axes.set_xlabel('Number of beads used for refinement')
        self.axes.set_ylabel('RMS error [nm]')
        self.axes.title.set_text('RMS error convergence')


class Scatter(QtWidgets.QMainWindow):
    def __init__(self, parent, base):
        super(Scatter, self).__init__(parent)
        self.parent = parent
        self.theme = self.parent.theme
        self.resize(800, 800)
        self.parent._set_theme(self.theme)
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        sc = Scatter_Plot(base)
        layout.addWidget(sc)


class Convergence(QtWidgets.QMainWindow):
    def __init__(self, parent, base):
        super(Convergence, self).__init__(parent)
        self.parent = parent
        self.theme = self.parent.theme
        self.resize(800, 800)
        self.parent._set_theme(self.theme)
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        sc = Convergence_Plot(base)
        layout.addWidget(sc)

class Peak_Params(QtWidgets.QMainWindow):
    def __init__(self, parent, fm):
        super(Peak_Params, self).__init__(parent)
        self.parent = parent
        self.fm = fm
        self.data = None
        self.roi = None
        self.data_roi = None
        self.orig_data_roi = None
        self.background_correction = False
        self.color_data = None
        self.peaks = []
        self.num_channels = self.fm.ops.num_channels
        self.theme = self.parent.theme
        self.resize(800, 800)
        self.parent._set_theme(self.theme)
        self._init_ui()
        self._calc_max_proj()

    def _init_ui(self):
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        self.peak_imview = pg.ImageView()
        self.peak_imview.ui.roiBtn.hide()
        self.peak_imview.ui.menuBtn.hide()
        self.peak_imview.scene.sigMouseClicked.connect(self._imview_clicked)
        layout.addWidget(self.peak_imview)
        # Options
        options = QtWidgets.QVBoxLayout()
        options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(options)
        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        label = QtWidgets.QLabel('Set Peak finding parameters:', self)
        line.addWidget(label)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        label = QtWidgets.QLabel('Select channel:', self)
        line.addWidget(label)
        self.ref_btn = QtWidgets.QComboBox()
        listview = QtWidgets.QListView(self)
        self.ref_btn.setView(listview)
        for i in range(self.num_channels):
            self.ref_btn.addItem('Channel ' + str(i+1))
        self.ref_btn.setCurrentIndex(self.num_channels - 1)
        self.ref_btn.currentIndexChanged.connect(self._change_ref)
        self.ref_btn.setMinimumWidth(100)
        line.addWidget(self.ref_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        label = QtWidgets.QLabel('Define ROI:', self)
        line.addWidget(label)
        self.draw_btn = QtWidgets.QPushButton('Draw ROI', self)
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self._draw_roi)
        self.reset_btn = QtWidgets.QPushButton('Reset ROI', self)
        self.reset_btn.clicked.connect(self._reset_roi)
        line.addWidget(self.draw_btn)
        line.addWidget(self.reset_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        label = QtWidgets.QLabel('Background correction:', self)
        line.addWidget(label)
        self.sigma_btn = QtWidgets.QDoubleSpinBox(self)
        self.sigma_btn.setRange(0,20)
        self.sigma_btn.setDecimals(1)
        self.sigma_btn.setValue(5)
        self.background_btn = QtWidgets.QCheckBox('Subtract background', self)
        self.background_btn.stateChanged.connect(self._subtract_background)
        line.addWidget(self.sigma_btn)
        line.addWidget(self.background_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        label = QtWidgets.QLabel('Noise threshold:')
        line.addWidget(label)
        self.t_noise = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.t_noise.setRange(0,200) # 10 times the actual value to allow floats
        self.t_noise.setFocusPolicy(QtCore.Qt.NoFocus)
        self.t_noise.setValue(100)
        self.t_noise.valueChanged.connect(lambda state, param=0: self._set_noise_threshold(param, state))
        self.t_noise_label = QtWidgets.QDoubleSpinBox(self)
        self.t_noise_label.setRange(0,20)
        self.t_noise_label.setDecimals(1)
        self.t_noise_label.editingFinished.connect(lambda param=1: self._set_noise_threshold(param))
        self.t_noise_label.setValue(10)
        line.addWidget(self.t_noise)
        line.addWidget(self.t_noise_label)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        self.plt = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.plt.setRange(0,100)
        self.plt.setFocusPolicy(QtCore.Qt.NoFocus)
        self.plt.valueChanged.connect(lambda state, param=0: self._set_plt_threshold(param, state))
        self.plt_label = QtWidgets.QSpinBox(self)
        self.plt_label.setRange(0,100)
        self.plt_label.editingFinished.connect(lambda param=1: self._set_plt_threshold(param))
        self.plt.setValue(10)
        label = QtWidgets.QLabel('Min number of pixels per peak:')
        line.addWidget(label)
        line.addWidget(self.plt)
        line.addWidget(self.plt_label)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        self.put = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.put.setRange(0,3000)
        self.put.setFocusPolicy(QtCore.Qt.NoFocus)
        self.put.valueChanged.connect(lambda state, param=0: self._set_put_threshold(param, state))
        self.put_label = QtWidgets.QSpinBox(self)
        self.put_label.setRange(0,3000)
        self.put_label.editingFinished.connect(lambda param=1: self._set_put_threshold(param))
        self.put.setValue(200)
        label = QtWidgets.QLabel('Max number of pixels per peak:')
        line.addWidget(label)
        line.addWidget(self.put)
        line.addWidget(self.put_label)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        self.flood_steps = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.flood_steps.setRange(0,20)
        self.flood_steps.setFocusPolicy(QtCore.Qt.NoFocus)
        self.flood_steps.valueChanged.connect(lambda state, param=0: self._set_flood_steps(param, state))
        self.flood_steps_label = QtWidgets.QSpinBox(self)
        self.flood_steps_label.setRange(0,20)
        self.flood_steps_label.editingFinished.connect(lambda param=1: self._set_flood_steps(param))
        self.flood_steps.setValue(10)
        label = QtWidgets.QLabel('Number of flood fill steps:')
        line.addWidget(label)
        line.addWidget(self.flood_steps)
        line.addWidget(self.flood_steps_label)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        self.peak_btn = QtWidgets.QPushButton('Show peaks', self)
        self.peak_btn.setCheckable(True)
        self.peak_btn.toggled.connect(self._show_peaks)
        line.addWidget(self.peak_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        line.addStretch(1)
        self.save_btn = QtWidgets.QPushButton('Save and close', self)
        self.save_btn.clicked.connect(self._save)
        line.addWidget(self.save_btn)

    def _update(self):
        self._calc_color_channels()
        self.peak_imview.setImage(self.color_data)
        vr = self.peak_imview.getImageItem().getViewBox().targetRect()
        self.peak_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _calc_color_channels(self):
        idx = self.ref_btn.currentIndex()
        if self.data_roi is None:
            self.color_data = np.zeros((1,) + self.data[:, :, 0].shape + (3,))
            my_channel = self.data[:, :, idx]
        else:
            self.color_data = np.zeros((1,) + self.data_roi[:, :, 0].shape + (3,))
            my_channel = self.data_roi[:, :, idx]
        my_channel_rgb = np.repeat(my_channel[:, :, np.newaxis], 3, axis=2)
        rgb = tuple([int(self.fm._colors[idx][1 + 2 * c:3 + 2 * c], 16) / 255. for c in range(3)])
        self.color_data[0, :, :, :] = my_channel_rgb * rgb

    def _calc_max_proj(self):
        if self.fm.ops.max_proj_data is None:
            self.fm.ops.calc_max_proj_data()
        self.data = self.fm.ops.max_proj_data
        self.data /= self.data.max()
        self.data *= self.fm.ops.norm_factor
        if self.data_roi is None:
            self.data_roi = self.data
            self.orig_data_roi = np.copy(self.data_roi)
        self._update()

    @utils.wait_cursor
    def _subtract_background(self, checked):
        if checked:
            self.background_correction = True
            self.data_roi = self.fm.ops.subtract_background(self.data_roi, sigma=self.sigma_btn.value())
        else:
            self.background_correction = False
            self.data_roi = self.orig_data_roi
        self._update()

    def _change_ref(self, state=None):
        num = self.ref_btn.currentIndex()
        self.fm.ops._peak_reference = num
        self.fm.ops.orig_tf_peak_slices = None
        self.fm.ops.tf_peak_slices = None
        self.fm.ops.peak_slices = None
        self.fm.ops._color_matrices = []
        [self.fm.ops._color_matrices.append(np.identity(3)) for i in range(self.num_channels)]
        self._update()

    def _imview_clicked(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
            return
        else:
            print('clicked')
            if self.draw_btn.isChecked():
                print('Drawing')

    def _draw_roi(self, checked):
        if checked:
            self.reset_btn.setEnabled(False)
            if self.roi is not None:
                self.peak_imview.removeItem(self.roi)
            print('Draw ROI in the image!')
            pos = np.array([self.data.shape[0]//2, self.data.shape[1]//2])
            size = np.array([self.data.shape[0]//4, self.data.shape[1]//4])
            self.roi = pg.ROI(pos=pos,size=size, rotatable=False, removable=False)
            self.roi.addScaleHandle(pos=[1,1], center=[0.5,0.5])
            self.peak_imview.addItem(self.roi)
        else:
            self.reset_btn.setEnabled(True)
            self.roi.movable = False
            self.roi.resizable = False
            self.data_roi = self.roi.getArrayRegion(self.data, self.peak_imview.getImageItem())
            self.orig_data_roi = np.copy(self.data_roi)
            print(self.data_roi.shape)
            print('Done drawing ROI!')
            self.peak_imview.removeItem(self.roi)
            self._update()

    def _reset_roi(self):
        self.data_roi = self.data
        self.orig_data_roi = np.copy(self.data_roi)
        if self.background_correction:
            self._subtract_background(checked=True)
        self._update()

    def _set_noise_threshold(self, param, state=None):
        if param == 0:
            float_value = float(self.t_noise.value()) / 10
            self.t_noise_label.blockSignals(True)
            self.t_noise_label.setValue(float_value)
            self.t_noise_label.blockSignals(False)
        else:
            float_value = float(self.t_noise_label.value())
            self.t_noise.blockSignals(True)
            self.t_noise.setValue(10*float_value)
            self.t_noise.blockSignals(False)
            self.t_noise_label.clearFocus()

    def _set_plt_threshold(self, param, state=None):
        if param == 0:
            value = self.plt.value()
            self.plt_label.blockSignals(True)
            self.plt_label.setValue(value)
            self.plt_label.blockSignals(False)
        else:
            value = self.plt_label.value()
            self.plt.blockSignals(True)
            self.plt.setValue(value)
            self.plt.blockSignals(False)
            self.plt_label.clearFocus()

    def _set_put_threshold(self, param, state=None):
        if param == 0:
            value = self.put.value()
            self.put_label.blockSignals(True)
            self.put_label.setValue(value)
            self.put_label.blockSignals(False)
        else:
            value = self.put_label.value()
            self.put.blockSignals(True)
            self.put.setValue(value)
            self.put.blockSignals(False)
            self.put_label.clearFocus()

    def _set_flood_steps(self, param, state=None):
        if param == 0:
            value = self.flood_steps.value()
            self.flood_steps_label.blockSignals(True)
            self.flood_steps_label.setValue(value)
            self.flood_steps_label.blockSignals(False)
        else:
            value = self.flood_steps_label.value()
            self.flood_steps.blockSignals(True)
            self.flood_steps.setValue(value)
            self.flood_steps.blockSignals(False)
            self.flood_steps_label.clearFocus()

    @utils.wait_cursor
    def _show_peaks(self, state=None):
        if not self.peak_btn.isChecked():
            [self.peak_imview.removeItem(point) for point in self.peaks]
            self.peaks = []
            return
        self.fm.ops.threshold = self.t_noise_label.value()
        self.fm.ops.pixel_lower_threshold = self.plt.value()
        self.fm.ops.pixel_upper_threshold = self.put.value()
        self.fm.ops.flood_steps = self.flood_steps.value()
        self.fm.ops._peak_reference = self.ref_btn.currentIndex()
        self.fm.ops.peak_finding(self.data_roi[:,:,self.ref_btn.currentIndex()], transformed=False, curr_slice=0,
                                 test=True)
        peaks_2d = self.fm.ops.peaks_test
        if len(peaks_2d.shape) > 0:
            for i in range(len(peaks_2d)):
                pos = QtCore.QPointF(peaks_2d[i][0] - self.fm.size / 2, peaks_2d[i][1] - self.fm.size / 2)
                point_obj = pg.CircleROI(pos, self.fm.size, parent=self.peak_imview.getImageItem(), movable=False,
                                         removable=True)
                point_obj.removeHandle(0)
                self.peak_imview.addItem(point_obj)
                self.peaks.append(point_obj)

    def _save(self):
        if self.fm.ops is not None:
            self.fm.ops.background_correction = self.background_correction
            self.fm.ops.sigma = self.sigma_btn.value()
            self.fm.ops.threshold = self.t_noise_label.value()
            self.fm.ops.pixel_lower_threshold = self.plt.value()
            self.fm.ops.pixel_upper_threshold = self.put.value()
            self.fm.ops.flood_steps = self.flood_steps.value()
            self.fm.ops._peak_reference = self.ref_btn.currentIndex()
            self.fm.ops.orig_tf_peak_slices = None
            self.fm.ops.tf_peak_slices = None
            self.fm.ops.peak_slices = None
            self.fm.ops._color_matrices = []
            [self.fm.ops._color_matrices.append(np.identity(3)) for i in range(self.num_channels)]
        self.close()

class Merge(QtGui.QMainWindow):
    def __init__(self, parent):
        super(Merge, self).__init__(parent)
        self.parent = parent
        self.theme = self.parent.theme
        self.fm_copy = None
        self.other = self.parent.fmcontrols.other

        self.curr_mrc_folder_popup = self.parent.emcontrols.curr_folder
        self.num_slices_popup = self.parent.fmcontrols.num_slices
        self.downsampling = 2  # per dimension
        self.color_data_popup = None
        self.color_overlay_popup = None
        self.annotations_popup = []
        self.coordinates = []
        self.counter_popup = 0
        self.stage_positions_popup = None
        self.settings = QtCore.QSettings('MPSD-CNI', 'CLEMGui', self)
        self.max_help = False
        self.fib = False
        self.size = 10

        self._channels_popup = []
        self._colors_popup = list(np.copy(self.parent.fmcontrols._colors))
        self._current_slice_popup = self.parent.fmcontrols._current_slice
        self._overlay_popup = True
        self._clicked_points_popup = []

        if self.parent.fibcontrols.fib:
            merged_data = self.parent.em.merged_3d
            self.fib = True
            self.downsampling = 1
        else:
            merged_data = self.parent.em.merged_2d
            self.fib = False
        if merged_data is not None:
            print(self._colors_popup)
            self.data_popup = np.copy(merged_data)
            for i in range(merged_data.shape[2]):
                self._channels_popup.append(True)
            while len(self._colors_popup) < len(self._channels_popup):
                self._colors_popup.append('#808080')
            self.parent.colors = copy.copy(self._colors_popup)
        else:
            self.data_popup = np.copy(self.parent.fm.data)

        self.data_orig_popup = np.copy(self.data_popup)

        self._init_ui()
        self._copy_poi()

    def _init_ui(self):
        self.resize(800, 800)
        #self.parent._set_theme(self.theme)
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
        action = QtWidgets.QAction('&Save data', self)
        action.triggered.connect(self._save_data_popup)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Quit', self)
        action.triggered.connect(self.close)
        filemenu.addAction(action)
        #self._set_theme_popup(self.theme)

        self.imview_popup = pg.ImageView()
        self.imview_popup.ui.roiBtn.hide()
        self.imview_popup.ui.menuBtn.hide()
        self.imview_popup.scene.sigMouseClicked.connect(lambda evt: self._imview_clicked_popup(evt))
        # self.imview_popup.setImage(np.sum(self.data_popup,axis=2), levels=(self.data_popup.min(), self.data_popup.max()//3))
        layout.addWidget(self.imview_popup)

        options = QtWidgets.QHBoxLayout()
        options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(options)
        self._init_options_popup(options)
        self._calc_color_channels_popup()
        if self.overlay_btn_popup.isChecked():
            self.imview_popup.setImage(self.color_overlay_popup, levels=(self.data_popup.min(), self.data_popup.mean()))
        else:
            self.imview_popup.setImage(self.color_data_popup, levels=(self.data_popup.min(), self.data_popup.mean()))

    def _init_options_popup(self, parent_layout):
        vbox = QtWidgets.QVBoxLayout()
        parent_layout.addLayout(vbox)
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)

        label = QtWidgets.QLabel('FM image:', self)
        line.addWidget(label)
        self.fm_fname_popup = self.parent.fmcontrols.fm_fname.text()
        label = QtWidgets.QLabel(self.fm_fname_popup, self)
        line.addWidget(label, stretch=1)
        self.max_proj_btn_popup = QtWidgets.QCheckBox('Max projection')
        self.max_proj_btn_popup.stateChanged.connect(self._show_max_projection_popup)
        line.addWidget(self.max_proj_btn_popup)

        self.slice_select_btn_popup = QtWidgets.QSpinBox(self)
        self.slice_select_btn_popup.editingFinished.connect(self._slice_changed_popup)
        self.slice_select_btn_popup.setRange(0, self.parent.fm.num_slices)
        self.slice_select_btn_popup.setValue(self.parent.fmcontrols.slice_select_btn.value())
        line.addWidget(self.slice_select_btn_popup)
        if self.fib:
            self.max_proj_btn_popup.setEnabled(False)
            self.max_proj_btn_popup.blockSignals(True)
            self.max_proj_btn_popup.setChecked(True)
            self.max_proj_btn_popup.blockSignals(False)
            self.slice_select_btn_popup.setEnabled(False)

        if self.parent.fmcontrols.max_proj_btn.isChecked():
            self.max_help = True
            self.max_proj_btn_popup.setChecked(True)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('EM Image:', self)
        line.addWidget(label)
        self.em_fname_popup = self.parent.emcontrols.mrc_fname.text()
        label = QtWidgets.QLabel(self.em_fname_popup, self)
        line.addWidget(label, stretch=1)

        self.channel_line = QtWidgets.QHBoxLayout()
        vbox.addLayout(self.channel_line)
        label = QtWidgets.QLabel('Colors:', self)
        self.channel_line.addWidget(label)

        self.channel_btns_popup = []
        self.color_btns_popup = []
        for i in range(len(self._channels_popup)):
            channel_btn = QtWidgets.QCheckBox(' ', self)
            channel_btn.setChecked(True)
            channel_btn.stateChanged.connect(lambda state, channel=i: self._show_channels_popup(state, channel))
            self.channel_btns_popup.append(channel_btn)
            color_btn = QtWidgets.QPushButton(' ', self)
            color_btn.clicked.connect(lambda state, channel=i: self._sel_color_popup(state, channel))
            width = color_btn.fontMetrics().boundingRect(' ').width() + 24
            color_btn.setFixedWidth(width)
            color_btn.setMaximumHeight(width)
            color_btn.setStyleSheet('background-color: {}'.format(self._colors_popup[i]))
            self.color_btns_popup.append(color_btn)
            self.channel_line.addWidget(color_btn)
            self.channel_line.addWidget(channel_btn)

        self.overlay_btn_popup = QtWidgets.QCheckBox('Overlay', self)
        self.overlay_btn_popup.setChecked(True)
        self.overlay_btn_popup.stateChanged.connect(self._show_overlay_popup)
        self.channel_line.addWidget(self.overlay_btn_popup)
        self.channel_line.addStretch(1)

        # Select and save coordinates
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Select and save coordinates', self)
        line.addWidget(label)
        self.select_btn_popup = QtWidgets.QPushButton('Select points of interest', self)
        self.select_btn_popup.setCheckable(True)
        self.select_btn_popup.toggled.connect(self._calc_stage_positions_popup)
        self.save_btn_popup = QtWidgets.QPushButton('Save data')
        self.save_btn_popup.clicked.connect(self._save_data_popup)
        line.addWidget(self.select_btn_popup)
        line.addWidget(self.save_btn_popup)
        line.addStretch(1)

    def _copy_poi(self):
        corr_points = self.other._points_corr
        if len(corr_points) != 0:
            self.select_btn_popup.setChecked(True)
        if not self.other.fib:
            for point in corr_points:
                init = np.array([point.pos().x() + self.size/2, point.pos().y() + self.size/2, 1])
                transf = (np.linalg.inv(self.parent.emcontrols.ops.tf_matrix) @ init) / self.downsampling
                pos = QtCore.QPointF(transf[0] - self.size/2, transf[1] - self.size/2)
                self._draw_correlated_points_popup(pos, self.imview_popup.getImageItem())
        else:
            for i in range(len(corr_points)):
                pos = corr_points[i].pos()
                self._draw_correlated_points_popup(pos, self.imview_popup.getImageItem())

    def _update_poi(self, pos, fib):
        if not fib:
            init = np.array([pos.x() + self.size/2, pos.y() + self.size/2, 1])
            transf = (np.linalg.inv(self.parent.emcontrols.ops.tf_matrix) @ init) / self.downsampling
            pos = QtCore.QPointF(transf[0] - self.size / 2, transf[1] - self.size / 2)

        self._draw_correlated_points_popup(pos, self.imview_popup.getImageItem())

    def _imview_clicked_popup(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
            return

        item = self.imview_popup.getImageItem()
        pos = self.imview_popup.getImageItem().mapFromScene(event.pos())
        pos.setX(pos.x() - self.size / 2)
        pos.setY(pos.y() - self.size / 2)

        if self.select_btn_popup.isChecked():
            self._draw_correlated_points_popup(pos, item)

    def _draw_correlated_points_popup(self, pos, item):
        point = pg.CircleROI(pos, self.size, parent=item, movable=False, removable=True)
        point.setPen(0, 255, 0)
        point.removeHandle(0)
        self.imview_popup.addItem(point)
        self._clicked_points_popup.append(point)

        self.counter_popup += 1
        annotation = pg.TextItem(str(self.counter_popup), color=(0, 255, 0), anchor=(0, 0))
        annotation.setPos(pos.x() + 5, pos.y() + 5)
        self.annotations_popup.append(annotation)
        self.imview_popup.addItem(annotation)
        point.sigRemoveRequested.connect(lambda: self._remove_correlated_points_popup(point, annotation))

    def _remove_correlated_points_popup(self, pt, anno):
        idx = self._clicked_points_popup.index(pt)

        for i in range(idx+1, len(self._clicked_points_popup)):
            self.annotations_popup[i].setText(str(i))
        self.counter_popup -= 1

        self.imview_popup.removeItem(pt)
        self._clicked_points_popup.remove(pt)
        self.imview_popup.removeItem(anno)
        self.annotations_popup.remove(anno)

    @utils.wait_cursor
    def _show_overlay_popup(self, checked):
        self._overlay_popup = not self._overlay_popup
        self._update_imview_popup()

    @utils.wait_cursor
    def _show_channels_popup(self, checked, my_channel):
        self._channels_popup[my_channel] = not self._channels_popup[my_channel]
        self._update_imview_popup()

    def _sel_color_popup(self, state, index):
        button = self.color_btns_popup[index]
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            cname = color.name()
            self._colors_popup[index] = cname
            button.setStyleSheet('background-color: {}'.format(cname))
            self._update_imview_popup()
        else:
            print('Invalid color')

    def _calc_color_channels_popup(self):
        self.color_data_popup = np.zeros(
            (len(self._channels_popup), int(np.ceil(self.data_popup.shape[0] / self.downsampling)),
             int(np.ceil(self.data_popup.shape[1] / self.downsampling)), 3))
        if not self.parent.fm._show_mapping:
            for i in range(len(self._channels_popup)):
                if self._channels_popup[i]:
                    my_channel = self.data_popup[::self.downsampling, ::self.downsampling, i]
                    my_channel_rgb = np.repeat(my_channel[:, :, np.newaxis], 3, axis=2)
                    rgb = tuple([int(self._colors_popup[i][1 + 2 * c:3 + 2 * c], 16) / 255. for c in range(3)])
                    self.color_data_popup[i, :, :, :] = my_channel_rgb * rgb
                else:
                    self.color_data_popup[i, :, :, :] = np.zeros(
                        (self.data_popup[::self.downsampling, ::self.downsampling, 0].shape + (3,)))
        else:
            self.color_data_popup[0, :, :, :] = hsv2rgb(self.data_popup[::self.downsampling, ::self.downsampling, :3])
            em_img = self.data_popup[::self.downsampling, ::self.downsampling, -1]
            em_img_rgb = np.repeat(em_img[:, :, np.newaxis], 3, axis=2)
            rgb = tuple([int(self._colors_popup[-1][1 + 2 * c:3 + 2 * c], 16) / 255. for c in range(3)])
            self.color_data_popup[-1, :, :, :] = em_img_rgb * rgb

        if self.overlay_btn_popup.isChecked():
            self.color_overlay_popup = np.sum(self.color_data_popup, axis=0)
            self.color_data_popup = np.sum(self.color_data_popup, axis=0)

    def _update_imview_popup(self):
        self._calc_color_channels_popup()
        vr = self.imview_popup.getImageItem().getViewBox().targetRect()
        levels = self.imview_popup.getHistogramWidget().item.getLevels()
        self.imview_popup.setImage(self.color_data_popup, levels=levels)
        self.imview_popup.getImageItem().getViewBox().setRange(vr, padding=0)

    def _calc_stage_positions_popup(self, checked):
        if checked:
            print('Select points of interest!')
            if len(self._clicked_points_popup) != 0:
                [self.imview_popup.removeItem(point) for point in self._clicked_points_popup]
                [self.imview_popup.removeItem(annotation) for annotation in self.annotations_popup]
                self._clicked_points_popup = []
                self.annotations_popup = []
                self.counter_popup = 0
                self.stage_positions_popup = None
                self.coordinates = []
        else:
            self.coordinates = [self.downsampling * np.array([point.x() + self.size / 2, point.y() + self.size / 2]) for point in
                           self._clicked_points_popup]

            #if self.fib:
            #    self.stage_positions_popup = np.copy(coordinates)
            #else:
            #    self.stage_positions_popup = self.parent.emcontrols.ops.calc_stage_positions(coordinates,
            #                                                                                 self.downsampling)
            print('Done selecting points of interest!')

    @utils.wait_cursor
    def _save_data_popup(self, state=None):
        if self.curr_mrc_folder_popup is None:
            self.curr_mrc_folder_popup = os.getcwd()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Merged Image and Coordinates', self.curr_mrc_folder_popup)
        if file_name != '':
            try:
                file_name = file_name.split('.', 1)[0]
                screenshot = self.imview_popup.grab()
                screenshot.save(file_name, 'tif')
                self._save_merge_popup(file_name)
                self._save_coordinates_popup(file_name)
            except PermissionError:
                print('Permission error! Choose a different directory!')

    def _save_merge_popup(self, fname):
        with mrc.new(fname + '.mrc', overwrite=True) as f:
            f.set_data(self.data_popup.astype(np.float32))
            f.update_header_stats()

    def _save_coordinates_popup(self, fname):
        #if self.stage_positions_popup is not None:
            #for i in range(len(self.stage_positions_popup)):
        enumerated = []
        for i in range(len(self.coordinates)):
            #enumerated.append((i + 1, self.stage_positions_popup[i][0], self.stage_positions_popup[i][1]))
            enumerated.append((i + 1, self.coordinates[i][0], self.coordinates[i][1]))
        with open(fname + '.txt', 'w', newline='') as f:
            csv.writer(f, delimiter=' ').writerows(enumerated)

    @utils.wait_cursor
    def _show_max_projection_popup(self, state=None):
        if self.max_help:
            self.max_help = False
            self.slice_select_btn_popup.setEnabled(False)
            return
        else:
            if self.fm_copy is None:
                self.fm_copy = copy.copy(self.parent.fm)
            self.slice_select_btn_popup.setEnabled(not self.max_proj_btn_popup.isChecked())
            self.fm_copy.calc_max_projection()
            self.fm_copy.apply_merge()
            self.data_popup = np.copy(self.fm_copy.merged)
            self._update_imview_popup()

    @utils.wait_cursor
    def _slice_changed_popup(self):
        if self.fm_copy is None:
            self.fm_copy = copy.copy(self.parent.fm)
        num = self.slice_select_btn_popup.value()
        if num != self._current_slice_popup:
            self.fm_copy.parse(fname=self.fm_fname_popup, z=num, reopen=False)
            self.fm_copy.apply_merge()
            self.data_popup = np.copy(self.fm_copy.merged)
            self._update_imview_popup()
            fname, indstr = self.fm_fname_popup.split()
            self.fm_fname_popup = (fname + ' [%d/%d]' % (num, self.fm_copy.num_slices))
            self._current_slice_popup = num
            self.slice_select_btn_popup.clearFocus()

    def _set_theme_popup(self, name):
        if name == 'none':
            self.setStyleSheet('')
        else:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'styles/%s.qss' % name), 'r') as f:
                self.setStyleSheet(f.read())
        self.settings.setValue('theme', name)

    def _reset_init(self):
        self.parent = None
        self.theme = None
        self.fm_copy = None

        self.curr_mrc_folder_popup = None
        self.num_slices_popup = None
        self.color_data_popup = None
        self.color_overlay_popup = None
        self.annotations_popup = []
        self.counter_popup = 0
        self.stage_positions_popup = None
        self.settings = None
        self.max_help = False

        self._channels_popup = []
        self._colors_popup = []
        self._current_slice_popup = None
        self._overlay_popup = True
        self._clicked_points_popup = []

        self.data_popup = None

        self.data_orig_popup = None
        for i in range(len(self.channel_btns_popup)):
            self.channel_line.removeWidget(self.channel_btns_popup[i])
            self.channel_line.removeWidget(self.color_btns_popup[i])

    def closeEvent(self, event):
        if self.parent is not None:
            if self.parent.fmcontrols.other == self.other:
                self.parent.fmcontrols.progress_bar.setValue(0)
            self.other.progress = 0
            self.parent.project.merged = False
            self.other.show_merge = False
        self._reset_init()
        event.accept()
