import sys
import os
import warnings
from PyQt5 import QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg
import csv
import mrcfile as mrc
import copy
from skimage.color import hsv2rgb
from scipy import ndimage as ndi
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
    def __init__(self, base, printer):
        MplCanvas.__init__(self)
        self.base = base
        self.print = printer
        self._scatter_plot()

    @utils.wait_cursor('print')
    def _scatter_plot(self):
        diff = self.base._err
        max = np.max(np.abs(diff))
        x = np.linspace(-max, max)
        y = np.linspace(-max, max)
        X, Y = np.meshgrid(x, y)
        levels = np.array([0.5, 0.75, 0.95])
        cset = self.axes.contour(X, Y, 1 - self.base._dist, colors='k', levels=levels)
        self.axes.clabel(cset, fmt='%1.2f', inline=1, fontsize=10)
        if self.base.other.tab_index == 1:
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
    def __init__(self, base, printer):
        MplCanvas.__init__(self)
        self.base = base
        self.min_points = self.base.min_conv_points
        self.print = printer
        self._convergence_plot()

    @utils.wait_cursor('print')
    def _convergence_plot(self, state=None):
        refined, free, all = self.base._conv
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
    def __init__(self, parent, base, printer):
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

        sc = Scatter_Plot(base, printer)
        layout.addWidget(sc)


class Convergence(QtWidgets.QMainWindow):
    def __init__(self, parent, base, printer):
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
        sc = Convergence_Plot(base, printer)
        layout.addWidget(sc)

class Peak_Params(QtWidgets.QMainWindow):
    def __init__(self, parent, fm, printer, logger):
        super(Peak_Params, self).__init__(parent)
        self.parent = parent
        self.fm = fm
        self.roi = None
        self.data_roi = None
        self._roi_pos = None
        self._roi_angle = None
        self._roi_shape = None
        self.orig_data_roi = None
        self.background_correction = False
        self.invert = False
        self.color_data = None
        self.levels = None
        self.coor = None
        self.peaks = []
        self.num_channels = self.fm.ops.num_channels
        self.recover_transformed = False
        self.theme = self.parent.theme
        self.resize(800, 800)
        self.parent._set_theme(self.theme)
        self.print = printer
        self.log = logger
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
        layout.addWidget(self.peak_imview)
        # Options
        hlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(hlayout)
        options = QtWidgets.QVBoxLayout()
        options.setContentsMargins(4, 0, 4, 4)
        hlayout.addLayout(options)
        align_options = QtWidgets.QVBoxLayout()
        align_options.setContentsMargins(4, 0, 4, 4)
        hlayout.addLayout(align_options)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        label = QtWidgets.QLabel('Set Peak finding parameters:', self)
        line.addWidget(label)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        label = QtWidgets.QLabel('Select channel:', self)
        line.addWidget(label)
        self.peak_channel_btn = QtWidgets.QComboBox()
        listview = QtWidgets.QListView(self)
        self.peak_channel_btn.setView(listview)
        for i in range(self.num_channels):
            self.peak_channel_btn.addItem('Channel ' + str(i+1))
        self.peak_channel_btn.setCurrentIndex(self.num_channels - 1)
        self.peak_channel_btn.currentIndexChanged.connect(self._change_peak_ref)
        self.peak_channel_btn.setMinimumWidth(100)
        line.addWidget(self.peak_channel_btn)
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
        self.reset_btn.setEnabled(False)
        line.addWidget(self.draw_btn)
        line.addWidget(self.reset_btn)
        line.addStretch(1)

        #line = QtWidgets.QHBoxLayout()
        #options.addLayout(line)
        #label = QtWidgets.QLabel('Background correction:', self)
        #line.addWidget(label)
        #label = QtWidgets.QLabel('Sigma:', self)
        #line.addWidget(label)
        #self.sigma_btn = QtWidgets.QDoubleSpinBox(self)
        #self.sigma_btn.setRange(0,20)
        #self.sigma_btn.setDecimals(1)
        #self.sigma_btn.setValue(10)
        #self.background_btn = QtWidgets.QCheckBox('Subtract background', self)
        #self.background_btn.stateChanged.connect(self._subtract_background)
        #self.invert_btn = QtWidgets.QCheckBox('Invert', self)
        #self.invert_btn.stateChanged.connect(self._invert_channel)
        #line.addWidget(self.sigma_btn)
        #line.addWidget(self.background_btn)
        #line.addWidget(self.invert_btn)
        #line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        options.addLayout(line)
        label = QtWidgets.QLabel('Thresholds:')
        line.addWidget(label)
        label = QtWidgets.QLabel('Low:')
        line.addWidget(label)
        self.t_low = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.t_low.setRange(0,1000)
        self.t_low.setFocusPolicy(QtCore.Qt.NoFocus)
        self.t_low.valueChanged.connect(lambda state, b=1, param=0: self._set_thresholds(param, b, state))
        self.t_low_label = QtWidgets.QDoubleSpinBox(self)
        self.t_low_label.setRange(0,100)
        self.t_low_label.setDecimals(1)
        self.t_low_label.editingFinished.connect(lambda b=0, param=0: self._set_thresholds(param, b))
        self.t_low.setValue(0)
        self.t_low_label.setValue(0)
        line.addWidget(self.t_low)
        line.addWidget(self.t_low_label)
        label = QtWidgets.QLabel('High:')
        line.addWidget(label)
        self.t_high = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.t_high.setRange(0,1000)
        self.t_high.setFocusPolicy(QtCore.Qt.NoFocus)
        self.t_high.setValue(1000)
        self.t_high.valueChanged.connect(lambda state, b=1, param=1: self._set_thresholds(param, b, state))
        self.t_high_label = QtWidgets.QDoubleSpinBox(self)
        self.t_high_label.setRange(0,100)
        self.t_high_label.setDecimals(1)
        self.t_high_label.editingFinished.connect(lambda b=0, param=1: self._set_thresholds(param, b))
        self.t_high_label.setValue(100)
        line.addWidget(self.t_high)
        line.addWidget(self.t_high_label)
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
        self.peak_btn = QtWidgets.QPushButton('Find peaks', self)
        self.peak_btn.setCheckable(True)
        self.peak_btn.toggled.connect(self._show_peaks)
        line.addWidget(self.peak_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        align_options.addLayout(line)
        label = QtWidgets.QLabel('Align color channels:')
        line.addWidget(label)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        align_options.addLayout(line)
        label = QtWidgets.QLabel('Reference channel:', self)
        line.addWidget(label)
        self.ref_btn = QtWidgets.QLabel('')
        #listview = QtWidgets.QListView(self)
        #self.ref_btn.setView(listview)
        #for i in range(self.num_channels):
        #    self.ref_btn.addItem('Channel ' + str(i+1))
        #self.ref_btn.setCurrentIndex(self.num_channels - 1)
        #self.ref_btn.currentIndexChanged.connect(self._change_ref)
        #self.ref_btn.setMinimumWidth(100)
        self.ref_btn.setText('Channel {}'.format(self.peak_channel_btn.currentIndex()+1))
        line.addWidget(self.ref_btn)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        align_options.addLayout(line)
        label = QtWidgets.QLabel('Select channel:', self)
        line.addWidget(label)
        self.align_btn = QtWidgets.QPushButton('---Align channels---')
        self.align_menu = QtWidgets.QMenu()
        self.align_btn.setMenu(self.align_menu)
        self.align_btn.setMinimumWidth(150)
        self.action_btns = []

        for i in range(self.num_channels):
            self.action_btns.append(QtWidgets.QAction('Channel ' + str(i+1), self.align_menu, checkable=True))
            self.align_menu.addAction(self.action_btns[i])
            self.action_btns[i].toggled.connect(lambda state, i=i: self._align_channel(i, state))

        self.align_btn.setEnabled(False)
        line.addWidget(self.align_btn)
        line.addStretch(1)
        align_options.setAlignment(QtCore.Qt.AlignTop)

        line = QtWidgets.QHBoxLayout()
        line.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(line)
        line.addStretch(1)
        self.save_btn = QtWidgets.QPushButton('Save and close', self)
        self.save_btn.clicked.connect(self._save)
        line.addWidget(self.save_btn)

    @utils.wait_cursor('print')
    def _update_imview(self, state=None):
        self._calc_color_channels()
        if self.peak_imview.image is not None:
            old_shape = self.peak_imview.image.shape
        else:
            old_shape = None
        new_shape = self.color_data.shape
        if self.levels is None:
            levels = self.parent.fm_controls.imview.getHistogramWidget().item.getLevels()
        else:
            levels = self.imview.getHistogramWidget().item.getLevels()
        if old_shape == new_shape:
            vr = self.peak_imview.getImageItem().getViewBox().targetRect()
        self.peak_imview.setImage(self.color_data, levels=levels)
        if old_shape == new_shape:
            self.peak_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    @utils.wait_cursor('print')
    def _update_data(self, state=None):
        if self.fm.ops._transformed:
            self.fm.show_btn.setChecked(True)
            self.recover_transformed = True
        self.data_roi = np.copy(self.fm.ops.data)
        if self.orig_data_roi is None:
            self.orig_data_roi = np.copy(self.data_roi)
            self.data_roi = (self.data_roi - self.data_roi.min()) / (self.data_roi.max() - self.data_roi.min()) \
                            * self.fm.ops.norm_factor
        else:
            if self.roi is not None:
                self._update_imview()
                self.peak_imview.addItem(self.roi)
                self.data_roi, self.coor = self.roi.getArrayRegion(self.data_roi, self.peak_imview.getImageItem(), returnMappedCoords=True)
                self.peak_imview.removeItem(self.roi)
                #self._update_imview()

        #if self.background_correction:
        #    self.data_roi = self.fm.ops.subtract_background(self.data_roi, sigma=self.sigma_btn.value())
        #if self.invert:
        #    self.data_roi = self.data_roi.max() - self.data_roi

        self.data_roi[self.data_roi < self.t_low_label.value()] = 0
        self.data_roi[self.data_roi > self.t_high_label.value()] = 0
        self._update_imview()

    @utils.wait_cursor('print')
    def _calc_color_channels(self, state=None):
        idx = self.peak_channel_btn.currentIndex()
        self.color_data = np.zeros((1,) + self.data_roi[:, :, 0].shape + (3,))
        my_channel = self.data_roi[:, :, idx]
        my_channel_rgb = np.repeat(my_channel[:, :, np.newaxis], 3, axis=2)
        rgb = tuple([int(self.fm._colors[idx][1 + 2 * c:3 + 2 * c], 16) / 255. for c in range(3)])
        self.color_data[0, :, :, :] = my_channel_rgb * rgb

    @utils.wait_cursor('print')
    def _calc_max_proj(self, state=None):
        if self.fm.ops.max_proj_data is None:
            self.fm.ops.calc_max_proj_data()
        self._update_data()

    @utils.wait_cursor('print')
    def _subtract_background(self, checked):
        self.background_correction = self.background_btn.isChecked()
        self._update_data()

    @utils.wait_cursor('print')
    def _invert_channel(self, checked):
        self.invert = self.invert_btn.isChecked()
        self._update_data()

    @utils.wait_cursor('print')
    def _change_peak_ref(self, state=None):
        self.peak_btn.setChecked(False)
        self._reset_peaks()
        self.ref_btn.setText('Channel {}'.format(self.peak_channel_btn.currentIndex()+1))
        [btn.setChecked(False) for btn in self.action_btns]
        [btn.setEnabled(True) for btn in self.action_btns]
        self.action_btns[self.peak_channel_btn.currentIndex()].setEnabled(False)
        [self.fm.ops._color_matrices.append(np.identity(3)) for i in range(self.num_channels)]
        self._update_data()

    def _align_channel(self, idx, state):
        recalc_peaks = False
        if self.peak_btn.isChecked():
            recalc_peaks = True
        self.fm._align_colors(idx, state)
        self._update_data()
        if recalc_peaks:
            self.peak_btn.setChecked(False)
            self.peak_btn.setChecked(True)

    @utils.wait_cursor('print')
    def _draw_roi(self, checked):
        if checked:
            self.peak_btn.setChecked(False)
            self.reset_btn.setEnabled(False)
            if self.roi is not None:
                self.peak_imview.removeItem(self.roi)
            self.print('Draw ROI in the image!')
            pos = np.array([self.data_roi.shape[0]//2, self.data_roi.shape[1]//2])
            size = np.array([self.data_roi.shape[0]//2, self.data_roi.shape[1]//2])
            qrect = None
            self.roi = pg.ROI(pos=pos,size=size, maxBounds=qrect, resizable=True, rotatable=True, removable=False)
            self.roi.addScaleHandle(pos=[1,1], center=[0.5,0.5])
            self.roi.addRotateHandle(pos=[0.5,0], center=[0.5,0.5])
            self.peak_imview.addItem(self.roi)
        else:
            [self.peak_imview.removeItem(point) for point in self.peaks]
            self.reset_btn.setEnabled(True)
            self.draw_btn.setEnabled(False)
            self.roi.movable = False
            self.roi.resizable = False
            self._roi_pos = np.array([self.roi.pos().x(), self.roi.pos().y()])
            self._roi_angle = float(self.roi.angle())
            self._roi_shape = np.array([self.roi.size().x(), self.roi.size().y()])
            self.log(self.data_roi.shape)
            self.print('Done drawing ROI at position ', self._roi_pos)
            self.peak_imview.removeItem(self.roi)
            if len(self.peaks) > 0:
                self._reset_peaks()
            self._update_data()

    @utils.wait_cursor('print')
    def _reset_roi(self, state=None):
        self.peak_btn.setChecked(False)
        self.draw_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)
        self.data_roi = None
        self.orig_data_roi = None
        self.roi = None
        self.coor = None
        self._roi_pos = None
        if len(self.peaks) > 0:
            self._reset_peaks()
        self._update_data()

    @utils.wait_cursor('print')
    def _set_thresholds(self, param, b, state=None):
        if b == 1:
            if param == 0:
                float_value = float(self.t_low.value())
                self.t_low_label.blockSignals(True)
                self.t_low_label.setValue(10*float_value)
                self.t_low_label.blockSignals(False)
            else:
                float_value = float(self.t_high.value())
                self.t_high_label.blockSignals(True)
                self.t_high_label.setValue(10*float_value)
                self.t_high_label.blockSignals(False)
        else:
            if param == 0:
                float_value = float(self.t_low_label.value())
                self.t_low.blockSignals(True)
                self.t_low.setValue(10*float_value)
                self.t_low.blockSignals(False)
                self.t_low_label.clearFocus()
            else:
                float_value = float(self.t_high_label.value())
                self.t_high.blockSignals(True)
                self.t_high.setValue(10*float_value)
                self.t_high.blockSignals(False)
                self.t_high_label.clearFocus()
        self.peak_btn.setChecked(False)
        self._reset_peaks()
        self.data_roi = np.copy(self.orig_data_roi)
        self._update_data()

    @utils.wait_cursor('print')
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



    @utils.wait_cursor('print')
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

    @utils.wait_cursor('print')
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

    @utils.wait_cursor('print')
    def _show_peaks(self, state=None):
        if self.draw_btn.isChecked():
            self.print('You have to confirm the ROI!')
            self.peak_btn.setChecked(False)
            return
        if not self.peak_btn.isChecked():
            [self.peak_imview.removeItem(point) for point in self.peaks]
            self.align_btn.setEnabled(False)
            return

        if len(self.peaks) > 0:
            [self.peak_imview.addItem(p) for p in self.peaks]
            self.fm.ops.adjusted_params = True
            self.align_btn.setEnabled(True)
            return

        self._reset_peaks()
        self.fm.ops.pixel_lower_threshold = self.plt.value()
        self.fm.ops.pixel_upper_threshold = self.put.value()
        self.fm.ops.flood_steps = self.flood_steps.value()
        self.fm.ops._peak_reference = self.peak_channel_btn.currentIndex()
        self.fm.ops.peak_finding(self.data_roi[:,:,self.peak_channel_btn.currentIndex()], transformed=False,
                                 roi_pos= self._roi_pos, background_correction=False)

        peaks_2d = copy.copy(self.fm.ops.peaks)
        if peaks_2d is None:
            self.print('No peaks have been found! Adjust parameters!')
            return
        if self._roi_pos is not None:
            peaks_2d -= self._roi_pos #correct for roi_pos bug out of range, qrect for roi not resizable
        if len(peaks_2d.shape) > 0:
            for i in range(len(peaks_2d)):
                pos = QtCore.QPointF(peaks_2d[i][0] - self.fm.size / 2, peaks_2d[i][1] - self.fm.size / 2)
                point_obj = pg.CircleROI(pos, self.fm.size, parent=self.peak_imview.getImageItem(), movable=False,
                                         removable=True)
                point_obj.removeHandle(0)
                self.peak_imview.addItem(point_obj)
                self.peaks.append(point_obj)
            if self.coor is not None:
                for i in range(len(self.fm.ops.peaks)):
                    self.fm.ops.peaks_orig[i] = self.coor[:,np.round(peaks_2d[i][0]).astype(int), np.round(peaks_2d[i][1]).astype(int)]
                    self.fm.ops.peaks = np.copy(self.fm.ops.peaks_orig)
        self.fm.ops.adjusted_params = True
        self.align_btn.setEnabled(True)

    @utils.wait_cursor('print')
    def _reset_peaks(self, state=None):
        self.peaks = []
        self.fm.ops.reset_peaks()

    @utils.wait_cursor('print')
    def _save(self, state=None):
        if self.fm.ops is not None and len(self.peaks) > 0:
            self.fm.ops.adjusted_params = True
            #self.fm.ops.background_correction = self.background_correction
            #self.fm.ops.sigma_background = self.sigma_btn.value()
            self.fm.ops.pixel_lower_threshold = self.plt.value()
            self.fm.ops.pixel_upper_threshold = self.put.value()
            self.fm.ops.flood_steps = self.flood_steps.value()
            self.fm.poi_ref_btn.setCurrentIndex(self.peak_channel_btn.currentIndex())
            self.fm.peak_btn.setChecked(True)
            if self.recover_transformed:
                self.fm.show_btn.setChecked(False)
        self.close()

class Merge(QtWidgets.QMainWindow):
    def __init__(self, parent, printer, logger):
        super(Merge, self).__init__(parent)
        self.parent = parent
        self.theme = self.parent.theme
        self.fm_copy = None
        self.other = self.parent.fm_controls.other
        self.print = printer
        self.log = logger

        self.curr_mrc_folder_popup = self.parent.fm_controls.other.curr_folder
        self.num_slices_popup = self.parent.fm_controls.num_slices
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
        self.size = self.parent.fm_controls.size
        self.lines = None
        self.pixel_size = self.parent.fm_controls.other.ops.pixel_size
        self.lambda_list = []
        self.theta = None
        self.lamella_pos = None
        self.lamella_size = None

        self._channels_popup = []
        self._colors_popup = list(np.copy(self.parent.fm_controls._colors))
        self._current_slice_popup = self.parent.fm_controls._current_slice
        self._overlay_popup = True
        self._clicked_points_popup = []
        self._clicked_points_popup_base_indices = []

        merged_data = self.parent.em.merged[self.parent.fm_controls.tab_index]
        if self.parent.fm_controls.tab_index == 1:
            self.fib = True
            self.downsampling = 1
        else:
            self.fib = False
            self.downsampling = 1
        if merged_data is not None:
            self.log(self._colors_popup)
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
        self._copy_pois()

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
        layout.addWidget(self.imview_popup)

        options = QtWidgets.QHBoxLayout()
        options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(options)

        print_layout = QtWidgets.QHBoxLayout()
        print_layout.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(print_layout)
        self.print_line = QtWidgets.QLabel('')
        print_layout.addWidget(self.print_line)
        print_layout.addStretch(1)

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
        self.fm_fname_popup = self.parent.fm_controls.fm_fname.text()
        label = QtWidgets.QLabel(self.fm_fname_popup, self)
        line.addWidget(label, stretch=1)
        self.max_proj_btn_popup = QtWidgets.QCheckBox('Max projection')
        self.max_proj_btn_popup.stateChanged.connect(self._show_max_projection_popup)
        line.addWidget(self.max_proj_btn_popup)

        self.slice_select_btn_popup = QtWidgets.QSpinBox(self)
        self.slice_select_btn_popup.editingFinished.connect(self._slice_changed_popup)
        self.slice_select_btn_popup.setRange(0, self.parent.fm.num_slices)
        self.slice_select_btn_popup.setValue(self.parent.fm_controls.slice_select_btn.value())
        line.addWidget(self.slice_select_btn_popup)
        if self.fib:
            self.max_proj_btn_popup.setEnabled(False)
            self.max_proj_btn_popup.blockSignals(True)
            self.max_proj_btn_popup.setChecked(True)
            self.max_proj_btn_popup.blockSignals(False)
            self.slice_select_btn_popup.setEnabled(False)

        if self.parent.fm_controls.max_proj_btn.isChecked():
            self.max_help = True
            self.max_proj_btn_popup.setChecked(True)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('EM Image:', self)
        line.addWidget(label)
        self.em_fname_popup = self.parent.fm_controls.other.mrc_fname.text()
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

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Lamella thickness [nm]: ')
        self.lamella_btn = QtWidgets.QLineEdit('')
        self.lamella_btn.setMaximumWidth(40)
        self.lamella_btn.setEnabled(self.fib)
        self.lamella_btn.textChanged.connect(self._set_lamella_size)
        self.draw_lines_btn = QtWidgets.QCheckBox('Draw lamella')
        self.draw_lines_btn.setEnabled(self.fib)
        self.draw_lines_btn.stateChanged.connect(self._toggle_lines)
        self.lamella_btn.setText('300')
        if self.parent.fm_controls.tab_index == 1 or self.parent.fm_controls.tab_index == 2:
            line.addWidget(label)
            line.addWidget(self.lamella_btn)
            line.addWidget(self.draw_lines_btn)
            line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        line.addStretch(1)
        self.save_btn_popup = QtWidgets.QPushButton('Save data')
        self.save_btn_popup.clicked.connect(self._save_data_popup)
        line.addWidget(self.save_btn_popup)

    def _calc_ellipses_popup(self, cov):
        cov_matrix = np.copy(self.parent.fm_controls.other.cov_matrix)
        self.log('Cov matrix: \n', cov_matrix)
        self.log('Cov_i matrix: \n', cov[:2,:2]*(self.pixel_size[0]**2))
        self.print('Fitting precision in merged frame: ', np.sqrt(np.diag(cov[:2, :2])) * self.pixel_size[0])
        cov_matrix += (cov[:2, :2] * (self.pixel_size[0]**2))

        self.log('Total covariance: \n', cov_matrix)
        self.log('Total transformed errors [nm]: ', np.sqrt(np.diag(cov_matrix)))
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[order]
        vx, vy = eigvecs[0,0], eigvecs[0,1]
        theta = -np.arctan2(vy, vx) * 180 / np.pi

        #scale eigenvalues to 75% confidence interval and pixel size
        #lambda_1, lambda_2 = 2 * np.sqrt(2.77*eigvals)[:2] / np.array(self.pixel_size[:2])
        #scale eigenvalues to 68.3% (1sigma) confidence interval and pixel size
        lambda_1, lambda_2 = 2 * np.sqrt(2.3*eigvals)[:2] / np.array(self.pixel_size[:2])
        #scale eigenvalues to 50%  confidence interval and pixel size
        #lambda_1, lambda_2 = 2 * np.sqrt(1.39*eigvals)[:2] / np.array(self.pixel_size[:2])
        self.log(lambda_1, lambda_2)
        self.print('Lambdas [nm]: ', lambda_1*self.pixel_size[0], lambda_2*self.pixel_size[0])
        return lambda_1, lambda_2, theta

    def _convert_pois(self, points=None):
        if points is None:
            points = [p.pos() for p in self.parent.fm_controls.pois]
            z_values = self.parent.fm_controls.pois_z
            covs = self.parent.fm_controls.pois_cov
            sizes = self.parent.fm_controls.pois_sizes
        else:
            z_values = [self.parent.fm_controls.pois_z[-1]]
            covs = [self.parent.fm_controls.pois_cov[-1]]
            sizes = [self.parent.fm_controls.pois_sizes[-1]]

        if points is None or len(points) == 0:
            return [], []

        transf_points = []
        transf_covs = []
        if self.parent.fm_controls.tab_index == 1 or self.parent.fm_controls.tab_index == 2:
            tr_matrix = np.copy(self.parent.fm_controls.tr_matrices)
            tr_matrix = np.insert(np.insert(tr_matrix, 2, 0, axis=0), 2, 0, axis=1)
            tr_matrix[2, 2] = 1
            if self.parent.fm_controls.tab_index == 1:
                refine_matrix = np.insert(np.insert(self.other.ops._refine_matrix, 2, 0, axis=0), 2, 0, axis=1)
            else:
                refine_matrix = np.insert(np.insert(self.other.ops.gis_transf @ self.other.ops._refine_matrix, 2, 0, axis=0), 2, 0, axis=1)
            refine_matrix[2, 2] = 1
            tot_matrix = refine_matrix @ self.other.ops.fib_matrix @ tr_matrix
            for i in range(len(points)):
                pos = points[i]
                size = sizes[i]
                point = np.array([pos.x() + size[0] // 2 - 0.5, pos.y() + size[1] // 2 - 0.5, z_values[i], 1]) #0.5 pixel correction for drawing ellipse
                transf = tot_matrix @ point
                transf_points.append(transf[:2])
                cov_i = np.insert(np.insert(np.array(covs[i]), 3, 0, axis=0), 3, 0, axis=1)
                tf_cov = tot_matrix @ np.array(cov_i) @ tot_matrix.T
                transf_covs.append(tf_cov)
        else:
            tot_matrix = self.parent.fm_controls.tr_matrices
            for i in range(len(points)):
                pos = points[i]
                size = sizes[i]
                point = np.array([pos.x() + size[0] // 2 - 0.5, pos.y() + size[1] // 2 - 0.5, 1])
                transf = tot_matrix @ point
                transf_points.append(transf[:2] / self.downsampling)
                cov_i = np.array(covs[i])
                cov_i[-1,-1] = 0
                tf_cov = tot_matrix @ cov_i @ tot_matrix.T
                transf_covs.append(tf_cov[:3, :3])

        return transf_points, transf_covs

    @utils.wait_cursor('print')
    def _copy_pois(self, state=None):
        tf_points, tf_covs = self._convert_pois()
        for i in range(len(tf_points)):
            pos = QtCore.QPointF(tf_points[i][0], tf_points[i][1])
            self._draw_correlated_points_popup(pos, self.imview_popup.getImageItem(), tf_covs[i])
        [self._clicked_points_popup_base_indices.append(i) for i in range(len(tf_points))]

    @utils.wait_cursor('print')
    def _update_poi(self, pos):
        tf_points, tf_covs = self._convert_pois([pos])
        pos = QtCore.QPointF(tf_points[0][0], tf_points[0][1])
        self._draw_correlated_points_popup(pos, self.imview_popup.getImageItem(), tf_covs[0])
        self._clicked_points_popup_base_indices.append(self.counter_popup-1)

    @utils.wait_cursor('print')
    def _imview_clicked_popup(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
            return
        if self.draw_lines_btn.isChecked():
            pos = self.imview_popup.getImageItem().mapFromScene(event.pos())
            idx = self._check_ellipse_index(pos)
            if idx is not None:
                self._draw_lines(idx)
            else:
                self.print('You have to select a POI!')

    @utils.wait_cursor('print')
    def _draw_correlated_points_popup(self, pos, item, tf_cov):
        img_center = np.array(self.data_popup.shape)/2
        lambda_1, lambda_2, theta = self._calc_ellipses_popup(tf_cov)
        point = pg.EllipseROI(img_center, size=[lambda_1, lambda_2], angle=0, parent=item,
                              movable=False, removable=True, resizable=False, rotatable=False)

        pos = [pos.x() - lambda_1 / 2 + 0.5, pos.y() - lambda_2 / 2 + 0.5]
        self.print('Total error estimate: ', lambda_1/2, lambda_2/2)
        point.setTransformOriginPoint(QtCore.QPointF(lambda_1/2, lambda_2/2))
        point.setRotation(theta)
        point.setPos(pos)
        point.setPen(0, 255, 0)
        point.removeHandle(0)
        point.removeHandle(0)
        self.imview_popup.addItem(point)
        self._clicked_points_popup.append(point)
        self.lambda_list.append((lambda_1, lambda_2))
        self.counter_popup += 1
        annotation = pg.TextItem(str(self.counter_popup), color=(0, 255, 0), anchor=(0, 0))
        annotation.setPos(pos[0] + 5, pos[1] + 5)
        self.annotations_popup.append(annotation)
        self.imview_popup.addItem(annotation)
        point.sigRemoveRequested.connect(lambda: self._remove_correlated_points_popup(point, annotation))

    @utils.wait_cursor('print')
    def _remove_correlated_points_popup(self, pt, anno):
        idx = self._clicked_points_popup.index(pt)

        for i in range(idx+1, len(self._clicked_points_popup)):
            self.annotations_popup[i].setText(str(i))
        self.counter_popup -= 1

        self.imview_popup.removeItem(pt)
        self._clicked_points_popup.remove(pt)
        self.imview_popup.removeItem(anno)
        self.annotations_popup.remove(anno)
        if idx in self._clicked_points_popup_base_indices:
            self._clicked_points_popup_base_indices.remove(idx)

    @utils.wait_cursor('print')
    def _show_overlay_popup(self, checked):
        self._overlay_popup = not self._overlay_popup
        self._update_imview_popup()

    @utils.wait_cursor('print')
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
            self.print('Invalid color')

    @utils.wait_cursor('print')
    def _calc_color_channels_popup(self, state=None):
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

    @utils.wait_cursor('print')
    def _update_imview_popup(self, state=None):
        self._calc_color_channels_popup()
        old_shape = self.imview_popup.image.shape
        new_shape = self.color_data_popup.shape
        if old_shape == new_shape:
            vr = self.imview_popup.getImageItem().getViewBox().targetRect()
        levels = self.imview_popup.getHistogramWidget().item.getLevels()
        self.imview_popup.setImage(self.color_data_popup, levels=levels)
        if old_shape == new_shape:
            self.imview_popup.getImageItem().getViewBox().setRange(vr, padding=0)

    @utils.wait_cursor('print')
    def _calc_stage_positions_popup(self, checked):
        if checked:
            self.print('Select points of interest!')
            if len(self._clicked_points_popup) != 0:
                [self.imview_popup.removeItem(point) for point in self._clicked_points_popup]
                [self.imview_popup.removeItem(annotation) for annotation in self.annotations_popup]
                self._clicked_points_popup = []
                self._clicked_points_popup_base_indices = []
                self.annotations_popup = []

                self.counter_popup = 0
                self.stage_positions_popup = None
                self.coordinates = []
        else:
            coordinates = []
            for i in range(len(self._clicked_points_popup)):
                coordinates.append(self.downsampling * np.array([self._clicked_points_popup[i].pos().x() + self.lambda_list[i][0] / 2 - 0.5,
                                                                 self._clicked_points_popup[i].pos().y() + self.lambda_list[i][1] / 2] - 0.5))
            self.coordinates = np.array(coordinates)
            #self.coordinates[:,1] = self.data_popup.shape[1] - self.coordinates[:,1]

            #if self.fib:
            #    self.stage_positions_popup = np.copy(coordinates)
            #else:
            #    self.stage_positions_popup = self.parent.sem_controls.ops.calc_stage_positions(coordinates,
            #                                                                                 self.downsampling)
            self.print('Done selecting points of interest!')

    def _save_data_popup(self, state=None):
        #if self.select_btn_popup.isChecked():
        #    self.print('You have to confirm selected points first!')
        #    return
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
                self.print('Permission error! Choose a different directory!')

    @utils.wait_cursor('print')
    def _save_merge_popup(self, fname):
        with mrc.new(fname + '.mrc', overwrite=True) as f:
            f.set_data(self.data_popup.astype(np.float32))
            f.update_header_stats()

    @utils.wait_cursor('print')
    def _save_coordinates_popup(self, fname):
        #if self.stage_positions_popup is not None:
            #for i in range(len(self.stage_positions_popup)):
        enumerated = []
        for i in range(len(self.coordinates)):
            #enumerated.append((i + 1, self.stage_positions_popup[i][0], self.stage_positions_popup[i][1]))
            enumerated.append((i + 1, self.coordinates[i][0], self.coordinates[i][1]))

        lamella_pos = []
        lamella_strings = ['Lamella upper boundary:', 'Lamella lower boundary:' ]
        if self.lines is not None:
            for i in range(len(self.lines)):
                lamella_pos.append([lamella_strings[i], self.lines[i].pos().y()])

        with open(fname + '.txt', 'w', newline='') as f:
            csv.writer(f, delimiter=' ').writerows(enumerated)
            csv.writer(f, delimiter=' ').writerows(lamella_pos)

    @utils.wait_cursor('print')
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

    @utils.wait_cursor('print')
    def _slice_changed_popup(self, state=None):
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

    @utils.wait_cursor('print')
    def _set_lamella_size(self, state=None):
        if self.lamella_btn.text() is not '':
            self.lamella_size = int(self.lamella_btn.text())
            if self.draw_lines_btn.isChecked():
                self.draw_lines_btn.setChecked(False)
                self.draw_lines_btn.setChecked(True)

    def _update_lines(self, i):
        pos = self.lines[i].pos().y()
        k = self.lamella_size / self.pixel_size[1]
        if i == 0:
            self.lines[1].setPos(pos-k)
            self.lamella_pos[1] = pos - k / 2
        else:
            self.lines[0].setPos(pos+k)
            self.lamella_pos[1] = pos + k / 2

        self.print_lamella_pos()

    def _check_ellipse_index(self, pos):
        point = np.array([pos.x(), pos.y()])
        idx = None
        for i in range(len(self.lambda_list)):
            poi = np.array([self._clicked_points_popup[i].pos().x() + self.lambda_list[i][0] / 2,
                            self._clicked_points_popup[i].y() + self.lambda_list[i][1] / 2])
            diff = point - poi
            dist = np.sqrt(diff[0]**2 + diff[1]**2)
            ref = np.sqrt(self.lambda_list[i][0]**2 + self.lambda_list[i][1]**2)
            if dist <= ref:
                idx = i
                break
        return idx

    @utils.wait_cursor('print')
    def _toggle_lines(self, checked):
        if checked:
            self.print('Selct a POI!')
            self.lines = []
        else:
            [self.imview_popup.removeItem(line) for line in self.lines]

    @utils.wait_cursor('print')
    def _draw_lines(self, idx):
        point = self._clicked_points_popup[idx].pos()
        self.lamella_pos = [point.x() + self.lambda_list[idx][0] / 2, point.y() + self.lambda_list[idx][1] / 2]

        k = self.lamella_size / self.pixel_size[1] / 2
        line = pg.InfiniteLine(pos=self.lamella_pos[1] + k, angle=0, pen='r', hoverPen='c', movable=True,
                               bounds=[0, self.data_popup.shape[0]])
        line2 = pg.InfiniteLine(pos=self.lamella_pos[1] - k, angle=0, pen='r', hoverPen='c', movable=True,
                                bounds=[0, self.data_popup.shape[0]])

        line.sigPositionChanged.connect(lambda : self._update_lines(0))
        line2.sigPositionChanged.connect(lambda : self._update_lines(1))
        for i in range(len(self.lines)):
            self.imview_popup.removeItem(self.lines[i])
        self.lines = []
        self.lines.append(line)
        self.lines.append(line2)
        for i in range(len(self.lines)):
            self.imview_popup.addItem(self.lines[i])

        self.print_lamella_pos()

    def print_lamella_pos(self):
        center = np.array(self.data_popup[:,:,-1].shape) / 2  #y-axis showing upwards
        coor = (np.array(self.lamella_pos) - center) * self.pixel_size[:2] * np.array([1, -1]) * 1e-3 #coor in microns
        self.print_line.setText('Lamella position [\u03BCm]: [{}, {}]'.format(coor[0], coor[1]))

    @utils.wait_cursor('print')
    def _set_theme_popup(self, name):
        if name == 'none':
            self.setStyleSheet('')
        else:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'styles/%s.qss' % name), 'r') as f:
                self.setStyleSheet(f.read())
        self.settings.setValue('theme', name)

    @utils.wait_cursor('print')
    def _reset_init(self, state=None):
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
            if self.parent.fm_controls.other == self.other:
                self.parent.fm_controls.progress_bar.setValue(0)
                self.parent.fm_controls.poi_btn.setEnabled(True)
                print('heere')
                print(self._colors_popup)
                print(self.parent.fm_controls._colors)
                self.parent.fm_controls._colors[-1] = self._colors_popup[-1]
            self.other.progress = 0
            self.parent.project.merged = [False, False, False, False]
            self.other.show_merge = False
        self._reset_init()
        event.accept()
