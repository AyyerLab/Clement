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


class Merge(QtGui.QMainWindow, ):
    def __init__(self, parent):
        super(Merge, self).__init__(parent)
        self.parent = parent
        self.theme = self.parent.theme
        self.fm_copy = None

        self.curr_mrc_folder_popup = self.parent.emcontrols.curr_folder
        self.num_slices_popup = self.parent.fmcontrols.num_slices
        self.downsampling = 4  # per dimension
        self.color_data_popup = None
        self.color_overlay_popup = None
        self.annotations_popup = []
        self.counter_popup = 0
        self.stage_positions_popup = None
        self.settings = QtCore.QSettings('MPSD-CNI', 'CLEMGui', self)
        self.max_help = False
        self.fib = False
        self.size = 10

        self._channels_popup = [True, True, True, True]
        self._colors_popup = list(np.copy(self.parent.colors))
        self._current_slice_popup = self.parent.fmcontrols._current_slice
        self._overlay_popup = True
        self._clicked_points_popup = []

        if self.parent.fibcontrols.fib:
            merged_data = self.parent.fm.merged_3d
            self.fib = True
            self.downsampling = 1
        else:
            merged_data = self.parent.fm.merged_2d
            self.fib = False
        if merged_data is not None:
            self.data_popup = np.copy(merged_data)
            self._channels_popup.append(True)
            self._colors_popup.append('#808080')
        else:
            self.data_popup = np.copy(self.parent.fm.data)

        self.data_orig_popup = np.copy(self.data_popup)

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
        action = QtWidgets.QAction('&Save data', self)
        action.triggered.connect(self._save_data_popup)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Quit', self)
        action.triggered.connect(self.close)
        filemenu.addAction(action)
        self._set_theme(self.theme)

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

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Colors:', self)
        line.addWidget(label)
        self.channel1_btn_popup = QtWidgets.QCheckBox(' ', self)
        self.channel2_btn_popup = QtWidgets.QCheckBox(' ', self)
        self.channel3_btn_popup = QtWidgets.QCheckBox(' ', self)
        self.channel4_btn_popup = QtWidgets.QCheckBox(' ', self)
        self.channel5_btn_popup = QtWidgets.QCheckBox(' ', self)
        self.overlay_btn_popup = QtWidgets.QCheckBox('Overlay', self)
        self.channel1_btn_popup.setChecked(True)
        self.channel2_btn_popup.setChecked(True)
        self.channel3_btn_popup.setChecked(True)
        self.channel4_btn_popup.setChecked(True)
        self.channel5_btn_popup.setChecked(True)
        self.overlay_btn_popup.setChecked(True)
        self.channel1_btn_popup.stateChanged.connect(lambda state, channel=0: self._show_channels_popup(state, channel))
        self.channel2_btn_popup.stateChanged.connect(lambda state, channel=1: self._show_channels_popup(state, channel))
        self.channel3_btn_popup.stateChanged.connect(lambda state, channel=2: self._show_channels_popup(state, channel))
        self.channel4_btn_popup.stateChanged.connect(lambda state, channel=3: self._show_channels_popup(state, channel))

        self.channel5_btn_popup.stateChanged.connect(lambda state, channel=4: self._show_channels_popup(state, channel))
        self.overlay_btn_popup.stateChanged.connect(self._show_overlay_popup)

        self.c1_btn_popup = QtWidgets.QPushButton(' ', self)
        self.c1_btn_popup.clicked.connect(lambda: self._sel_color_popup(0, self.c1_btn_popup))
        width = self.c1_btn_popup.fontMetrics().boundingRect(' ').width() + 24
        self.c1_btn_popup.setFixedWidth(width)
        self.c1_btn_popup.setMaximumHeight(width)
        self.c1_btn_popup.setStyleSheet('background-color: {}'.format(self._colors_popup[0]))
        self.c2_btn_popup = QtWidgets.QPushButton(' ', self)
        self.c2_btn_popup.clicked.connect(lambda: self._sel_color_popup(1, self.c2_btn_popup))
        self.c2_btn_popup.setMaximumHeight(width)
        self.c2_btn_popup.setFixedWidth(width)
        self.c2_btn_popup.setStyleSheet('background-color: {}'.format(self._colors_popup[1]))
        self.c3_btn_popup = QtWidgets.QPushButton(' ', self)
        self.c3_btn_popup.setMaximumHeight(width)
        self.c3_btn_popup.setFixedWidth(width)
        self.c3_btn_popup.clicked.connect(lambda: self._sel_color_popup(2, self.c3_btn_popup))
        self.c3_btn_popup.setStyleSheet('background-color: {}'.format(self._colors_popup[2]))
        self.c4_btn_popup = QtWidgets.QPushButton(' ', self)
        self.c4_btn_popup.setMaximumHeight(width)
        self.c4_btn_popup.setFixedWidth(width)
        self.c4_btn_popup.clicked.connect(lambda: self._sel_color_popup(3, self.c4_btn_popup))
        self.c4_btn_popup.setStyleSheet('background-color: {}'.format(self._colors_popup[3]))
        self.c5_btn_popup = QtWidgets.QPushButton(' ', self)
        self.c5_btn_popup.setMaximumHeight(width)
        self.c5_btn_popup.setFixedWidth(width)
        if len(self._channels_popup) == 5:
            self.c5_btn_popup.clicked.connect(lambda: self._sel_color_popup(4, self.c5_btn_popup))
            self.c5_btn_popup.setStyleSheet('background-color: {}'.format(self._colors_popup[4]))

        line.addWidget(self.c1_btn_popup)
        line.addWidget(self.channel1_btn_popup)
        line.addWidget(self.c2_btn_popup)
        line.addWidget(self.channel2_btn_popup)
        line.addWidget(self.c3_btn_popup)
        line.addWidget(self.channel3_btn_popup)
        line.addWidget(self.c4_btn_popup)
        line.addWidget(self.channel4_btn_popup)
        line.addWidget(self.c5_btn_popup)
        line.addWidget(self.channel5_btn_popup)
        line.addWidget(self.overlay_btn_popup)
        line.addStretch(1)

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
        self.imview_popup.removeItem(pt)
        self._clicked_points_popup.remove(pt)
        self.imview_popup.removeItem(anno)
        self.annotations_popup.remove(anno)

    def _show_overlay_popup(self, checked):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self._overlay_popup = not self._overlay_popup
        self._update_imview_popup()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _show_channels_popup(self, checked, my_channel):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self._channels_popup[my_channel] = not self._channels_popup[my_channel]
        self._update_imview_popup()
        QtWidgets.QApplication.restoreOverrideCursor()

    def _sel_color_popup(self, index, button):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            cname = color.name()
            self._colors_popup[index] = cname
            button.setStyleSheet('background-color: {}'.format(cname))
            self._update_imview_popup()
        else:
            print('Invalid color')

    def _calc_color_channels_popup(self):
        print(self.data_popup[:, :, 0].shape)
        print(len(self._channels_popup))
        # self.color_data_popup = np.zeros((len(self._channels_popup),) + self.data_popup[:,:,0].shape + (3,))
        self.color_data_popup = np.zeros(
            (len(self._channels_popup), int(np.ceil(self.data_popup.shape[0] / self.downsampling)),
             int(np.ceil(self.data_popup.shape[1] / self.downsampling)), 3))
        print(self.color_data_popup.shape)
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
        else:
            coordinates = [np.array([point.x() + self.size / 2, point.y() + self.size / 2]) for point in
                           self._clicked_points_popup]
            if self.fib:
                self.stage_positions_popup = np.copy(coordinates)
            else:
                self.stage_positions_popup = self.parent.emcontrols.ops.calc_stage_positions(coordinates,
                                                                                             self.downsampling)
            print('Done selecting points of interest!')

    def _save_data_popup(self):
        if self.curr_mrc_folder_popup is None:
            self.curr_mrc_folder_popup = os.getcwd()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Merged Image', self.curr_mrc_folder_popup)
        if file_name != '':
            try:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                file_name = file_name.split('.', 1)[0]
                screenshot = self.imview_popup.grab()
                screenshot.save(file_name, 'tif')
                self._save_merge_popup(file_name)
                self._save_coordinates_popup(file_name)
                QtWidgets.QApplication.restoreOverrideCursor()
            except PermissionError:
                print('Permission error! Choose a different directory!')
                QtWidgets.QApplication.restoreOverrideCursor()

    def _save_merge_popup(self, fname):
        with mrc.new(fname + '.mrc', overwrite=True) as f:
            f.set_data(self.data_popup.astype(np.float32))
            f.update_header_stats()

    def _save_coordinates_popup(self, fname):
        if self.stage_positions_popup is not None:
            enumerated = []
            for i in range(len(self.stage_positions_popup)):
                enumerated.append((i + 1, self.stage_positions_popup[i][0], self.stage_positions_popup[i][1]))
                with open(fname + '.txt', 'a', newline='') as f:
                    csv.writer(f, delimiter=' ').writerows(enumerated)

    def _show_max_projection_popup(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.max_help:
            self.max_help = False
            self.slice_select_btn_popup.setEnabled(False)
            QtWidgets.QApplication.restoreOverrideCursor()
            return
        else:
            if self.fm_copy is None:
                self.fm_copy = copy.copy(self.parent.fm)
            self.slice_select_btn_popup.setEnabled(not self.max_proj_btn_popup.isChecked())
            self.fm_copy.calc_max_projection()
            self.fm_copy.apply_merge()
            self.data_popup = np.copy(self.fm_copy.merged)
            self._update_imview_popup()
            QtWidgets.QApplication.restoreOverrideCursor()

    def _slice_changed_popup(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
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
        QtWidgets.QApplication.restoreOverrideCursor()

    def _set_theme(self, name):
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

    def closeEvent(self, event):
        if self.parent is not None:
            self.parent.fmcontrols.progress.setValue(0)
            self.parent.project.merged = False
        self._reset_init()
        event.accept()
