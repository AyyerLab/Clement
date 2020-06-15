#!/usr/bin/env python
import sys
import os
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from . import res_styles
from .em_controls import EMControls
from .fm_controls import FMControls
from .fib_controls import FIBControls
from .project import Project
from .popup import Merge, Scatter, Convergence

warnings.simplefilter('ignore', category=FutureWarning)


def resource_path(rel_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, rel_path)


class GUI(QtGui.QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        self.settings = QtCore.QSettings('MPSD-CNI', 'CLEMGui', self)
        self.colors = self.settings.value('channel_colors', defaultValue=['#ff0000', '#00ff00', '#0000ff', '#808080'])
        self._init_ui()

    def _init_ui(self):
        geom = self.settings.value('geometry')
        if geom is None:
            self.resize(1600, 800)
        else:
            self.setGeometry(geom)

        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        # Image views
        splitter_images = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter_images, stretch=1)

        # -- FM Image view
        # self.fm_imview = pg.ImageView()
        self.fm_stacked_imview = QtWidgets.QStackedWidget()
        self.fm_imview = pg.ImageView()
        self.fm_imview.ui.roiBtn.hide()
        self.fm_imview.ui.menuBtn.hide()
        self.fm_stacked_imview.addWidget(self.fm_imview)
        splitter_images.addWidget(self.fm_stacked_imview)

        # -- EM Image view
        self.em_imview = QtWidgets.QStackedWidget()
        self.tem_imview = pg.ImageView()
        self.tem_imview.ui.roiBtn.hide()
        self.tem_imview.ui.menuBtn.hide()
        self.fib_imview = pg.ImageView()
        self.fib_imview.ui.roiBtn.hide()
        self.fib_imview.ui.menuBtn.hide()
        self.em_imview.addWidget(self.tem_imview)
        self.em_imview.addWidget(self.fib_imview)
        splitter_images.addWidget(self.em_imview)

        # Options
        options = QtWidgets.QHBoxLayout()
        options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(options)

        self.fmcontrols = FMControls(self.fm_imview, self.colors)
        self.fm_imview.getImageItem().getViewBox().sigRangeChanged.connect(self.fmcontrols._couple_views)
        self.fmcontrols.curr_folder = self.settings.value('fm_folder', defaultValue=os.getcwd())
        options.addWidget(self.fmcontrols)

        vbox = QtWidgets.QVBoxLayout()
        options.addLayout(vbox)
        self.tabs = QtWidgets.QTabWidget()
        self.tab_2d = QtWidgets.QWidget()
        self.tab_3d = QtWidgets.QWidget()
        self.tabs.resize(300, 200)
        self.tabs.addTab(self.tab_2d, 'TEM/SEM')
        self.tabs.addTab(self.tab_3d, 'FIB')
        vbox.addWidget(self.tabs)

        vbox_2d = QtWidgets.QVBoxLayout()
        vbox_3d = QtWidgets.QVBoxLayout()
        self.tab_2d.setLayout(vbox_2d)
        self.tab_3d.setLayout(vbox_3d)

        self.emcontrols = EMControls(self.tem_imview, vbox_2d)
        self.tem_imview.getImageItem().getViewBox().sigRangeChanged.connect(self.emcontrols._couple_views)
        self.emcontrols.curr_folder = self.settings.value('em_folder', defaultValue=os.getcwd())
        vbox_2d.addWidget(self.emcontrols)
        self.fibcontrols = FIBControls(self.fib_imview, vbox_3d, self.emcontrols.ops)
        self.fibcontrols.curr_folder = self.settings.value('em_folder', defaultValue=os.getcwd())
        vbox_3d.addWidget(self.fibcontrols)

        self.tabs.currentChanged.connect(self.select_tab)
        # Connect controllers
        self.emcontrols.err_plt_btn.clicked.connect(lambda: self._show_scatter(idx=0))
        self.emcontrols.convergence_btn.clicked.connect(lambda: self._show_convergence(idx=0))
        self.emcontrols.other = self.fmcontrols
        self.fibcontrols.err_plt_btn.clicked.connect(lambda: self._show_scatter(idx=1))
        self.fibcontrols.convergence_btn.clicked.connect(lambda: self._show_convergence(idx=1))
        self.fibcontrols.other = self.fmcontrols
        self.fmcontrols.other = self.emcontrols
        self.fmcontrols.merge_btn.clicked.connect(self.merge)

        self.popup = None
        self.scatter = None
        self.convergence = None
        self.project = Project(self.fmcontrols, self.emcontrols, self.fibcontrols, self)
        # Menu Bar
        self._init_menubar()

        self.theme = self.settings.value('theme')
        if self.theme is None:
            self.theme = 'none'
        self._set_theme(self.theme)

        self.show()

    def _init_menubar(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        # -- File menu
        filemenu = menubar.addMenu('&File')
        action = QtWidgets.QAction('Load &FM image(s)', self)
        action.triggered.connect(self.fmcontrols._load_fm_images)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Load &EM montage', self)
        action.triggered.connect(self.emcontrols._load_mrc)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Load project', self)
        action.triggered.connect(self._load_p)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Save project', self)
        action.triggered.connect(self._save_p)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Save binned montage', self)
        action.triggered.connect(self.emcontrols._save_mrc_montage)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Quit', self)
        action.triggered.connect(self.close)
        filemenu.addAction(action)

        # -- Theme menu
        thememenu = menubar.addMenu('&Theme')
        agroup = QtWidgets.QActionGroup(self)
        agroup.setExclusive(True)
        # action = QtWidgets.QAction('None', self)
        # action.triggered.connect(lambda: self._set_theme('none'))
        # thememenu.addAction(action)
        # agroup.addAction(action)
        action = QtWidgets.QAction('Dark', self)
        action.triggered.connect(lambda: self._set_theme('dark'))
        thememenu.addAction(action)
        agroup.addAction(action)
        action = QtWidgets.QAction('Solarized', self)
        action.triggered.connect(lambda: self._set_theme('solarized'))
        thememenu.addAction(action)
        agroup.addAction(action)

        self.show()

    def _save_p(self):
        self.project._save_project()

    def _load_p(self):
        self.project._load_project()

    def select_tab(self, idx):
        self.fmcontrols.select_btn.setChecked(False)
        for i in range(len(self.fmcontrols._points_corr)):
            self.fmcontrols._remove_correlated_points(self.fmcontrols._points_corr[0])
        if idx == 0:
            self.em_imview.setCurrentIndex(0)
            self.emcontrols._update_imview()
            self.fmcontrols.other = self.emcontrols
            if self.fmcontrols._refined:
                self.fmcontrols.undo_refine_btn.setEnabled(True)
            else:
                self.fmcontrols.undo_refine_btn.setEnabled(False)
            self.fibcontrols.fib = False
        else:
            if self.emcontrols.ops is not None and self.fmcontrols.ops is not None:
                if self.fmcontrols.ops.points is not None and self.emcontrols.ops.points is not None:
                    self.fmcontrols._calc_tr_matrices()
            self.fibcontrols.fib = True
            self.em_imview.setCurrentIndex(1)
            if self.fibcontrols._refined:
                self.fmcontrols.undo_refine_btn.setEnabled(True)
            else:
                self.fmcontrols.undo_refine_btn.setEnabled(False)
            self.fibcontrols._update_imview()
            self.fibcontrols.sem_ops = self.emcontrols.ops
            self.fmcontrols.other = self.fibcontrols
            if self.emcontrols.ops is not None:
                if self.emcontrols.ops._orig_points is not None:
                    self.fibcontrols.enable_buttons(enable=True)
                else:
                    self.fibcontrols.enable_buttons(enable=False)
                if self.fibcontrols.ops is not None and self.emcontrols.ops._tf_points is not None:
                    self.fibcontrols.ops._transformed = True

            if self.fibcontrols.num_slices is None:
                self.fibcontrols.num_slices = self.fmcontrols.num_slices
                if self.fibcontrols.ops is not None:
                    if self.fibcontrols.ops.fib_matrix is not None and self.fmcontrols.num_slices is not None:
                        self.fibcontrols.correct_grid_z()

    def _show_scatter(self, idx):
        if idx == 0:
            self.scatter = Scatter(self, self.emcontrols)
        else:
            self.scatter = Scatter(self, self.fibcontrols)
        self.scatter.show()

    def _show_convergence(self, idx):
        if len(self.fmcontrols.other._conv[idx]) == 3:
            if idx == 0:
                self.convergence = Convergence(self, self.emcontrols)
            else:
                self.convergence = Convergence(self, self.fibcontrols)
            self.convergence.show()
        else:
            print('To use this feature, you have to use at least 10 points for the refinement!')

    def merge(self, project=None):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.fm = self.fmcontrols.ops
        if self.fibcontrols.fib:
            self.em = self.fibcontrols.ops
            em = self.fibcontrols.sem_ops
        else:
            self.em = self.emcontrols.ops
            em = self.em

        if self.fm is not None and self.em is not None:
            if self.fibcontrols.fib and self.fibcontrols.sem_ops.data is None:
                print('You have to calculate the FM to TEM/SEM correlation first!')
            else:
                if self.fm._tf_points is not None and (em._tf_points is not None or em._tf_points_region is not None):
                    condition = self.fmcontrols.merge()
                    if condition:
                        if self.popup is not None:
                            self.popup.close()
                        self.popup = Merge(self)
                        self.project.merged = True
                        self.project.popup = self.popup
                        if self.project.load_merge:
                            self.project._load_merge(project)
                            self.project.load_merge = False
                        QtWidgets.QApplication.restoreOverrideCursor()
                        self.popup.show()
                    QtWidgets.QApplication.restoreOverrideCursor()
                else:
                    print('You have to transform the FM and the TEM/SEM images first!')
                    QtWidgets.QApplication.restoreOverrideCursor()
        else:
            print('Select FM and EM data first!')
            QtWidgets.QApplication.restoreOverrideCursor()

    def _set_theme(self, name):
        self.setStyleSheet('')
        with open(resource_path('styles/%s.qss' % name), 'r') as f:
            self.setStyleSheet(f.read())

        if name != 'none':
            if name == 'solarized':
                c = (203, 76, 22, 80)
                bc = '#002b36'
            else:
                c = (0, 0, 255, 80)
                bc = (0, 0, 0)

            for imview in [self.fib_imview, self.tem_imview, self.fm_imview]:
                imview.view.setBackgroundColor(bc)
                hwidget = imview.getHistogramWidget()
                hwidget.setBackground(bc)
                hwidget.item.region.setBrush(c)
                hwidget.item.fillHistogram(color=c)
        self.settings.setValue('theme', name)

    def keyPressEvent(self, event):
        key = event.key()
        mod = int(event.modifiers())

        if QtGui.QKeySequence(mod + key) == QtGui.QKeySequence('Ctrl+P'):
            self._prev_file()
        elif QtGui.QKeySequence(mod + key) == QtGui.QKeySequence('Ctrl+N'):
            self._next_file()
        elif QtGui.QKeySequence(mod + key) == QtGui.QKeySequence('Ctrl+W'):
            self.close()
        else:
            event.ignore()

    def closeEvent(self, event):
        self.settings.setValue('channel_colors', self.colors)
        self.settings.setValue('geometry', self.geometry())
        self.settings.setValue('fm_folder', self.fmcontrols.curr_folder)
        self.settings.setValue('em_folder', self.emcontrols.curr_folder)
        self.settings.setValue('fib_folder', self.fibcontrols.curr_folder)
        event.accept()


def main():
    app = QtWidgets.QApplication([])
    gui = GUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
