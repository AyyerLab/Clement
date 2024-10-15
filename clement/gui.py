#!/usr/bin/env python
import sys
import os
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from . import res_styles
from .sem_controls import SEMControls
from .tem_controls import TEMControls
from .fm_controls import FMControls
from .fib_controls import FIBControls
from .gis_controls import GISControls
from .project import Project
from .popup import Merge, Scatter, Convergence, Peak_Params
from . import utils

warnings.simplefilter('ignore', category=FutureWarning)

def resource_path(rel_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, rel_path)


class GUI(QtWidgets.QMainWindow):
    def __init__(self, project_fname=None, no_restore=False):
        super(GUI, self).__init__()
        if not no_restore:
            self.settings = QtCore.QSettings('MPSD-CNI', 'CLEMGui', self)
        else:
            self.settings = QtCore.QSettings()

        self.colors_tmp = ['#ff0000', '#00ff00', '#0000ff', '#808080', '#808080']
        self.colors = self.settings.value('channel_colors', defaultValue=self.colors_tmp)
        #self.colors = ['#ff0000', '#00ff00', '#0000ff', '#808080', '#808080', '#808080']
        self._init_ui()
        if project_fname is not None:
            self.project._load_project(project_fname)

    def _init_ui(self):
        geom = self.settings.value('geometry')
        if geom is None:
            self.resize(1600, 800)
        else:
            self.setGeometry(geom)

        self.setWindowTitle('Clement')
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
        self.fm_stacked_imview.addWidget(self.fm_imview)
        splitter_images.addWidget(self.fm_stacked_imview)

        # -- EM Image view
        self.em_imview = QtWidgets.QStackedWidget()
        self.sem_imview = pg.ImageView()
        self.sem_imview.ui.roiBtn.hide()
        self.sem_imview.ui.menuBtn.hide()
        self.fib_imview = pg.ImageView()
        self.fib_imview.ui.roiBtn.hide()
        self.fib_imview.ui.menuBtn.hide()
        self.gis_imview = pg.ImageView()
        self.gis_imview.ui.roiBtn.hide()
        self.gis_imview.ui.menuBtn.hide()
        self.tem_imview = pg.ImageView()
        self.tem_imview.ui.roiBtn.hide()
        self.tem_imview.ui.menuBtn.hide()
        self.em_imview.addWidget(self.sem_imview)
        self.em_imview.addWidget(self.fib_imview)
        self.em_imview.addWidget(self.gis_imview)
        self.em_imview.addWidget(self.tem_imview)
        splitter_images.addWidget(self.em_imview)

        # Options
        options = QtWidgets.QHBoxLayout()
        options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(options)

        refine_options = QtWidgets.QHBoxLayout()
        refine_options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(refine_options)

        merge_options = QtWidgets.QHBoxLayout()
        merge_options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(merge_options)

        print_layout = QtWidgets.QHBoxLayout()
        print_layout.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(print_layout)
        self.print_label = QtWidgets.QLabel('')
        print_layout.addWidget(self.print_label)
        print_layout.addStretch(1)
        self.worker = utils.PrintGUI(self.print_label)
        self.workerThread = QtCore.QThread()
        self.workerThread.started.connect(self.worker.run)
        self.worker.moveToThread(self.workerThread)
        self.print = self.worker.print
        self.log = self.worker.log
        self.workerThread.start()

        vbox = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        tab_sem = QtWidgets.QWidget()
        tab_fib = QtWidgets.QWidget()
        tab_tem = QtWidgets.QWidget()
        tab_gis = QtWidgets.QWidget()
        self.tabs.resize(300, 200)
        self.tabs.addTab(tab_sem, 'SEM')
        self.tabs.addTab(tab_fib, 'FIB')
        self.tabs.addTab(tab_gis, 'FIB-GIS')
        self.tabs.addTab(tab_tem, 'TEM')
        vbox.addWidget(self.tabs)

        self.vbox_sem = QtWidgets.QVBoxLayout()
        self.vbox_fib = QtWidgets.QVBoxLayout()
        self.vbox_gis = QtWidgets.QVBoxLayout()
        self.vbox_tem = QtWidgets.QVBoxLayout()
        tab_sem.setLayout(self.vbox_sem)
        tab_fib.setLayout(self.vbox_fib)
        tab_gis.setLayout(self.vbox_gis)
        tab_tem.setLayout(self.vbox_tem)

        self.sem_controls = SEMControls(self.sem_imview, self.vbox_sem, self.print, self.log)
        self.sem_imview.getImageItem().getViewBox().sigRangeChanged.connect(self.sem_controls._couple_views)
        self.sem_controls.curr_folder = self.settings.value('sem_folder', defaultValue=os.getcwd())
        self.vbox_sem.addWidget(self.sem_controls)
        self.fib_controls = FIBControls(self.fib_imview, self.vbox_fib, self.sem_controls.ops, self.print, self.log)
        self.fib_controls.curr_folder = self.settings.value('fib_folder', defaultValue=os.getcwd())
        self.vbox_fib.addWidget(self.fib_controls)
        self.gis_controls = GISControls(self.gis_imview, self.vbox_gis, self.sem_controls.ops, self.fib_controls.ops, self.print, self.log)
        self.gis_controls.curr_folder = self.settings.value('fib_folder', defaultValue=os.getcwd())
        self.vbox_gis.addWidget(self.gis_controls)
        self.tem_controls = TEMControls(self.tem_imview, self.vbox_tem, self.print, self.log)
        self.tem_imview.getImageItem().getViewBox().sigRangeChanged.connect(self.tem_controls._couple_views)
        self.tem_controls.curr_folder = self.settings.value('tem_folder', defaultValue=os.getcwd())
        self.vbox_tem.addWidget(self.tem_controls)

        self.fm_controls = FMControls(self.fm_imview, self.colors, refine_options, merge_options, self.sem_controls, self.fib_controls,
                                      self.gis_controls, self.tem_controls, self.tabs.currentIndex(), self.print, self.log)
        self.fm_imview.getImageItem().getViewBox().sigRangeChanged.connect(self.fm_controls._couple_views)
        self.fm_controls.curr_folder = self.settings.value('fm_folder', defaultValue=os.getcwd())
        options.addWidget(self.fm_controls)
        options.addLayout(vbox)

        self.tabs.currentChanged.connect(self.change_tab)
        # Connect controllers
        self.fm_controls.err_plt_btn.clicked.connect(lambda: self._show_scatter(idx=self.tabs.currentIndex()))
        self.fm_controls.convergence_btn.clicked.connect(lambda: self._show_convergence(idx=self.tabs.currentIndex()))
        self.fm_controls.set_params_btn.clicked.connect(lambda: self._show_peak_params())
        self.sem_controls.other = self.fm_controls
        self.fib_controls.other = self.fm_controls
        self.gis_controls.other = self.fm_controls
        self.tem_controls.other = self.fm_controls
        self.fm_controls.other = self.sem_controls
        self.fm_controls.merge_btn.clicked.connect(self.merge)

        self.sem_popup = None
        self.fib_popup = None
        self.tem_popup = None
        self.scatter = None
        self.convergence = None
        self.peak_params = None
        self.project = Project(self.fm_controls, self.sem_controls, self.fib_controls, self.gis_controls, self.tem_controls, self, self.print, self.log)
        self.project._project_folder = self.settings.value('project_folder', defaultValue=os.getcwd())
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
        action = QtWidgets.QAction('Load FM', self)
        action.triggered.connect(self.fm_controls._load_fm_images)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Load SEM', self)
        action.triggered.connect(lambda idx: self.load_and_select_tab(idx=0))
        filemenu.addAction(action)
        action = QtWidgets.QAction('Load FIB', self)
        action.triggered.connect(lambda idx: self.load_and_select_tab(idx=1))
        filemenu.addAction(action)
        action = QtWidgets.QAction('Load FIB-GIS', self)
        action.triggered.connect(lambda idx: self.load_and_select_tab(idx=2))
        filemenu.addAction(action)
        action = QtWidgets.QAction('Load TEM', self)
        action.triggered.connect(lambda idx: self.load_and_select_tab(idx=3))
        filemenu.addAction(action)
        action = QtWidgets.QAction('Load project', self)
        action.triggered.connect(self._load_p)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Save project', self)
        action.triggered.connect(self._save_p)
        filemenu.addAction(action)
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

    def load_and_select_tab(self, idx):
        if idx == 0:
            self.sem_controls._load_mrc()
        elif idx == 1:
            self.fib_controls._load_mrc()
        elif idx == 2:
            self.gis_controls._load_mrc()
        else:
            self.tem_controls._load_mrc()
        self.tabs.setCurrentIndex(idx)

    @utils.wait_cursor('print')
    def change_tab(self, idx):
        show_grid = self.fm_controls.other.show_grid_btn.isChecked()
        self.fm_controls.other.show_grid_btn.setChecked(False)
        self.em_imview.setCurrentIndex(idx)
        self.fm_controls.select_tab(idx, show_grid)

    @utils.wait_cursor('print')
    def _show_scatter(self, idx):
        self.scatter = None
        if self.fm_controls.other._err is None:
            return
        if idx == 0:
            controls = self.sem_controls
        elif idx == 1:
            controls = self.fib_controls
        elif idx == 2:
            controls = self.gis_controls
        else:
            controls = self.tem_controls

        self.scatter = Scatter(self, controls, self.print)
        self.scatter.show()

    @utils.wait_cursor('print')
    def _show_convergence(self, idx):
        self.convergence = None
        if self.fm_controls.other._conv is None:
            return
        if len(self.fm_controls.other._conv) == 3:
            if idx == 0:
                controls = self.sem_controls
            elif idx == 1:
                controls = self.fib_controls
            elif idx == 2:
                controls = self.gis_controls
            else:
                controls = self.tem_controls

            self.convergence = Convergence(self, controls, self.print)
            self.convergence.show()
        else:
            self.print('To use this feature, you have to use at least 10 points for the refinement!')

    @utils.wait_cursor('print')
    def _show_peak_params(self, state=None):
        self.fm_controls.peak_btn.setChecked(False)
        if self.peak_params is None:
            self.peak_params = Peak_Params(self, self.fm_controls, self.print, self.log)
            self.fm_controls.peak_controls = self.peak_params
        self.peak_params.show()

    @utils.wait_cursor('print')
    def merge(self, project=None):
        self.fm = self.fm_controls.ops
        self.em = self.fm_controls.other.ops
        if self.tabs.currentIndex() == 0:
            self.em = self.sem_controls.ops
            ops = self.em
            popup = self.sem_popup
            controls = self.sem_controls
        elif self.tabs.currentIndex() == 1:
            ops = self.fib_controls.sem_ops
            popup = self.fib_popup
            controls = self.fib_controls
        elif self.tabs.currentIndex() == 2:
            ops = self.gis_controls.sem_ops
            popup = self.fib_popup
            controls = self.gis_controls
        else:
            ops = self.tem_controls.ops
            popup = self.tem_popup
            controls = self.tem_controls

        if self.fm is not None and self.em is not None:
            if self.fm_controls.tab_index == 1 and self.fib_controls.sem_ops.data is None:
                self.print('You have to calculate the FM to TEM/SEM correlation first!')
            else:
                if self.fm._tf_points is not None and (ops._tf_points is not None or ops._tf_points_region is not None):
                    condition = self.fm_controls.perform_merge()
                    #condition = controls.merge()
                    if condition:
                        if popup is not None:
                            self.fm_controls.poi_btn.setEnabled(True)
                            popup.close()
                        popup = Merge(self, self.print, self.log)
                        controls.popup = popup
                        self.project.merged[self.tabs.currentIndex()] = True
                        self.project.popup = popup
                        if self.project.load_merge:
                            self.project._load_merge(project)
                            self.project.load_merge = False
                        self.fm_controls.poi_btn.setChecked(False)
                        self.fm_controls.poi_btn.setEnabled(False)
                        popup.show()
                else:
                    self.print('You have to transform the FM and the TEM/SEM images first!')
        else:
            self.print('Select FM and EM data first!')

    @utils.wait_cursor('print')
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

            for imview in [self.fib_imview, self.sem_imview, self.fm_imview]:
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
        self.settings.setValue('fm_folder', self.fm_controls._curr_folder)
        self.settings.setValue('sem_folder', self.sem_controls._curr_folder)
        self.settings.setValue('tem_folder', self.tem_controls._curr_folder)
        self.settings.setValue('fib_folder', self.fib_controls._curr_folder)
        self.settings.setValue('project_folder', self.project._project_folder)
        event.accept()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Clement: GUI for Correlative Light and Electron Microscopy')
    parser.add_argument('-p', '--project_fname', help='Path to project .yml file')
    parser.add_argument('--no-restore', help='Do not restore QSettings from last time Clement closed', action='store_true')
    args, unknown_args = parser.parse_known_args()

    app = QtWidgets.QApplication(unknown_args)
    app.setStyle('fusion')
    gui = GUI(args.project_fname, args.no_restore)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
